/** Feature selection tab, against a mocked API. */
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { FeaturesTab } from '../features/analysis.jsx'

const feat = (name, mi_rel, over = {}) => ({ name, kind: 'numeric', missing_pct: 0, unique: 50, mi: mi_rel / 10, mi_rel, rank: null, constant: false, id_like: false, skew: 0.2, outlier_pct: 0, ...over })
const REPORT = {
  target: 'churn', task: 'classification', rows_sampled: 500,
  features: [feat('tenure', 1, { rank: 1 }), feat('monthly', 0.4, { rank: 2 }), feat('income', 0.01, { rank: 3, skew: 3.2, missing_pct: 12 }), feat('noise', 0.0, { rank: 4, outlier_pct: 9 }), feat('customer_id', 0, { kind: 'categorical', id_like: true })],
  redundant_pairs: [{ a: 'tenure', b: 'monthly', r: 0.93 }],
  auto_excluded: { id_like: ['customer_id'], constant: [] },
  notes: [],
}
const KINDS = { tenure: 'numeric', monthly: 'numeric', income: 'numeric', noise: 'numeric', customer_id: 'categorical', churn: 'categorical', signup: 'datetime' }

function mockApi(routes) {
  globalThis.fetch = vi.fn(async (url, init) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, '')
    const hit = Object.entries(routes).find(([k]) => path.startsWith(k))
    if (!hit) return new Response(JSON.stringify({ error_code: 'NOT_FOUND', message: 'nope' }), { status: 404 })
    const body = typeof hit[1] === 'function' ? hit[1](path, init && init.body ? JSON.parse(init.body) : null) : hit[1]
    return new Response(JSON.stringify(body), { status: 200 })
  })
}
const result = (steps, extra = {}) => ({ steps: steps.map((s) => ({ op: 'x', summary: s })), rows_before: 500, rows_after: 500, columns_before: 6, columns_after: 6, missing_total: 0, column_kinds: KINDS, ...extra })

beforeEach(() => localStorage.clear())
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('FeaturesTab', () => {
  const setup = (over = {}) => {
    mockApi({ '/api/v1/sessions/s1/features': REPORT })
    const onApply = vi.fn(async () => ({ ok: true, result: result(['done']) }))
    const onUndo = vi.fn(async () => {})
    render(<FeaturesTab id="s1" ver="1" target="churn" kinds={KINDS} missingTotal={0} onApply={onApply} onUndo={onUndo} {...over} />)
    return { onApply, onUndo }
  }

  it('asks for a dataset and then a target before it can rank anything', () => {
    mockApi({})
    const { rerender } = render(<FeaturesTab id={null} ver="0" target={null} kinds={{}} onApply={vi.fn()} onUndo={vi.fn()} />)
    expect(screen.getByText(/NO DATASET/)).toBeTruthy()
    rerender(<FeaturesTab id="s1" ver="0" target={null} kinds={{}} onApply={vi.fn()} onUndo={vi.fn()} />)
    expect(screen.getByTestId('features-needs-target')).toBeTruthy()
  })

  it('ranks features by relevance and flags ids, skew, outliers and missing values', async () => {
    setup()
    const table = await screen.findByTestId('features-table')
    const rows = within(table).getAllByRole('row').slice(1).map((r) => r.textContent)
    expect(rows[0]).toMatch(/tenure/)
    expect(within(screen.getByTestId('feat-customer_id')).getByText('ID')).toBeTruthy()
    expect(within(screen.getByTestId('feat-customer_id')).getByText('NOT SCORED')).toBeTruthy()
    expect(within(screen.getByTestId('feat-income')).getByText('SKEW +3.2')).toBeTruthy()
    expect(within(screen.getByTestId('feat-income')).getByText('12%')).toBeTruthy()
    expect(within(screen.getByTestId('feat-noise')).getByText('OUT')).toBeTruthy()
    expect(screen.getByText(/AUTO-EXCLUDED FROM TRAINING: customer_id/)).toBeTruthy()
    expect(screen.getByText(/REDUNDANT PAIRS: tenure \/ monthly \(\+0\.93\)/)).toBeTruthy()
  })

  it('drops the ticked columns in one call and reports what happened', async () => {
    const { onApply } = setup()
    await screen.findByTestId('features-table')
    expect(screen.getByTestId('feat-drop').disabled).toBe(true)
    fireEvent.click(screen.getByTestId('feat-pick-noise'))
    fireEvent.click(screen.getByTestId('feat-pick-income'))
    expect(screen.getByTestId('feat-drop').textContent).toBe('DROP SELECTED (2)')
    fireEvent.click(screen.getByTestId('feat-drop'))
    await waitFor(() => expect(onApply).toHaveBeenCalledWith([{ op: 'drop_columns', columns: ['noise', 'income'] }]))
    await waitFor(() => expect(screen.getByTestId('features-msg').textContent).toMatch(/DROPPED: done \/\/ ROWS 500 > 500 \/\/ COLUMNS 6 > 6/))
  })

  it('"select weak" ticks the low-relevance columns', async () => {
    setup()
    await screen.findByTestId('features-table')
    fireEvent.click(screen.getByTestId('feat-pick-weak'))
    expect(screen.getByTestId('feat-pick-noise').checked).toBe(true)
    expect(screen.getByTestId('feat-pick-tenure').checked).toBe(false)
  })

  it('has no new-column, suggestion or in-place transform controls', async () => {
    setup()
    await screen.findByTestId('features-table')
    expect(screen.queryByTestId('engineer-form')).toBeNull()
    expect(screen.queryByTestId('features-suggestions')).toBeNull()
    expect(screen.queryByTestId('inplace-form')).toBeNull()
  })

  it('shows a backend refusal in red and warns about new missing values', async () => {
    const onApply = vi.fn(async () => ({ ok: false, error: "Cannot drop every feature column." }))
    setup({ onApply, missingTotal: 14 })
    await screen.findByTestId('features-table')
    fireEvent.click(screen.getByTestId('feat-pick-income'))
    fireEvent.click(screen.getByTestId('feat-drop'))
    await waitFor(() => expect(screen.getByTestId('features-msg').textContent).toMatch(/Cannot drop every feature column/))
    expect(screen.getByText(/14 MISSING VALUES IN THE DATA/)).toBeTruthy()
  })

  it('undo restores the raw file', async () => {
    const { onUndo } = setup()
    await screen.findByTestId('features-table')
    fireEvent.click(screen.getByTestId('feat-undo'))
    await waitFor(() => expect(onUndo).toHaveBeenCalled())
    await waitFor(() => expect(screen.getByTestId('features-msg').textContent).toMatch(/RESTORED THE RAW FILE/))
  })
})

/**
 * Live end-to-end test: drives the real UI against a running backend (http://localhost:8000, or VITE_API_BASE_URL).
 * Skipped automatically when no backend answers, so `npm test` stays green offline.
 */
import React from 'react'
import fs from 'node:fs'
import path from 'node:path'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterAll, describe, expect, it } from 'vitest'
import App from '../App.jsx'

const BASE = process.env.VITE_API_BASE_URL || 'http://localhost:8000'
const live = await fetch(`${BASE}/health`).then((r) => r.ok).catch(() => false)
const csvPath = process.env.SAMPLE_CSV || path.resolve(process.cwd(), '../sample_churn.csv')

afterAll(cleanup)

const click = (id) => fireEvent.click(screen.getByTestId(id))
const byText = (re, tag = 'button') => [...document.querySelectorAll(tag)].find((e) => re.test(e.textContent.trim()))

describe.skipIf(!live || !fs.existsSync(csvPath))('live pipeline', () => {
  it('upload -> target -> impute -> train (tuned) -> summary, curves, importance, SHAP, tuning, runs', async () => {
    const csv = fs.readFileSync(csvPath, 'utf8').split('\n').slice(0, 1201).join('\n')
    render(<App />)
    await waitFor(() => expect(screen.getByText('API OK')).toBeTruthy(), { timeout: 10000 })

    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [new File([csv], 'churn_small.csv', { type: 'text/csv' })] } })
    await waitFor(() => expect(screen.getByTestId('file-summary').textContent).toMatch(/1,200 x/), { timeout: 15000 })

    // data preview
    click('p1tab-preview')
    await waitFor(() => expect(screen.getByTestId('preview-table')).toBeTruthy(), { timeout: 15000 })
    expect(within(screen.getByTestId('preview-table')).getAllByRole('row').length).toBeGreaterThan(5)
    click('p1tab-status')

    fireEvent.change(screen.getByTestId('target-select'), { target: { value: 'churn' } })
    click('set-target')
    await waitFor(() => expect(screen.getByTestId('task-chip').textContent).toBe('CLASSIFICATION'), { timeout: 15000 })

    // feature selection: rank, log-transform a column, drop a column; the toolbar's column list follows
    click('p1tab-features')
    await waitFor(() => expect(screen.getByTestId('features-table')).toBeTruthy(), { timeout: 20000 })
    expect(within(screen.getByTestId('features-table')).getAllByRole('row').length).toBeGreaterThan(5)
    expect(screen.getByTestId('feat-customer_id').textContent).toMatch(/ID/)
    fireEvent.change(screen.getByTestId('inplace-col'), { target: { value: 'monthly_charges' } })
    fireEvent.click(screen.getByTestId('inplace-apply'))
    await waitFor(() => expect(screen.getByTestId('features-msg').textContent).toMatch(/LOG: applied log1p/), { timeout: 20000 })
    fireEvent.click(await screen.findByTestId('feat-pick-support_calls', {}, { timeout: 20000 }))
    fireEvent.click(screen.getByTestId('feat-drop'))
    await waitFor(() => expect([...screen.getByTestId('target-select').options].map((o) => o.value)).not.toContain('support_calls'), { timeout: 20000 })
    click('p1tab-status')

    // training is refused while NaNs remain; impute fixes it
    await waitFor(() => expect(byText(/^CLEAN$/)).toBeTruthy(), { timeout: 15000 })
    click('run-pipeline')
    await waitFor(() => expect(screen.getByTestId('error-banner').textContent).toMatch(/missing values/i), { timeout: 15000 })
    fireEvent.click(byText(/^CLEAN$/))
    await waitFor(() => expect(screen.getByTestId('panel-01').textContent).toMatch(/MISSING \d+\s*(→|->)\s*0/), { timeout: 15000 })

    // quick run: LR + decision tree, tuning on with few trials
    fireEvent.click(screen.getByTestId('tab-REGISTRY'))
    for (const abbr of ['XGB', 'RFR', 'SVM', 'MLP']) {
      const box = screen.getByTestId(`reg-sel-${abbr}`)
      if (box.getAttribute('aria-checked') === 'true') fireEvent.click(box)
    }
    fireEvent.click(screen.getByTestId('reg-sel-DT'))
    fireEvent.click(screen.getByTestId('tab-LEADERBOARD'))
    click('opt-tune')
    fireEvent.change(screen.getByTestId('opt-trials'), { target: { value: '5' } })
    click('run-pipeline')
    await waitFor(() => expect(screen.getByTestId('lb-row-LR')).toBeTruthy(), { timeout: 120000 })
    await waitFor(() => expect(screen.getByTestId('lb-row-DT')).toBeTruthy(), { timeout: 120000 })
    await waitFor(() => expect(screen.getByTestId('run-pipeline').disabled).toBe(false), { timeout: 120000 })

    // summary names a recommendation with metrics
    click('p5tab-summary')
    await waitFor(() => expect(screen.getByTestId('summary-verdict')).toBeTruthy())
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/RECOMMENDED MODEL FOR \[churn\]/)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/PRECISION .* RECALL/)
    expect(within(screen.getByTestId('summary-table')).getAllByRole('row')).toHaveLength(3)

    // every model carries cross-validation, and the summary leads with it
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/5-FOLD CROSS-VALIDATION/)
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/CV F1 [0-9.]+/)
    expect(screen.getByTestId('cv-row-LR')).toBeTruthy()
    expect(screen.getByTestId('cv-row-DT')).toBeTruthy()
    expect(screen.getByTestId('summary-why').textContent).toMatch(/CROSS-VALIDATED F1/)
    expect(screen.getByTestId('summary-caveats').textContent).toMatch(/NEVER USED/)

    // ROC / PR / decision threshold
    click('lb-row-LR')
    click('p4tab-curves')
    await waitFor(() => expect(screen.getByTestId('curves')).toBeTruthy(), { timeout: 20000 })
    const before = screen.getByTestId('confusion').textContent
    fireEvent.change(screen.getByTestId('dthreshold-slider'), { target: { value: '0.1' } })
    expect(screen.getByTestId('dthreshold-value').textContent).toBe('0.10')
    expect(screen.getByTestId('confusion').textContent).not.toBe(before)

    // tuning, ROC overlay, importance, SHAP
    click('p4tab-tune')
    await waitFor(() => expect(screen.getByTestId('tuning')).toBeTruthy(), { timeout: 20000 })
    click('p5tab-roc')
    await waitFor(() => expect(screen.getByTestId('roc-overlay')).toBeTruthy(), { timeout: 30000 })
    click('p6tab-imp')
    await waitFor(() => expect(screen.getByTestId('importance')).toBeTruthy(), { timeout: 30000 })
    click('p6tab-shap')
    await waitFor(() => expect(screen.getByTestId('shap')).toBeTruthy(), { timeout: 90000 })
    expect(screen.getByTestId('swarm')).toBeTruthy()

    // saved runs
    click('p5tab-runs')
    await waitFor(() => expect(screen.getByTestId('runs-table')).toBeTruthy(), { timeout: 20000 })
  })
})

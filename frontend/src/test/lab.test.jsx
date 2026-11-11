import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App.jsx'
import ModelLab from '../features/lab.jsx'

const CATALOG = {
  models: [
    ['linear_regression', 'LNR', 'Linear regression', ['regression']],
    ['logistic_regression', 'LR', 'Logistic regression', ['classification']],
    ['knn', 'KNN', 'K-nearest neighbors', ['classification', 'regression']],
    ['xgboost', 'XGB', 'XGBoost', ['classification', 'regression']],
    ['decision_tree', 'DT', 'Decision tree', ['classification', 'regression']],
    ['random_forest', 'RFR', 'Random forest', ['classification', 'regression']],
    ['kmeans', 'KMN', 'K-means', ['clustering']],
    ['svm', 'SVM', 'Support vector machine', ['classification', 'regression']],
  ].map(([key, abbr, name, tasks]) => ({
    key, abbr, name, tasks, summary: `${name} summary`, scaled: true, task_note: null, default_sweep: ['p'],
    params: [{ name: 'p', label: 'P', kind: 'int', default: 3, help: 'help', lo: 1, hi: 10, step: 1, choices: [], when: {}, tasks: [], zero_means: null, sweep: [1, 2] }],
  })),
}

function mockFetch() {
  globalThis.fetch = vi.fn(async (url) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, '')
    if (path.startsWith('/health')) return new Response(JSON.stringify({ status: 'ok' }), { status: 200 })
    if (path.startsWith('/api/v1/lab/models')) return new Response(JSON.stringify(CATALOG), { status: 200 })
    if (path.startsWith('/api/v1/runs')) return new Response(JSON.stringify({ runs: [], total: 0 }), { status: 200 })
    if (path.startsWith('/api/v1/datasets')) return new Response(JSON.stringify({ datasets: [], total: 0 }), { status: 200 })
    return new Response(JSON.stringify({ error_code: 'NOT_FOUND', message: 'nope' }), { status: 404 })
  })
  globalThis.WebSocket = class { close() {} }
}

beforeEach(() => {
  localStorage.clear()
  window.history.replaceState(null, '', '/')
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('model lab', () => {
  it('switches between the dashboard and the lab and offers the eight models in one dropdown', async () => {
    mockFetch()
    render(<App />)
    expect(screen.getByTestId('panel-01')).toBeTruthy()
    fireEvent.click(screen.getByTestId('view-lab'))
    expect(window.location.hash).toBe('#lab')
    const select = await screen.findByTestId('lab-model')
    await waitFor(() => expect(select.options.length).toBe(8))
    expect([...select.options].map((o) => o.value)).toEqual(CATALOG.models.map((m) => m.key))
    expect(screen.getByTestId('lab-empty').textContent).toMatch(/UPLOAD A FILE/)
    // the dashboard stays mounted (hidden) so its state survives the switch
    expect(screen.getByTestId('panel-01')).toBeTruthy()
    fireEvent.click(screen.getByTestId('view-dashboard'))
    expect(window.location.hash).toBe('')
    expect(screen.queryByTestId('model-lab')).toBeNull()
  })

  it('opens straight into the lab from the #lab link', async () => {
    window.history.replaceState(null, '', '/#lab')
    mockFetch()
    render(<App />)
    expect(await screen.findByTestId('model-lab')).toBeTruthy()
  })

  it('downloads the shown model with the settings it was fitted with', async () => {
    mockFetch()
    const base = globalThis.fetch
    const FIT = {
      model: 'plain', abbr: 'PLN', task: 'regression', params: { p: 7 }, metric: 'R2', fit_seconds: 0.1, notes: [], charts: {},
      train: { r2: 0.9, adj_r2: 0.9, rmse: 1, mae: 1, target_std: 2 }, test: { r2: 0.8, adj_r2: 0.8, rmse: 1.5, mae: 1.2, target_std: 2 }, cv: null,
      data: { rows_total: 100, rows_used: 100, capped: false, train_rows: 80, test_rows: 20, features_available: ['a', 'b'], features_used: ['a', 'b'], class_labels: [] },
      performance: { scatter: [], qq: [], histogram: { edges: [0, 1], counts: [1] }, bias: 0, residual_std: 1, skew: 0 },
      summary: { verdict: 'good', title: 'Generalises well', lines: [{ label: 'WHAT', text: 'Plain predicting `y`' }, { label: 'NEXT', text: 'Looks solid.' }] },
    }
    globalThis.fetch = vi.fn(async (url, init) => {
      const path = String(url).replace(/^https?:\/\/[^/]+/, '')
      if (path.startsWith('/api/v1/lab/models')) {
        return new Response(JSON.stringify({ models: [{ ...CATALOG.models[3], key: 'plain', abbr: 'PLN', name: 'Plain' }] }), { status: 200 })
      }
      if (path.endsWith('/lab/plain/fit')) return new Response(JSON.stringify(FIT), { status: 200 })
      if (path.endsWith('/lab/plain/export')) {
        return new Response(new Blob(['zip']), { status: 200, headers: { 'Content-Type': 'application/zip', 'Content-Disposition': 'attachment; filename="d_y_xgboost_lab.zip"' } })
      }
      return base(url, init)
    })
    URL.createObjectURL = vi.fn(() => 'blob:x')
    URL.revokeObjectURL = vi.fn()
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(<ModelLab sessionId="s1" target="y" task="regression" dataKey="k" columns={['a', 'b', 'y']} />)
    expect(await screen.findByTestId('card-export')).toBeTruthy()
    expect(screen.getByTestId('lab-export-params').textContent).toContain('7')
    expect(screen.getByTestId('lab-summary-verdict').textContent).toBe('Generalises well')
    expect(screen.getByTestId('card-summary').textContent).toContain('Plain predicting y')
    fireEvent.click(screen.getByTestId('lab-export'))
    expect(await screen.findByTestId('lab-export-done')).toBeTruthy()
    const call = globalThis.fetch.mock.calls.find(([u]) => String(u).endsWith('/api/v1/sessions/s1/lab/plain/export'))
    expect(JSON.parse(call[1].body)).toEqual({ params: { p: 3 }, test_size: 0.2, scale: true, features: null, cv: true })
    expect(click).toHaveBeenCalled()
    expect(screen.getByTestId('lab-export-done').textContent).toContain('d_y_xgboost_lab.zip')
  })
})

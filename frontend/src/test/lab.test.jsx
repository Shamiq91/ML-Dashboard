import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App.jsx'

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
})

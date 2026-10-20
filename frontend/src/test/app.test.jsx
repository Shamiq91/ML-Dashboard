import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App.jsx'

function mockFetch(health = true) {
  globalThis.fetch = vi.fn(async (url) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, '')
    if (path.startsWith('/health') && health) return new Response(JSON.stringify({ status: 'ok' }), { status: 200 })
    if (path.startsWith('/api/v1/runs')) return new Response(JSON.stringify({ runs: [], total: 0 }), { status: 200 })
    if (path.startsWith('/api/v1/datasets')) return new Response(JSON.stringify({ datasets: [], total: 0 }), { status: 200 })
    return new Response(JSON.stringify({ error_code: 'NOT_FOUND', message: 'nope' }), { status: 404 })
  })
  globalThis.WebSocket = class { close() {} }
}

beforeEach(() => localStorage.clear())
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('empty dashboard', () => {
  it('shows the six numbered panels, the legend and a connected API chip', async () => {
    mockFetch()
    render(<App />)
    for (const id of ['panel-01', 'panel-02', 'panel-03', 'panel-04', 'panel-05', 'panel-06', 'panel-07']) expect(screen.getByTestId(id)).toBeTruthy()
    expect(screen.getByText('ML PIPELINE')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('API OK')).toBeTruthy())
    expect(screen.getAllByText(/NO DATASET/).length).toBeGreaterThanOrEqual(4)
  })

  it('reports an unreachable API', async () => {
    mockFetch(false)
    render(<App />)
    await waitFor(() => expect(screen.getByText('API DOWN')).toBeTruthy())
  })

  it('offers only classify / regress tasks (no time series) and disables run until a target is set', () => {
    mockFetch()
    render(<App />)
    const options = [...screen.getByTestId('opt-task').options].map((o) => o.value)
    expect(options).toEqual(['auto', 'classification', 'regression'])
    expect(screen.queryByTestId('time-select')).toBeNull()
    expect(screen.getByTestId('run-pipeline').disabled).toBe(true)
    expect(screen.getByTestId('set-target').disabled).toBe(true)
  })

  it('tuning is off by default and reveals a trials picker when switched on', () => {
    mockFetch()
    render(<App />)
    expect(screen.getByTestId('opt-tune').textContent).toBe('TUNE OFF')
    expect(screen.queryByTestId('opt-trials')).toBeNull()
    fireEvent.click(screen.getByTestId('opt-tune'))
    expect(screen.getByTestId('opt-tune').textContent).toBe('TUNE ON')
    expect(screen.getByTestId('opt-trials').value).toBe('15')
  })
})

describe('panel tabs', () => {
  it('each panel exposes its tabs and tabs fall back to a prompt when there is nothing to show', () => {
    mockFetch()
    render(<App />)
    const tabs = (id) => within(screen.getByTestId(id)).getAllByRole('button').map((b) => b.textContent)
    expect(tabs('panel-01')).toEqual(expect.arrayContaining(['STATUS', 'PREVIEW', 'FEATURES']))
    expect(tabs('panel-04')).toEqual(expect.arrayContaining(['TELEMETRY', 'ROC/PR/THR', 'TUNING']))
    expect(tabs('panel-05')).toEqual(expect.arrayContaining(['METRICS', 'SUMMARY', 'ROC', 'RUNS']))
    expect(tabs('panel-06')).toEqual(expect.arrayContaining(['CORRELATION', 'IMPORTANCE', 'SHAP']))
    expect(tabs('panel-06')).not.toContain('SERIES')
    expect(tabs('panel-06')).not.toContain('WHAT-IF')
    expect(tabs('panel-05')).not.toContain('FUTURE')

    fireEvent.click(screen.getByTestId('p5tab-summary'))
    expect(within(screen.getByTestId('panel-05')).getByText(/NO DATASET/)).toBeTruthy()
    fireEvent.click(screen.getByTestId('p6tab-shap'))
    expect(within(screen.getByTestId('panel-06')).getByText(/NO DATASET/)).toBeTruthy()
    fireEvent.click(screen.getByTestId('p1tab-features'))
    expect(within(screen.getByTestId('panel-01')).getByText(/NO DATASET/)).toBeTruthy()
    fireEvent.click(screen.getByTestId('p1tab-preview'))
    expect(within(screen.getByTestId('panel-01')).getByText(/NO DATASET/)).toBeTruthy()
  })

  it('the legend and thresholds menu open and close', () => {
    mockFetch()
    render(<App />)
    expect(screen.queryByText(/Optuna/)).toBeNull()
    fireEvent.click(screen.getByTestId('legend-toggle'))
    expect(screen.getByText(/Tuned with Optuna/)).toBeTruthy()
    fireEvent.click(screen.getByTestId('threshold-toggle'))
    expect(screen.getByTestId('threshold-menu')).toBeTruthy()
    fireEvent.change(screen.getByTestId('threshold-overfitGap'), { target: { value: '0.25' } })
    expect(screen.getByTestId('threshold-overfitGap-value').textContent).toBe('0.25')
  })
})

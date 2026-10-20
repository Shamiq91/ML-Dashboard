import { describe, expect, it } from 'vitest'
import { downsample, fCell, fPct, fSig, fSigned } from '../lib/fmt'
import { BACKEND_MODEL_KEYS, THRESH_DEFAULT, deriveBanner, fitFlags, modelAbbrs, pickHeatColumns, resizeTracks } from '../App.jsx'

describe('formatters', () => {
  it('formats numbers compactly and never throws on null', () => {
    expect(fSig(1234.567)).toBe('1,235')
    expect(fSig(0.12345)).toBe('0.123')
    expect(fSig(0.00001234)).toBe('1.2e-5')
    expect(fSig(null)).toBe('—')
    expect(fPct(0.1234)).toBe('12.3%')
    expect(fSigned(0.5)).toBe('+0.500')
    expect(fCell(3.14159265)).toBe('3.1416')
    expect(fCell(null)).toBe('')
  })
  it('downsamples keeping first and last', () => {
    const out = downsample(Array.from({ length: 100 }, (_, i) => i), 10)
    expect(out).toHaveLength(10)
    expect(out[0]).toBe(0)
    expect(out[9]).toBe(99)
  })
})

describe('model catalog', () => {
  it('maps every UI model to a backend key, including PyTorch MLP and TabNet', () => {
    expect(BACKEND_MODEL_KEYS.PTM).toBe('torch_mlp')
    expect(BACKEND_MODEL_KEYS.TAB).toBe('tabnet')
    expect(Object.keys(BACKEND_MODEL_KEYS)).toHaveLength(23)
  })
  it('hides classification-only models for a numeric target but keeps the neural nets', () => {
    const reg = modelAbbrs('regression')
    for (const a of ['LR', 'NB', 'LDA', 'QDA']) expect(reg).not.toContain(a)
    for (const a of ['XGB', 'LNR', 'TAB', 'PTM']) expect(reg).toContain(a)
    expect(modelAbbrs('classification')).toContain('LR')
  })
})

describe('overfit / underfit flags', () => {
  const clf = { task: 'classification', train_score: 0.9, train_test_gap: 0.3, gap_check: true, confusion_matrix: [[5, 1], [1, 5]] }
  it('flags a wide train-test gap as overfit using the threshold', () => {
    expect(fitFlags(clf, THRESH_DEFAULT).overfit).toBe(true)
    expect(fitFlags(clf, { ...THRESH_DEFAULT, overfitGap: 0.5 }).overfit).toBe(false)
  })
  it('skips the gap rule for models that memorise by design, but still honours curve overfit', () => {
    expect(fitFlags({ ...clf, gap_check: false }, THRESH_DEFAULT).overfit).toBe(false)
    expect(fitFlags({ ...clf, gap_check: false, curve_overfit: true }, THRESH_DEFAULT).overfit).toBe(true)
  })
  it('flags chance-level classification and R2 near zero as underfit', () => {
    expect(fitFlags({ ...clf, train_score: 0.52, train_test_gap: 0 }, THRESH_DEFAULT).underfit).toBe(true)
    expect(fitFlags({ task: 'regression', score_metric: 'r2', train_score: 0.05, train_test_gap: 0 }, THRESH_DEFAULT).underfit).toBe(true)
    expect(fitFlags({ task: 'regression', score_metric: 'r2', train_score: 0.8, train_test_gap: 0 }, THRESH_DEFAULT).underfit).toBe(false)
  })
})

describe('layout helpers', () => {
  it('resizeTracks moves a boundary but keeps every panel above the minimum', () => {
    const out = resizeTracks([1, 1, 1], 0, 0.5)
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    expect(out[0]).toBeCloseTo(0.5)
    expect(resizeTracks([1, 1, 1], 0, 0.0)[0]).toBeGreaterThanOrEqual(0.12)
    expect(resizeTracks([1, 1, 1], 0, 5)[1]).toBeGreaterThanOrEqual(0.12)
  })
  it('puts multicollinear columns first in the heatmap', () => {
    const corr = { columns: ['a', 'b', 'c', 'd'], multicollinearity_flags: [{ feature_a: 'c', feature_b: 'd', r: 0.9 }] }
    expect(pickHeatColumns(corr, 3)).toEqual(['c', 'd', 'a'])
  })
  it('summarises assumption results as a banner', () => {
    expect(deriveBanner(null).label).toBe('AWAITING DATA')
    expect(deriveBanner({ overall_status: 'GREEN' }).label).toBe('PIPELINE CLEAR')
    expect(deriveBanner({ overall_status: 'RED', independence: { status: 'RED' } }).label).toBe('PIPELINE FAIL')
  })
})

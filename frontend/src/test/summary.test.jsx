import React from 'react'
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { SummaryTab } from '../features/analysis.jsx'

afterEach(cleanup)

const T = { overfitGap: 0.1, underfitSkill: 0.2 }
const clf = (abbr, rank, over = {}) => ({
  abbr,
  flags: { overfit: false, underfit: false },
  row: { rank, abbr, name: `${abbr} model`, score: 0.5 - rank * 0.05, score_name: 'F1', accuracy: 0.75, precision: 0.6, recall: 0.3, roc_auc: 0.75, avg_precision: 0.5, train_score: 0.62, test_score: 0.61, train_test_gap: 0.01, train_time_sec: 0.5 * rank, gap_check: true, tuned: false, confusion_matrix: [[900, 100], [250, 150]], ...over },
})
const ctx = { reg: false, target: 'churn', thresholds: T, split: { train_rows: 5600, test_rows: 1400, stratified: true }, average: 'binary', failed: [], warnings: [] }

describe('SummaryTab', () => {
  it('asks for a run when there are no results', () => {
    render(<SummaryTab rows={[]} ctx={ctx} />)
    expect(screen.getByText(/run the pipeline/i)).toBeTruthy()
  })

  it('recommends the top-ranked model and explains it with its metrics', () => {
    render(<SummaryTab rows={[clf('LR', 1), clf('RFR', 2)]} ctx={ctx} />)
    const verdict = screen.getByTestId('summary-verdict').textContent
    expect(verdict).toMatch(/LR/)
    expect(verdict).toMatch(/CLASSIFICATION/)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/PRECISION 0\.600/)
    expect(why).toMatch(/RECALL 0\.300/)
    expect(why).toMatch(/ROC AUC 0\.750/)
    expect(why).toMatch(/NO OVERFIT OR UNDERFIT DETECTED/)
    expect(why).toMatch(/\(THE FASTEST\)/)
    expect(screen.queryByTestId('summary-skipped')).toBeNull()
  })

  it('warns when accuracy is inflated by a dominant class', () => {
    render(<SummaryTab rows={[clf('LR', 1)]} ctx={ctx} />)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/87% ONE CLASS|ONE CLASS/)
  })

  it('skips a higher-ranked model that is flagged overfit and says why', () => {
    const rows = [clf('MLP', 1, { train_test_gap: 0.4 }), clf('LR', 2)]
    rows[0].flags = { overfit: true, underfit: false }
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/RECOMMENDED MODEL[\s\S]*LRLR model/)
    expect(screen.getByTestId('summary-skipped').textContent).toMatch(/MLP .* FLAGGED OVERFIT, SO LR IS RECOMMENDED/)
  })

  it('lists every model with the best value in each column highlighted', () => {
    render(<SummaryTab rows={[clf('LR', 1), clf('RFR', 2, { roc_auc: 0.9 })]} ctx={ctx} />)
    const table = screen.getByTestId('summary-table')
    expect(within(table).getByTestId('summary-row-LR')).toBeTruthy()
    expect(within(table).getByTestId('summary-row-RFR')).toBeTruthy()
    const best = within(screen.getByTestId('summary-row-RFR')).getByText('0.900')
    expect(best.style.fontWeight).toBe('700')
  })

  it('uses R2 / RMSE / MAE columns and the target spread for a numeric target', () => {
    const reg = (abbr, rank, r2) => ({ abbr, flags: { overfit: false, underfit: false }, row: { rank, abbr, name: abbr, score: r2, score_name: 'R2', r2, rmse: 8.5, mae: 6, train_score: r2, test_score: r2, train_test_gap: 0.01, train_time_sec: 1, gap_check: true, tuned: true } })
    render(<SummaryTab rows={[reg('XGB', 1, 0.9), reg('LNR', 2, 0.6)]} ctx={{ ...ctx, reg: true, target: 'price', targetStd: 28, split: null }} />)
    const heads = [...screen.getByTestId('summary-table').querySelectorAll('th')].map((t) => t.textContent)
    expect(heads).toEqual(expect.arrayContaining(['R2', 'RMSE', 'MAE']))
    expect(heads).not.toContain('PREC')
    expect(screen.getByTestId('summary-why').textContent).toMatch(/EXPLAINS 90\.0% OF THE TARGET'S VARIANCE/)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/RMSE IS 30% OF THE TARGET'S STANDARD DEVIATION/)
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/TUNED/)
  })

  it('surfaces failed models, warnings and a small-test-set caution', () => {
    render(<SummaryTab rows={[clf('LR', 1)]} ctx={{ ...ctx, split: { train_rows: 80, test_rows: 20 }, failed: [{ model: 'tabnet' }], warnings: ['SVM SAMPLED 10000 ROWS'] }} />)
    const c = screen.getByTestId('summary-caveats').textContent
    expect(c).toMatch(/SMALL TEST SET/)
    expect(c).toMatch(/FAILED: tabnet/)
    expect(c).toMatch(/SVM SAMPLED 10000 ROWS/)
  })
})

/* ---------------------------------------------------------------- cross-validation */
import { recommend } from '../features/analysis.jsx'

const cvRep = (score, std = 0.02, folds = 5, over = {}) => ({
  folds, folds_ok: folds, rows_used: 4000, metric: 'f1', score, score_std: std, score_sem: std / Math.sqrt(folds), seconds: 1, note: null,
  metrics: {
    f1: { mean: score, std, min: score - std, max: score + std, folds: [] },
    precision: { mean: 0.55, std: 0.04, min: 0.5, max: 0.6, folds: [] },
    recall: { mean: 0.32, std: 0.02, min: 0.3, max: 0.34, folds: [] },
    roc_auc: { mean: 0.77, std: 0.02, min: 0.75, max: 0.79, folds: [] },
  },
  ...over,
})
const withCv = (abbr, rank, holdout, cvScore, std, over = {}) => {
  const m = clf(abbr, rank, { score: holdout, ...over })
  m.row.cv = cvRep(cvScore, std)
  m.row.cv_score = cvScore
  m.row.cv_std = std
  return m
}

describe('cross-validated recommendation', () => {
  it('prefers the higher cross-validated score over a lucky hold-out split', () => {
    const rows = [withCv('MLP', 1, 0.43, 0.38, 0.02), withCv('LR', 2, 0.40, 0.41, 0.02)]
    expect(recommend(rows).rec.abbr).toBe('LR')
    render(<SummaryTab rows={rows} ctx={ctx} />)
    const verdict = screen.getByTestId('summary-verdict').textContent
    expect(verdict).toMatch(/LRLR model/)
    expect(verdict).toMatch(/5-FOLD CROSS-VALIDATION/)
    expect(verdict).toMatch(/CV F1 0\.410/)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/SINGLE HOLD-OUT SPLIT RANKS MLP FIRST .* PREFERS LR/)
    expect(why).toMatch(/CROSS-VALIDATED F1 0\.410 \+\/- 0\.020 OVER 5 FOLDS \(HOLD-OUT 0\.400\)/)
  })

  it('says when the hold-out split and cross-validation agree', () => {
    render(<SummaryTab rows={[withCv('LR', 1, 0.42, 0.41, 0.02), withCv('RFR', 2, 0.30, 0.31, 0.02)]} ctx={ctx} />)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/HOLD-OUT SPLIT AND CROSS-VALIDATION AGREE/)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/OUTSIDE ONE STANDARD ERROR, A REAL LEAD/)
  })

  it('reports models within one standard error as tied and points at the fastest', () => {
    const rows = [withCv('LR', 1, 0.41, 0.410, 0.03, { train_time_sec: 2 }), withCv('XGB', 2, 0.40, 0.405, 0.03, { train_time_sec: 0.2 })]
    const pick = recommend(rows)
    expect(pick.rec.abbr).toBe('LR')
    expect(pick.tied.map((m) => m.abbr)).toEqual(['XGB'])
    render(<SummaryTab rows={rows} ctx={ctx} />)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/XGB IS STATISTICALLY TIED WITH LR/)
    expect(why).toMatch(/XGB \(0\.20s\) IS A FAIR CHOICE/)
    expect(why).toMatch(/WITHIN NOISE/)
  })

  it('flags an unstable model and never recommends an overfit one, even with the best CV score', () => {
    const over = withCv('XGB', 1, 0.42, 0.45, 0.02)
    over.flags = { overfit: true, underfit: false }
    const rows = [over, withCv('LR', 2, 0.40, 0.41, 0.09)]
    const pick = recommend(rows)
    expect(pick.rec.abbr).toBe('LR')
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-skipped').textContent).toMatch(/XGB \(F1 CV 0\.450\) RANKED HIGHER BUT IS FLAGGED OVERFIT, SO LR IS RECOMMENDED/)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/UNSTABLE/)
  })

  it('draws a cross-validation bar per model and shows mean +/- std in the table', () => {
    render(<SummaryTab rows={[withCv('LR', 1, 0.42, 0.41, 0.02), withCv('RFR', 2, 0.30, 0.31, 0.03)]} ctx={ctx} />)
    expect(within(screen.getByTestId('summary-cv')).getByTestId('cv-row-LR').textContent).toMatch(/0\.410 \+\/- 0\.020/)
    expect(screen.getByTestId('cv-row-RFR')).toBeTruthy()
    const heads = [...screen.getByTestId('summary-table').querySelectorAll('th')].map((t) => t.textContent)
    expect(heads).toEqual(expect.arrayContaining(['CV F1', 'F1']))
    expect(within(screen.getByTestId('summary-row-LR')).getByText('0.410+/-0.020')).toBeTruthy()
    expect(screen.getByTestId('summary-caveats').textContent).toMatch(/5 FOLDS ON 4,000 TRAINING ROWS ONLY/)
  })

  it('uses cross-validated precision / recall / AUC in the explanation', () => {
    render(<SummaryTab rows={[withCv('LR', 1, 0.42, 0.41, 0.02)]} ctx={ctx} />)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/PRECISION 0\.550 \+\/- 0\.040/)
    expect(why).toMatch(/RECALL 0\.320 \+\/- 0\.020/)
    expect(why).toMatch(/ROC AUC 0\.770 \+\/- 0\.020/)
  })

  it('falls back to the hold-out ranking, and says so, when a model has no cross-validation', () => {
    const rows = [clf('MLP', 1), withCv('LR', 2, 0.40, 0.41, 0.02)]
    const pick = recommend(rows)
    expect(pick.useCv).toBe(false)
    expect(pick.rec.abbr).toBe('MLP')
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-caveats').textContent).toMatch(/NO CROSS-VALIDATION FOR AT LEAST ONE MODEL/)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/RESTS ON ONE SPLIT/)
  })

  it('reports cross-validated regression metrics in the target units', () => {
    const reg = (abbr, rank, r2, cvR2) => ({
      abbr, flags: { overfit: false, underfit: false },
      row: { rank, abbr, name: abbr, score: r2, score_name: 'R2', r2, rmse: 8.5, mae: 6, train_score: r2, test_score: r2, train_test_gap: 0.01, train_time_sec: 1, gap_check: true, tuned: false,
        cv: { folds: 5, rows_used: 3000, metric: 'r2', score: cvR2, score_std: 0.01, score_sem: 0.0045, metrics: { r2: { mean: cvR2, std: 0.01, min: cvR2 - 0.01, max: cvR2 + 0.01, folds: [] }, rmse: { mean: 9.2, std: 0.4, min: 8.8, max: 9.6, folds: [] }, mae: { mean: 6.5, std: 0.3, min: 6.2, max: 6.8, folds: [] } } } },
    })
    render(<SummaryTab rows={[reg('XGB', 1, 0.9, 0.88), reg('LNR', 2, 0.6, 0.62)]} ctx={{ ...ctx, reg: true, target: 'price', targetStd: 28, split: null }} />)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/EXPLAINS 88\.0% OF THE TARGET'S VARIANCE \(R2 0\.880 \+\/- 0\.010\)/)
    expect(why).toMatch(/RMSE 9\.2 \+\/- 0\.4, MAE 6\.5 \+\/- 0\.3/)
    expect(why).toMatch(/RMSE IS 33% OF THE TARGET'S STANDARD DEVIATION/)
  })
})

/* ---------------------------------------------------------------- paired fold comparison */
import { clearlyBetter } from '../features/analysis.jsx'

const foldsOf = (abbr, rank, folds) => {
  const mean = folds.reduce((a, b) => a + b, 0) / folds.length
  const m = withCv(abbr, rank, mean, mean, 0.02)
  m.row.cv.metrics.f1.folds = folds
  return m
}

describe('paired fold comparison', () => {
  it('calls a small but consistent lead real, which the one-standard-error rule misses', () => {
    const a = foldsOf('LR', 1, [0.62, 0.55, 0.70, 0.60, 0.66])
    const b = foldsOf('XGB', 2, [0.61, 0.54, 0.69, 0.59, 0.65])
    expect(clearlyBetter(a.row, b.row)).toBe(true)
  })

  it('does not call a lead real when the fold differences flip sign', () => {
    const a = foldsOf('LR', 1, [0.62, 0.55, 0.70, 0.60, 0.66])
    const b = foldsOf('XGB', 2, [0.60, 0.57, 0.71, 0.58, 0.64])
    expect(clearlyBetter(a.row, b.row)).toBe(false)
  })

  it('returns null when fold scores are missing or the fold counts differ', () => {
    const a = foldsOf('LR', 1, [0.62, 0.55, 0.70, 0.60, 0.66])
    expect(clearlyBetter(a.row, withCv('X', 2, 0.5, 0.5, 0.02).row)).toBeNull()
    expect(clearlyBetter(a.row, foldsOf('Y', 2, [0.6, 0.5, 0.7]).row)).toBeNull()
  })

  it('uses the paired test for ties and says so', () => {
    const rows = [foldsOf('LR', 1, [0.62, 0.55, 0.70, 0.60, 0.66]), foldsOf('XGB', 2, [0.60, 0.57, 0.71, 0.58, 0.64])]
    const pick = recommend(rows)
    expect(pick.tied.map((m) => m.abbr)).toEqual(['XGB'])
    expect(pick.paired).toBe(true)
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/NO CLEAR DIFFERENCE FOLD BY FOLD/)
  })
})

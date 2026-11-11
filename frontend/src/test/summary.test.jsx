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

  it('names the model, rates it against a naive guess and says what it does in plain words', () => {
    render(<SummaryTab rows={[clf('LR', 1), clf('RFR', 2)]} ctx={{ ...ctx, positive: 'Yes' }} />)
    const verdict = screen.getByTestId('summary-verdict').textContent
    expect(verdict).toMatch(/RECOMMENDED MODEL FOR \[churn\]/)
    expect(verdict).toMatch(/LRLR model/)
    expect(verdict).toMatch(/test F1 0\.450/)
    // 400 of 1400 test rows are positive: flagging all of them scores F1 2p/(1+p) = 0.444, so 0.45 is barely better
    expect(screen.getByTestId('summary-level').textContent).toBe('WEAK')
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/Flagging every row as Yes would already score F1 0\.444\. This model scores 0\.450, which is barely any better/)
    expect(why).toMatch(/catches 30% of the actual Yes cases \(recall\), and 60% of the rows it flags are right \(precision\)/)
    expect(why).toMatch(/lower the threshold in panel 04/)
    expect(why).toMatch(/It generalises: train balanced accuracy 0\.620 vs test 0\.610/)
    expect(screen.getByTestId('summary-why').querySelectorAll('li').length).toBeLessThanOrEqual(4)
    expect(screen.queryByTestId('summary-skipped')).toBeNull()
  })

  it('passes over a higher-scoring overfit model and says so in one line', () => {
    const rows = [clf('MLP', 1, { train_test_gap: 0.4 }), clf('LR', 2)]
    rows[0].flags = { overfit: true, underfit: false }
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/LRLR model/)
    expect(screen.getByTestId('summary-skipped').textContent).toBe('MLP scored higher but is overfitting, so it is not recommended.')
    expect(within(screen.getByTestId('summary-row-MLP')).getByText('OVERFIT')).toBeTruthy()
    expect(within(screen.getByTestId('summary-row-LR')).getByText('RECOMMENDED')).toBeTruthy()
  })

  it('keeps the table to a few columns', () => {
    render(<SummaryTab rows={[clf('LR', 1), clf('RFR', 2)]} ctx={ctx} />)
    const heads = [...screen.getByTestId('summary-table').querySelectorAll('th')].map((t) => t.textContent)
    expect(heads).toEqual(['MODEL', 'TEST F1', 'FIT', 'TIME'])  // no CV column when no model has CV
  })

  it('rates a regression model by R2 and gives its typical error', () => {
    const reg = (abbr, rank, r2) => ({ abbr, flags: { overfit: false, underfit: false }, row: { rank, abbr, name: abbr, score: r2, score_name: 'R2', r2, rmse: 8.5, mae: 6, train_score: r2, test_score: r2, train_test_gap: 0.01, train_time_sec: 1, gap_check: true, tuned: true } })
    render(<SummaryTab rows={[reg('XGB', 1, 0.9), reg('LNR', 2, 0.6)]} ctx={{ ...ctx, reg: true, target: 'price', split: null }} />)
    const heads = [...screen.getByTestId('summary-table').querySelectorAll('th')].map((t) => t.textContent)
    expect(heads).toEqual(['MODEL', 'TEST R2', 'MAE', 'FIT', 'TIME'])
    expect(screen.getByTestId('summary-level').textContent).toBe('GOOD')
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/It explains 90% of the variation in price/)
    expect(why).toMatch(/typically off by 6 \(MAE\)/)
    expect(screen.getByTestId('summary-verdict').textContent).toMatch(/TUNED/)
  })

  it('surfaces failed models, warnings and a small-test-set caution', () => {
    render(<SummaryTab rows={[clf('LR', 1)]} ctx={{ ...ctx, split: { train_rows: 80, test_rows: 20 }, failed: [{ model: 'tabnet' }], warnings: ['SVM SAMPLED 10000 ROWS'] }} />)
    const c = screen.getByTestId('summary-caveats').textContent
    expect(c).toMatch(/Small test set/)
    expect(c).toMatch(/Failed: tabnet/)
    expect(c).toMatch(/SVM SAMPLED 10000 ROWS/)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/only model trained/)
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
    expect(verdict).toMatch(/CV F1 0\.410 \+\/- 0\.020 · test F1 0\.400/)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/Best cross-validated F1 of the 2 models\. It is clearly ahead of the next one, MLP \(0\.380\)/)
    expect(why).toMatch(/scores are steady across folds/)
  })

  it('calls models within noise just as good and points at the faster one', () => {
    const rows = [withCv('LR', 1, 0.41, 0.410, 0.03, { train_time_sec: 2 }), withCv('XGB', 2, 0.40, 0.405, 0.03, { train_time_sec: 0.2 })]
    const pick = recommend(rows)
    expect(pick.rec.abbr).toBe('LR')
    expect(pick.tied.map((m) => m.abbr)).toEqual(['XGB'])
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/XGB \(0\.405\) is about as good: the difference is within noise, and XGB trains faster \(0\.20s vs 2\.00s\)/)
  })

  it('does not call a model tied when its CV score is clearly lower, even if the paired test cannot separate them', () => {
    // the real churn case: DT 0.531 +/- 0.026 vs RFR 0.484 (5 folds): 0.047 apart, far outside one standard error (0.012)
    const a = foldsOf('DT', 1, [0.50, 0.56, 0.52, 0.55, 0.525])
    const b = foldsOf('RFR', 2, [0.47, 0.50, 0.48, 0.49, 0.48])
    b.row.cv.score = 0.484
    a.row.cv.score_std = 0.026
    a.row.cv.score_sem = 0.026 / Math.sqrt(5)
    const pick = recommend([a, b])
    expect(pick.rec.abbr).toBe('DT')
    expect(pick.tied).toEqual([])
    render(<SummaryTab rows={[a, b]} ctx={ctx} />)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/clearly ahead of the next one, RFR \(0\.484\)/)
  })

  it('compares with the next clean model, not a flagged one', () => {
    // the real churn run: RFR recommended; MLP has a higher CV score than LR but is overfit
    const mlp = withCv('MLP', 2, 0.514, 0.440, 0.039)
    mlp.flags = { overfit: true, underfit: false }
    const rows = [withCv('RFR', 1, 0.531, 0.484, 0.012), mlp, withCv('LR', 3, 0.447, 0.426, 0.027)]
    render(<SummaryTab rows={rows} ctx={ctx} />)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/clearly ahead of the next one, LR \(0\.426\)/)
    expect(why).not.toMatch(/MLP/)
  })

  it('names a coded positive class with its column', () => {
    render(<SummaryTab rows={[clf('LR', 1)]} ctx={{ ...ctx, target: 'Churn', positive: '1' }} />)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/Flagging every row as Churn = 1/)
    expect(why).toMatch(/catches 30% of the actual Churn = 1 cases/)
  })

  it('never recommends an overfit model, even with the best CV score, and flags unstable scores', () => {
    const over = withCv('XGB', 1, 0.42, 0.45, 0.02)
    over.flags = { overfit: true, underfit: false }
    const rows = [over, withCv('LR', 2, 0.40, 0.41, 0.09)]
    expect(recommend(rows).rec.abbr).toBe('LR')
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-skipped').textContent).toMatch(/XGB scored higher but is overfitting/)
    const why = screen.getByTestId('summary-why').textContent
    expect(why).toMatch(/Best cross-validated F1 of the 1 model \(out of 2\) that is not overfitting or underfitting/)
    expect(why).toMatch(/This model scores 0\.410 \(cross-validated\), which is no better/)
    expect(why).toMatch(/scores swing a lot between folds/)
  })

  it('shows CV and test side by side in the table and the folds in the footnote', () => {
    render(<SummaryTab rows={[withCv('LR', 1, 0.42, 0.41, 0.02), withCv('RFR', 2, 0.30, 0.31, 0.03)]} ctx={ctx} />)
    const heads = [...screen.getByTestId('summary-table').querySelectorAll('th')].map((t) => t.textContent)
    expect(heads).toEqual(['MODEL', 'CV F1', 'TEST F1', 'FIT', 'TIME'])
    expect(screen.getByTestId('summary-row-LR').textContent).toMatch(/0\.410 \+\/- 0\.020/)
    expect(screen.getByTestId('summary-caveats').textContent).toMatch(/Cross-validated with 5 folds on 4,000 training rows, then tested on 1,400 held-out rows/)
  })

  it('falls back to the hold-out ranking, and says so, when a model has no cross-validation', () => {
    const rows = [clf('MLP', 1), withCv('LR', 2, 0.40, 0.41, 0.02)]
    const pick = recommend(rows)
    expect(pick.useCv).toBe(false)
    expect(pick.rec.abbr).toBe('MLP')
    render(<SummaryTab rows={rows} ctx={ctx} />)
    expect(screen.getByTestId('summary-why').textContent).toMatch(/rests on a single test split/)
  })

  it('rates a regression model by its cross-validated R2', () => {
    const reg = (abbr, rank, r2, cvR2) => ({
      abbr, flags: { overfit: false, underfit: false },
      row: { rank, abbr, name: abbr, score: r2, score_name: 'R2', r2, rmse: 8.5, mae: 6, train_score: r2, test_score: r2, train_test_gap: 0.01, train_time_sec: 1, gap_check: true, tuned: false,
        cv: { folds: 5, rows_used: 3000, metric: 'r2', score: cvR2, score_std: 0.01, score_sem: 0.0045, metrics: { r2: { mean: cvR2, std: 0.01, min: cvR2 - 0.01, max: cvR2 + 0.01, folds: [] } } } },
    })
    render(<SummaryTab rows={[reg('XGB', 1, 0.9, 0.55), reg('LNR', 2, 0.6, 0.5)]} ctx={{ ...ctx, reg: true, target: 'price', split: null }} />)
    expect(screen.getByTestId('summary-level').textContent).toBe('FAIR')
    expect(screen.getByTestId('summary-why').textContent).toMatch(/It explains 55% of the variation in price/)
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
    expect(screen.getByTestId('summary-why').textContent).toMatch(/XGB \(0\.620\) is about as good/)
  })
})

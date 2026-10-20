/**
 * ML PIPELINE DASHBOARD
 * Single-file React + Tailwind client for the FastAPI backend.
 * Only static metadata is constant here (API URL, colors, registry catalog, legend text, thresholds).
 * Every number, status, curve and log line shown in panels 01-06 and the radio strip is a backend response.
 */
import React, { createContext, useContext, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'
import { CurvesTab, FeaturesTab, ImportanceTab, K as FK, PreviewTab, ResidualsTab, RocOverlayTab, RunsTab, ShapTab, SummaryTab, TabBar, TuningTab } from './features/analysis'

/* ============================================================================================
 * STATIC METADATA
 * ========================================================================================== */
export const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  'http://localhost:8000'
const WS_BASE_URL = API_BASE_URL.replace(/^http/i, 'ws')

const C = {
  purple: '#b800e6',
  green: '#00e600',
  yellow: '#e6e600',
  red: '#ff0000',
  loss: '#00d5ff',
  lossVal: '#7fe9ff',
  acc: '#ff2a2a',
  cpu: '#3b82f6',
  gpu: '#22c55e',
  pending: '#3f3f46',
}

const POLL_MS = 1000
const RADIO_SHOWN = 50
const RADIO_KEEP = 300
const HEAT_THRESHOLD = 0.85
const HEAT_DEFAULT_N = 6
const HEAT_MAX_N = 8
const SPLITS = [0.1, 0.2, 0.3]
const MODEL_PALETTE = ['#00d5ff', '#fb923c', '#a3e635', '#f472b6', '#facc15', '#94a3b8', '#2dd4bf', '#818cf8', '#fda4af', '#d4d4d8']

/** The ONE place that says which UI models the backend can run. */
export const BACKEND_MODEL_KEYS = {
  XGB: 'xgboost',
  LGB: 'lightgbm',
  RFR: 'random_forest',
  MLP: 'mlp',
  SVM: 'svm',
  LR: 'logistic_regression',
  CAT: 'catboost',
  ETC: 'extra_trees',
  GBM: 'gradient_boosting',
  ADA: 'adaboost',
  BAG: 'bagging',
  STK: 'stacking',
  VOT: 'voting',
  DT: 'decision_tree',
  KNN: 'knn',
  NB: 'naive_bayes',
  LDA: 'lda',
  RDG: 'ridge',
  SGD: 'sgd',
  QDA: 'qda',
  LNR: 'linear_regression',
  TAB: 'tabnet',
  PTM: 'torch_mlp',
}
const ALL_MODEL_KEYS = BACKEND_MODEL_KEYS
const KEY_TO_ABBR = Object.fromEntries(Object.entries(ALL_MODEL_KEYS).map(([a, k]) => [k, a]))
/** True for the numeric-target task. */
export const isRegTask = (task) => task === 'regression'
/** Models that only make sense with a categorical target. */
const CLASSIFICATION_ONLY = ['LR', 'NB', 'LDA', 'QDA']
/** Models pre-checked after a target is set; the rest are one click away in the registry. */
export const CORE_MODELS = ['XGB', 'RFR', 'SVM', 'MLP', 'LR', 'LNR']
export const modelAbbrs = (task) => Object.keys(BACKEND_MODEL_KEYS).filter((a) => !(isRegTask(task) && CLASSIFICATION_ONLY.includes(a)))
const defaultSelection = (task) => Object.fromEntries(modelAbbrs(task).map((x) => [x, CORE_MODELS.includes(x) && !(x === 'LNR' && !isRegTask(task))]))
/** Registry name/hyperparameters as they apply to the active task. */
const REG_OVERRIDES = {
  SGD: { name: 'SGD Regressor', hp: 'max_iter=1000, tol=1e-3' },
  STK: { hp: 'cv=3, base=RF+Ridge, final=Ridge' },
  VOT: { hp: 'base=Ridge+RF+KNN (averaged)' },
}
const modelInfo = (m, task) => {
  if (!isRegTask(task)) return m
  return { ...m, name: m.name.replace('Classifier', 'Regressor'), ...(REG_OVERRIDES[m.abbr] || {}) }
}

export const CATALOG = [
  { abbr: 'XGB', name: 'XGBoost', fam: 'ENS', hw: 'GPU', iterative: true, hp: 'n_estimators=200, max_depth=6, lr=0.1', est: '~3 s' },
  { abbr: 'LGB', name: 'LightGBM', fam: 'ENS', hw: 'CPU', iterative: true, hp: 'n_estimators=200, num_leaves=31, lr=0.1', est: '~2 s' },
  { abbr: 'RFR', name: 'Random Forest', fam: 'ENS', hw: 'CPU', hp: 'n_estimators=200, max_depth=12', est: '~4 s' },
  { abbr: 'MLP', name: 'Multi-Layer Perceptron', fam: 'NN', hw: 'GPU', iterative: true, hp: 'hidden=64-32, lr=0.001, max_epochs=200', est: '~10 s' },
  { abbr: 'SVM', name: 'Support Vector Machine', fam: 'SVM', hw: 'CPU', hp: 'kernel=rbf, C=1.0', est: '~12 s' },
  { abbr: 'LR', name: 'Logistic Regression', fam: 'LIN', hw: 'CPU', hp: 'C=1.0, max_iter=1000', est: '~1 s' },
  { abbr: 'CAT', name: 'CatBoost', fam: 'ENS', hw: 'GPU', hp: 'iterations=300, depth=6, lr=0.1', est: '~8 s' },
  { abbr: 'ETC', name: 'Extra Trees', fam: 'ENS', hw: 'CPU', hp: 'n_estimators=200, max_depth=12', est: '~3 s' },
  { abbr: 'GBM', name: 'Gradient Boosting', fam: 'ENS', hw: 'CPU', hp: 'n_estimators=150, max_depth=3, lr=0.1', est: '~9 s' },
  { abbr: 'ADA', name: 'AdaBoost', fam: 'ENS', hw: 'CPU', hp: 'n_estimators=100, lr=1.0', est: '~3 s' },
  { abbr: 'BAG', name: 'Bagging Classifier', fam: 'ENS', hw: 'CPU', hp: 'n_estimators=50', est: '~4 s' },
  { abbr: 'STK', name: 'Stacking Ensemble', fam: 'ENS', hw: 'CPU', hp: 'cv=3, base=RF+LR, final=LR', est: '~15 s' },
  { abbr: 'VOT', name: 'Voting Ensemble', fam: 'ENS', hw: 'CPU', hp: 'voting=soft, base=LR+RF+NB', est: '~5 s' },
  { abbr: 'DT', name: 'Decision Tree', fam: 'TREE', hw: 'CPU', hp: 'max_depth=6', est: '<1 s' },
  { abbr: 'KNN', name: 'K-Nearest Neighbors', fam: 'KNN', hw: 'CPU', hp: 'n_neighbors=15', est: '~1 s' },
  { abbr: 'NB', name: 'Gaussian Naive Bayes', fam: 'NB', hw: 'CPU', hp: 'var_smoothing=1e-9', est: '<1 s' },
  { abbr: 'LDA', name: 'Linear Discriminant Analysis', fam: 'LIN', hw: 'CPU', hp: 'solver=svd', est: '<1 s' },
  { abbr: 'RDG', name: 'Ridge Classifier', fam: 'LIN', hw: 'CPU', hp: 'alpha=1.0', est: '<1 s' },
  { abbr: 'SGD', name: 'SGD Classifier', fam: 'LIN', hw: 'CPU', hp: 'loss=log_loss, max_iter=1000', est: '~1 s' },
  { abbr: 'QDA', name: 'Quadratic Discriminant Analysis', fam: 'LIN', hw: 'CPU', hp: 'reg_param=0.1', est: '<1 s' },
  { abbr: 'LNR', name: 'Linear Regression', fam: 'LIN', hw: 'CPU', hp: 'fit_intercept=True', est: '<1 s' },
  { abbr: 'TAB', name: 'TabNet', fam: 'NN', hw: 'GPU', iterative: true, hp: 'n_d=8, n_a=8, n_steps=3, max_epochs=60', est: '~30 s' },
  { abbr: 'PTM', name: 'PyTorch MLP', fam: 'NN', hw: 'GPU', iterative: true, hp: 'hidden=64-32, lr=0.001, dropout=0.1, max_epochs=100', est: '~10 s' },
]
const CAT = Object.fromEntries(CATALOG.map((m) => [m.abbr, m]))
const FAMILIES = ['ALL', 'LIN', 'TREE', 'ENS', 'SVM', 'NN', 'NB', 'KNN']
const STATES = ['ALL', 'TRAINED', 'AVAIL', 'QUEUED', 'N/A']
const isSupported = (abbr, task = 'classification') => modelAbbrs(task).includes(abbr)
const naReason = (abbr, task = 'classification') =>
  isRegTask(task) && CLASSIFICATION_ONLY.includes(abbr)
      ? 'classification only // needs a categorical target'
      : 'not implemented in backend'
const abbrOfRow = (row) => KEY_TO_ABBR[row.model] || row.abbr || String(row.model || '?').toUpperCase()

const LEGEND = [
  { title: 'COLUMNS', items: [['POS', 'Position'], ['MDL', 'Model'], ['HW', 'Hardware'], ['F1', 'F1 Score'], ['GAP', 'Metric delta to Leader'], ['CTE', 'Clean/Train/Eval pipeline status']] },
  { title: 'HARDWARE', items: [[{ dot: C.cpu }, 'CPU'], [{ dot: C.gpu }, 'GPU']] },
  { title: 'CTE BLOCKS', items: [[{ sq: C.green }, 'Done'], [{ sq: C.yellow }, 'Running/Warn'], [{ sq: C.red }, 'Failed'], [{ sq: C.pending }, 'Pending']] },
  {
    title: 'MODELS',
    items: [['XGB', 'XGBoost'], ['LGB', 'LightGBM'], ['RFR', 'Random Forest'], ['SVM', 'Support Vector Machine'], ['LR', 'Logistic Regression'], ['LNR', 'Linear Regression'], ['MLP', 'Neural Net'], ['CAT', 'CatBoost'], ['ETC', 'Extra Trees'], ['GBM', 'Gradient Boosting'], ['ADA', 'AdaBoost'], ['BAG', 'Bagging'], ['STK', 'Stacking'], ['VOT', 'Voting'], ['DT', 'Decision Tree'], ['KNN', 'K-Nearest Neighbors'], ['NB', 'Naive Bayes'], ['LDA/QDA', 'Linear/Quadratic Discriminant'], ['RDG', 'Ridge'], ['SGD', 'Stochastic Gradient Descent'], ['TAB', 'TabNet'], ['PTM', 'PyTorch MLP']],
  },
  {
    title: 'REGISTRY',
    items: [['ABR', 'Abbreviation'], ['FAM', 'Family (LIN/TREE/ENS/SVM/NN/NB/KNN)'], ['SEL', 'Selected for race'], ['TRAINED', 'Has results'], ['AVAIL', 'Available, not run'], ['QUEUED', 'Waiting to run'], ['N/A', 'Unavailable']],
  },
  { title: 'STATS', items: [['STD', 'Std Deviation'], ['NULL', 'Missing count'], ['OVF', 'Overfit (val loss rising, or train-test gap > 0.10; gap not used for RFR/ETC/BAG)'], ['UNF', 'Underfit (chance-level, or R2 near 0, on training data)'], ['MARG', 'Marginal'], ['SKIP', 'Test skipped'], ['N/A', 'Not applicable (categorical target)'], ['ERR', 'Model failed'], ['DW', 'Durbin-Watson'], ['SW', 'Shapiro-Wilk'], ['BP', 'Breusch-Pagan']] },
  { title: 'ANALYSIS', items: [['TUN', 'Tuned with Optuna (best of N cross-validated trials)'], ['ROC', 'True-positive vs false-positive rate'], ['PR', 'Precision vs recall'], ['THR', 'Decision threshold: score at or above it flags the positive class'], ['SHAP', 'Per-feature push on a prediction'], ['PERM', 'Permutation importance: score drop when a column is shuffled'], ['ADF', 'Stationarity test (p < 0.05 = stationary)'], ['STL', 'Trend / seasonal / residual split']] },
  { title: 'REGRESSION', items: [['R2', 'Variance explained (1 = perfect, 0 = predicts the mean)'], ['RMSE', 'Root mean squared error, target units'], ['MAE', 'Mean absolute error, target units']] },
  { title: 'RADIO', items: [['T+MM:SS', 'Time since session start'], [{ text: '#e4e4e7', label: 'INFO' }, 'zinc/white'], [{ text: C.yellow, label: 'WARN' }, 'yellow'], [{ text: C.red, label: 'ERROR' }, 'red'], [{ text: C.green, label: 'SUCCESS' }, 'green']] },
  { title: 'COLORS', items: [[{ sq: C.purple }, 'Leader/Best'], [{ sq: C.green }, 'Pass/OK'], [{ sq: C.yellow }, 'Warning'], [{ sq: C.red }, 'Fail/Multicollinear (|r| > 0.85)']] },
]

/* ============================================================================================
 * API HELPER
 * ========================================================================================== */
export class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

function describeError(body, status) {
  const base = body && typeof body.message === 'string' && body.message ? body.message : `REQUEST FAILED (HTTP ${status})`
  if (body && Array.isArray(body.details) && body.error_code === 'VALIDATION_ERROR') {
    const extra = body.details
      .slice(0, 2)
      .map((d) => `${(d.loc || []).join('.')} ${d.msg || ''}`.trim())
      .join('; ')
    return extra ? `${base} ${extra}` : base
  }
  return base
}

async function request(path, { method = 'GET', json, form } = {}) {
  let res
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: json !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: json !== undefined ? JSON.stringify(json) : form,
    })
  } catch {
    throw new ApiError(0, 'NETWORK_ERROR', `CANNOT REACH API AT ${API_BASE_URL}`)
  }
  const text = await res.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  if (!res.ok) throw new ApiError(res.status, (body && body.error_code) || `HTTP_${res.status}`, describeError(body, res.status), body && body.details)
  return body
}

export const api = {
  get: (path) => request(path),
  post: (path, json) => request(path, { method: 'POST', json: json === undefined ? {} : json }),
  del: (path) => request(path, { method: 'DELETE' }),
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/api/v1/sessions/upload', { method: 'POST', form })
  },
}
const sx = (id) => `/api/v1/sessions/${id}`

/* ============================================================================================
 * FORMATTERS
 * ========================================================================================== */
const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const fmtNum = (v) => {
  if (!isNum(v)) return '--'
  if (Math.abs(v) >= 1000) return v.toFixed(1)
  return String(Number(v.toFixed(2)))
}
const fmt3 = (v) => (isNum(v) ? v.toFixed(3) : '---')
const fmtInt = (v) => (isNum(v) ? v.toLocaleString('en-US') : '--')
const fmtPct1 = (r) => (isNum(r) ? `${(r * 100).toFixed(1)}%` : '--')
const fmtP = (p) => (!isNum(p) ? '--' : p < 0.001 ? '<0.001' : p.toFixed(3))
const fmtBytes = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(2)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`)
const fmtMissingPct = (p) => (!isNum(p) ? '--' : p === 0 ? '0%' : p < 0.01 ? '<0.01%' : `${p.toFixed(2)}%`)
const stageLabel = (s) => (s ? String(s).replace(/_/g, ' ') : '--')
const levelColor = (l) => (l === 'WARN' ? C.yellow : l === 'ERROR' ? C.red : l === 'SUCCESS' ? C.green : '#e4e4e7')
const STATUS_META = {
  GREEN: { label: 'PASS', bg: C.green, fg: '#000' },
  YELLOW: { label: 'MARG', bg: C.yellow, fg: '#000' },
  RED: { label: 'FAIL', bg: C.red, fg: '#fff' },
}
const CTE_COLOR = { DONE: C.green, RUNNING: C.yellow, WARN: C.yellow, FAILED: C.red, PENDING: C.pending }
const CTE_NAMES = ['CLEAN', 'TRAIN', 'EVAL']

/* ============================================================================================
 * SHARED REDUCER STATE
 * ========================================================================================== */
const R0 = { status: 'idle', data: null, error: null }
const PANEL_KEYS = ['missing', 'balance', 'summary', 'corr', 'assump']
const idlePanels = () => Object.fromEntries(PANEL_KEYS.map((k) => [k, R0]))

const initialState = {
  api: 'unknown',
  radioConn: 'offline',
  uploading: false,
  file: null,
  session: null,
  stage: null,
  target: null,
  targetInfo: null,
  targetBusy: false,
  expired: false,
  banner: null,
  flash: 0,
  dataVer: 0,
  opts: { testSize: 0.2, scale: true, imbalance: 'none', avg: 'auto', task: 'auto', tune: false, trials: 15 },
  data: idlePanels(),
  assumpColumn: '',
  heatN: HEAT_DEFAULT_N,
  clean: { strategy: 'impute', method: 'median', k: 5, columns: [], busy: false, result: null, error: null },
  sel: {},
  job: null,
  board: null,
  selected: null,
  tele: {},
  radio: [],
}

const freshSession = (s) => ({ ...initialState, api: s.api, opts: s.opts, radioConn: s.radioConn })
const clearResults = { board: null, job: null, selected: null, tele: {} }
const boardRows = (board) => (board && board.leaderboard ? board.leaderboard : [])
/** Cache-buster for analysis endpoints: changes on every training job and every cleaning step. */
const verOf = (s) => `${s.job ? s.job.id : '0'}:${s.dataVer}:${s.stage || ''}:${s.data.missing.data ? s.data.missing.data.total_missing : ''}`
/** 'classification' or 'regression', as decided by the backend when the target was set. */
const taskOf = (s) => (s.targetInfo && s.targetInfo.task) || 'classification'
const scoreOf = (r) => (isNum(r.score) ? r.score : r.f1)
const scoreName = (r) => r.score_name || 'F1'
const trainScoreOf = (r) => (isNum(r.train_score) ? r.train_score : r.train_balanced_accuracy)
const testScoreOf = (r) => (isNum(r.test_score) ? r.test_score : r.test_balanced_accuracy)
const fitMetricLabel = (r) => (r.score_metric === 'r2' || isRegTask(r.task) ? 'R2' : 'BAL ACC')
const trainedSet = (s) => new Set(boardRows(s.board).map(abbrOfRow))

function applyRadioToCte(job, message) {
  if (!job || job.status !== 'RUNNING') return job
  let m = /^([A-Z0-9]+) TRAINING STARTED$/.exec(message)
  let next = null
  if (m && job.cte[m[1]]) next = [m[1], ['DONE', 'RUNNING', 'PENDING']]
  if (!next && (m = /^([A-Z0-9]+) EVALUATION COMPLETE/.exec(message)) && job.cte[m[1]]) next = [m[1], ['DONE', 'DONE', 'DONE']]
  if (!next && (m = /^([A-Z0-9]+) FAILED/.exec(message)) && job.cte[m[1]]) next = [m[1], ['DONE', 'FAILED', 'PENDING']]
  return next ? { ...job, cte: { ...job.cte, [next[0]]: next[1] } } : job
}

function reducer(s, a) {
  switch (a.type) {
    case 'API_STATUS':
      return { ...s, api: a.value }
    case 'RADIO_CONN':
      return { ...s, radioConn: a.value }
    case 'UPLOAD_START':
      return { ...s, uploading: true, banner: null, expired: false }
    case 'UPLOAD_OK':
      return { ...freshSession(s), session: { id: a.upload.session_id, upload: a.upload }, file: a.file, stage: 'UPLOADED' }
    case 'UPLOAD_FAIL':
      return { ...s, uploading: false, banner: { kind: 'error', text: a.text } }
    case 'SESSION_CLEAR':
      return freshSession(s)
    case 'EXPIRED':
      return { ...freshSession(s), expired: true }
    case 'STAGE':
      return { ...s, stage: a.stage }
    case 'TARGET_BUSY':
      return { ...s, targetBusy: a.value }
    case 'TARGET_OK':
      return {
        ...s,
        ...clearResults,
        targetBusy: false,
        target: a.target,
        targetInfo: a.info,
        data: idlePanels(),
        assumpColumn: '',
        clean: {
          ...s.clean,
          result: null,
          error: null,
          // a series is filled along time; other tasks never offer interpolation
          method: s.clean.method === 'interpolate' ? 'median' : s.clean.method,
        },
        sel: defaultSelection((a.info && a.info.task) || 'classification'),
      }
    case 'PANEL': {
      const prev = s.data[a.key]
      const next =
        a.status === 'loading'
          ? { status: 'loading', data: prev.data, error: null }
          : a.status === 'ok'
            ? { status: 'ok', data: a.data, error: null }
            : { status: 'error', data: null, error: a.error }
      return { ...s, data: { ...s.data, [a.key]: next } }
    }
    case 'ASSUMP_COLUMN':
      return { ...s, assumpColumn: a.value }
    case 'HEAT_N':
      return { ...s, heatN: a.value }
    case 'CLEAN_FIELD':
      return { ...s, clean: { ...s.clean, [a.k]: a.v } }
    case 'CLEAN_BUSY':
      return { ...s, clean: { ...s.clean, busy: true, error: null } }
    case 'CLEAN_OK': {
      // undoing (result null) restores the raw file, so also restore its column list after feature engineering
      const session = a.result === null && s.session && s.session.baseUpload ? { ...s.session, upload: s.session.baseUpload, baseUpload: null } : s.session
      return { ...s, ...clearResults, session, dataVer: a.result === null ? s.dataVer + 1 : s.dataVer, clean: { ...s.clean, busy: false, result: a.result, error: null } }
    }
    case 'CLEAN_FAIL':
      return { ...s, clean: { ...s.clean, busy: false, error: a.error } }
    case 'TRANSFORM_OK': {
      // column set / row count changed: refresh what the toolbar and panels read from the upload summary
      const up = s.session && s.session.upload
      const kinds = a.result.column_kinds
      const session = up
        ? { ...s.session, baseUpload: s.session.baseUpload || up, upload: { ...up, total_columns: Object.keys(kinds).length, total_rows: a.result.rows_after, column_kinds: kinds, dtypes: { ...kinds } } }
        : s.session
      return { ...s, ...clearResults, session, banner: null, dataVer: s.dataVer + 1 }
    }
    case 'BANNER':
      return { ...s, banner: { kind: a.kind || 'error', text: a.text } }
    case 'BANNER_CLEAR':
      return { ...s, banner: null }
    case 'FLASH':
      return { ...s, flash: s.flash + 1 }
    case 'OPT':
      return { ...s, opts: { ...s.opts, [a.k]: a.v } }
    case 'SEL_TOGGLE':
      return isSupported(a.abbr, taskOf(s)) ? { ...s, sel: { ...s.sel, [a.abbr]: !s.sel[a.abbr] } } : s
    case 'SEL_ALL_AVAIL': {
      const trained = trainedSet(s)
      const sel = { ...s.sel }
      modelAbbrs(taskOf(s)).forEach((x) => {
        if (!sel[x] && !trained.has(x)) sel[x] = true
      })
      return { ...s, sel }
    }
    case 'SEL_CLEAR_QUEUED': {
      const trained = trainedSet(s)
      const sel = { ...s.sel }
      Object.keys(sel).forEach((x) => {
        if (sel[x] && !trained.has(x)) sel[x] = false
      })
      return { ...s, sel }
    }
    case 'JOB_START':
      return {
        ...s,
        banner: null,
        job: { id: a.id, status: 'RUNNING', abbrs: a.abbrs, cte: Object.fromEntries(a.abbrs.map((x) => [x, ['DONE', 'PENDING', 'PENDING']])), error: null },
      }
    case 'JOB_POLL':
      return s.job && s.job.id === a.id && a.modelStatus
        ? { ...s, job: { ...s.job, cte: { ...s.job.cte, ...a.modelStatus } } }
        : s
    case 'JOB_DONE': {
      if (!s.job || s.job.id !== a.id) return s
      const board = a.result || s.board
      const rows = boardRows(board)
      const failed = new Set(((board && board.failed_models) || []).map((f) => KEY_TO_ABBR[f.model] || f.model))
      const sel = { ...s.sel }
      failed.forEach((x) => {
        sel[x] = false
      })
      const have = new Set(rows.map(abbrOfRow))
      const selected = s.selected && (have.has(s.selected) || failed.has(s.selected)) ? s.selected : rows.length ? abbrOfRow(rows[0]) : null
      return {
        ...s,
        board,
        sel,
        selected,
        job: { ...s.job, status: a.status, error: a.error || null, cte: { ...s.job.cte, ...(a.modelStatus || {}) } },
      }
    }
    case 'SELECT':
      return { ...s, selected: a.abbr }
    case 'TELE':
      return { ...s, tele: { ...s.tele, [a.abbr]: a.value } }
    case 'RADIO_LINE': {
      const key = `${a.line.ts_iso}|${a.line.message}`
      if (s.radio.some((l) => l.key === key)) return s
      const radio = [...s.radio, { ...a.line, key }].slice(-RADIO_KEEP)
      return { ...s, radio, job: applyRadioToCte(s.job, a.line.message) }
    }
    default:
      return s
  }
}

const Ctx = createContext(null)
const useCtx = () => useContext(Ctx)

/* Overfit / underfit thresholds. Defaults mirror backend config (overfit_max_gap, underfit_min_skill);
 * flags are re-evaluated in the browser from each row's train/test scores, so no retraining is needed. */
export const THRESH_DEFAULT = { overfitGap: 0.1, underfitSkill: 0.2 }
const THRESH_KEY = 'mldash.thresholds.v1'
const ThrCtx = createContext({ thresholds: THRESH_DEFAULT, setThresholds: () => {} })
const useThresholds = () => useContext(ThrCtx)

export function fitFlags(r, t) {
  const train = isNum(r.train_score) ? r.train_score : r.train_balanced_accuracy
  const regression = r.score_metric === 'r2' || isRegTask(r.task)
  // classification: balanced accuracy rescaled so chance = 0; regression: R2 already has the mean-predictor at 0
  const k = Array.isArray(r.confusion_matrix) ? Math.max(r.confusion_matrix.length, 2) : 2
  const chance = 1 / k
  const skill = !isNum(train) ? null : regression ? train : (train - chance) / (1 - chance)
  const gapOverfit = r.gap_check !== false && isNum(r.train_test_gap) && r.train_test_gap > t.overfitGap
  return {
    overfit: Boolean(r.curve_overfit) || gapOverfit,
    gapOverfit,
    underfit: skill !== null && skill < t.underfitSkill,
    skill,
  }
}

function loadThresholds() {
  try {
    const v = JSON.parse(localStorage.getItem(THRESH_KEY))
    if (v && isNum(v.overfitGap) && isNum(v.underfitSkill)) return v
  } catch {
    /* fall back to defaults */
  }
  return THRESH_DEFAULT
}

/** Derive the leaderboard rows (ranked results, then queued/pending, then ERR) from the shared state. */
function buildRows(s) {
  const results = boardRows(s.board).map((r) => ({ kind: 'result', abbr: abbrOfRow(r), row: r }))
  const have = new Set(results.map((r) => r.abbr))
  const failedList = ((s.board && s.board.failed_models) || []).map((f) => ({ abbr: KEY_TO_ABBR[f.model] || f.model, error: f.error }))
  const failedSet = new Set(failedList.map((f) => f.abbr))
  const running = s.job && s.job.status === 'RUNNING'
  const queuedAbbrs = modelAbbrs(taskOf(s)).filter(
    (x) => !have.has(x) && !failedSet.has(x) && (running ? s.job.abbrs.includes(x) : s.sel[x]),
  )
  const pending = queuedAbbrs.map((abbr) => ({ kind: 'pending', abbr }))
  const failed = failedList.filter((f) => !have.has(f.abbr)).map((f) => ({ kind: 'failed', abbr: f.abbr, error: f.error }))
  return [...results, ...pending, ...failed]
}

function cteFor(s, item) {
  const job = s.job
  if (job && job.cte && job.cte[item.abbr] && (job.status === 'RUNNING' || item.kind !== 'result')) return job.cte[item.abbr]
  if (item.kind === 'result') return item.row.cte && item.row.cte.length === 3 ? item.row.cte : ['DONE', 'DONE', 'DONE']
  if (item.kind === 'failed') return ['DONE', 'FAILED', 'PENDING']
  return ['PENDING', 'PENDING', 'PENDING']
}

function registryState(s, abbr) {
  if (!isSupported(abbr, taskOf(s))) return 'N/A'
  const running = s.job && s.job.status === 'RUNNING' && s.job.abbrs.includes(abbr)
  if (running) return 'QUEUED'
  if (trainedSet(s).has(abbr)) return 'TRAINED'
  return s.sel[abbr] ? 'QUEUED' : 'AVAIL'
}
const modelSwatch = (i) => (i < MODEL_PALETTE.length ? MODEL_PALETTE[i] : `hsl(${((i - MODEL_PALETTE.length) * 67 + 20) % 360} 65% 62%)`)
const modelColor = (abbr, leader) => (abbr === leader ? C.purple : modelSwatch(Math.max(0, Object.keys(ALL_MODEL_KEYS).indexOf(abbr))))

/* ============================================================================================
 * ACTIONS (REST orchestration) AND HOOKS
 * ========================================================================================== */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const PANEL_PATHS = { missing: '/missing', balance: '/class-balance', summary: '/eda/summary', corr: '/eda/correlation' }
const REGRESSION_BALANCE_PATH = '/target-summary'

function makeActions(getState, dispatch) {
  const sid = () => {
    const s = getState()
    return s.session ? s.session.id : null
  }
  const isExpired = (e) => e instanceof ApiError && e.status === 404 && e.code === 'SESSION_NOT_FOUND'
  const messageOf = (e) => (e instanceof ApiError ? e.message : 'UNEXPECTED ERROR')

  /** Route any error to the right UI: expired screen, red banner, or flashing prep strip. */
  const handle = (e) => {
    if (isExpired(e)) {
      dispatch({ type: 'EXPIRED' })
      return
    }
    dispatch({ type: 'BANNER', kind: 'error', text: messageOf(e) })
    if (e instanceof ApiError && e.status === 409 && e.code === 'NAN_REMAINING') dispatch({ type: 'FLASH' })
  }

  const checkHealth = async () => {
    try {
      await api.get('/health')
      dispatch({ type: 'API_STATUS', value: 'ok' })
    } catch {
      dispatch({ type: 'API_STATUS', value: 'down' })
    }
  }

  const refreshStage = async () => {
    const id = sid()
    if (!id) return
    try {
      const info = await api.get(sx(id))
      if (sid() === id) dispatch({ type: 'STAGE', stage: info.stage })
    } catch (e) {
      if (isExpired(e)) handle(e)
    }
  }

  const loadPanel = async (key, columnOverride, taskOverride) => {
    const id = sid()
    if (!id) return
    const s = getState()
    const column = columnOverride !== undefined ? columnOverride : s.assumpColumn
    let path = PANEL_PATHS[key]
    if (key === 'balance' && isRegTask(taskOverride || taskOf(s))) path = REGRESSION_BALANCE_PATH
    if (key === 'assump') path = `/assumptions${column ? `?column=${encodeURIComponent(column)}` : ''}`
    dispatch({ type: 'PANEL', key, status: 'loading' })
    try {
      const data = await api.get(`${sx(id)}${path}`)
      if (sid() === id) dispatch({ type: 'PANEL', key, status: 'ok', data })
    } catch (e) {
      if (sid() !== id) return
      if (isExpired(e)) handle(e)
      else dispatch({ type: 'PANEL', key, status: 'error', error: messageOf(e) })
    }
  }
  const loadPanels = (task) => Promise.allSettled(PANEL_KEYS.map((k) => loadPanel(k, undefined, task)))

  const uploadFile = async (file) => {
    if (!file) return
    if (!/\.(csv|parquet)$/i.test(file.name || '')) {
      dispatch({ type: 'BANNER', kind: 'error', text: 'ONLY .CSV OR .PARQUET FILES ARE ACCEPTED' })
      return
    }
    const old = sid()
    dispatch({ type: 'UPLOAD_START' })
    try {
      const upload = await api.upload(file)
      if (old) api.del(sx(old)).catch(() => {})
      dispatch({ type: 'UPLOAD_OK', upload, file: { name: file.name, size: file.size } })
      refreshStage()
    } catch (e) {
      dispatch({ type: 'UPLOAD_FAIL', text: messageOf(e) })
    }
  }

  const setTarget = async (column) => {
    const id = sid()
    if (!id || !column) return
    dispatch({ type: 'TARGET_BUSY', value: true })
    try {
      const o = getState().opts
      const info = await api.post(`${sx(id)}/target`, { target_column: column, task: o.task })
      if (sid() !== id) return
      dispatch({ type: 'TARGET_OK', target: column, info })
      refreshStage()
      loadPanels(info.task)
    } catch (e) {
      dispatch({ type: 'TARGET_BUSY', value: false })
      handle(e)
    }
  }

  const clean = async () => {
    const id = sid()
    const { clean: c } = getState()
    if (!id) return
    const body = { strategy: c.strategy, knn_neighbors: Number(c.k) || 5 }
    if (c.strategy === 'impute') body.method = c.method
    if (c.columns.length) body.columns = c.columns
    dispatch({ type: 'CLEAN_BUSY' })
    try {
      const result = await api.post(`${sx(id)}/clean`, body)
      if (sid() !== id) return
      dispatch({ type: 'CLEAN_OK', result })
      refreshStage()
      loadPanels()
    } catch (e) {
      dispatch({ type: 'CLEAN_FAIL', error: messageOf(e) })
      if (isExpired(e)) handle(e)
    }
  }

  /** Run feature-engineering / selection operations (POST /transform). Resolves to { ok, result | error } instead of throwing. */
  const transform = async (operations) => {
    const id = sid()
    if (!id) return { ok: false, error: 'NO DATASET' }
    try {
      const result = await api.post(`${sx(id)}/transform`, { operations })
      if (sid() !== id) return { ok: false, error: 'SESSION CHANGED' }
      dispatch({ type: 'TRANSFORM_OK', result })
      refreshStage()
      loadPanels()
      return { ok: true, result }
    } catch (e) {
      if (isExpired(e)) handle(e)
      return { ok: false, error: messageOf(e) }
    }
  }

  const undoClean = async () => {
    const id = sid()
    if (!id) return
    dispatch({ type: 'CLEAN_BUSY' })
    try {
      await api.post(`${sx(id)}/reset`)
      if (sid() !== id) return
      dispatch({ type: 'CLEAN_OK', result: null })
      refreshStage()
      loadPanels()
    } catch (e) {
      dispatch({ type: 'CLEAN_FAIL', error: messageOf(e) })
      if (isExpired(e)) handle(e)
    }
  }

  const loadTelemetry = async (abbr) => {
    const id = sid()
    if (!id || !abbr) return
    if (!CAT[abbr] || !isSupported(abbr, taskOf(getState()))) {
      dispatch({ type: 'TELE', abbr, value: { status: 'none' } })
      return
    }
    dispatch({ type: 'TELE', abbr, value: { status: 'loading' } })
    try {
      const data = await api.get(`${sx(id)}/telemetry/${ALL_MODEL_KEYS[abbr]}`)
      if (sid() === id) dispatch({ type: 'TELE', abbr, value: { status: 'ok', data } })
    } catch (e) {
      if (sid() !== id) return
      if (isExpired(e)) handle(e)
      else if (e instanceof ApiError && (e.status === 404 || e.status === 409)) dispatch({ type: 'TELE', abbr, value: { status: 'none' } })
      else dispatch({ type: 'TELE', abbr, value: { status: 'error', error: messageOf(e) } })
    }
  }

  const runPipeline = async () => {
    const id = sid()
    const s = getState()
    if (!id || !s.target || (s.job && s.job.status === 'RUNNING')) return
    const task = taskOf(s)
    const abbrs = modelAbbrs(task).filter((x) => s.sel[x])
    if (!abbrs.length) {
      dispatch({ type: 'BANNER', kind: 'error', text: 'SELECT AT LEAST ONE SUPPORTED MODEL IN THE REGISTRY' })
      return
    }
    const body = {
      models: abbrs.map((x) => ALL_MODEL_KEYS[x]),
      test_size: s.opts.testSize,
      scale_features: s.opts.scale,
      handle_imbalance: isRegTask(task) ? 'none' : s.opts.imbalance,
      tune: s.opts.tune,
      tune_trials: s.opts.trials,
    }
    if (!isRegTask(task) && s.opts.avg !== 'auto') body.average = s.opts.avg
    try {
      const started = await api.post(`${sx(id)}/train`, body)
      dispatch({ type: 'JOB_START', id: started.job_id, abbrs })
      for (;;) {
        await sleep(POLL_MS)
        const cur = getState()
        if (!cur.job || cur.job.id !== started.job_id || sid() !== id) return
        const job = await api.get(`${sx(id)}/train/${started.job_id}`)
        if (job.status === 'RUNNING') {
          dispatch({ type: 'JOB_POLL', id: started.job_id, modelStatus: job.model_status })
          continue
        }
        dispatch({ type: 'JOB_DONE', id: started.job_id, status: job.status, result: job.result, error: job.error, modelStatus: job.model_status })
        if (job.status === 'FAILED') dispatch({ type: 'BANNER', kind: 'error', text: job.error || 'TRAINING FAILED' })
        refreshStage()
        const rows = job.result ? job.result.leaderboard : []
        await Promise.all(rows.map(abbrOfRow).filter((x) => CAT[x] && ALL_MODEL_KEYS[x]).map(loadTelemetry))
        const sel = getState().selected
        if (sel && !getState().tele[sel]) loadTelemetry(sel)
        return
      }
    } catch (e) {
      handle(e)
      const cur = getState()
      if (cur.job && cur.job.status === 'RUNNING') dispatch({ type: 'JOB_DONE', id: cur.job.id, status: 'FAILED', result: null, error: messageOf(e) })
    }
  }

  const resetSession = async () => {
    const id = sid()
    dispatch({ type: 'SESSION_CLEAR' })
    if (id) api.del(sx(id)).catch(() => {})
  }

  return { checkHealth, refreshStage, loadPanel, loadPanels, uploadFile, setTarget, clean, transform, undoClean, loadTelemetry, runPipeline, resetSession }
}

/** Per-session WebSocket with capped exponential backoff; heartbeat/error frames are not log lines. */
function useRadio(sessionId, dispatch) {
  useEffect(() => {
    if (!sessionId || typeof WebSocket === 'undefined') return undefined
    let ws = null
    let timer = null
    let attempt = 0
    let stopped = false
    const connect = () => {
      ws = new WebSocket(`${WS_BASE_URL}/ws/radio?session_id=${sessionId}`)
      ws.onopen = () => {
        attempt = 0
        dispatch({ type: 'RADIO_CONN', value: 'live' })
      }
      ws.onmessage = (ev) => {
        let m
        try {
          m = JSON.parse(ev.data)
        } catch {
          return
        }
        if (!m || m.type === 'heartbeat') return
        if (m.type === 'error') {
          if (m.error_code === 'SESSION_NOT_FOUND') {
            stopped = true
            dispatch({ type: 'EXPIRED' })
          }
          return
        }
        if (typeof m.message === 'string') dispatch({ type: 'RADIO_LINE', line: m })
      }
      ws.onerror = () => {
        try {
          ws.close()
        } catch {
          /* already closed */
        }
      }
      ws.onclose = () => {
        dispatch({ type: 'RADIO_CONN', value: 'offline' })
        if (!stopped) {
          const delay = Math.min(15000, 1000 * 2 ** attempt)
          attempt += 1
          timer = setTimeout(connect, delay)
        }
      }
    }
    connect()
    return () => {
      stopped = true
      clearTimeout(timer)
      if (ws) {
        ws.onclose = null
        ws.close()
      }
      dispatch({ type: 'RADIO_CONN', value: 'offline' })
    }
  }, [sessionId, dispatch])
}

/* ============================================================================================
 * SMALL UI ATOMS
 * ========================================================================================== */
const ctl = 'h-6 bg-zinc-950 border border-zinc-700 text-white text-[10px] px-1 rounded-none disabled:opacity-40 disabled:cursor-not-allowed'
const lbl = 'text-[10px] uppercase text-zinc-400'

function Dot({ color, title }) {
  return <span title={title} className="inline-block w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
}
function Sq({ color, size = 8, title }) {
  return <span title={title} className="inline-block shrink-0" style={{ background: color, width: size, height: size }} />
}
function Btn({ children, onClick, disabled, bg = '#27272a', fg = '#fff', title, testid, className = '' }) {
  return (
    <button
      type="button"
      data-testid={testid}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`h-6 px-2 text-[10px] font-bold uppercase border border-zinc-700 rounded-none whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed ${className}`}
      style={disabled ? { background: '#27272a', color: '#a1a1aa' } : { background: bg, color: fg }}
    >
      {children}
    </button>
  )
}
function Chip({ text, color, fg = '#000', testid }) {
  return (
    <span data-testid={testid} className="inline-flex items-center h-5 px-1 text-[10px] font-bold uppercase whitespace-nowrap" style={{ background: color, color: fg }}>
      {text}
    </span>
  )
}
function Placeholder({ children, testid }) {
  return (
    <div data-testid={testid} className="h-full min-h-[80px] flex items-center justify-center text-center text-[10px] uppercase text-zinc-500 px-2 py-4">
      {children}
    </div>
  )
}
function Panel({ n, title, right, children, testid, className = '' }) {
  return (
    <section data-testid={testid} className={`min-h-[300px] lg:min-h-0 min-w-0 overflow-auto bg-zinc-900 ${className}`}>
      <header className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-zinc-800 bg-zinc-900 px-2 py-1 text-[10px] uppercase tracking-wide text-zinc-400">
        <span className="truncate">
          {n} // {title}
        </span>
        {right}
      </header>
      <div className="p-1 min-w-0">{children}</div>
    </section>
  )
}
const noData = (s) => !s.session
const needTarget = (s) => s.session && !s.target
function PanelGate({ s, children }) {
  if (noData(s)) return <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
  if (needTarget(s)) return <Placeholder testid="placeholder-target">SELECT A TARGET COLUMN TO LOAD THIS PANEL</Placeholder>
  return children
}
function PanelState({ r, children }) {
  if (r.status === 'loading' && !r.data) return <Placeholder>LOADING...</Placeholder>
  if (r.status === 'error') return <div className="p-2 text-[10px] uppercase" style={{ color: C.red }}>{r.error}</div>
  if (!r.data) return <Placeholder>AWAITING DATA</Placeholder>
  return children(r.data)
}

/* ============================================================================================
 * TOP BAR + RADIO STRIP
 * ========================================================================================== */
function ConnChip({ label, ok }) {
  return (
    <span className="inline-flex items-center gap-1 h-5 px-1 border border-zinc-700 text-[10px] font-bold uppercase text-white whitespace-nowrap">
      <Sq color={ok ? C.green : C.red} />
      {label}
    </span>
  )
}

function UploadZone() {
  const { state: s, actions } = useCtx()
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const pick = (f) => f && actions.uploadFile(f)
  const up = s.session && s.session.upload
  return (
    <div
      data-testid="dropzone"
      onDragOver={(e) => {
        e.preventDefault()
        setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        pick(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0])
      }}
      className={`flex items-center gap-2 h-7 px-1 border border-dashed min-w-0 ${drag ? 'border-white bg-zinc-800' : 'border-zinc-600'}`}
    >
      <Btn testid="choose-file" onClick={() => inputRef.current && inputRef.current.click()} disabled={s.uploading}>
        {s.uploading ? 'UPLOADING...' : 'CHOOSE FILE'}
      </Btn>
      <input
        ref={inputRef}
        data-testid="file-input"
        type="file"
        accept=".csv,.parquet"
        className="hidden"
        onChange={(e) => {
          pick(e.target.files && e.target.files[0])
          try {
            e.target.value = ''
          } catch {
            /* some environments forbid resetting file inputs */
          }
        }}
      />
      <span data-testid="file-summary" className="truncate text-[10px] text-white max-w-[260px]">
        {up ? `${s.file ? s.file.name : up.filename} | ${s.file ? fmtBytes(s.file.size) : ''} | ${fmtInt(up.total_rows)} x ${up.total_columns}` : <span className="text-zinc-500">DROP .CSV / .PARQUET HERE</span>}
      </span>
    </div>
  )
}

function TopBar() {
  const { state: s, dispatch, actions } = useCtx()
  const [pick, setPick] = useState('')
  const up = s.session && s.session.upload
  const columns = up ? Object.keys(up.dtypes) : []
  const running = s.job && s.job.status === 'RUNNING'
  const bal = s.data.balance.data
  return (
    <div className="shrink-0 border-b border-zinc-800 bg-zinc-950 px-2 py-1 flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="text-[11px] font-bold text-white whitespace-nowrap">ML PIPELINE</span>
      <ConnChip label={s.api === 'ok' ? 'API OK' : s.api === 'down' ? 'API DOWN' : 'API ...'} ok={s.api === 'ok'} />
      <ConnChip label={s.radioConn === 'live' ? 'RADIO LIVE' : 'RADIO OFFLINE'} ok={s.radioConn === 'live'} />
      <UploadZone />
      <div className="flex items-center gap-1">
        <span className={lbl}>TARGET</span>
        <select data-testid="target-select" className={`${ctl} max-w-[140px]`} disabled={!up} value={pick} onChange={(e) => setPick(e.target.value)}>
          <option value="">SELECT...</option>
          {columns.map((c) => (
            <option key={c} value={c}>
              {c} ({up.column_kinds[c]})
            </option>
          ))}
        </select>
        <select data-testid="opt-task" title="AUTO: a numeric target with many distinct values is regression, otherwise classification" className={ctl} value={s.opts.task} onChange={(e) => dispatch({ type: 'OPT', k: 'task', v: e.target.value })}>
          <option value="auto">TASK: AUTO</option>
          <option value="classification">CLASSIFY</option>
          <option value="regression">REGRESS</option>
        </select>
        <Btn testid="set-target" disabled={!up || !pick || s.targetBusy} onClick={() => actions.setTarget(pick)}>
          SET TARGET
        </Btn>
        {s.target && <span className="text-[10px] text-white truncate max-w-[100px]" title={s.target}>[{s.target}]</span>}
        {s.target && <Chip testid="task-chip" text={taskOf(s) === 'regression' ? 'REGRESSION' : 'CLASSIFICATION'} color={isRegTask(taskOf(s)) ? C.purple : '#e4e4e7'} fg={isRegTask(taskOf(s)) ? '#fff' : '#000'} />}
      </div>
      <Chip testid="stage-chip" text={`STAGE ${stageLabel(s.stage)}`} color={s.stage ? '#e4e4e7' : '#3f3f46'} fg={s.stage ? '#000' : '#a1a1aa'} />
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className={lbl}>TEST SPLIT</span>
        <select data-testid="opt-split" className={ctl} value={s.opts.testSize} onChange={(e) => dispatch({ type: 'OPT', k: 'testSize', v: Number(e.target.value) })}>
          {SPLITS.map((v) => (
            <option key={v} value={v}>
              {Math.round(v * 100)}%
            </option>
          ))}
        </select>
        <button type="button" data-testid="opt-scale" onClick={() => dispatch({ type: 'OPT', k: 'scale', v: !s.opts.scale })} className={`${ctl} font-bold`}>
          SCALE {s.opts.scale ? 'ON' : 'OFF'}
        </button>
        {!isRegTask(taskOf(s)) && <span className={lbl}>IMBALANCE</span>}
        {!isRegTask(taskOf(s)) && <select data-testid="opt-imbalance" className={ctl} value={s.opts.imbalance} onChange={(e) => dispatch({ type: 'OPT', k: 'imbalance', v: e.target.value })}>
          <option value="none">NONE</option>
          <option value="class_weight">CLASS_WEIGHT</option>
          <option value="smote">SMOTE</option>
        </select>}
        {!isRegTask(taskOf(s)) && bal && bal.imbalance_warning && s.opts.imbalance === 'none' && (
          <Chip testid="imbalance-hint" text="IMBALANCE STRICT // TRY CLASS_WEIGHT OR SMOTE" color={C.yellow} />
        )}
        {!isRegTask(taskOf(s)) && <span className={lbl}>AVG</span>}
        {!isRegTask(taskOf(s)) && <select data-testid="opt-avg" className={ctl} value={s.opts.avg} onChange={(e) => dispatch({ type: 'OPT', k: 'avg', v: e.target.value })}>
          <option value="auto">AUTO</option>
          <option value="binary">BINARY</option>
          <option value="macro">MACRO</option>
          <option value="weighted">WEIGHTED</option>
        </select>}
        <button type="button" data-testid="opt-tune" title="Optuna hyper-parameter search per model (cross-validated on training rows only; defaults are kept unless a trial beats them)" onClick={() => dispatch({ type: 'OPT', k: 'tune', v: !s.opts.tune })} className={`${ctl} font-bold`} style={s.opts.tune ? { background: C.purple, borderColor: C.purple } : undefined}>
          TUNE {s.opts.tune ? 'ON' : 'OFF'}
        </button>
        {s.opts.tune && (
          <select data-testid="opt-trials" title="Trials per model" className={ctl} value={s.opts.trials} onChange={(e) => dispatch({ type: 'OPT', k: 'trials', v: Number(e.target.value) })}>
            {[5, 10, 15, 25, 40].map((v) => <option key={v} value={v}>{v} TRIALS</option>)}
          </select>
        )}
      </div>
      <div className="flex items-center gap-1 ml-auto">
        <Btn testid="run-pipeline" bg={C.green} fg="#000" disabled={!s.target || running} onClick={actions.runPipeline}>
          {running ? 'RUNNING...' : 'RUN PIPELINE'}
        </Btn>
        <Btn testid="reset-session" bg={C.red} fg="#fff" disabled={!s.session} onClick={actions.resetSession}>
          RESET SESSION
        </Btn>
      </div>
      {(s.expired || s.banner || (s.targetInfo && s.targetInfo.warnings.length > 0)) && (
        <div className="basis-full flex flex-col gap-1">
          {s.expired && (
            <div data-testid="expired-banner" className="px-2 py-1 text-[10px] font-bold uppercase" style={{ background: C.red, color: '#fff' }}>
              SESSION EXPIRED // UPLOAD AGAIN
            </div>
          )}
          {s.banner && (
            <div data-testid="error-banner" className="flex items-start justify-between gap-2 px-2 py-1 text-[10px] font-bold uppercase" style={{ background: s.banner.kind === 'error' ? C.red : C.yellow, color: s.banner.kind === 'error' ? '#fff' : '#000' }}>
              <span className="min-w-0 break-words">{s.banner.text}</span>
              <button type="button" aria-label="dismiss" onClick={() => dispatch({ type: 'BANNER_CLEAR' })}>
                X
              </button>
            </div>
          )}
          {s.targetInfo &&
            s.targetInfo.warnings.map((w) => (
              <div key={w} data-testid="target-warning" className="px-2 py-1 text-[10px] font-bold uppercase break-words" style={{ background: C.yellow, color: '#000' }}>
                TARGET WARNING: {w}
              </div>
            ))}
        </div>
      )}
    </div>
  )
}

function RadioStrip() {
  const { state: s } = useCtx()
  const [open, setOpen] = useState(false)
  const boxRef = useRef(null)
  const shown = s.radio.slice(-RADIO_SHOWN)
  const last = shown[shown.length - 1]
  useEffect(() => {
    if (open && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [open, shown.length])
  const line = (l) => (
    <>
      <span className="text-zinc-400">{l.timestamp}</span> <span style={{ color: levelColor(l.level) }} className="font-bold">{l.level}</span> <span style={{ color: levelColor(l.level) }}>{l.message}</span>
    </>
  )
  return (
    <div className="relative shrink-0 border-b border-zinc-800 bg-zinc-950">
      <button type="button" data-testid="radio-strip" onClick={() => setOpen((o) => !o)} className="w-full flex items-center gap-2 px-2 h-6 text-left text-[10px] uppercase">
        <span className="text-zinc-400 shrink-0">RADIO</span>
        <span className="truncate min-w-0 flex-1">{last ? line(last) : <span className="text-zinc-500">NO RADIO TRAFFIC</span>}</span>
        <span className="text-zinc-400 shrink-0">{open ? '[-]' : '[+]'}</span>
      </button>
      {open && (
        <div ref={boxRef} data-testid="radio-log" className="absolute left-0 right-0 top-6 z-30 max-h-64 overflow-auto border border-zinc-700 bg-zinc-950 p-1 text-[10px] uppercase">
          {shown.length === 0 && <div className="text-zinc-500">NO RADIO TRAFFIC</div>}
          {shown.map((l) => (
            <div key={l.key} className="break-words">
              {line(l)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ============================================================================================
 * 01 // DATASET STATUS
 * ========================================================================================== */
function ColumnPicker({ columns, value, onChange }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <button type="button" data-testid="col-picker" className={`${ctl} max-w-[110px] truncate`} onClick={() => setOpen((o) => !o)}>
        COLS: {value.length ? value.length : 'ALL'} v
      </button>
      {open && (
        <div className="absolute left-0 top-6 z-20 max-h-40 w-44 overflow-auto border border-zinc-700 bg-zinc-950 p-1">
          <button type="button" className="text-[10px] uppercase text-zinc-400 underline" onClick={() => onChange([])}>
            ALL COLUMNS
          </button>
          {columns.map((c) => (
            <label key={c} className="flex items-center gap-1 text-[10px] text-white truncate">
              <input type="checkbox" checked={value.includes(c)} onChange={() => onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c])} />
              <span className="truncate">{c}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

function PrepStrip() {
  const { state: s, dispatch, actions } = useCtx()
  const ref = useRef(null)
  const [flashOn, setFlashOn] = useState(false)
  useEffect(() => {
    if (!s.flash) return undefined
    if (ref.current && ref.current.scrollIntoView) ref.current.scrollIntoView({ block: 'nearest' })
    setFlashOn(true)
    const t = setTimeout(() => setFlashOn(false), 2500)
    return () => clearTimeout(t)
  }, [s.flash])
  const c = s.clean
  const miss = s.data.missing.data
  const set = (k, v) => dispatch({ type: 'CLEAN_FIELD', k, v })
  const columns = s.session ? Object.keys(s.session.upload.dtypes) : []
  const canClean = Boolean(miss && miss.has_missing) && !c.busy
  const r = c.result
  return (
    <div ref={ref} data-testid="prep-strip" className={`mb-1 border p-1 ${flashOn ? 'border-[#e6e600] bg-zinc-800' : 'border-zinc-800'}`}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className={lbl}>STRATEGY</span>
        <select data-testid="clean-strategy" className={ctl} value={c.strategy} onChange={(e) => set('strategy', e.target.value)}>
          <option value="drop">DROP</option>
          <option value="impute">IMPUTE</option>
        </select>
        <span className={lbl}>METHOD</span>
        <select data-testid="clean-method" className={ctl} disabled={c.strategy === 'drop'} value={c.method} onChange={(e) => set('method', e.target.value)}>
          {['mean', 'median', 'mode', 'knn', ].map((m) => (
            <option key={m} value={m}>
              {m.toUpperCase()}
            </option>
          ))}
        </select>
        {c.strategy === 'impute' && c.method === 'knn' && (
          <>
            <span className={lbl}>K</span>
            <input data-testid="clean-k" type="number" min={1} max={100} className={`${ctl} w-12`} value={c.k} onChange={(e) => set('k', e.target.value)} />
          </>
        )}
        <ColumnPicker columns={columns} value={c.columns} onChange={(v) => set('columns', v)} />
        <Btn testid="clean-btn" bg={C.yellow} fg="#000" disabled={!canClean} title={miss && !miss.has_missing ? 'No missing values' : ''} onClick={actions.clean}>
          CLEAN
        </Btn>
        <Btn testid="undo-btn" disabled={c.busy} onClick={actions.undoClean}>
          UNDO (RESET)
        </Btn>
      </div>
      {c.error && <div data-testid="clean-error" className="mt-1 text-[10px] uppercase" style={{ color: C.red }}>{c.error}</div>}
      {r && (
        <div data-testid="clean-result" className="mt-1 text-[10px] uppercase text-white break-words">
          ROWS {fmtInt(r.rows_before)} -&gt; {fmtInt(r.rows_after)} | MISSING {fmtInt(r.missing_before)} -&gt; {fmtInt(r.missing_after)}
          {r.fallbacks.map((f) => (
            <div key={f} style={{ color: C.yellow }}>FALLBACK: {f}</div>
          ))}
          {r.warnings.map((w) => (
            <div key={w} style={{ color: C.yellow }}>WARN: {w}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function TargetChart({ bal }) {
  const classes = [...bal.classes].sort((a, b) => b.count - a.count)
  const counts = classes.map((x) => x.count)
  const ratio = counts.length > 1 && Math.min(...counts) > 0 ? Math.max(...counts) / Math.min(...counts) : null
  const max = Math.max(...classes.map((x) => x.ratio), 0.0001)
  return (
    <div data-testid="target-chart" className="border border-zinc-800 p-1">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <span className={lbl}>TARGET DISTRIBUTION [{bal.target}]</span>
        {bal.single_class ? (
          <Chip text={bal.message} color={C.yellow} />
        ) : bal.imbalance_warning ? (
          <>
            <Chip testid="imbalance-badge" text={`IMBALANCE ${bal.severity} // MIN ${fmtPct1(bal.minority_ratio)}`} color={C.yellow} />
            {ratio && <Chip text={`${ratio.toFixed(1)}:1`} color={C.yellow} />}
          </>
        ) : (
          <Chip testid="balanced-badge" text="BALANCED" color={C.green} />
        )}
      </div>
      {classes.map((k, i) => (
        <div key={k.label} className="mb-1">
          <div className="text-[10px] text-white truncate" title={k.label}>
            {k.label.toUpperCase()} {fmtPct1(k.ratio)} ({fmtInt(k.count)})
          </div>
          <div className="h-2 bg-zinc-800">
            <div className="h-2" style={{ width: `${(k.ratio / max) * 100}%`, background: i === 0 ? '#a1a1aa' : '#fb923c' }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function TargetSummary({ sum }) {
  const h = sum.histogram
  const max = Math.max(...h.counts, 1)
  const fmt = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : Number(v.toPrecision(4)).toString())
  const skewNote = Math.abs(sum.skew) > 1 ? (sum.skew > 0 ? 'RIGHT-SKEWED' : 'LEFT-SKEWED') : 'ROUGHLY SYMMETRIC'
  return (
    <div data-testid="target-summary" className="border border-zinc-800 p-1">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <span className={lbl}>TARGET DISTRIBUTION [{sum.target}]</span>
        <Chip text="REGRESSION" color={C.purple} fg="#fff" />
        <Chip text={skewNote} color={Math.abs(sum.skew) > 1 ? C.yellow : C.green} />
      </div>
      <div className="grid grid-cols-4 gap-x-2 gap-y-1 text-[10px] uppercase mb-1">
        {[['MEAN', sum.mean], ['STD', sum.std], ['MEDIAN', sum.median], ['SKEW', sum.skew], ['MIN', sum.min], ['Q1', sum.q1], ['Q3', sum.q3], ['MAX', sum.max]].map(([k, v]) => (
          <div key={k}>
            <div className="text-zinc-500">{k}</div>
            <div className="text-white">{fmt(v)}</div>
          </div>
        ))}
      </div>
      <div className="flex items-end gap-px h-12" title={`${fmtInt(sum.count)} values, ${fmtInt(sum.n_unique)} unique`}>
        {h.counts.map((c, i) => (
          <div key={i} className="flex-1 bg-zinc-400" style={{ height: `${Math.max((c / max) * 100, c > 0 ? 4 : 0)}%` }} title={`${fmt(h.edges[i])} to ${fmt(h.edges[i + 1])}: ${fmtInt(c)}`} />
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-zinc-500"><span>{fmt(h.edges[0])}</span><span>{fmt(h.edges[h.edges.length - 1])}</span></div>
    </div>
  )
}

/** Header tab strip shared by panels 01 / 04 / 05 / 06. */
function usePanelTabs(testid, tabs, initial) {
  const [tab, setTab] = useState(initial)
  const live = tabs.some(([v]) => v === tab) ? tab : initial
  return [live, <TabBar key="tabs" tabs={tabs} value={live} onChange={setTab} testid={testid} />]
}
const selectedModel = (s) => {
  const abbr = s.selected
  const row = abbr ? boardRows(s.board).find((r) => abbrOfRow(r) === abbr) : null
  return { abbr, row, key: row ? ALL_MODEL_KEYS[abbr] : null }
}

function Panel01() {
  const { state: s, actions } = useCtx()
  const [tab, tabs] = usePanelTabs('p1tab', [['status', 'STATUS'], ['preview', 'PREVIEW'], ['features', 'FEATURES']], 'status')
  const up = s.session && s.session.upload
  const miss = s.data.missing.data
  const rows = miss ? miss.rows : up && up.total_rows
  const cats = up ? Object.entries(up.column_kinds).filter(([, k]) => k !== 'numeric') : []
  return (
    <Panel n="01" title="DATASET STATUS" right={tabs} testid="panel-01">
      {tab === 'features' ? (
        <FeaturesTab id={s.session ? s.session.id : null} ver={verOf(s)} target={s.target} kinds={s.session ? s.session.upload.column_kinds : {}} missingTotal={s.data.missing.data ? s.data.missing.data.total_missing : 0} onApply={actions.transform} onUndo={actions.undoClean} />
      ) : tab === 'preview' ? (
        noData(s) ? <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder> : <PreviewTab id={s.session.id} ver={verOf(s)} />
      ) : noData(s) ? (
        <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-px bg-zinc-800 mb-1" data-testid="summary-line">
            {[
              ['ROWS', fmtInt(rows), '#fff'],
              ['COLUMNS', fmtInt(up.total_columns), '#fff'],
              ['MEMORY', `${up.memory_usage_mb.toFixed(3)} MB`, '#fff'],
              ['MISSING', miss ? fmtMissingPct(miss.total_percent) : '--', miss && miss.total_missing > 0 ? C.yellow : '#fff'],
            ].map(([k, v, color]) => (
              <div key={k} className="bg-zinc-900 p-1 min-w-0">
                <div className={lbl}>{k}</div>
                <div className="text-xs truncate" style={{ color }}>{v}</div>
              </div>
            ))}
          </div>
          {needTarget(s) ? (
            <Placeholder testid="placeholder-target">SELECT A TARGET COLUMN TO LOAD STATISTICS, PREP AND CLASS BALANCE</Placeholder>
          ) : (
            <>
              <PrepStrip />
              <PanelState r={s.data.summary}>
                {(sum) => {
                  const entries = Object.entries(sum.features)
                  const grid = 'grid grid-cols-[1fr_52px_52px_52px_52px_36px] gap-x-1 text-[10px] items-center'
                  return (
                    <div data-testid="stats-table" className="mb-1 border border-zinc-800">
                      <div className={`${grid} px-1 text-zinc-400 uppercase border-b border-zinc-800`}>
                        <span>FEATURE</span>
                        {['MEAN', 'STD', 'MIN', 'MAX', 'NULL'].map((h) => (
                          <span key={h} className="text-right">{h}</span>
                        ))}
                      </div>
                      {entries.length === 0 && <div className="p-1 text-zinc-500 uppercase">NO NUMERIC FEATURES</div>}
                      {entries.map(([name, f]) => {
                        const nulls = miss && miss.columns[name] ? miss.columns[name].count : f.null_count
                        return (
                          <div key={name} className={`${grid} px-1 text-white hover:bg-zinc-800`}>
                            <span className="truncate" title={f.constant ? `${name} (constant)` : name}>{name}</span>
                            <span className="text-right">{fmtNum(f.mean)}</span>
                            <span className="text-right">{f.std === 0 ? '--' : fmtNum(f.std)}</span>
                            <span className="text-right">{fmtNum(f.min)}</span>
                            <span className="text-right">{fmtNum(f.max)}</span>
                            <span className="text-right" style={nulls > 0 ? { color: C.yellow } : undefined}>{fmtInt(nulls)}</span>
                          </div>
                        )
                      })}
                    </div>
                  )
                }}
              </PanelState>
              <div data-testid="cat-line" className="mb-1 text-[10px] uppercase text-zinc-400 truncate" title={cats.map(([n]) => n).join(', ')}>
                CATEGORICAL: {cats.length} {cats.length ? `(${cats.map(([n]) => n).join(', ')})` : ''}
              </div>
              <PanelState r={s.data.balance}>{(bal) => (isRegTask(taskOf(s)) ? <TargetSummary sum={bal} /> : <TargetChart bal={bal} />)}</PanelState>
            </>
          )}
        </>
      )}
    </Panel>
  )
}

/* ============================================================================================
 * 02 // ASSUMPTION CHECK
 * ========================================================================================== */
const TESTS = [
  { key: 'homoscedasticity', label: 'HOMOSCEDASTICITY', short: 'BP', warn: 'HETEROSKEDASTICITY' },
  { key: 'normality', label: 'NORMALITY', short: 'SW', warn: 'NORMALITY' },
  { key: 'independence', label: 'INDEPENDENCE', short: 'DW', warn: 'INDEPENDENCE' },
]
const ROW_ORDER = ['normality', 'independence', 'homoscedasticity']

export function deriveBanner(a) {
  if (!a) return { label: 'AWAITING DATA', bg: '#3f3f46', fg: '#fff' }
  const na = TESTS.filter((t) => a[t.key] && a[t.key].result === 'N/A').map((t) => t.short)
  if (a.overall_status === 'GREEN') return { label: 'PIPELINE CLEAR', extra: na.length ? `${na.join(' / ')} N/A // CATEGORICAL TARGET` : undefined, bg: C.green, fg: '#000' }
  if (a.overall_status === 'RED') {
    const failing = TESTS.filter((t) => a[t.key] && a[t.key].status === 'RED').map((t) => t.warn)
    return { label: 'PIPELINE FAIL', extra: failing.join(' / '), bg: C.red, fg: '#fff' }
  }
  const worst = TESTS.find((t) => a[t.key] && a[t.key].status === 'YELLOW')
  return { label: `${worst ? worst.warn : 'ASSUMPTION'} WARNING`, bg: C.yellow, fg: '#000' }
}

function assumptionValue(key, t) {
  if (t.result === 'SKIPPED') return 'SKIPPED'
  if (t.result === 'N/A') return 'CATEGORICAL TARGET'
  if (key === 'normality') return `p=${fmtP(t.p_value)}`
  if (key === 'independence') return `DW=${isNum(t.statistic) ? t.statistic.toFixed(2) : '--'}`
  return `p=${fmtP(t.lm_p_value)}`
}

function Panel02() {
  const { state: s, dispatch, actions } = useCtx()
  const [open, setOpen] = useState(null)
  const a = s.data.assump.data
  const banner = deriveBanner(a)
  const numeric = s.data.summary.data ? Object.keys(s.data.summary.data.features) : []
  return (
    <Panel n="02" title="ASSUMPTION CHECK" testid="panel-02">
      <PanelGate s={s}>
        <div data-testid="assump-banner" className="px-2 py-1 mb-1 text-xs font-bold uppercase" style={{ background: banner.bg, color: banner.fg }}>
          {banner.label}
          {banner.extra ? <span className="ml-2 text-[10px]">[{banner.extra}]</span> : null}
        </div>
        <PanelState r={s.data.assump}>
          {(data) => (
            <div data-testid="assump-rows">
              {ROW_ORDER.map((key) => {
                const t = data[key]
                const skipped = t.result === 'SKIPPED'
                const na = t.result === 'N/A'
                const meta = skipped ? { label: 'SKIP', bg: C.yellow, fg: '#000' } : na ? { label: 'N/A', bg: C.pending, fg: '#d4d4d8' } : STATUS_META[t.status]
                const tag = TESTS.find((x) => x.key === key)
                return (
                  <div key={key} className="border-b border-zinc-800">
                    <button type="button" title={t.detail} onClick={() => setOpen(open === key ? null : key)} className="w-full grid grid-cols-[1fr_16px_40px_1fr] items-center gap-x-1 px-1 py-1 text-left text-[10px] hover:bg-zinc-800">
                      <span className="truncate text-zinc-400 uppercase">{tag.label} <span className="text-zinc-500">({tag.short})</span></span>
                      <Sq color={meta.bg} size={10} />
                      <span className="font-bold px-1 text-center" style={{ background: meta.bg, color: meta.fg }}>{meta.label}</span>
                      <span className="truncate text-white text-right">{assumptionValue(key, t)}</span>
                    </button>
                    {(open === key || skipped) && (
                      <div className="px-1 pb-1 text-[10px] text-zinc-300 break-words" data-testid={`detail-${key}`}>{t.detail}</div>
                    )}
                  </div>
                )
              })}
              <div className="mt-1 text-[9px] uppercase text-zinc-500">
                RESIDUAL BASIS // OLS OF TARGET ON NUMERIC FEATURES // N={fmtInt(data.n_observations)}
                {data.dropped_columns && (data.dropped_columns.constant.length > 0 || data.dropped_columns.collinear.length > 0)
                  ? ` // DROPPED: ${[...data.dropped_columns.constant, ...data.dropped_columns.collinear].join(', ')}`
                  : ''}
              </div>
            </div>
          )}
        </PanelState>
        {numeric.length > 0 && (
          <div className="mt-1 flex items-center gap-1">
            <span className={lbl}>NORMALITY ON</span>
            <select
              data-testid="assump-column"
              className={`${ctl} max-w-[150px]`}
              value={s.assumpColumn}
              onChange={(e) => {
                dispatch({ type: 'ASSUMP_COLUMN', value: e.target.value })
                actions.loadPanel('assump', e.target.value)
              }}
            >
              <option value="">RESIDUALS</option>
              {numeric.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        )}
      </PanelGate>
    </Panel>
  )
}

/* ============================================================================================
 * 03 // LEADERBOARD + REGISTRY
 * ========================================================================================== */
const LB_GRID = 'grid grid-cols-[20px_minmax(110px,1fr)_20px_50px_50px_60px] min-w-[340px] gap-x-1 items-center text-[10px]'
const REG_GRID = 'grid grid-cols-[16px_36px_minmax(90px,1fr)_44px_20px_56px] min-w-[300px] gap-x-1 items-center text-[10px]'

function CteBlocks({ states }) {
  return (
    <span className="inline-flex gap-[2px]" data-testid="cte">
      {states.map((st, i) => (
        <Sq key={CTE_NAMES[i]} size={8} color={CTE_COLOR[st] || C.pending} title={`${CTE_NAMES[i]}: ${st}`} />
      ))}
    </span>
  )
}

function Leaderboard() {
  const { state: s, dispatch, actions } = useCtx()
  const { thresholds } = useThresholds()
  const rows = buildRows(s)
  const board = s.board
  const leader = boardRows(board)[0]
  return (
    <div data-testid="leaderboard">
      <div className={`${LB_GRID} px-1 py-1 border-b border-zinc-800 text-zinc-400 uppercase`}>
        <span>POS</span>
        <span>MDL</span>
        <span>HW</span>
        <span className="text-right">{leader ? scoreName(leader) : isRegTask(taskOf(s)) ? 'R2' : 'F1'}</span>
        <span className="text-right">GAP</span>
        <span>CTE</span>
      </div>
      {rows.length === 0 && <Placeholder testid="lb-empty">{noData(s) ? 'NO DATASET // UPLOAD A FILE TO BEGIN' : 'NO RESULTS // RUN THE PIPELINE'}</Placeholder>}
      {rows.map((item) => {
        const r = item.row
        const isLeader = item.kind === 'result' && r.rank === 1
        const selected = s.selected === item.abbr
        const flags = item.kind === 'result' ? fitFlags(r, thresholds) : null
        const ovf = Boolean(flags && flags.overfit)
        const f1Color = ovf ? C.red : isLeader ? C.purple : '#fff'
        const hw = CAT[item.abbr] ? CAT[item.abbr].hw : 'CPU'
        return (
          <div
            key={item.abbr}
            data-testid={`lb-row-${item.abbr}`}
            role="button"
            tabIndex={0}
            onClick={() => {
              dispatch({ type: 'SELECT', abbr: item.abbr })
              if (!s.tele[item.abbr]) actions.loadTelemetry(item.abbr)
            }}
            onKeyDown={(e) => e.key === 'Enter' && dispatch({ type: 'SELECT', abbr: item.abbr })}
            className={`${LB_GRID} px-1 py-[3px] border-l-2 cursor-pointer hover:bg-zinc-800 ${selected ? 'bg-zinc-700' : ''} ${item.kind === 'failed' ? 'opacity-90' : ''}`}
            style={{ borderLeftColor: isLeader ? C.purple : selected ? '#fff' : 'transparent' }}
          >
            <span className="text-zinc-400">{item.kind === 'result' ? r.rank : '-'}</span>
            <span className="flex items-center gap-1 min-w-0 text-white font-bold whitespace-nowrap">
              <span className="shrink-0">{item.abbr}</span>
              {ovf && <span title={`OVERFIT DETECTED${isNum(r.train_test_gap) ? ` // TRAIN ${fmt3(trainScoreOf(r))} VS TEST ${fmt3(testScoreOf(r))} ${fitMetricLabel(r)}` : ''}`} className="px-[2px] text-[8px]" style={{ background: C.red, color: '#fff' }}>OVF</span>}
              {flags && flags.underfit && (
                <span data-testid={`unf-${item.abbr}`} title={`UNDERFIT DETECTED // TRAIN ${fitMetricLabel(r)} ${fmt3(trainScoreOf(r))} (${r.score_metric === 'r2' ? 'NO BETTER THAN PREDICTING THE MEAN' : 'CHANCE-LEVEL'})`} className="px-[2px] text-[8px]" style={{ background: C.yellow, color: '#000' }}>UNF</span>
              )}
              {item.kind === 'result' && r.tuned && <span data-testid={`tun-${item.abbr}`} title="HYPER-PARAMETERS TUNED WITH OPTUNA" className="px-[2px] text-[8px]" style={{ background: C.purple, color: '#fff' }}>TUN</span>}
              {item.kind === 'failed' && <span title={item.error} className="px-[2px] text-[8px]" style={{ background: C.red, color: '#fff' }}>ERR</span>}
            </span>
            <Dot color={hw === 'GPU' ? C.gpu : C.cpu} title={hw} />
            <span className="text-right font-bold" style={{ color: f1Color }}>{item.kind === 'result' ? fmt3(scoreOf(r)) : '---'}</span>
            <span className="text-right text-white">{item.kind === 'result' && !isLeader ? `+${fmt3(r.gap_to_top)}` : '---'}</span>
            <CteBlocks states={cteFor(s, item)} />
          </div>
        )
      })}
      {board && (
        <div data-testid="lb-footer" className="mt-1 border-t border-zinc-800 pt-1 text-[10px] uppercase text-zinc-400 break-words">
          {board.split && (
            <div style={board.split.stratified ? { color: '#fff' } : { color: C.yellow }}>
              TRAIN {fmtInt(board.split.train_rows)} | TEST {fmtInt(board.split.test_rows)} | {`STRATIFIED ${board.split.stratified ? 'YES' : 'NO'}`}
              {board.average ? ` | AVG ${String(board.average).toUpperCase()}` : ''}
            </div>
          )}
          {board.exclusions && (board.exclusions.id_like.length > 0 || board.exclusions.constant.length > 0) && (
            <div>
              EXCLUDED: {[...board.exclusions.id_like.map((c) => `${c} (id-like)`), ...board.exclusions.constant.map((c) => `${c} (constant)`)].join(', ')}
            </div>
          )}
          {board.warnings && board.warnings.map((w) => (
            <div key={w} style={{ color: C.yellow }}>{w}</div>
          ))}
          {leader && <div>LEADER: {abbrOfRow(leader)} {scoreName(leader)} {fmt3(scoreOf(leader))}{isNum(leader.rmse) ? ` | RMSE ${fmt3(leader.rmse)} | MAE ${fmt3(leader.mae)}` : ''}</div>}
        </div>
      )}
    </div>
  )
}

function Registry() {
  const { state: s, dispatch, actions } = useCtx()
  const [q, setQ] = useState('')
  const [fam, setFam] = useState('ALL')
  const [stf, setStf] = useState('ALL')
  const [pick, setPick] = useState(null)
  const states = Object.fromEntries(CATALOG.map((m) => [m.abbr, registryState(s, m.abbr)]))
  const count = (name) => CATALOG.filter((m) => states[m.abbr] === name).length
  const shown = CATALOG.filter(
    (m) =>
      (fam === 'ALL' || m.fam === fam) &&
      (stf === 'ALL' || states[m.abbr] === stf) &&
      (!q.trim() || `${m.abbr} ${modelInfo(m, taskOf(s)).name}`.toLowerCase().includes(q.trim().toLowerCase())),
  )
  const checked = modelAbbrs(taskOf(s)).filter((x) => s.sel[x])
  const running = s.job && s.job.status === 'RUNNING'
  const detail = pick ? modelInfo(CAT[pick], taskOf(s)) : null
  const result = detail ? boardRows(s.board).find((r) => abbrOfRow(r) === detail.abbr) : null
  const chip = (active, text, onClick) => (
    <button key={text} type="button" onClick={onClick} className={`h-5 px-1 text-[10px] uppercase border ${active ? 'bg-white text-black border-white font-bold' : 'bg-zinc-950 text-zinc-300 border-zinc-700'}`}>
      {text}
    </button>
  )
  return (
    <div data-testid="registry">
      <div className="flex flex-wrap items-center gap-1 mb-1">
        <input data-testid="reg-search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="SEARCH..." className="h-5 w-24 bg-zinc-950 border border-zinc-700 px-1 text-[10px] text-white" />
        <div className="flex flex-wrap gap-[2px]">{FAMILIES.map((f) => chip(fam === f, f, () => setFam(f)))}</div>
        <div className="flex flex-wrap gap-[2px]">{STATES.map((f) => chip(stf === f, f, () => setStf(f)))}</div>
      </div>
      <div data-testid="reg-counter" className="mb-1 text-[10px] uppercase text-zinc-300">
        {CATALOG.length} MODELS | {count('TRAINED')} TRAINED | {count('AVAIL')} AVAIL | {count('QUEUED')} QUEUED | {count('N/A')} N/A
      </div>
      <div className={`${REG_GRID} px-1 py-1 border-b border-zinc-800 text-zinc-400 uppercase`}>
        <span>SEL</span>
        <span>ABR</span>
        <span>MODEL</span>
        <span>FAM</span>
        <span>HW</span>
        <span>STATE</span>
      </div>
      {shown.map((m) => {
        const st = states[m.abbr]
        const na = st === 'N/A'
        const color = st === 'TRAINED' ? C.green : st === 'QUEUED' ? C.yellow : st === 'N/A' ? C.red : '#a1a1aa'
        const checkedNow = !na && Boolean(s.sel[m.abbr])
        return (
          <div
            key={m.abbr}
            data-testid={`reg-row-${m.abbr}`}
            onClick={() => setPick(m.abbr)}
            title={na ? naReason(m.abbr, taskOf(s)) : modelInfo(m, taskOf(s)).name}
            className={`${REG_GRID} px-1 py-[3px] cursor-pointer hover:bg-zinc-800 ${na ? 'opacity-50' : ''} ${pick === m.abbr ? 'bg-zinc-800' : ''}`}
          >
            <button
              type="button"
              data-testid={`reg-sel-${m.abbr}`}
              aria-label={`select ${m.abbr}`}
              aria-checked={checkedNow}
              role="checkbox"
              disabled={na || running}
              onClick={(e) => {
                e.stopPropagation()
                dispatch({ type: 'SEL_TOGGLE', abbr: m.abbr })
              }}
              className="w-3 h-3 border border-zinc-400 flex items-center justify-center disabled:cursor-not-allowed"
              style={{ background: checkedNow ? '#fff' : 'transparent', color: '#000' }}
            >
              {checkedNow ? <span className="text-[9px] leading-none font-bold">x</span> : null}
            </button>
            <span className="text-white font-bold">{m.abbr}</span>
            <span className="truncate text-zinc-400">{modelInfo(m, taskOf(s)).name}</span>
            <span className="text-zinc-300">{m.fam}</span>
            <Dot color={m.hw === 'GPU' ? C.gpu : C.cpu} title={m.hw} />
            <span className="font-bold" style={{ color }}>{st}</span>
          </div>
        )
      })}
      {shown.length === 0 && <div className="p-2 text-center text-[10px] uppercase text-zinc-500">NO MODELS MATCH</div>}
      <div data-testid="reg-detail" className="mt-1 border border-zinc-800 p-1 text-[10px] break-words min-h-[34px]">
        {detail ? (
          <>
            <div className="text-white font-bold">{detail.abbr} // {detail.name}</div>
            {states[detail.abbr] === 'N/A' ? (
              <div style={{ color: C.red }}>N/A: {naReason(detail.abbr, taskOf(s))}</div>
            ) : (
              <>
                <div className="text-zinc-300">HP: {detail.hp} | EST: {detail.est}</div>
                {result && <div className="text-white">LAST {scoreName(result)} {fmt3(scoreOf(result))}{isNum(result.rmse) ? ` | RMSE ${fmt3(result.rmse)} | MAE ${fmt3(result.mae)}` : ''} | TRAIN TIME {isNum(result.train_time_sec) ? `${result.train_time_sec.toFixed(2)} s` : '--'}</div>}
              </>
            )}
            {detail.abbr === 'LNR' && <div className="text-zinc-400">{isRegTask(taskOf(s)) ? 'PLAIN LEAST-SQUARES REGRESSION.' : 'OUTPUT THRESHOLDED AT 0.5 FOR BINARY TARGETS.'}</div>}
          </>
        ) : (
          <span className="text-zinc-500 uppercase">CLICK A ROW FOR DETAILS</span>
        )}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        <Btn testid="reg-select-all" disabled={!s.target || running} onClick={() => dispatch({ type: 'SEL_ALL_AVAIL' })}>SELECT ALL AVAIL</Btn>
        <Btn testid="reg-clear" disabled={running} onClick={() => dispatch({ type: 'SEL_CLEAR_QUEUED' })}>CLEAR QUEUED</Btn>
        <Btn testid="reg-run" bg={C.green} fg="#000" disabled={!s.target || running || checked.length === 0} onClick={actions.runPipeline}>
          RUN QUEUED ({checked.length})
        </Btn>
      </div>
    </div>
  )
}

function Panel03() {
  const { state: s } = useCtx()
  const [tab, setTab] = useState('LEADERBOARD')
  const tabs = (
    <div className="flex">
      {['LEADERBOARD', 'REGISTRY'].map((t) => (
        <button
          key={t}
          type="button"
          data-testid={`tab-${t}`}
          onClick={() => setTab(t)}
          className={`px-2 py-[2px] text-[10px] uppercase border-b-2 ${tab === t ? 'text-white border-[#b800e6] font-bold' : 'text-zinc-500 border-transparent'}`}
        >
          {t}
        </button>
      ))}
    </div>
  )
  return (
    <Panel n="03" title="" right={tabs} testid="panel-03">
      {tab === 'LEADERBOARD' ? <Leaderboard /> : <Registry />}
    </Panel>
  )
}

/* ============================================================================================
 * 04 // TRAINING TELEMETRY
 * ========================================================================================== */
const axisProps = { stroke: '#a1a1aa', tick: { fill: '#a1a1aa', fontSize: 10 }, tickLine: { stroke: '#a1a1aa' }, axisLine: { stroke: '#52525b' } }

function Panel04() {
  const { state: s } = useCtx()
  const { thresholds } = useThresholds()
  const reg = isRegTask(taskOf(s))
  const [tab, tabs] = usePanelTabs('p4tab', [['tele', 'TELEMETRY'], ['curves', reg ? 'RESIDUALS' : 'ROC/PR/THR'], ['tune', 'TUNING']], 'tele')
  const m = selectedModel(s)
  const base = s.session ? sx(s.session.id) : null
  const sel = s.selected
  const t = sel ? s.tele[sel] : null
  const selRow = sel ? boardRows(s.board).find((r) => r.abbr === sel) : null
  const flags = selRow ? fitFlags(selRow, thresholds) : null
  const fit = selRow ? (
    <>
      {flags.underfit && (
        <div data-testid="unf-tag" className="mb-1 px-1 text-[10px] font-bold uppercase" style={{ background: C.yellow, color: '#000' }}>
          UNDERFIT DETECTED // {sel} TRAIN {fitMetricLabel(selRow)} {fmt3(trainScoreOf(selRow))} - {selRow.score_metric === 'r2' ? 'NO BETTER THAN PREDICTING THE MEAN' : 'BARELY ABOVE CHANCE'} ON ITS OWN TRAINING DATA
        </div>
      )}
      {flags.gapOverfit && (
        <div data-testid="ovf-gap-tag" className="mb-1 px-1 text-[10px] font-bold uppercase" style={{ background: C.red, color: '#fff' }}>
          OVERFIT DETECTED // {sel} TRAIN {fmt3(trainScoreOf(selRow))} VS TEST {fmt3(testScoreOf(selRow))} {fitMetricLabel(selRow)} (GAP {fmt3(selRow.train_test_gap)})
        </div>
      )}
      {selRow.gap_check === false && (
        <div data-testid="gap-na-tag" className="mb-1 px-1 text-[10px] uppercase text-zinc-300 bg-zinc-800">
          OVERFIT GAP CHECK N/A // {sel} FITS ITS TRAINING DATA ~PERFECTLY BY DESIGN (TRAIN {fmt3(trainScoreOf(selRow))}) - JUDGE IT BY ITS TEST {scoreName(selRow)}
        </div>
      )}
    </>
  ) : null
  let body
  if (noData(s)) body = <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
  else if (!s.board) body = <Placeholder testid="tele-empty">{s.job && s.job.status === 'RUNNING' ? 'TRAINING...' : 'NO RESULTS // RUN THE PIPELINE'}</Placeholder>
  else if (!sel || !t || t.status === 'loading') body = <Placeholder>LOADING TELEMETRY...</Placeholder>
  else if (t.status === 'none') body = <>{fit}<Placeholder testid="tele-none">NO TELEMETRY // TOO LITTLE DATA FOR A VALIDATION SPLIT</Placeholder></>
  else if (t.status === 'error') body = <div className="p-2 text-[10px] uppercase" style={{ color: C.red }}>{t.error}</div>
  else if (t.data.kind === 'learning_curve') {
    const d = t.data
    const data = d.train_sizes.map((n, i) => ({ rows: n, train: d.train_score[i], val: d.val_score[i] }))
    const last = data[data.length - 1]
    const gap = last.train - last.val
    const isR2 = d.metric === 'r2'
    const lowCut = isR2 ? thresholds.underfitSkill : 0.6 // below this a model is not learning (R2 0 = predicts the mean)
    const memorizes = selRow && selRow.gap_check === false
    const verdict = memorizes
      ? last.val < lowCut
        ? 'VAL STAYS NEAR CHANCE: THE MODEL IS NOT GENERALISING (TRAIN LINE IS ~PERFECT BY DESIGN FOR THIS MODEL - IGNORE THE GAP)'
        : 'TRAIN LINE IS ~PERFECT BY DESIGN FOR THIS MODEL - JUDGE IT BY THE VAL LINE'
      : last.val < lowCut && last.train < lowCut
        ? 'BOTH LOW: UNDERFIT - THE MODEL CANNOT LEARN THIS SIGNAL; TRY A MORE FLEXIBLE MODEL OR BETTER FEATURES'
        : gap > thresholds.overfitGap
          ? 'WIDE GAP: OVERFIT - IF VAL STILL RISES AS ROWS GROW, MORE DATA HELPS; OTHERWISE SIMPLIFY THE MODEL'
          : 'LINES CLOSE TOGETHER: GOOD FIT AT THIS SIZE'
    body = (
      <div data-testid="tele-chart">
        {fit}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-white mb-1" data-testid="tele-readout">
          <span>{sel}</span>
          <span>LEARNING CURVE</span>
          <span>ROWS {fmtInt(last.rows)}</span>
          <span>TRAIN {fmt3(last.train)}</span>
          <span>VAL {fmt3(last.val)}</span>
          <span>GAP {fmt3(gap)}</span>
        </div>
        <div className="h-52 min-w-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="#27272a" />
              <XAxis dataKey="rows" type="number" domain={['dataMin', 'dataMax']} label={{ value: 'TRAINING ROWS', position: 'insideBottom', offset: -8, fill: '#a1a1aa', fontSize: 10 }} {...axisProps} />
              <YAxis domain={[isR2 ? Math.min(0, Math.floor(Math.min(...data.map((x) => Math.min(x.train, x.val))) * 10) / 10) : 0, 1]} tickFormatter={(v) => v.toFixed(1)} label={{ value: isR2 ? 'R2' : 'BAL ACC', angle: -90, position: 'insideLeft', fill: '#a1a1aa', fontSize: 10, dx: 4 }} width={44} {...axisProps} />
              <ReferenceLine y={isR2 ? 0 : 0.5} stroke="#52525b" strokeDasharray="2 4" />
              <Line type="monotone" dataKey="train" name="Train" stroke={C.loss} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
              <Line type="monotone" dataKey="val" name="Val" stroke={C.lossVal} strokeWidth={2} strokeDasharray="5 3" dot={{ r: 2 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-x-3 text-[10px] uppercase text-zinc-300">
          <span className="flex items-center gap-1"><Sq color={C.loss} /> TRAIN</span>
          <span className="flex items-center gap-1"><Sq color={C.lossVal} /> VAL (DASHED)</span>
          <span className="text-zinc-500">{isR2 ? 'DOTTED LINE = PREDICTING THE MEAN (R2 = 0)' : 'DOTTED LINE = CHANCE (2 CLASSES)'}</span>
        </div>
        <div data-testid="lc-verdict" className="mt-1 text-[10px] uppercase text-zinc-400">{verdict}</div>
      </div>
    )
  } else {
    const d = t.data
    const data = d.epochs.map((e, i) => ({ epoch: e, train: d.train_loss[i], val: d.val_loss[i], acc: d.val_accuracy[i] }))
    const last = data.length - 1
    body = (
      <div data-testid="tele-chart">
        {fit}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-white mb-1" data-testid="tele-readout">
          <span>{sel}</span>
          <span>EPOCH {data[last].epoch}/{d.total_epochs}</span>
          <span>LOSS {fmt3(data[last].train)}</span>
          <span>VAL {fmt3(data[last].val)}</span>
          <span>{d.val_metric === 'r2' ? 'R2' : 'ACC'} {fmt3(data[last].acc)}</span>
          {d.overfit_warning && (
            <span data-testid="ovf-tag" className="px-1 font-bold" style={{ background: C.red, color: '#fff' }}>
              {d.overfit_warning}
              {isNum(d.first_detected_epoch) ? ` @ EPOCH ${d.first_detected_epoch}` : ''}
            </span>
          )}
        </div>
        <div className="h-52 min-w-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="#27272a" />
              <XAxis dataKey="epoch" type="number" domain={['dataMin', 'dataMax']} label={{ value: 'EPOCH', position: 'insideBottom', offset: -8, fill: '#a1a1aa', fontSize: 10 }} {...axisProps} />
              <YAxis domain={[(min) => (d.val_metric === 'r2' ? Math.min(0, Math.floor(min * 10) / 10) : 0), (max) => Math.max(1, Math.ceil(max * 10) / 10)]} tickFormatter={(v) => v.toFixed(1)} label={{ value: d.val_metric === 'r2' ? 'LOSS / R2' : 'LOSS / ACC', angle: -90, position: 'insideLeft', fill: '#a1a1aa', fontSize: 10, dx: 4 }} width={44} {...axisProps} />
              <Line type="monotone" dataKey="train" name="Train" stroke={C.loss} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="val" name="Val" stroke={C.lossVal} strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="acc" name={d.val_metric === 'r2' ? 'R2' : 'Acc'} stroke={C.acc} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-3 text-[10px] uppercase text-zinc-300">
          <span className="flex items-center gap-1"><Sq color={C.loss} /> TRAIN</span>
          <span className="flex items-center gap-1"><Sq color={C.lossVal} /> VAL (DASHED)</span>
          <span className="flex items-center gap-1"><Sq color={C.acc} /> {d.val_metric === 'r2' ? 'R2' : 'ACC'}</span>
        </div>
      </div>
    )
  }
  let content = body
  if (tab !== 'tele') {
    if (noData(s)) content = <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
    else if (!s.board || !m.row) content = <Placeholder>{s.job && s.job.status === 'RUNNING' ? 'TRAINING...' : 'NO RESULTS // RUN THE PIPELINE'}</Placeholder>
    else {
      content = (
        <div key={`${m.abbr}-${verOf(s)}`}>
          <div className="mb-1 text-[10px] uppercase text-zinc-400">MODEL <b className="text-white">{m.abbr}</b> // PICK ANOTHER IN THE LEADERBOARD</div>
          {tab === 'curves' ? (reg ? <ResidualsTab base={base} mkey={m.key} /> : <CurvesTab base={base} mkey={m.key} />) : <TuningTab base={base} mkey={m.key} row={m.row} />}
        </div>
      )
    }
  }
  return (
    <Panel n="04" title="TRAINING TELEMETRY" right={tabs} testid="panel-04">
      {content}
    </Panel>
  )
}

/* ============================================================================================
 * 05 // MODEL COMPARISON
 * ========================================================================================== */
const METRICS = [['accuracy', 'ACCURACY'], ['precision', 'PRECISION'], ['recall', 'RECALL'], ['f1', 'F1-SCORE']]
const REG_METRICS = [['r2', 'R2']]

function Panel05() {
  const { state: s } = useCtx()
  const { thresholds } = useThresholds()
  const rows = boardRows(s.board)
  const [tab, tabs] = usePanelTabs('p5tab', [['metrics', 'METRICS'], ['summary', 'SUMMARY'], ...(isRegTask(taskOf(s)) ? [] : [['roc', 'ROC']]), ['runs', 'RUNS']], 'metrics')
  const m = selectedModel(s)
  const base = s.session ? sx(s.session.id) : null
  let body
  if (noData(s)) body = <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
  else if (rows.length === 0) body = <Placeholder testid="cmp-empty">NO RESULTS // RUN THE PIPELINE</Placeholder>
  else {
    const abbrs = rows.map(abbrOfRow)
    const leader = abbrs[0]
    const regression = isRegTask(rows[0].task)
    const metrics = regression ? REG_METRICS : METRICS
    const data = metrics.map(([k, label]) => ({ metric: label, ...Object.fromEntries(rows.map((r) => [abbrOfRow(r), r[k]])) }))
    const all = rows.flatMap((r) => metrics.map(([k]) => r[k])).filter(isNum)
    const floor = regression ? Math.min(0, Math.floor((Math.min(...all) - 0.05) * 20) / 20) : Math.max(0, Math.floor((Math.min(...all) - 0.05) * 20) / 20)
    body = (
      <div data-testid="cmp-chart">
        <div className="h-52 min-w-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 4, right: 8, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="#27272a" />
              <XAxis dataKey="metric" label={{ value: 'METRIC', position: 'insideBottom', offset: -8, fill: '#a1a1aa', fontSize: 10 }} {...axisProps} />
              <YAxis domain={[floor, 1]} tickFormatter={(v) => v.toFixed(2)} label={{ value: regression ? 'R2' : 'SCORE', angle: -90, position: 'insideLeft', fill: '#a1a1aa', fontSize: 10, dx: 4 }} width={44} {...axisProps} />
              {abbrs.map((a) => (
                <Bar key={a} dataKey={a} fill={modelColor(a, leader)} fillOpacity={s.selected && s.selected !== a ? 0.3 : 1} isAnimationActive={false} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        {regression && (
          <div data-testid="cmp-errors" className="mb-1 grid grid-cols-[auto_1fr_1fr] gap-x-3 text-[10px] uppercase text-zinc-300">
            <span className="text-zinc-500">MDL</span><span className="text-zinc-500 text-right">RMSE</span><span className="text-zinc-500 text-right">MAE</span>
            {rows.map((r) => (
              <React.Fragment key={r.model}>
                <span className="text-white">{abbrOfRow(r)}</span><span className="text-right">{fmt3(r.rmse)}</span><span className="text-right">{fmt3(r.mae)}</span>
              </React.Fragment>
            ))}
            <span className="col-span-3 normal-case text-zinc-500">RMSE / MAE are in the target&apos;s own units (lower is better).</span>
          </div>
        )}
        <div data-testid="cmp-legend" className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase text-white">
          {abbrs.map((a) => (
            <span key={a} className="flex items-center gap-1" style={{ opacity: s.selected && s.selected !== a ? 0.5 : 1 }}>
              <Sq color={modelColor(a, leader)} size={10} /> {a}
              {a === leader ? ' (LEADER)' : ''}
            </span>
          ))}
        </div>
      </div>
    )
  }
  let content = body
  if (tab === 'runs') content = <RunsTab refreshKey={verOf(s)} />
  else if (tab === 'summary') {
    const items = rows.map((r) => ({ abbr: abbrOfRow(r), row: r, flags: fitFlags(r, thresholds) }))
    const tsum = s.data.balance.data
    content = noData(s)
      ? <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
      : <SummaryTab rows={items} ctx={{ reg: isRegTask(taskOf(s)), target: s.target, thresholds, split: s.board && s.board.split, average: s.board && s.board.average, failed: (s.board && s.board.failed_models) || [], warnings: (s.board && s.board.warnings) || [], targetStd: isRegTask(taskOf(s)) && tsum ? tsum.std : null }} />
  }
  else if (tab !== 'metrics') {
    if (noData(s)) content = <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
    else if (rows.length === 0) content = <Placeholder testid="cmp-empty">NO RESULTS // RUN THE PIPELINE</Placeholder>
    else if (tab === 'roc') {
      const leader = abbrOfRow(rows[0])
      content = <RocOverlayTab key={verOf(s)} base={base} rows={rows.map((r) => ({ abbr: abbrOfRow(r), key: ALL_MODEL_KEYS[abbrOfRow(r)] })).filter((r) => r.key)} colorOf={(a) => modelColor(a, leader)} />
    }
  }
  return (
    <Panel n="05" title="MODEL COMPARISON" right={tabs} testid="panel-05">
      {content}
    </Panel>
  )
}

/* ============================================================================================
 * 06 // FEATURE ANALYTICS
 * ========================================================================================== */
function heatColor(v) {
  const t = Math.min(1, Math.abs(v))
  const base = [39, 39, 42]
  const tint = v < 0 ? [59, 130, 246] : [249, 115, 22]
  const rgb = base.map((b, i) => Math.round(b + (tint[i] - b) * t))
  const lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
  return { bg: `rgb(${rgb.join(',')})`, fg: lum > 0.5 ? '#000' : '#fff' }
}
const abbrevCol = (n) => (n.length > 6 ? `${n.slice(0, 5)}~` : n)

export function pickHeatColumns(corr, n) {
  const flagged = []
  corr.multicollinearity_flags.forEach((f) => {
    ;[f.feature_a, f.feature_b].forEach((c) => {
      if (!flagged.includes(c)) flagged.push(c)
    })
  })
  const rest = corr.columns.filter((c) => !flagged.includes(c))
  return [...flagged, ...rest].slice(0, n)
}

function Heatmap({ corr }) {
  const { state: s, dispatch } = useCtx()
  if (corr.columns.length < 2) return <Placeholder testid="heat-needs">NEEDS 2+ NUMERIC FEATURES</Placeholder>
  const maxN = Math.min(HEAT_MAX_N, corr.columns.length)
  const n = Math.min(s.heatN, maxN)
  const cols = pickHeatColumns(corr, n)
  const idx = cols.map((c) => corr.columns.indexOf(c))
  const flagged = new Set()
  corr.multicollinearity_flags.forEach((f) => {
    flagged.add(`${f.feature_a}|${f.feature_b}`)
    flagged.add(`${f.feature_b}|${f.feature_a}`)
  })
  const cell = 36
  return (
    <div data-testid="heatmap">
      <div className="flex items-center gap-1 mb-1">
        <span className={lbl}>FEATURES</span>
        <select data-testid="heat-n" className={ctl} value={n} onChange={(e) => dispatch({ type: 'HEAT_N', value: Number(e.target.value) })}>
          {Array.from({ length: maxN - 1 }, (_, i) => i + 2).map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <span className="text-[10px] text-zinc-500 uppercase">OF {corr.columns.length}</span>
      </div>
      <div className="overflow-auto">
        <div className="grid gap-px w-max" style={{ gridTemplateColumns: `56px repeat(${cols.length}, ${cell}px)` }}>
          <div />
          {cols.map((c) => (
            <div key={c} title={c} className="text-[9px] text-zinc-400 text-center uppercase truncate">{abbrevCol(c)}</div>
          ))}
          {cols.map((rowName, i) => (
            <React.Fragment key={rowName}>
              <div title={rowName} className="text-[9px] text-zinc-400 uppercase truncate flex items-center">{abbrevCol(rowName)}</div>
              {cols.map((colName, j) => {
                const v = corr.matrix[idx[i]][idx[j]]
                const diag = i === j
                const hot = !diag && flagged.has(`${rowName}|${colName}`)
                let st
                if (v === null || v === undefined) st = { bg: '#27272a', fg: '#a1a1aa' }
                else if (diag) st = { bg: '#52525b', fg: '#fff' }
                else if (hot) st = { bg: C.red, fg: '#fff' }
                else st = heatColor(v)
                return (
                  <div
                    key={colName}
                    data-testid={`heat-${rowName}-${colName}`}
                    data-hot={hot ? '1' : '0'}
                    title={`${rowName} x ${colName}: ${isNum(v) ? v.toFixed(2) : '--'}`}
                    className={`flex items-center justify-center text-[10px] ${hot ? 'font-bold' : ''}`}
                    style={{ width: cell, height: cell, background: st.bg, color: st.fg }}
                  >
                    {isNum(v) ? v.toFixed(2) : '--'}
                  </div>
                )
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
      <div className="mt-1 text-[10px] uppercase text-zinc-300 break-words">
        <span className="px-1 font-bold" style={{ background: C.red, color: '#fff' }}>RED = |r| &gt; {HEAT_THRESHOLD}</span>
        <div data-testid="heat-flags" className="mt-1">
          {corr.multicollinearity_flags.length
            ? `FLAGGED: ${corr.multicollinearity_flags.map((f) => `${f.feature_a}/${f.feature_b} ${f.r.toFixed(2)}`).join(', ')}`
            : 'NO MULTICOLLINEARITY'}
        </div>
      </div>
    </div>
  )
}

function Panel06() {
  const { state: s } = useCtx()
  const reg = isRegTask(taskOf(s))
  const [tab, tabs] = usePanelTabs('p6tab', [['corr', 'CORRELATION'], ['imp', 'IMPORTANCE'], ['shap', 'SHAP'], ], 'corr')
  const m = selectedModel(s)
  const base = s.session ? sx(s.session.id) : null
  let content
  if (tab === 'corr') {
    content = (
      <PanelGate s={s}>
        <PanelState r={s.data.corr}>{(corr) => <Heatmap corr={corr} />}</PanelState>
      </PanelGate>
    )
  } else if (noData(s)) {
    content = <Placeholder testid="placeholder-empty">NO DATASET // UPLOAD A FILE TO BEGIN</Placeholder>
  } else if (!s.board || !m.row) {
    content = <Placeholder>{s.job && s.job.status === 'RUNNING' ? 'TRAINING...' : 'NO RESULTS // RUN THE PIPELINE'}</Placeholder>
  } else {
    content = (
      <div key={`${m.abbr}-${verOf(s)}`}>
        <div className="mb-1 text-[10px] uppercase text-zinc-400">MODEL <b className="text-white">{m.abbr}</b> // PICK ANOTHER IN THE LEADERBOARD</div>
        {tab === 'imp' ? <ImportanceTab base={base} mkey={m.key} /> : <ShapTab base={base} mkey={m.key} reg={reg} />}
      </div>
    )
  }
  return (
    <Panel n="06" title="FEATURE ANALYTICS" right={tabs} testid="panel-06">
      {content}
    </Panel>
  )
}

/* ============================================================================================
 * 07 // LEGEND (static, always populated)
 * ========================================================================================== */
function LegendItem({ k, v }) {
  let key
  if (typeof k === 'string') key = <b className="text-white">{k}</b>
  else if (k.dot) key = <Dot color={k.dot} />
  else if (k.sq) key = <Sq color={k.sq} size={10} />
  else key = <b style={{ color: k.text }}>{k.label}</b>
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-zinc-300 whitespace-nowrap">
      {key}
      <span>= {v}</span>
    </span>
  )
}

const LEGEND_KEY = 'mldash.legend.open'

const SLIDERS = [
  { key: 'overfitGap', label: 'OVERFIT GAP', min: 0.02, max: 0.4, step: 0.01, hint: (v) => `flag OVF when train score beats test by > ${v.toFixed(2)}` },
  { key: 'underfitSkill', label: 'UNDERFIT SKILL', min: 0.05, max: 0.6, step: 0.01, hint: (v, reg) => (reg ? `flag UNF when train R2 < ${v.toFixed(2)}` : `flag UNF when train bal acc < ${(0.5 + v / 2).toFixed(2)} (2 classes)`) },
]

function ThresholdMenu() {
  const { state: s } = useCtx()
  const { thresholds, setThresholds } = useThresholds()
  const [open, setOpen] = useState(false)
  const rows = boardRows(s.board)
  const flagged = rows.map((r) => fitFlags(r, thresholds))
  const changed = SLIDERS.some((x) => thresholds[x.key] !== THRESH_DEFAULT[x.key])
  return (
    <div className="relative shrink-0 border-l border-zinc-800">
      <button
        type="button"
        data-testid="threshold-toggle"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-2 py-1 text-[10px] uppercase tracking-wide text-zinc-400 hover:bg-zinc-800"
      >
        <span>THRESHOLDS{changed ? ' *' : ''}</span>
        <span>{open ? '[-]' : '[+]'}</span>
      </button>
      {open && (
        <div data-testid="threshold-menu" className="absolute right-0 bottom-full z-30 w-72 max-w-[calc(100vw-16px)] border border-zinc-600 bg-zinc-900 p-2 shadow-lg">
          <div className="mb-1 text-[10px] font-bold uppercase text-zinc-300">OVERFIT / UNDERFIT THRESHOLDS</div>
          {SLIDERS.map((x) => (
            <label key={x.key} className="mb-2 block text-[10px] uppercase text-zinc-400">
              <span className="flex justify-between">
                <span>{x.label}</span>
                <b data-testid={`threshold-${x.key}-value`} className="text-white">{thresholds[x.key].toFixed(2)}</b>
              </span>
              <input
                data-testid={`threshold-${x.key}`}
                type="range"
                min={x.min}
                max={x.max}
                step={x.step}
                value={thresholds[x.key]}
                onChange={(e) => setThresholds({ ...thresholds, [x.key]: Number(e.target.value) })}
                className="w-full accent-zinc-300"
              />
              <span className="block normal-case text-zinc-500">{x.hint(thresholds[x.key], isRegTask(taskOf(s)))}</span>
            </label>
          ))}
          <div className="text-[10px] uppercase text-zinc-400">
            {rows.length > 0
              ? `NOW: ${flagged.filter((f) => f.overfit).length} OVF // ${flagged.filter((f) => f.underfit).length} UNF OF ${rows.length} MODELS`
              : 'RUN THE PIPELINE TO SEE FLAGS UPDATE LIVE'}
          </div>
          <div className="mt-1 text-[9px] normal-case text-zinc-500">
            The gap rule skips RFR / ETC / BAG (they memorise training data). XGB / LGB / MLP are also flagged when validation loss rises, which this does not change.
          </div>
          <button type="button" data-testid="threshold-reset" disabled={!changed} onClick={() => setThresholds(THRESH_DEFAULT)} className={`${ctl} mt-1 px-2 uppercase`}>
            RESET DEFAULTS
          </button>
        </div>
      )}
    </div>
  )
}

function Legend() {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(LEGEND_KEY) === '1'
    } catch {
      return false
    }
  })
  const toggle = () =>
    setOpen((o) => {
      try {
        localStorage.setItem(LEGEND_KEY, o ? '0' : '1')
      } catch {
        /* ignore */
      }
      return !o
    })
  return (
    <section data-testid="panel-07" className="md:col-span-2 lg:col-span-3 bg-zinc-900 border-t-2 border-zinc-600 min-w-0">
      <div className={`flex items-stretch ${open ? 'border-b border-zinc-800' : ''}`}>
        <button
          type="button"
          data-testid="legend-toggle"
          aria-expanded={open}
          onClick={toggle}
          className="flex flex-1 items-center justify-between px-2 py-1 text-left text-[10px] uppercase tracking-wide text-zinc-400 hover:bg-zinc-800"
        >
          <span>07 // LEGEND</span>
          <span className="shrink-0">{open ? '[-]' : '[+]'}</span>
        </button>
        <ThresholdMenu />
      </div>
      {open && (
        <div className="flex flex-wrap gap-x-5 gap-y-[2px] px-2 py-1 lg:max-h-32 overflow-y-auto">
          {LEGEND.map((seg) => (
            <div key={seg.title} className="flex flex-wrap items-center gap-x-2 gap-y-[2px] min-w-0">
              <span className="text-[10px] font-bold uppercase text-zinc-500">{seg.title}</span>
              {seg.items.map(([k, v], i) => (
                <LegendItem key={`${seg.title}-${i}`} k={k} v={v} />
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

/* ============================================================================================
 * RESIZABLE GRID (lg+ only): drag the gutters between panels, double-click a gutter to reset
 * ========================================================================================== */
const GRID_KEY = 'mldash.grid.v1'
const GRID_DEFAULT = { cols: [1, 1, 1], rows: [1, 1] }
const GRID_MIN = 0.12 // smallest share of the track a panel may shrink to

function loadGrid() {
  try {
    const g = JSON.parse(localStorage.getItem(GRID_KEY))
    const ok = (a, n) => Array.isArray(a) && a.length === n && a.every((v) => isNum(v) && v > 0)
    if (g && ok(g.cols, 3) && ok(g.rows, 2)) return g
  } catch {
    /* storage unavailable or corrupt: fall back to defaults */
  }
  return GRID_DEFAULT
}

// Move the boundary after track i so that it sits at `pos` (0..1 of the whole axis).
export function resizeTracks(tracks, i, pos) {
  const total = tracks.reduce((a, b) => a + b, 0)
  const before = tracks.slice(0, i).reduce((a, b) => a + b, 0) / total
  const pair = (tracks[i] + tracks[i + 1]) / total
  const left = Math.min(Math.max(pos - before, GRID_MIN), pair - GRID_MIN)
  const out = tracks.map((t) => t / total)
  out[i] = left
  out[i + 1] = pair - left
  return out
}

function useResizableGrid() {
  const [grid, setGrid] = useState(loadGrid)
  const mainRef = useRef(null)
  const [box, setBox] = useState(null) // { w, h } of the six-panel area in px

  useEffect(() => {
    try {
      localStorage.setItem(GRID_KEY, JSON.stringify(grid))
    } catch {
      /* ignore */
    }
  }, [grid])

  useEffect(() => {
    const el = mainRef.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const measure = () => {
      const legend = el.querySelector('[data-testid="panel-07"]')
      const lh = legend ? legend.getBoundingClientRect().height : 0
      setBox({ w: el.clientWidth, h: el.clientHeight - lh })
    }
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    const legend = el.querySelector('[data-testid="panel-07"]')
    if (legend) ro.observe(legend)
    measure()
    return () => ro.disconnect()
  }, [])

  const startDrag = (axis, i) => (e) => {
    e.preventDefault()
    const el = mainRef.current
    if (!el || !box) return
    const rect = el.getBoundingClientRect()
    const move = (ev) => {
      const pos = axis === 'cols' ? (ev.clientX - rect.left) / box.w : (ev.clientY - rect.top) / box.h
      setGrid((g) => ({ ...g, [axis]: resizeTracks(g[axis], i, pos) }))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.body.style.cursor = axis === 'cols' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const reset = () => setGrid(GRID_DEFAULT)
  const style = {
    '--grid-cols': grid.cols.map((c) => `minmax(0,${c}fr)`).join(' '),
    '--grid-rows': `${grid.rows.map((r) => `minmax(0,${r}fr)`).join(' ')} auto`,
  }
  return { grid, box, mainRef, style, startDrag, reset }
}

function GridHandles({ grid, box, startDrag, reset }) {
  if (!box) return null
  const frac = (tracks, i) => tracks.slice(0, i + 1).reduce((a, b) => a + b, 0) / tracks.reduce((a, b) => a + b, 0)
  const handle = 'absolute z-20 hidden lg:block bg-transparent hover:bg-zinc-500/60 active:bg-zinc-400/70 transition-colors'
  return (
    <>
      {[0, 1].map((i) => (
        <div
          key={`c${i}`}
          data-testid={`grid-col-handle-${i}`}
          title="Drag to resize // double-click to reset"
          onPointerDown={startDrag('cols', i)}
          onDoubleClick={reset}
          className={`${handle} cursor-col-resize`}
          style={{ left: box.w * frac(grid.cols, i) - 3, top: 0, width: 6, height: box.h }}
        />
      ))}
      <div
        data-testid="grid-row-handle-0"
        title="Drag to resize // double-click to reset"
        onPointerDown={startDrag('rows', 0)}
        onDoubleClick={reset}
        className={`${handle} cursor-row-resize`}
        style={{ top: box.h * frac(grid.rows, 0) - 3, left: 0, height: 6, width: box.w }}
      />
    </>
  )
}

/* ============================================================================================
 * APP ROOT
 * ========================================================================================== */
export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const stateRef = useRef(state)
  stateRef.current = state
  const actions = useMemo(() => makeActions(() => stateRef.current, dispatch), [])
  const ctx = useMemo(() => ({ state, dispatch, actions }), [state, actions])
  const layout = useResizableGrid()
  const [thresholds, setThresholdsState] = useState(loadThresholds)
  const setThresholds = (v) => {
    setThresholdsState(v)
    try {
      localStorage.setItem(THRESH_KEY, JSON.stringify(v))
    } catch {
      /* ignore */
    }
  }
  const thr = useMemo(() => ({ thresholds, setThresholds }), [thresholds])

  useEffect(() => {
    actions.checkHealth()
    const t = setInterval(actions.checkHealth, 15000)
    return () => clearInterval(t)
  }, [actions])

  useRadio(state.session ? state.session.id : null, dispatch)

  // Telemetry for the selected model on demand.
  useEffect(() => {
    if (state.selected && state.board && !state.tele[state.selected] && !(state.job && state.job.status === 'RUNNING')) {
      actions.loadTelemetry(state.selected)
    }
  }, [state.selected, state.board, state.tele, state.job, actions])

  // A radio "OVERFIT DETECTED ON <MODEL>" line refreshes that model's telemetry.
  const handledOvf = useRef(null)
  useEffect(() => {
    const last = state.radio[state.radio.length - 1]
    if (!last || handledOvf.current === last.key) return
    const m = /^OVERFIT DETECTED ON ([A-Z0-9]+)/.exec(last.message)
    if (m) {
      handledOvf.current = last.key
      if (!(state.job && state.job.status === 'RUNNING')) actions.loadTelemetry(m[1])
    }
  }, [state.radio, state.job, actions])

  return (
    <Ctx.Provider value={ctx}>
      <ThrCtx.Provider value={thr}>
        <div className="min-h-screen lg:h-screen flex flex-col bg-zinc-950 text-zinc-200 font-mono text-xs overflow-x-hidden">
          <TopBar />
          <RadioStrip />
          <main
            ref={layout.mainRef}
            style={layout.style}
            className="relative grid grid-cols-1 md:grid-cols-2 lg:[grid-template-columns:var(--grid-cols)] lg:[grid-template-rows:var(--grid-rows)] lg:flex-1 lg:min-h-0 gap-px bg-zinc-800"
          >
            <Panel01 />
            <Panel02 />
            <Panel03 />
            <Panel04 />
            <Panel05 />
            <Panel06 />
            <Legend />
            <GridHandles {...layout} />
          </main>
        </div>
      </ThrCtx.Provider>
    </Ctx.Provider>
  )
}

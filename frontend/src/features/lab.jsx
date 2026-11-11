/**
 * MODEL LAB
 * One model at a time: hand-set hyper-parameters, train/test/CV metrics, plain-language notes and graphs that are
 * specific to the model (coefficients, regularization paths, the fitted tree, boosting loss, out-of-bag growth,
 * neighbour explanations, decision surfaces, cluster profiles) plus parameter sweeps and a learning curve.
 * Controls are drawn from GET /api/v1/lab/models; every number shown is a backend response. The model on screen
 * downloads as a zip (model, predict.py, model card) trained with exactly the lab's settings.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, download, messageOf, sx } from '../lib/api'
import { f3, fInt, fPct, fSig, fSigned, isNum } from '../lib/fmt'

const K = {
  purple: '#b800e6', green: '#00e600', yellow: '#e6e600', red: '#ff0000', cyan: '#00d5ff', orange: '#fb923c',
  dim: '#a1a1aa', grid: '#27272a', text: '#e4e4e7', pink: '#f472b6', lime: '#a3e635',
}
const PALETTE = ['#00d5ff', '#fb923c', '#a3e635', '#f472b6', '#facc15', '#94a3b8', '#2dd4bf', '#818cf8', '#fda4af', '#d4d4d8']
const axis = { stroke: '#a1a1aa', tick: { fill: '#a1a1aa', fontSize: 10 }, tickLine: { stroke: '#a1a1aa' }, axisLine: { stroke: '#52525b' } }
const tip = { contentStyle: { background: '#09090b', border: '1px solid #52525b', fontSize: 10, borderRadius: 0 }, labelStyle: { color: '#a1a1aa' }, itemStyle: { padding: 0 } }
const ctl = 'h-6 bg-zinc-950 border border-zinc-700 text-white text-[10px] px-1 rounded-none disabled:opacity-40 disabled:cursor-not-allowed'
const lbl = 'text-[10px] uppercase text-zinc-400'
const gridLines = <CartesianGrid stroke={K.grid} />
/** bars coloured per bar have no series colour, so their tooltip text needs one */
const barTip = { ...tip, itemStyle: { padding: 0, color: K.text } }
const MODEL_KEY = 'mldash.lab.model.v1'
const M = { top: 8, right: 12, bottom: 4, left: 4 }
/** x-axis title in its own band under the tick labels, so it never runs into the legend */
const xAx = (value) => ({ height: 40, label: { value, position: 'insideBottom', offset: 4, fill: '#a1a1aa', fontSize: 10 } })
/** rotated y-axis title on the outer edge of the axis band (each axis is 16 px wider to make room) */
const yAx = (value, right = false) => ({
  label: { value, angle: right ? 90 : -90, position: right ? 'insideRight' : 'insideLeft', offset: 0, fill: '#a1a1aa', fontSize: 10, style: { textAnchor: 'middle' } },
})
/** legend below the x-axis title with a clear gap */
const legend = { wrapperStyle: { fontSize: 10, paddingTop: 10, lineHeight: '16px' } }
const clipName = (v) => (String(v).length > 22 ? `${String(v).slice(0, 21)}…` : v)

/** How to read each graph (shown under its title). */
const HELP = {
  sweep: 'Cross-validated score (training rows only) as one setting changes, every other setting held. Where validation peaks is the sweet spot; where train pulls away from validation, the model starts to overfit.',
  grid: 'Validation score for every pair of values. Brighter is better; the outlined cell is your current setting. Click a cell to use it.',
  clusterSweep: 'Inertia always falls as k grows. Look for the elbow where it stops falling fast, and for the k with the highest silhouette.',
  learning: 'Score as the training set grows. A big lasting gap means overfitting (more data or a simpler model helps); two low curves that meet mean underfitting (a more flexible model helps).',
  roc: 'True-positive rate against false-positive rate across every threshold. The area under it (AUC) is the chance a random positive scores above a random negative; 0.5 is guessing.',
  pr: 'Precision against recall. More honest than ROC when the positive class is rare; the dashed line is what guessing gets.',
  threshold: 'Move the threshold to trade precision (fewer false alarms) for recall (fewer misses). Nothing is retrained.',
  calibration: 'Do predicted probabilities mean what they say? Points on the diagonal are well calibrated; below it the model is overconfident.',
  separation: 'Predicted probability of the positive class for rows that are really negative vs really positive. Less overlap = easier to separate.',
  confusion: 'Rows are the true class, columns the prediction. The diagonal is correct; read across a row to see what a class gets mistaken for.',
  actualPred: 'Each dot is a test row. On the dashed line the prediction is exact; distance from it is the error.',
  residuals: 'Error against the prediction. A shapeless band around 0 is good; a funnel means errors grow with the value, a curve means a missed pattern.',
  residHist: 'Distribution of errors on the test rows. Centred on 0 and symmetric is ideal; a long tail flags rows the model gets badly wrong.',
  qq: 'Residual quantiles against a normal distribution. On the line = normal errors (needed for valid p-values and intervals); bends at the ends = heavy tails.',
  coef: 'Change in the target for one unit (one standard deviation, when scaled) of each feature, others held fixed. Whiskers are 95% confidence intervals; one that crosses 0 means the effect may be noise.',
  odds: 'Log-odds change per unit (one standard deviation, when scaled). Right of 0 pushes towards the class, left away; hover for the odds ratio.',
  path: 'Every coefficient as the penalty strength changes. Strong penalties shrink them towards 0 (lasso sets them to exactly 0). The dashed line is your setting.',
  boundary: 'The data squashed onto its first 2 principal components, with a copy of the model refit on just those 2 so its decision regions can be drawn. The real model works in all dimensions, so treat this as intuition.',
  knnDist: 'Distance from each test row to its k-th neighbour. Long distances mean sparse regions where the vote is shaky; the bars show how accuracy changes with distance.',
  explain: 'Pick a test row to see the k training rows that voted on it, their distance and their labels.',
  loss: 'Loss on the rows each tree was fitted to, and on held-back validation rows, after every boosting round. Validation bottoming out then rising = overfitting from that round on.',
  xgbImp: 'gain = total loss reduction from splits on the feature (usually the most meaningful). splits = how often it is used. cover = how many rows those splits touch.',
  tree: 'Follow a row from the top: each box asks a question (left = yes, right = no) until it reaches a leaf. The coloured header is what the tree predicts there; the bar is the real mix of training rows that reached the box.',
  rules: 'The largest leaves written out as if-then rules.',
  pruning: 'Cost-complexity pruning cuts branches whose gain is below alpha. Watch test score as the tree shrinks: the best alpha is often a much smaller tree.',
  importance: 'Share of total impurity reduction from splits on each feature.',
  growth: 'Scores as trees are added. Out-of-bag uses rows each tree never saw, so it is a free validation score. Once the curve flattens, more trees only cost time.',
  forestImp: 'MDI comes from the trees (biased towards features with many values); permutation is the drop in test score when a column is shuffled. Disagreement is a warning sign.',
  agreement: 'How much the trees agree on each test row. Rows where the forest is split are where it is most often wrong.',
  spread: 'How much the trees disagree on each test row (std of their predictions). High spread = an uncertain prediction.',
  margins: 'Signed distance of test rows from the boundary. Rows between -1 and 1 sit inside the margin; overlap of the two classes there is where errors happen.',
  marginsMulti: 'Gap between the best and second-best class score. Small gaps are close calls.',
  tube: 'Prediction errors on test rows (in standard deviations of the target). Errors inside the epsilon tube cost nothing during training.',
  clusters: 'Rows on their first 2 principal components, coloured by cluster (or by target). Crosses are the cluster centres.',
  silhouette: 'One thin bar per row, sorted within each cluster: how much closer it is to its own cluster than to the next. Bars below 0 are probably in the wrong cluster.',
  profiles: 'Cluster centres per feature. Scaled numeric features read in standard deviations from the overall mean; one-hot levels read as the share of rows. Hover for raw means.',
  target: 'How the clusters line up with the target (K-means never saw it).',
}

/* ------------------------------------------------------------------ atoms */
function Msg({ children, color, testid }) {
  return (
    <div data-testid={testid} className="min-h-[80px] flex items-center justify-center text-center text-[10px] uppercase text-zinc-500 px-2 py-4" style={color ? { color } : undefined}>
      {children}
    </div>
  )
}

function Btn({ children, onClick, disabled, title, testid, bg = '#27272a', fg = '#fff', className = '' }) {
  return (
    <button type="button" data-testid={testid} title={title} disabled={disabled} onClick={onClick}
      className={`h-6 px-2 text-[10px] font-bold uppercase border border-zinc-700 rounded-none whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed ${className}`}
      style={disabled ? { background: '#27272a', color: '#a1a1aa' } : { background: bg, color: fg }}>
      {children}
    </button>
  )
}

function Seg({ value, onChange, options, testid }) {
  return (
    <span className="inline-flex flex-wrap" data-testid={testid}>
      {options.map(([v, label]) => (
        <button key={String(v)} type="button" onClick={() => onChange(v)}
          className={`h-5 px-1.5 text-[10px] uppercase border ${value === v ? 'bg-white text-black border-white font-bold' : 'bg-zinc-950 text-zinc-300 border-zinc-700'}`}>
          {label}
        </button>
      ))}
    </span>
  )
}

function Toggle({ on, onClick, children, testid, title }) {
  return (
    <button type="button" data-testid={testid} title={title} onClick={onClick} className={`${ctl} font-bold`} style={on ? { background: K.purple, borderColor: K.purple } : undefined}>
      {children} {on ? 'ON' : 'OFF'}
    </button>
  )
}

function Card({ title, help, right, children, wide, testid, busy }) {
  return (
    <section data-testid={testid} className={`border border-zinc-800 bg-zinc-900 min-w-0 ${wide ? 'xl:col-span-2' : ''}`}>
      <header className="flex flex-wrap items-center justify-between gap-1 border-b border-zinc-800 px-2 py-1">
        <span className="text-[10px] uppercase tracking-wide text-zinc-300 font-bold">{title}{busy && <span className="ml-2 font-normal text-zinc-500">UPDATING...</span>}</span>
        {right}
      </header>
      <div className="p-2">
        {help && <p className="mb-1 text-[10px] leading-snug text-zinc-500">{help}</p>}
        {children}
      </div>
    </section>
  )
}

const Box = ({ h = 'h-60', children }) => (
  <div className={h}>
    <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
  </div>
)

function Stats({ items }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase text-white mt-1">
      {items.filter(Boolean).map(([k, v, color]) => (
        <span key={k}><span className="text-zinc-500">{k} </span><span style={color ? { color } : undefined} className="font-bold">{v}</span></span>
      ))}
    </div>
  )
}

const pValue = (p) => (!isNum(p) ? '—' : p < 0.001 ? '<0.001' : p.toFixed(3))
const labelOf = (v, p) => (p && p.zero_means && v === 0 ? p.zero_means : typeof v === 'number' ? fSig(v) : String(v))

/* ------------------------------------------------------------------ hyper-parameter controls */
function ParamControl({ p, value, onChange }) {
  const id = `lab-param-${p.name}`
  let input = null
  if (p.kind === 'bool') {
    input = <Toggle testid={id} on={Boolean(value)} onClick={() => onChange(!value)}>{''}</Toggle>
  } else if (p.kind === 'cat') {
    const short = p.choices.length <= 4 && p.choices.join('').length <= 26
    input = short ? (
      <Seg testid={id} value={value} onChange={onChange} options={p.choices.map((c) => [c, String(c)])} />
    ) : (
      <select data-testid={id} className={`${ctl} w-full`} value={String(value)} onChange={(e) => onChange(p.choices.find((c) => String(c) === e.target.value))}>
        {p.choices.map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
      </select>
    )
  } else {
    const log = p.kind === 'log'
    const lo = log ? Math.log10(p.lo) : p.lo
    const hi = log ? Math.log10(p.hi) : p.hi
    const pos = log ? Math.log10(Math.max(value, p.lo)) : value
    const step = log ? 0.01 : p.step || (p.kind === 'int' ? 1 : (p.hi - p.lo) / 100)
    const fromSlider = (raw) => {
      const v = Number(raw)
      if (log) return Number(Number(10 ** v).toPrecision(3))
      return p.kind === 'int' ? Math.round(v) : Number(v.toFixed(6))
    }
    input = (
      <div className="flex items-center gap-1">
        <input data-testid={id} type="range" min={lo} max={hi} step={step} value={pos} onChange={(e) => onChange(fromSlider(e.target.value))} className="flex-1 min-w-0" />
        <NumBox value={value} onCommit={(v) => onChange(Math.min(p.hi, Math.max(p.lo, p.kind === 'int' ? Math.round(v) : v)))} />
      </div>
    )
  }
  return (
    <div className="mb-2" data-testid={`ctl-${p.name}`}>
      <div className="flex items-center justify-between gap-1 text-[10px] uppercase">
        <span className="text-zinc-200 font-bold">{p.label}</span>
        {p.kind !== 'cat' && p.kind !== 'bool' && <span className="text-zinc-400">{labelOf(value, p)}</span>}
      </div>
      <div className="mt-[2px]">{input}</div>
      <p className="mt-[2px] text-[10px] leading-snug text-zinc-500">{p.help}</p>
    </div>
  )
}

function NumBox({ value, onCommit }) {
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])
  const commit = () => {
    const v = Number(text)
    if (Number.isFinite(v)) onCommit(v)
    else setText(String(value))
  }
  return (
    <input value={text} onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()}
      className="h-5 w-14 bg-zinc-950 border border-zinc-700 px-1 text-[10px] text-white" />
  )
}

const visible = (p, values) => Object.entries(p.when || {}).every(([k, allowed]) => allowed.includes(values[k]))

/* ------------------------------------------------------------------ metrics strip + notes */
function MetricCard({ label, value, sub, delta, good, testid, title }) {
  return (
    <div data-testid={testid} title={title} className="border border-zinc-800 bg-zinc-950 px-2 py-1 min-w-0">
      <div className="text-[9px] uppercase text-zinc-500 truncate">{label}</div>
      <div className="text-[15px] font-bold text-white leading-tight">{value}</div>
      <div className="flex gap-2 text-[9px] uppercase">
        {sub && <span className="text-zinc-500 truncate">{sub}</span>}
        {delta !== undefined && delta !== null && (
          <span style={{ color: delta === 0 ? K.dim : (delta > 0) === good ? K.green : K.red }}>A {fSigned(delta, 3)}</span>
        )}
      </div>
    </div>
  )
}

function MetricsStrip({ d, pin }) {
  const dv = (v, a) => (pin && isNum(v) && isNum(a) ? v - a : null)
  if (d.task === 'clustering') {
    const m = d.metrics
    const a = pin && pin.metrics
    return (
      <div data-testid="lab-metrics" className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-1">
        <MetricCard label="Silhouette" value={f3(m.silhouette)} sub="-1..1, higher better" delta={dv(m.silhouette, a && a.silhouette)} good />
        <MetricCard label="Inertia" value={fSig(m.inertia)} sub="within-cluster SS" delta={null} />
        <MetricCard label="Davies-Bouldin" value={f3(m.davies_bouldin)} sub="lower better" delta={dv(m.davies_bouldin, a && a.davies_bouldin)} good={false} />
        <MetricCard label="Calinski-Harabasz" value={fSig(m.calinski_harabasz)} sub="higher better" />
        <MetricCard label="Clusters" value={m.k} sub={`${m.n_iter} iterations`} />
        <MetricCard label="Rows" value={fInt(d.data.rows_used)} sub={`${d.fit_seconds}s fit`} />
      </div>
    )
  }
  const reg = d.task === 'regression'
  const t = d.test
  const tr = d.train
  const at = pin && pin.test
  const key = reg ? 'r2' : 'f1'
  const gap = isNum(tr[key]) && isNum(t[key]) ? tr[key] - t[key] : null
  const cards = reg
    ? [
        ['R2 (test)', f3(t.r2), `train ${f3(tr.r2)}`, dv(t.r2, at && at.r2), true],
        ['Adjusted R2', f3(t.adj_r2), 'penalises features', dv(t.adj_r2, at && at.adj_r2), true],
        ['RMSE', fSig(t.rmse), `target sd ${fSig(t.target_std)}`, dv(t.rmse, at && at.rmse), false],
        ['MAE', fSig(t.mae), `train ${fSig(tr.mae)}`, dv(t.mae, at && at.mae), false],
      ]
    : [
        [`${d.metric} (test)`, f3(t.f1), `train ${f3(tr.f1)}`, dv(t.f1, at && at.f1), true],
        ['ROC AUC', f3(t.roc_auc), 'ranking quality', dv(t.roc_auc, at && at.roc_auc), true],
        ['Accuracy', f3(t.accuracy), `balanced ${f3(t.balanced_accuracy)}`, dv(t.accuracy, at && at.accuracy), true],
        ['Precision / recall', `${f3(t.precision)} / ${f3(t.recall)}`, isNum(t.log_loss) ? `log loss ${f3(t.log_loss)}` : 'no probabilities', null, true],
      ]
  return (
    <div data-testid="lab-metrics" className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-1">
      {cards.map(([label, value, sub, delta, good]) => <MetricCard key={label} label={label} value={value} sub={sub} delta={delta} good={good} />)}
      <MetricCard label="Train - test gap" value={isNum(gap) ? fSigned(gap, 3) : '—'} sub={isNum(gap) ? (gap > 0.1 ? 'overfitting' : 'ok') : ''} title="Training score minus test score. Above 0.10 the model is memorising rather than learning." />
      <MetricCard testid="lab-cv" label={`CV ${d.cv ? d.cv.metric : ''}`} value={d.cv ? f3(d.cv.mean) : '—'} sub={d.cv ? `± ${f3(d.cv.std)} over ${d.cv.folds.length} folds` : 'off'} delta={d.cv && pin && pin.cv ? d.cv.mean - pin.cv.mean : null} good />
    </div>
  )
}

function Notes({ notes }) {
  if (!notes || !notes.length) return null
  return (
    <div data-testid="lab-notes" className="border border-zinc-800 bg-zinc-950 px-2 py-1">
      <div className="text-[10px] uppercase text-zinc-400 font-bold mb-[2px]">What the model is telling you</div>
      <ul className="space-y-[2px]">
        {notes.map((n) => (
          <li key={n} className="text-[11px] leading-snug text-zinc-200 flex gap-1">
            <span style={{ color: /overfit|underfit|off\.|weak|unstable|iteration limit/i.test(n) ? K.yellow : K.green }}>{'>'}</span>
            <span>{n}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ------------------------------------------------------------------ sweeps + learning curve */
function heat(t) {
  // 0 -> dark, 1 -> bright purple / pink
  const c = Math.max(0, Math.min(1, isNum(t) ? t : 0))
  return `rgba(${Math.round(60 + 180 * c)}, ${Math.round(20 + 60 * c)}, ${Math.round(90 + 140 * c)}, ${0.25 + 0.75 * c})`
}

function SweepCard({ model, task, values, sweep, sel, setSel, onRun, onApply, stale }) {
  const sweepable = model.params.filter((p) => p.sweep.length && (!p.tasks.length || p.tasks.includes(task)))
  const spec = (name) => model.params.find((p) => p.name === name && (!p.tasks.length || p.tasks.includes(task)))
  const d = sweep.data
  const right = (
    <span className="flex flex-wrap items-center gap-1">
      <select data-testid="sweep-param" className={ctl} value={sel.param} onChange={(e) => setSel({ ...sel, param: e.target.value, param2: e.target.value === sel.param2 ? '' : sel.param2 })}>
        {sweepable.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}
      </select>
      {task !== 'clustering' && (
        <select data-testid="sweep-param2" className={ctl} value={sel.param2 || ''} onChange={(e) => setSel({ ...sel, param2: e.target.value })}>
          <option value="">ONE PARAMETER</option>
          {sweepable.filter((p) => p.name !== sel.param).map((p) => <option key={p.name} value={p.name}>x {p.label}</option>)}
        </select>
      )}
      <Btn testid="sweep-run" bg={stale ? K.yellow : '#27272a'} fg={stale ? '#000' : '#fff'} disabled={sweep.status === 'loading'} onClick={() => onRun(sel)}>
        {sweep.status === 'loading' ? 'RUNNING...' : stale ? 'RERUN (STALE)' : 'RUN'}
      </Btn>
    </span>
  )
  let body
  if (sweep.error) body = <Msg color={K.red}>{sweep.error}</Msg>
  else if (!d) body = <Msg>{sweep.status === 'loading' ? 'REFITTING ACROSS VALUES...' : 'PICK A PARAMETER AND RUN'}</Msg>
  else if (d.kind === 'grid') {
    const p1 = spec(d.param)
    const p2 = spec(d.param2)
    const flat = d.val.flat().filter(isNum)
    const lo = Math.min(...flat)
    const hi = Math.max(...flat)
    body = (
      <div className="overflow-x-auto">
        <table data-testid="sweep-grid" className="text-[10px] border-collapse">
          <thead>
            <tr>
              <th className="px-1 text-left text-zinc-500 font-normal">{p1.label} \ {p2.label}</th>
              {d.values2.map((v) => <th key={String(v)} className="px-1 text-zinc-400 font-normal">{labelOf(v, p2)}</th>)}
            </tr>
          </thead>
          <tbody>
            {d.val.map((row, i) => (
              <tr key={String(d.values[i])}>
                <td className="px-1 text-zinc-400">{labelOf(d.values[i], p1)}</td>
                {row.map((v, j) => {
                  const cur = d.values[i] === values[d.param] && d.values2[j] === values[d.param2]
                  const best = d.best && d.best.value === d.values[i] && d.best.value2 === d.values2[j]
                  return (
                    <td key={j} className="p-0">
                      <button type="button" title={`train ${f3(d.train[i][j])} // validation ${f3(v)} // click to use`} onClick={() => onApply({ [d.param]: d.values[i], [d.param2]: d.values2[j], ...(d.changed || {}) })}
                        className={`w-14 h-7 text-[10px] ${best ? 'font-bold text-white' : 'text-zinc-200'}`}
                        style={{ background: heat(hi > lo ? (v - lo) / (hi - lo) : 1), outline: cur ? `2px solid ${K.yellow}` : 'none', outlineOffset: -2 }}>
                        {f3(v)}
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  } else if (d.kind === 'cluster_sweep') {
    const rows = d.values.map((v, i) => ({ k: String(v), inertia: d.inertia[i], silhouette: d.silhouette[i] }))
    body = (
      <Box>
        <ComposedChart data={rows} margin={M}>
          {gridLines}
          <XAxis dataKey="k" {...axis} {...xAx('k')} />
          <YAxis {...yAx('inertia')} yAxisId="l" {...axis} width={66} tickFormatter={fSig} />
          <YAxis {...yAx('silhouette', true)} yAxisId="r" orientation="right" {...axis} width={52} domain={[-0.1, 1]} />
          <Tooltip {...tip} formatter={(v, n) => [fSig(v), n]} />
          <Legend {...legend} />
          <ReferenceLine yAxisId="l" x={String(d.current)} stroke={K.yellow} strokeDasharray="3 3" />
          <Line yAxisId="l" dataKey="inertia" stroke={K.cyan} dot isAnimationActive={false} />
          <Line yAxisId="r" dataKey="silhouette" stroke={K.pink} dot isAnimationActive={false} />
        </ComposedChart>
      </Box>
    )
  } else {
    const p1 = spec(d.param)
    const rows = d.values.map((v, i) => ({ x: labelOf(v, p1), v, train: d.train[i], val: d.val[i], sd: d.val_std[i] }))
    const cur = rows.find((r) => r.v === d.current)
    body = (
      <Box>
        <LineChart data={rows} margin={M}>
          {gridLines}
          <XAxis dataKey="x" {...axis} {...xAx(p1 ? p1.label : d.param)} />
          <YAxis {...yAx(d.metric || 'score')} {...axis} width={56} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} />
          <Tooltip {...tip} formatter={(v, n, item) => [n.startsWith('validation') && item && isNum(item.payload.sd) ? `${f3(v)} ± ${f3(item.payload.sd)}` : f3(v), n]} />
          <Legend {...legend} />
          {cur && <ReferenceLine x={cur.x} stroke={K.yellow} strokeDasharray="3 3" label={{ value: 'NOW', fill: K.yellow, fontSize: 9, position: 'top' }} />}
          <Line dataKey="train" name="train" stroke={K.orange} dot isAnimationActive={false} />
          <Line dataKey="val" name="validation (CV)" stroke={K.cyan} dot isAnimationActive={false} />
          {d.best && <ReferenceDot x={labelOf(d.best.value, p1)} y={d.best.score} r={5} fill="none" stroke={K.green} strokeWidth={2} />}
        </LineChart>
      </Box>
    )
  }
  const best = d && (d.kind === 'cluster_sweep' ? (isNum(d.best_value) ? { [d.param]: d.best_value } : null) : d.best ? { [d.param]: d.best.value, ...(d.kind === 'grid' ? { [d.param2]: d.best.value2 } : {}) } : null)
  return (
    <Card testid="card-sweep" wide={d && d.kind === 'grid'} title={d && d.kind === 'grid' ? 'Score grid' : 'Validation curve'} busy={sweep.status === 'loading' && d} right={right}
      help={!d ? HELP.sweep : d.kind === 'grid' ? HELP.grid : d.kind === 'cluster_sweep' ? HELP.clusterSweep : HELP.sweep}>
      {body}
      {d && (
        <div className="flex flex-wrap items-center gap-2 mt-1">
          <Stats items={[
            d.metric && ['METRIC', d.metric],
            d.best && ['BEST', `${f3(d.best.score)} at ${Object.values(best || {}).map((v) => labelOf(v)).join(' / ')}`, K.green],
            d.kind === 'cluster_sweep' && isNum(d.best_value) && ['BEST SILHOUETTE AT K', d.best_value, K.green],
            ['ROWS', `${fInt(d.rows)}${d.folds ? ` x ${d.folds} folds` : ''}`],
            ['TIME', `${d.seconds}s`],
            d.truncated && ['STOPPED EARLY', 'time budget', K.yellow],
          ]} />
          {best && <Btn testid="sweep-apply" bg={K.green} fg="#000" onClick={() => onApply({ ...best, ...(d.changed || {}) })}>USE BEST</Btn>}
          {d.changed && Object.keys(d.changed).length > 0 && <span className="text-[10px] text-zinc-500">swept with {Object.entries(d.changed).map(([k, v]) => `${k}=${v}`).join(', ')}</span>}
        </div>
      )}
    </Card>
  )
}

function LearningCard({ lc, onRun, stale }) {
  const d = lc.data
  const rows = d ? d.values.map((n, i) => ({ n, train: d.train[i], val: d.val[i] })) : []
  return (
    <Card testid="card-learning" title="Learning curve" help={HELP.learning} busy={lc.status === 'loading' && d}
      right={<Btn bg={stale ? K.yellow : '#27272a'} fg={stale ? '#000' : '#fff'} disabled={lc.status === 'loading'} onClick={onRun}>{lc.status === 'loading' ? 'RUNNING...' : stale ? 'RERUN (STALE)' : 'RUN'}</Btn>}>
      {lc.error ? <Msg color={K.red}>{lc.error}</Msg> : !d ? <Msg>{lc.status === 'loading' ? 'REFITTING ON GROWING SAMPLES...' : 'NOT RUN'}</Msg> : (
        <>
          <Box>
            <LineChart data={rows} margin={M}>
              {gridLines}
              <XAxis dataKey="n" type="number" domain={['dataMin', 'dataMax']} {...axis} {...xAx('training rows')} />
              <YAxis {...yAx(d.metric)} {...axis} width={56} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip {...tip} formatter={(v, n) => [f3(v), n]} labelFormatter={(v) => `${fInt(v)} rows`} />
              <Legend {...legend} />
              <Line dataKey="train" stroke={K.orange} dot isAnimationActive={false} />
              <Line dataKey="val" name="validation (CV)" stroke={K.cyan} dot isAnimationActive={false} />
            </LineChart>
          </Box>
          <Stats items={[['METRIC', d.metric], ['FINAL GAP', fSigned(d.train[d.train.length - 1] - d.val[d.val.length - 1], 3)], ['TIME', `${d.seconds}s`]]} />
        </>
      )}
    </Card>
  )
}

/* ------------------------------------------------------------------ classification performance */
function RocCard({ perf }) {
  if (!perf.roc.length) return null
  return (
    <Card testid="card-roc" title="ROC curve" help={HELP.roc}>
      <Box>
        <LineChart margin={M}>
          {gridLines}
          <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...axis} {...xAx('false-positive rate')} />
          <YAxis {...yAx('true-positive rate')} type="number" domain={[0, 1]} {...axis} width={48} />
          <Tooltip {...tip} formatter={(v) => f3(v)} labelFormatter={(v) => `FPR ${f3(v)}`} />
          <Legend {...legend} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#52525b" strokeDasharray="3 3" />
          {perf.roc.map((c, i) => (
            <Line key={c.label} data={c.fpr.map((x, j) => ({ fpr: x, tpr: c.tpr[j] }))} dataKey="tpr" name={`${c.label} (AUC ${f3(c.auc)})`} stroke={PALETTE[i % PALETTE.length]} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </Box>
    </Card>
  )
}

function PrCard({ perf }) {
  if (!perf.pr) return null
  const rows = perf.pr.recall.map((r, i) => ({ recall: r, precision: perf.pr.precision[i] }))
  return (
    <Card testid="card-pr" title="Precision-recall" help={HELP.pr}>
      <Box>
        <LineChart data={rows} margin={M}>
          {gridLines}
          <XAxis type="number" dataKey="recall" domain={[0, 1]} {...axis} {...xAx('recall')} />
          <YAxis {...yAx('precision')} type="number" domain={[0, 1]} {...axis} width={48} />
          <Tooltip {...tip} formatter={(v) => f3(v)} labelFormatter={(v) => `recall ${f3(v)}`} />
          <ReferenceLine y={perf.pr.prevalence} stroke="#52525b" strokeDasharray="3 3" />
          <Line dataKey="precision" stroke={K.cyan} dot={false} isAnimationActive={false} />
        </LineChart>
      </Box>
      <Stats items={[['AVERAGE PRECISION', f3(perf.pr.ap)], ['GUESSING', f3(perf.pr.prevalence)]]} />
    </Card>
  )
}

function ThresholdCard({ perf }) {
  const t = perf.thresholds
  const [thr, setThr] = useState(0.5)
  if (!t) return null
  const i = Math.round(thr * (t.grid.length - 1))
  const bestF1 = t.f1.indexOf(Math.max(...t.f1))
  const rows = t.grid.map((g, j) => ({ g, precision: t.precision[j], recall: t.recall[j], f1: t.f1[j] }))
  const [neg, pos] = perf.labels
  const cell = (v, ok) => <td className="border border-zinc-800 px-2 py-1 text-center text-[11px] font-bold" style={{ color: ok ? K.green : K.red }}>{fInt(v)}</td>
  return (
    <Card testid="card-threshold" wide title="Decision threshold" help={HELP.threshold}
      right={<span className="flex items-center gap-1">
        <input data-testid="lab-threshold" type="range" min={0} max={1} step={0.01} value={thr} onChange={(e) => setThr(Number(e.target.value))} className="w-32" />
        <span className="text-[10px] text-white w-8">{thr.toFixed(2)}</span>
        <Btn onClick={() => setThr(t.grid[bestF1])}>BEST F1</Btn>
        <Btn onClick={() => setThr(0.5)}>0.50</Btn>
      </span>}>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_220px] gap-2">
        <Box>
          <LineChart data={rows} margin={M}>
            {gridLines}
            <XAxis type="number" dataKey="g" domain={[0, 1]} {...axis} {...xAx('threshold')} />
            <YAxis {...yAx('score')} domain={[0, 1]} {...axis} width={48} />
            <Tooltip {...tip} formatter={(v, n) => [f3(v), n]} labelFormatter={(v) => `threshold ${f3(v)}`} />
            <Legend {...legend} />
            <ReferenceLine x={thr} stroke={K.yellow} />
            <Line dataKey="precision" stroke={K.cyan} dot={false} isAnimationActive={false} />
            <Line dataKey="recall" stroke={K.orange} dot={false} isAnimationActive={false} />
            <Line dataKey="f1" name="F1" stroke={K.pink} dot={false} isAnimationActive={false} />
          </LineChart>
        </Box>
        <div>
          <table data-testid="threshold-confusion" className="w-full border-collapse text-[10px] mb-1">
            <thead><tr><th /><th className="text-zinc-500 font-normal">PRED {neg}</th><th className="text-zinc-500 font-normal">PRED {pos}</th></tr></thead>
            <tbody>
              <tr><td className="text-zinc-500 pr-1">TRUE {neg}</td>{cell(t.tn[i], true)}{cell(t.fp[i], false)}</tr>
              <tr><td className="text-zinc-500 pr-1">TRUE {pos}</td>{cell(t.fn[i], false)}{cell(t.tp[i], true)}</tr>
            </tbody>
          </table>
          <Stats items={[['PRECISION', f3(t.precision[i])], ['RECALL', f3(t.recall[i])], ['F1', f3(t.f1[i])], ['ACCURACY', f3(t.accuracy[i])], ['SPECIFICITY', f3(t.specificity[i])]]} />
          <p className="mt-1 text-[10px] text-zinc-500 leading-snug">At {thr.toFixed(2)} the model flags {fInt(t.tp[i] + t.fp[i])} rows as {pos}: it catches {fPct(t.recall[i], 0)} of real {pos} rows and {fPct(1 - t.precision[i], 0)} of its flags are false alarms.</p>
        </div>
      </div>
    </Card>
  )
}

function CalibrationCard({ perf }) {
  const c = perf.calibration
  if (!c) return null
  return (
    <Card testid="card-calibration" title="Calibration" help={HELP.calibration}>
      <Box>
        <ComposedChart data={c.bins} margin={M}>
          {gridLines}
          <XAxis type="number" dataKey="predicted" domain={[0, 1]} {...axis} {...xAx('mean predicted probability')} />
          <YAxis {...yAx('observed share positive')} type="number" domain={[0, 1]} {...axis} width={48} />
          <Tooltip {...tip} formatter={(v, n) => [n === 'count' ? fInt(v) : f3(v), n]} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#52525b" strokeDasharray="3 3" />
          <Line dataKey="observed" name="observed share positive" stroke={K.cyan} isAnimationActive={false} />
        </ComposedChart>
      </Box>
      <Stats items={[['BRIER SCORE', f3(c.brier)], ['', 'lower is better']]} />
    </Card>
  )
}

function SeparationCard({ perf }) {
  const s = perf.separation
  if (!s) return null
  const [neg, pos] = perf.labels
  const rows = s.negative.map((n, i) => ({ bin: s.edges[i].toFixed(2), [neg]: n, [pos]: s.positive[i] }))
  return (
    <Card testid="card-separation" title="Class separation" help={HELP.separation}>
      <Box>
        <BarChart data={rows} margin={M} barCategoryGap={1}>
          {gridLines}
          <XAxis dataKey="bin" {...axis} {...xAx(`predicted probability of ${pos}`)} />
          <YAxis {...yAx('rows')} {...axis} width={52} />
          <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} />
          <Legend {...legend} />
          <Bar dataKey={neg} fill={K.cyan} fillOpacity={0.7} isAnimationActive={false} />
          <Bar dataKey={pos} fill={K.orange} fillOpacity={0.8} isAnimationActive={false} />
        </BarChart>
      </Box>
    </Card>
  )
}

function ConfusionCard({ perf }) {
  const m = perf.confusion
  const labels = perf.labels
  return (
    <Card testid="card-confusion" title="Confusion matrix" help={HELP.confusion}>
      <div className="overflow-x-auto">
        <table className="border-collapse text-[10px]">
          <thead><tr><th className="px-1 text-zinc-500 font-normal">TRUE \ PRED</th>{labels.map((l) => <th key={l} className="px-1 text-zinc-400 font-normal truncate max-w-[70px]">{l}</th>)}</tr></thead>
          <tbody>
            {m.map((row, i) => {
              const total = row.reduce((a, b) => a + b, 0) || 1
              return (
                <tr key={labels[i]}>
                  <td className="px-1 text-zinc-400">{labels[i]}</td>
                  {row.map((v, j) => (
                    <td key={j} title={`${fPct(v / total)} of true ${labels[i]}`} className="w-14 h-7 text-center border border-zinc-900 text-white"
                      style={{ background: i === j ? `rgba(0,230,0,${0.15 + 0.6 * (v / total)})` : `rgba(255,0,0,${0.05 + 0.6 * (v / total)})` }}>{fInt(v)}</td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ------------------------------------------------------------------ regression performance */
function ScatterCard({ title, help, rows, x, y, xName, yName, diag, zero, testid }) {
  const xs = rows.map((r) => r[x])
  const lo = Math.min(...xs, ...(diag ? rows.map((r) => r[y]) : []))
  const hi = Math.max(...xs, ...(diag ? rows.map((r) => r[y]) : []))
  return (
    <Card testid={testid} title={title} help={help}>
      <Box>
        <ScatterChart margin={M}>
          {gridLines}
          <XAxis type="number" dataKey={x} name={xName} domain={diag ? [lo, hi] : ['auto', 'auto']} {...axis} tickFormatter={fSig} {...xAx(xName)} />
          <YAxis {...yAx(yName)} type="number" dataKey={y} name={yName} domain={diag ? [lo, hi] : ['auto', 'auto']} {...axis} width={64} tickFormatter={fSig} />
          <Tooltip {...tip} formatter={(v, n) => [fSig(v), n]} />
          {diag && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={K.yellow} strokeDasharray="3 3" />}
          {zero && <ReferenceLine y={0} stroke={K.yellow} strokeDasharray="3 3" />}
          <Scatter data={rows} fill={K.cyan} fillOpacity={0.55} isAnimationActive={false} shape={(p) => <circle cx={p.cx} cy={p.cy} r={2.2} fill={K.cyan} fillOpacity={0.6} />} />
        </ScatterChart>
      </Box>
    </Card>
  )
}

function HistCard({ title, help, hist, color = K.cyan, xName, extra, testid, marks = [] }) {
  const rows = hist.counts.map((c, i) => ({ x: (hist.edges[i] + hist.edges[i + 1]) / 2, c }))
  return (
    <Card testid={testid} title={title} help={help}>
      <Box>
        <BarChart data={rows} margin={M} barCategoryGap={1}>
          {gridLines}
          <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} {...axis} tickFormatter={fSig} {...xAx(xName)} />
          <YAxis {...yAx('rows')} {...axis} width={52} />
          <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v) => [fInt(v), 'rows']} labelFormatter={(v) => `~${fSig(v)}`} />
          {marks.map((m) => <ReferenceLine key={m} x={m} stroke={K.yellow} strokeDasharray="3 3" />)}
          <Bar dataKey="c" fill={color} isAnimationActive={false} />
        </BarChart>
      </Box>
      {extra}
    </Card>
  )
}

function RegressionPerf({ perf }) {
  return (
    <>
      <ScatterCard testid="card-actual-pred" title="Actual vs predicted" help={HELP.actualPred} rows={perf.scatter} x="actual" y="predicted" xName="actual" yName="predicted" diag />
      <ScatterCard testid="card-residuals" title="Residuals vs predicted" help={HELP.residuals} rows={perf.scatter} x="predicted" y="residual" xName="predicted" yName="residual" zero />
      <HistCard testid="card-resid-hist" title="Residual distribution" help={HELP.residHist} hist={perf.histogram} xName="residual (actual - predicted)"
        extra={<Stats items={[['BIAS', fSig(perf.bias)], ['STD', fSig(perf.residual_std)], ['SKEW', fSig(perf.skew)]]} />} />
      <ScatterCard testid="card-qq" title="Normal Q-Q" help={HELP.qq} rows={perf.qq} x="theoretical" y="sample" xName="theoretical quantile" yName="residual quantile" diag />
    </>
  )
}

/* ------------------------------------------------------------------ coefficient models */
function CoefBars({ rows, valueKey = 'coef', inference, xName = 'coefficient' }) {
  const data = rows.slice(0, 15).map((r) => ({ ...r, v: r[valueKey], err: inference && isNum(r.lo) ? [r[valueKey] - r.lo, r.hi - r[valueKey]] : undefined }))
  return (
    <div style={{ height: Math.max(180, data.length * 18 + 70) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={M}>
          {gridLines}
          <XAxis {...xAx(xName)} type="number" {...axis} tickFormatter={fSig} />
          <YAxis {...yAx('feature')} type="category" dataKey="feature" {...axis} width={146} tick={{ fill: '#d4d4d8', fontSize: 9 }} interval={0} tickFormatter={clipName} />
          <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} content={({ active, payload }) => {
            if (!active || !payload || !payload.length) return null
            const r = payload[0].payload
            return (
              <div className="bg-zinc-950 border border-zinc-600 px-2 py-1 text-[10px] text-zinc-200">
                <div className="font-bold text-white">{r.feature}</div>
                <div>coefficient {fSig(r.v)}</div>
                {isNum(r.oddsShown) && <div>odds ratio {fSig(r.oddsShown)} (x{fSig(r.oddsShown)} odds per unit)</div>}
                {isNum(r.lo) && <div>95% CI {fSig(r.lo)} to {fSig(r.hi)}</div>}
                {isNum(r.p) && <div>p-value {pValue(r.p)} {r.p < 0.05 ? '(significant)' : '(could be noise)'}</div>}
                {isNum(r.vif) && <div>VIF {fSig(r.vif)} {r.vif > 10 ? '(collinear)' : ''}</div>}
              </div>
            )
          }} />
          <ReferenceLine x={0} stroke="#71717a" />
          <Bar dataKey="v" isAnimationActive={false}>
            {data.map((r) => <Cell key={r.feature} fill={r.zero ? '#3f3f46' : r.v >= 0 ? K.cyan : K.orange} fillOpacity={isNum(r.p) && r.p >= 0.05 ? 0.35 : 0.9} />)}
            {inference && <ErrorBar dataKey="err" direction="x" width={3} stroke="#e4e4e7" />}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function PathCard({ path, xKey, xName, scoreKey, scoreName, testid, onPick, current }) {
  const rows = path.points.map((p) => ({ ...p, x: p[xKey] }))
  return (
    <Card testid={testid} title="Regularization path" help={HELP.path}>
      <Box h="h-72">
        <ComposedChart data={rows} margin={M} onClick={(e) => e && e.activePayload && onPick(e.activePayload[0].payload.x)}>
          {gridLines}
          <XAxis dataKey="x" type="number" scale="log" domain={['dataMin', 'dataMax']} {...axis} tickFormatter={fSig} {...xAx(`${xName} (log scale, click to use)`)} />
          <YAxis {...yAx('coefficient')} yAxisId="c" {...axis} width={60} tickFormatter={fSig} />
          <YAxis {...yAx(scoreName, true)} yAxisId="s" orientation="right" {...axis} width={50} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} />
          <Tooltip {...tip} formatter={(v, n) => [fSig(v), n]} labelFormatter={(v) => `${xName} ${fSig(v)}`} />
          <Legend {...legend} />
          {isNum(current) && <ReferenceLine yAxisId="c" x={current} stroke={K.yellow} strokeDasharray="3 3" />}
          {path.features.map((f, i) => <Line key={f} yAxisId="c" dataKey={f} stroke={PALETTE[i % PALETTE.length]} dot={false} strokeWidth={1.3} isAnimationActive={false} />)}
          <Line yAxisId="s" dataKey={scoreKey} name={scoreName} stroke="#fff" strokeDasharray="4 2" dot={false} isAnimationActive={false} />
        </ComposedChart>
      </Box>
    </Card>
  )
}

function LinearCards({ c, onParams, values }) {
  const L = c.linear
  return (
    <>
      <Card testid="card-coef" wide title="Coefficients" help={HELP.coef}>
        <div className="mb-1 text-[11px] text-zinc-200 break-words"><span className="text-zinc-500 text-[10px]">FITTED EQUATION </span>y = {L.equation}</div>
        <CoefBars rows={L.coefficients} inference={L.inference} xName="change in target per unit of feature" />
        <Stats items={[['TERMS', L.n_coefficients], L.inference && ['SIGNIFICANT (P<0.05)', L.n_significant], !L.inference && ['', 'intervals and p-values: set regularization to none']]} />
        <CoefTable rows={L.coefficients.slice(0, 12)} inference={L.inference} />
      </Card>
      <PathCard testid="card-path" path={L.path} xKey="alpha" xName="alpha" scoreKey="test_r2" scoreName="test R2" current={L.path.current}
        onPick={(a) => onParams({ alpha: Number(a.toPrecision(3)), ...(values.regularization === 'none' ? { regularization: L.path.kind } : {}) })} />
    </>
  )
}

function CoefTable({ rows, inference, odds }) {
  return (
    <div className="overflow-x-auto mt-1">
      <table className="w-full text-[10px] border-collapse">
        <thead>
          <tr className="text-zinc-500 text-left">
            <th className="font-normal pr-2">FEATURE</th><th className="font-normal pr-2">COEF</th>
            {odds && <th className="font-normal pr-2">ODDS RATIO</th>}
            {inference && <><th className="font-normal pr-2">SE</th><th className="font-normal pr-2">95% CI</th><th className="font-normal pr-2">P</th></>}
            {!odds && <th className="font-normal pr-2">VIF</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const v = Array.isArray(r.coef) ? r.coef[0] : r.coef
            return (
              <tr key={r.feature} className="border-t border-zinc-800 text-zinc-200">
                <td className="pr-2 truncate max-w-[160px]" title={r.feature}>{r.feature}</td>
                <td className="pr-2" style={{ color: v >= 0 ? K.cyan : K.orange }}>{fSig(v)}</td>
                {odds && <td className="pr-2">{fSig(Array.isArray(r.odds) ? r.odds[0] : r.odds)}</td>}
                {inference && <><td className="pr-2">{fSig(r.se)}</td><td className="pr-2">{fSig(r.lo)} .. {fSig(r.hi)}</td>
                  <td className="pr-2" style={{ color: isNum(r.p) && r.p < 0.05 ? K.green : K.dim }}>{pValue(r.p)}</td></>}
                {!odds && <td className="pr-2" style={{ color: isNum(r.vif) && r.vif > 10 ? K.yellow : undefined }}>{fSig(r.vif)}</td>}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function LogisticCards({ c, onParams, values }) {
  const L = c.logistic
  const [cls, setCls] = useState(0)
  const ci = Math.min(cls, L.classes.length - 1)
  const rows = L.coefficients.map((r) => ({ ...r, v: r.coef[ci], oddsShown: r.odds[ci], coef: r.coef[ci] }))
    .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
  return (
    <>
      <Card testid="card-odds" wide title="Coefficients and odds ratios" help={HELP.odds}
        right={L.classes.length > 1 ? <Seg value={ci} onChange={setCls} options={L.classes.map((l, i) => [i, l])} /> : <span className="text-[10px] text-zinc-500">TOWARDS {L.classes[0]}</span>}>
        <CoefBars rows={rows} valueKey="coef" inference={L.inference && L.classes.length === 1} xName="change in log-odds per unit of feature" />
        <Stats items={[['TERMS', L.n_coefficients], ['ZEROED BY PENALTY', L.n_zero], L.inference && ['SIGNIFICANT (P<0.05)', L.n_significant], !L.inference && ['', 'intervals and p-values: penalty none, binary target']]} />
        <CoefTable rows={rows.slice(0, 12)} inference={L.inference && L.classes.length === 1} odds />
      </Card>
      <PathCard testid="card-path" path={L.path} xKey="C" xName="C" scoreKey="test_score" scoreName="test F1" current={L.path.current}
        onPick={(v) => onParams({ C: Number(v.toPrecision(3)), ...(values.penalty === 'none' ? { penalty: L.path.penalty } : {}) })} />
    </>
  )
}

/* ------------------------------------------------------------------ decision surface */
function regColor(t) {
  const c = Math.max(0, Math.min(1, t))
  return `hsl(${Math.round(220 - 200 * c)}, 85%, ${Math.round(35 + 20 * c)}%)`
}

function BoundaryCard({ b, labels, task, title = 'Decision surface (2D view)' }) {
  if (!b) return null
  const W = 320
  const H = 240
  const { x, y, grid, n, points } = b
  const sx_ = (v) => ((v - x[0]) / (x[n - 1] - x[0])) * W
  const sy_ = (v) => H - ((v - y[0]) / (y[n - 1] - y[0])) * H
  const cw = W / n + 0.5
  const ch = H / n + 0.5
  const reg = task === 'regression'
  const vals = reg ? grid : []
  const lo = reg ? Math.min(...vals) : 0
  const hi = reg ? Math.max(...vals) : 1
  const color = (v) => (reg ? regColor((v - lo) / ((hi - lo) || 1)) : PALETTE[Math.round(v) % PALETTE.length])
  return (
    <Card testid="card-boundary" title={title} help={HELP.boundary}>
      <div className="flex items-stretch gap-1 max-w-[480px]">
        <div className="w-4 shrink-0 flex items-center justify-center">
          <span className="text-[10px] text-zinc-400 whitespace-nowrap -rotate-90">PC2 ({fPct(b.explained[1], 0)} of variance)</span>
        </div>
        <div className="flex-1 min-w-0">
          <svg viewBox={`0 0 ${W} ${H}`} className="block w-full h-auto bg-zinc-950">
            {grid.map((v, i) => (
              <rect key={i} x={(i % n) * (W / n)} y={H - (Math.floor(i / n) + 1) * (H / n)} width={cw} height={ch} fill={color(v)} fillOpacity={reg ? 0.45 : 0.22} />
            ))}
            {points.map((p, i) => (
              <circle key={i} cx={sx_(p.x)} cy={sy_(p.y)} r={p.sv ? 3 : 2} fill={color(p.v)} fillOpacity={0.9} stroke={p.sv ? '#fff' : '#000'} strokeWidth={p.sv ? 1 : 0.4} />
            ))}
          </svg>
          <div className="text-center text-[10px] text-zinc-400 mt-[2px]">PC1 ({fPct(b.explained[0], 0)} of variance)</div>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 mt-1 text-[10px]">
        {!reg && labels.map((l, i) => <span key={l} className="flex items-center gap-1"><span className="inline-block w-2 h-2" style={{ background: PALETTE[i % PALETTE.length] }} />{l}</span>)}
        {reg && <span className="text-zinc-400">blue = low, red = high ({fSig(lo)} .. {fSig(hi)})</span>}
        {b.n_support !== null && b.n_support !== undefined && <span className="text-zinc-300">white rings = support vectors</span>}
      </div>
      <Stats items={[['PC1 + PC2 VARIANCE', fPct(b.explained[0] + b.explained[1], 0)], [`2D COPY ${b.fidelity_metric}`, f3(b.fidelity)]]} />
    </Card>
  )
}

/* ------------------------------------------------------------------ KNN */
function KnnCards({ c, labels, task, row, setRow }) {
  const k = c.knn
  const ex = k.explain
  const buckets = k.by_distance.map((b) => ({ ...b, name: `${b.bucket} (${fSig(b.from)}-${fSig(b.to)})` }))
  return (
    <>
      <Card testid="card-knn-dist" title="Neighbour distances" help={HELP.knnDist}>
        <div className="grid grid-cols-2 gap-1">
          <Box h="h-56">
            <BarChart data={k.distance_hist.counts.map((cnt, i) => ({ x: (k.distance_hist.edges[i] + k.distance_hist.edges[i + 1]) / 2, cnt }))} margin={M} barCategoryGap={1}>
              <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} {...axis} tickFormatter={fSig} {...xAx('distance to k-th neighbour')} />
              <YAxis {...yAx('rows')} {...axis} width={46} />
              <Tooltip {...tip} formatter={(v) => [fInt(v), 'rows']} labelFormatter={(v) => `~${fSig(v)}`} />
              <Bar dataKey="cnt" fill={K.cyan} isAnimationActive={false} />
            </BarChart>
          </Box>
          <Box h="h-56">
            <BarChart data={buckets} margin={M}>
              <XAxis dataKey="bucket" {...axis} {...xAx('nearest -> farthest')} />
              <YAxis {...yAx(k.by_distance_metric)} {...axis} width={50} domain={task === 'regression' ? [0, 'auto'] : [0, 1]} tickFormatter={fSig} />
              <Tooltip {...tip} formatter={(v) => [fSig(v), k.by_distance_metric]} labelFormatter={(_, p) => (p && p[0] ? `${p[0].payload.name}, ${p[0].payload.rows} rows` : '')} />
              <Bar dataKey="score" fill={K.orange} isAnimationActive={false} />
            </BarChart>
          </Box>
        </div>
        <Stats items={[['MEAN DISTANCE', fSig(k.mean_distance)], ['RIGHT CHART', k.by_distance_metric]]} />
      </Card>
      <Card testid="card-knn-explain" wide title="Explain one prediction" help={HELP.explain}
        right={<span className="flex items-center gap-1">
          <span className={lbl}>TEST ROW</span>
          <Btn onClick={() => setRow(Math.max(0, row - 1))}>{'<'}</Btn>
          <NumBox value={ex.row} onCommit={(v) => setRow(Math.max(0, Math.min(ex.n_rows - 1, Math.round(v))))} />
          <Btn onClick={() => setRow(Math.min(ex.n_rows - 1, row + 1))}>{'>'}</Btn>
          <span className="text-[10px] text-zinc-500">of {fInt(ex.n_rows)}</span>
        </span>}>
        <Stats items={[['ACTUAL', ex.actual], ['PREDICTED', ex.predicted, ex.actual === ex.predicted ? K.green : K.yellow], Object.keys(ex.votes).length > 0 && ['VOTES', Object.entries(ex.votes).map(([l, n]) => `${l}: ${n}`).join('  ')]]} />
        <div className="overflow-x-auto mt-1">
          <table className="text-[10px] border-collapse w-full">
            <thead>
              <tr className="text-zinc-500 text-left"><th className="font-normal pr-2">#</th><th className="font-normal pr-2">DIST</th><th className="font-normal pr-2">LABEL</th>{ex.columns.map((col) => <th key={col} className="font-normal pr-2 truncate max-w-[90px]">{col}</th>)}</tr>
            </thead>
            <tbody>
              <tr className="text-white font-bold border-t border-zinc-700"><td className="pr-2">ROW</td><td className="pr-2">—</td><td className="pr-2">{String(ex.actual)}</td>{ex.columns.map((col) => <td key={col} className="pr-2">{String(ex.values[col])}</td>)}</tr>
              {ex.neighbours.map((nb, i) => (
                <tr key={i} className="border-t border-zinc-800 text-zinc-300">
                  <td className="pr-2">{i + 1}</td><td className="pr-2">{fSig(nb.distance)}</td>
                  <td className="pr-2" style={{ color: task === 'classification' ? PALETTE[labels.indexOf(nb.label) % PALETTE.length] : undefined }}>{String(nb.label)}</td>
                  {ex.columns.map((col) => <td key={col} className="pr-2" style={{ color: nb.values[col] === ex.values[col] ? K.text : '#a1a1aa' }}>{String(nb.values[col])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  )
}

/* ------------------------------------------------------------------ trees */
function layoutTree(root) {
  const nodes = []
  const edges = []
  let slot = 0
  const walk = (n, depth) => {
    let x
    if (n.children && n.children.length) {
      const xs = n.children.map((ch) => walk(ch, depth + 1))
      x = (xs[0] + xs[xs.length - 1]) / 2
      n.children.forEach((ch, i) => edges.push({ from: n, to: ch, yes: i === 0 }))
    } else {
      x = slot
      slot += 1
    }
    n._x = x
    n._d = depth
    nodes.push(n)
    return x
  }
  walk(root, 0)
  return { nodes, edges, slots: slot }
}

const DARK = '#0b0b0d'
/** low -> high colour for regression node values (cyan -> orange) */
const ramp = (v, lo, hi) => {
  const f = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0.5
  const a = [0, 213, 255]
  const b = [251, 146, 60]
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * f)).join(',')})`
}

function TreeLegend({ t, labels, task, xgb }) {
  const sw = (color, text, key) => (
    <span key={key} className="inline-flex items-center gap-1 mr-3">
      <span className="inline-block w-3 h-3" style={{ background: color }} />
      <span className="text-white font-bold">{text}</span>
    </span>
  )
  let swatches
  if (xgb) swatches = [sw(K.orange, 'PUSHES TOWARDS THE POSITIVE CLASS', 'p'), sw(K.cyan, 'PUSHES AWAY FROM IT', 'n')]
  else if (task === 'classification') {
    swatches = labels.map((l, i) => sw(PALETTE[i % PALETTE.length], `${String(l).toUpperCase()}${labels.length === 2 && i === 1 ? ' (POSITIVE)' : ''}`, l))
  } else if (t.pred_range) {
    swatches = [
      <span key="r" className="inline-flex items-center gap-1 mr-3 text-white font-bold">
        LOW {fSig(t.pred_range[0])}
        <span className="inline-block w-24 h-3" style={{ background: `linear-gradient(90deg, ${ramp(0, 0, 1)}, ${ramp(1, 0, 1)})` }} />
        HIGH {fSig(t.pred_range[1])}
      </span>,
    ]
  }
  return (
    <div className="mb-2 text-[10px] uppercase" data-testid="tree-legend">
      <div className="flex flex-wrap items-center gap-y-1">
        {swatches}
        <span className="text-zinc-400 normal-case">
          {xgb
            ? 'Colour = sign of the leaf value (log-odds added to the score).'
            : task === 'classification'
              ? 'Colour = what the tree predicts for rows that end in that box. The bar shows the real mix of training rows there.'
              : 'Colour = the average target value the tree predicts there.'}
          {' '}Left branch = <span className="text-[#a3e635]">yes</span>, right = <span className="text-[#f87171]">no</span>.
        </span>
      </div>
      {t.weighted && (
        <div data-testid="tree-weighted-note" className="mt-1 px-1 py-0.5 normal-case text-black font-bold" style={{ background: '#facc15' }}>
          Class weight is on: the tree counts each row of the rarer class more, so a box can predict it even when most of its
          rows are the other class. The bar and percentages show the real rows.
        </div>
      )}
    </div>
  )
}

/** SVG text never wraps: squeeze a line into ``w`` px when its estimated width (monospace, ~0.62em per glyph) is wider */
function FitText({ x, y, w, size, children, ...rest }) {
  const text = String(children ?? '')
  const tooWide = text.length * size * 0.64 > w
  return (
    <text x={x} y={y} fontSize={size} textAnchor="middle" {...rest}
      {...(tooWide ? { textLength: w, lengthAdjust: 'spacingAndGlyphs' } : {})}>
      {text}
    </text>
  )
}

function TreeView({ t, labels, task, xgb }) {
  const { nodes, edges, slots } = useMemo(() => layoutTree(JSON.parse(JSON.stringify(t.root))), [t])
  const BW = 240
  const BH = 92
  const GX = 256
  const GY = 128
  const TW = BW - 14 // usable text width inside a box
  const depth = Math.max(...nodes.map((n) => n._d))
  const W = Math.max(slots * GX, GX)
  const H = (depth + 1) * GY
  const px = (n) => n._x * GX + GX / 2
  const py = (n) => n._d * GY + 4
  const cls = task === 'classification' && !xgb
  const total = t.total || (t.root && t.root.samples) || 0
  const color = (n) => {
    if (xgb) return n.leaf ? (n.pred >= 0 ? K.orange : K.cyan) : '#71717a'
    if (cls) {
      const i = labels.indexOf(n.pred)
      return PALETTE[(i < 0 ? 0 : i) % PALETTE.length]
    }
    return t.pred_range ? ramp(n.pred, t.pred_range[0], t.pred_range[1]) : K.cyan
  }
  const clip = (s, m) => (s && s.length > m ? `${s.slice(0, m - 1)}…` : s)
  const header = (n) => {
    if (xgb) return n.leaf ? `LEAF  VALUE ${fSig(n.pred)}` : `SPLIT  GAIN ${fSig(n.gain)}`
    if (cls) return `${n.leaf ? 'LEAF → ' : ''}PREDICTS ${clip(String(n.pred).toUpperCase(), 18)}`
    return `${n.leaf ? 'LEAF → ' : ''}PREDICTS ${fSig(n.pred)}`
  }
  const mixText = (n) => {
    if (!n.mix) return isNum(n.confidence) ? `${fPct(n.confidence, 0)} weighted share` : ''
    if (labels.length === 2) return `${fPct(n.mix[1], 0)} are ${clip(String(labels[1]), 12)} (${fInt(n.counts[1])} of ${fInt(n.samples)})`
    const j = labels.indexOf(n.pred)
    return `${fPct(n.mix[j < 0 ? 0 : j], 0)} are ${clip(String(n.pred), 12)}`
  }
  return (
    <div data-testid="tree-wrap">
      <TreeLegend t={t} labels={labels} task={task} xgb={xgb} />
      <div className="overflow-x-auto" data-testid="tree-view">
        <svg width={W} height={H} className="block">
          {edges.map((e, i) => (
            <g key={i}>
              <line x1={px(e.from)} y1={py(e.from) + BH} x2={px(e.to)} y2={py(e.to)} stroke="#71717a" strokeWidth={1.2} />
              <text x={(px(e.from) + px(e.to)) / 2 + (e.yes ? -12 : 12)} y={(py(e.from) + BH + py(e.to)) / 2} fill={e.yes ? '#a3e635' : '#f87171'} fontSize={10} fontWeight="bold" textAnchor="middle">{e.yes ? 'yes' : 'no'}</text>
            </g>
          ))}
          {nodes.map((n) => {
            const c = color(n)
            const x0 = px(n) - BW / 2
            const y0 = py(n)
            const barW = BW - 16
            let acc = 0
            return (
              <g key={n.id} data-testid="tree-node">
                <title>{[
                  n.leaf ? 'Final group (leaf)' : `Question: ${n.yes} ?`,
                  `${n.samples} training rows${total ? ` (${fPct(n.samples / total, 1)} of all)` : ''}`,
                  `Prediction here: ${n.pred}`,
                  n.counts ? `Real rows: ${n.counts.map((k, i) => `${labels[i]} ${k} (${fPct(n.mix[i], 1)})`).join(', ')}` : '',
                  t.weighted && n.dist ? `Weighted shares the tree uses: ${n.dist.map((d, i) => `${labels[i]} ${fPct(d, 0)}`).join(', ')}` : '',
                  n.hidden ? `${n.hidden} more leaves below (raise SHOW DEPTH)` : '',
                ].filter(Boolean).join('\n')}</title>
                <rect x={x0} y={y0} width={BW} height={BH} fill="#141417" stroke={c} strokeWidth={n.leaf ? 3 : 1.5} />
                <rect x={x0} y={y0} width={BW} height={17} fill={c} />
                <FitText x={px(n)} y={y0 + 12.5} w={TW} size={10} fill={DARK} fontWeight="bold">{header(n)}</FitText>
                <FitText x={px(n)} y={y0 + 32} w={TW} size={10.5} fill="#fff" fontWeight="bold">{n.leaf ? 'final group' : `${clip(n.yes, 40)} ?`}</FitText>
                <FitText x={px(n)} y={y0 + 46} w={TW} size={9.5} fill="#a1a1aa">
                  {xgb ? `cover ${fSig(n.samples)}` : `${fInt(n.samples)} rows${total ? ` · ${fPct(n.samples / total, 0)} of data` : ''}`}
                </FitText>
                {n.mix && n.mix.map((m, i) => {
                  const w = m * barW
                  const el = <rect key={i} x={x0 + 8 + acc} y={y0 + 52} width={w} height={8} fill={PALETTE[i % PALETTE.length]} />
                  acc += w
                  return el
                })}
                <FitText x={px(n)} y={y0 + (n.mix ? 73 : 62)} w={TW} size={9.5} fill="#e4e4e7">{mixText(n)}</FitText>
                {n.hidden && (
                  <FitText x={px(n)} y={y0 + 86} w={TW} size={8.5} fill="#71717a">
                    {`+${n.hidden === true ? 'more' : fInt(n.hidden)} leaves below · raise SHOW DEPTH`}
                  </FitText>
                )}
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}

function TreeCard({ t, labels, task, xgb, depth, setDepth, index, setIndex, title }) {
  return (
    <Card testid="card-tree" wide title={title} help={HELP.tree}
      right={<span className="flex flex-wrap items-center gap-1">
        {setIndex && <><span className={lbl}>TREE</span>
          <Btn onClick={() => setIndex(Math.max(0, index - 1))}>{'<'}</Btn>
          <NumBox value={t.index} onCommit={(v) => setIndex(Math.max(0, Math.min(t.n_trees - 1, Math.round(v))))} />
          <Btn onClick={() => setIndex(Math.min(t.n_trees - 1, index + 1))}>{'>'}</Btn>
          <span className="text-[10px] text-zinc-500">of {fInt(t.n_trees)}</span></>}
        <span className={lbl}>SHOW DEPTH</span>
        <Seg value={depth} onChange={setDepth} options={[[2, '2'], [3, '3'], [4, '4'], [5, '5']]} />
      </span>}>
      <TreeView t={t} labels={labels} task={task} xgb={xgb} />
      <Stats items={[isNum(t.depth) && ['DEPTH', t.depth], ['LEAVES', fInt(t.leaves)], xgb && ['LEAF VALUES', t.units]]} />
    </Card>
  )
}

function ImportanceBars({ rows, valueKey = 'value', color = K.cyan, testid, title, help, right }) {
  const data = rows.slice(0, 15)
  return (
    <Card testid={testid} title={title} help={help} right={right}>
      <div style={{ height: Math.max(160, data.length * 18 + 60) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={M}>
            <XAxis {...xAx('share of importance')} type="number" {...axis} tickFormatter={(v) => fPct(v, 0)} />
            <YAxis {...yAx('feature')} type="category" dataKey="feature" {...axis} width={136} tick={{ fill: '#d4d4d8', fontSize: 9 }} interval={0} tickFormatter={clipName} />
            <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v) => [fPct(v, 1), 'share']} />
            <Bar dataKey={valueKey} fill={color} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function XgbCards({ c, labels, task, extras, setExtras }) {
  const L = c.loss
  const [metric, setMetric] = useState(L.metrics[L.metrics.length - 1])
  const [imp, setImp] = useState('gain')
  const m = L.curves[metric] ? metric : L.metrics[L.metrics.length - 1]
  const rows = L.rounds.map((r, i) => ({ r, train: L.curves[m].train[i], val: L.curves[m].val[i] }))
  return (
    <>
      <Card testid="card-loss" wide title="Boosting loss per round" help={HELP.loss} right={<Seg value={m} onChange={setMetric} options={L.metrics.map((x) => [x, x])} />}>
        <Box>
          <LineChart data={rows} margin={M}>
            {gridLines}
            <XAxis dataKey="r" type="number" domain={[1, 'dataMax']} {...axis} {...xAx('boosting round (trees)')} />
            <YAxis {...yAx(`loss (${m})`)} {...axis} width={60} domain={['auto', 'auto']} tickFormatter={fSig} />
            <Tooltip {...tip} formatter={(v, n) => [fSig(v), n]} labelFormatter={(v) => `round ${v}`} />
            <Legend {...legend} />
            <ReferenceLine x={L.best_iteration} stroke={K.green} strokeDasharray="3 3" label={{ value: `BEST ${L.best_iteration}`, fill: K.green, fontSize: 9, position: 'top' }} />
            <Line dataKey="train" name="fit rows" stroke={K.orange} dot={false} isAnimationActive={false} />
            <Line dataKey="val" name="validation rows" stroke={K.cyan} dot={false} isAnimationActive={false} />
          </LineChart>
        </Box>
        <Stats items={[['ROUNDS', L.n_rounds], ['BEST ROUND', L.best_iteration, K.green], ['EARLY STOPPED', L.early_stopped ? 'YES' : 'NO'], ['FIT / VALIDATION ROWS', `${fInt(L.inner_rows)} / ${fInt(L.val_rows)}`], L.loss_units && ['UNITS', L.loss_units]]} />
      </Card>
      <ImportanceBars testid="card-importance" title="Feature importance" help={HELP.xgbImp} rows={c.importance[imp]}
        right={<Seg value={imp} onChange={setImp} options={[['gain', 'GAIN'], ['splits', 'SPLITS'], ['cover', 'COVER']]} />} />
      <TreeCard t={c.tree} labels={labels} task={task} xgb title="Inspect a boosted tree" depth={extras.treeDepth} setDepth={(v) => setExtras({ treeDepth: v })}
        index={extras.treeIndex} setIndex={(v) => setExtras({ treeIndex: v })} />
    </>
  )
}

function DtCards({ c, labels, task, extras, setExtras, onParams, values }) {
  const P = c.pruning
  return (
    <>
      <TreeCard t={c.tree} labels={labels} task={task} title="The fitted tree" depth={extras.treeDepth} setDepth={(v) => setExtras({ treeDepth: v })} />
      <Card testid="card-pruning" title="Pruning path" help={HELP.pruning}>
        <Box>
          <ComposedChart data={P.points.map((p, i) => ({ ...p, i: String(i) }))} margin={M}
            onClick={(e) => e && e.activePayload && onParams({ ccp_alpha: Number(Math.min(0.05, e.activePayload[0].payload.alpha).toPrecision(3)) })}>
            {gridLines}
            <XAxis dataKey="i" {...axis} tickFormatter={(i) => fSig(P.points[Number(i)].alpha)} {...xAx('ccp_alpha (click to use)')} />
            <YAxis {...yAx('score')} yAxisId="s" {...axis} width={52} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} />
            <YAxis {...yAx('leaves', true)} yAxisId="l" orientation="right" {...axis} width={52} />
            <Tooltip {...tip} formatter={(v, n) => [n === 'leaves' ? fInt(v) : f3(v), n]} labelFormatter={(i) => `alpha ${fSig(P.points[Number(i)].alpha)}`} />
            <Legend {...legend} />
            <Bar yAxisId="l" dataKey="leaves" fill="#3f3f46" isAnimationActive={false} />
            <Line yAxisId="s" dataKey="train" stroke={K.orange} isAnimationActive={false} />
            <Line yAxisId="s" dataKey="test" stroke={K.cyan} isAnimationActive={false} />
          </ComposedChart>
        </Box>
        <Stats items={[['CURRENT ALPHA', fSig(values.ccp_alpha)]]} />
      </Card>
      <ImportanceBars testid="card-importance" title="Feature importance" help={HELP.importance} rows={c.importance} />
      <Card testid="card-rules" wide title="Decision rules" help={HELP.rules}>
        <div className="space-y-1">
          {c.rules.map((r, i) => (
            <div key={i} className="text-[10px] leading-snug border-t border-zinc-800 pt-1">
              <span className="text-zinc-500">IF </span>
              <span className="text-zinc-200">{r.conditions.join('  AND  ') || '(all rows)'}</span>
              <span className="text-zinc-500"> THEN </span>
              <span className="font-bold" style={{ color: task === 'classification' ? PALETTE[labels.indexOf(r.pred) % PALETTE.length] : K.cyan }}>{task === 'classification' ? r.pred : fSig(r.pred)}</span>
              <span className="text-zinc-500"> // {fInt(r.samples)} rows{isNum(r.confidence) && task === 'classification' ? `, ${fPct(r.confidence, 0)} of them are ${r.pred}` : ''}</span>
            </div>
          ))}
        </div>
      </Card>
    </>
  )
}

function RfCards({ c, labels, task, extras, setExtras }) {
  const g = c.growth
  const A = c.agreement
  const metric = task === 'classification' ? 'accuracy' : 'R2'
  return (
    <>
      <Card testid="card-growth" title="Score vs number of trees" help={HELP.growth}>
        <Box>
          <LineChart data={g} margin={M}>
            {gridLines}
            <XAxis dataKey="trees" type="number" domain={['dataMin', 'dataMax']} {...axis} {...xAx('trees')} />
            <YAxis {...yAx(metric)} {...axis} width={56} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(3)} />
            <Tooltip {...tip} formatter={(v, n) => [f3(v), n]} labelFormatter={(v) => `${v} trees`} />
            <Legend {...legend} />
            <Line dataKey="oob" name={`out-of-bag ${metric}`} stroke={K.pink} isAnimationActive={false} connectNulls />
            <Line dataKey="test" name={`test ${metric}`} stroke={K.cyan} isAnimationActive={false} />
          </LineChart>
        </Box>
      </Card>
      <Card testid="card-forest-imp" title="Importance: MDI vs permutation" help={HELP.forestImp}>
        <div style={{ height: Math.max(190, Math.min(15, c.importance.length) * 22 + 96) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={c.importance.slice(0, 15)} layout="vertical" margin={M}>
              <XAxis {...xAx('importance')} type="number" {...axis} tickFormatter={fSig} />
              <YAxis {...yAx('feature')} type="category" dataKey="feature" {...axis} width={126} tick={{ fill: '#d4d4d8', fontSize: 9 }} interval={0} tickFormatter={clipName} />
              <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v, n) => [f3(v), n]} />
              <Legend {...legend} />
              <Bar dataKey="mdi" name="MDI (share)" fill={K.orange} isAnimationActive={false} />
              <Bar dataKey="perm" name="permutation (score drop)" fill={K.cyan} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card testid="card-agreement" title={A.kind === 'agreement' ? 'Tree agreement' : 'Tree spread'} help={A.kind === 'agreement' ? HELP.agreement : HELP.spread}>
        <div className="grid grid-cols-2 gap-1">
          <Box h="h-56">
            <BarChart data={A.hist.counts.map((cnt, i) => ({ x: (A.hist.edges[i] + A.hist.edges[i + 1]) / 2, cnt }))} margin={M} barCategoryGap={1}>
              <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} {...axis} tickFormatter={A.kind === 'agreement' ? (v) => fPct(v, 0) : fSig} {...xAx(A.kind === 'agreement' ? 'trees agreeing' : 'std of tree predictions')} />
              <YAxis {...yAx('rows')} {...axis} width={46} />
              <Tooltip {...tip} formatter={(v) => [fInt(v), 'rows']} labelFormatter={(v) => `~${fSig(v)}`} />
              <Bar dataKey="cnt" fill={K.purple} isAnimationActive={false} />
            </BarChart>
          </Box>
          <Box h="h-56">
            <BarChart data={A.buckets} margin={M}>
              <XAxis dataKey="bucket" {...axis} {...xAx(A.kind === 'agreement' ? 'agreement' : 'spread quartile')} />
              <YAxis {...yAx(A.kind === 'agreement' ? 'accuracy' : 'MAE')} {...axis} width={50} domain={A.kind === 'agreement' ? [0, 1] : [0, 'auto']} tickFormatter={fSig} />
              <Tooltip {...tip} formatter={(v) => [fSig(v), A.kind === 'agreement' ? 'accuracy' : 'MAE']} />
              <Bar dataKey={A.kind === 'agreement' ? 'accuracy' : 'mae'} fill={K.orange} isAnimationActive={false} />
            </BarChart>
          </Box>
        </div>
        <Stats items={[[A.kind === 'agreement' ? 'MEAN AGREEMENT' : 'MEAN SPREAD', A.kind === 'agreement' ? fPct(A.mean, 0) : fSig(A.mean)], ['RIGHT CHART', A.kind === 'agreement' ? 'accuracy' : 'MAE']]} />
      </Card>
      <TreeCard t={c.tree} labels={labels} task={task} title="Inspect one tree of the forest" depth={extras.treeDepth} setDepth={(v) => setExtras({ treeDepth: v })}
        index={extras.treeIndex} setIndex={(v) => setExtras({ treeIndex: v })} />
    </>
  )
}

function SvmCards({ c, labels, task }) {
  const s = c.svm
  const M = s.margins
  let chart
  if (M.kind === 'binary') {
    const rows = M.edges.slice(0, -1).map((e, i) => ({ x: (e + M.edges[i + 1]) / 2, ...Object.fromEntries(Object.entries(M.by_class).map(([l, cs]) => [l, cs[i]])) }))
    chart = (
      <BarChart data={rows} margin={M} barCategoryGap={0}>
        {gridLines}
        <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} {...axis} tickFormatter={fSig} {...xAx('decision function (signed distance)')} />
        <YAxis {...yAx('rows')} {...axis} width={50} />
        <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} labelFormatter={(v) => `~${fSig(v)}`} />
        <Legend {...legend} />
        {[-1, 0, 1].map((v) => <ReferenceLine key={v} x={v} stroke={v === 0 ? K.yellow : '#71717a'} strokeDasharray="3 3" />)}
        {Object.keys(M.by_class).map((l, i) => <Bar key={l} dataKey={l} fill={PALETTE[i]} fillOpacity={0.75} isAnimationActive={false} />)}
      </BarChart>
    )
  }
  return (
    <>
      {M.kind === 'binary' ? (
        <Card testid="card-margins" title="Margin distribution" help={HELP.margins}>
          <Box>{chart}</Box>
          <Stats items={[['INSIDE MARGIN', fPct(M.inside_margin, 0)]]} />
        </Card>
      ) : (
        <HistCard testid="card-margins" title={M.kind === 'regression' ? 'Errors vs epsilon tube' : 'Class score gap'} help={M.kind === 'regression' ? HELP.tube : HELP.marginsMulti}
          hist={M} color={K.purple} xName={M.kind === 'regression' ? 'prediction error (target sd units)' : 'best minus second-best score'}
          marks={M.kind === 'regression' ? [-M.epsilon, M.epsilon] : []}
          extra={M.kind === 'regression' ? <Stats items={[['INSIDE TUBE', fPct(M.inside_tube, 0)], ['EPSILON', fSig(M.epsilon)]]} /> : null} />
      )}
      <Card testid="card-support" title="Support vectors" help="The training rows that sit on or inside the margin. Only these define the boundary; every other row could be deleted without changing the model.">
        <Stats items={[['SUPPORT VECTORS', fInt(s.n_support)], ['SHARE OF TRAINING ROWS', fPct(s.support_fraction, 0)]]} />
        {s.per_class && (
          <Box h="h-52">
            <BarChart data={Object.entries(s.per_class).map(([l, n]) => ({ l, n }))} margin={M}>
              <XAxis {...xAx('class')} dataKey="l" {...axis} />
              <YAxis {...yAx('support vectors')} {...axis} width={56} />
              <Tooltip {...barTip} cursor={{ fill: '#ffffff10' }} formatter={(v) => [fInt(v), 'support vectors']} />
              <Bar dataKey="n" isAnimationActive={false}>{Object.keys(s.per_class).map((l, i) => <Cell key={l} fill={PALETTE[labels.indexOf(l) >= 0 ? labels.indexOf(l) : i]} />)}</Bar>
            </BarChart>
          </Box>
        )}
      </Card>
    </>
  )
}

/* ------------------------------------------------------------------ K-means */
function KmeansCards({ d }) {
  const c = d.charts
  const S = c.scatter
  const [by, setBy] = useState('cluster')
  const targetLabels = useMemo(() => [...new Set(S.points.map((p) => p.t).filter((t) => t !== undefined))], [S])
  const numericTarget = targetLabels.length && typeof targetLabels[0] === 'number'
  const tLo = numericTarget ? Math.min(...targetLabels) : 0
  const tHi = numericTarget ? Math.max(...targetLabels) : 1
  const colorOf = (p) => (by === 'cluster' || p.t === undefined ? PALETTE[p.c % PALETTE.length]
    : numericTarget ? regColor((p.t - tLo) / ((tHi - tLo) || 1)) : PALETTE[targetLabels.indexOf(p.t) % PALETTE.length])
  const groups = by === 'cluster' ? [...new Set(S.points.map((p) => p.c))].sort((a, b) => a - b) : numericTarget ? ['all'] : targetLabels
  const P = c.profiles
  const k = d.metrics.k
  return (
    <>
      <Card testid="card-clusters" wide title="Clusters (2D view)" help={HELP.clusters}
        right={S.target_name ? <Seg value={by} onChange={setBy} options={[['cluster', 'BY CLUSTER'], ['target', 'BY TARGET']]} /> : null}>
        <Box h="h-72">
          <ScatterChart margin={M}>
            {gridLines}
            <XAxis type="number" dataKey="x" {...axis} tickFormatter={fSig} {...xAx(`PC1${S.explained ? ` (${fPct(S.explained[0], 0)})` : ''}`)} />
            <YAxis {...yAx(`PC2${S.explained ? ` (${fPct(S.explained[1], 0)})` : ''}`)} type="number" dataKey="y" {...axis} width={56} tickFormatter={fSig} />
            <Tooltip {...tip} formatter={(v, n) => [fSig(v), n]} />
            {groups.map((g) => (
              <Scatter key={String(g)} name={by === 'cluster' ? `cluster ${g}` : String(g)} isAnimationActive={false}
                data={S.points.filter((p) => (by === 'cluster' ? p.c === g : numericTarget || p.t === g))}
                shape={(p) => <circle cx={p.cx} cy={p.cy} r={2.2} fill={colorOf(p.payload)} fillOpacity={0.7} />} />
            ))}
            <Scatter name="centres" data={S.centroids} isAnimationActive={false}
              shape={(p) => <g><path d={`M${p.cx - 6} ${p.cy}h12M${p.cx} ${p.cy - 6}v12`} stroke="#fff" strokeWidth={3} /><path d={`M${p.cx - 5} ${p.cy}h10M${p.cx} ${p.cy - 5}v10`} stroke={PALETTE[p.payload.c % PALETTE.length]} strokeWidth={1.5} /></g>} />
          </ScatterChart>
        </Box>
        <div className="flex flex-wrap gap-2 text-[10px]">
          {by === 'cluster' && d.sizes.map((n, i) => <span key={i} className="flex items-center gap-1"><span className="inline-block w-2 h-2" style={{ background: PALETTE[i % PALETTE.length] }} />cluster {i}: {fInt(n)} rows ({fPct(n / d.data.rows_used, 0)})</span>)}
          {by === 'target' && !numericTarget && targetLabels.map((l, i) => <span key={l} className="flex items-center gap-1"><span className="inline-block w-2 h-2" style={{ background: PALETTE[i % PALETTE.length] }} />{l}</span>)}
          {by === 'target' && numericTarget && <span className="text-zinc-400">blue = low target, red = high</span>}
        </div>
      </Card>
      <Card testid="card-silhouette" title="Silhouette per cluster" help={HELP.silhouette}>
        {/* x runs from -0.5 to 1; one strip per cluster, one thin bar per sampled row */}
        <div className="flex items-stretch gap-1">
          <div className="w-4 shrink-0 flex items-center justify-center">
            <span className="text-[10px] text-zinc-400 whitespace-nowrap -rotate-90">rows (sorted)</span>
          </div>
        <div className="flex-1 min-w-0 space-y-[3px]">
          {c.silhouette.map((s) => {
            const xs = (v) => ((v + 0.5) / 1.5) * 300
            return (
              <div key={s.cluster} className="flex items-center gap-1">
                <span className="w-24 shrink-0 text-[10px]" style={{ color: PALETTE[s.cluster % PALETTE.length] }}>C{s.cluster} avg {f3(s.mean)}</span>
                <svg viewBox={`0 0 300 ${Math.max(1, s.values.length)}`} preserveAspectRatio="none" className="flex-1 h-9 bg-zinc-950">
                  {s.values.map((v, i) => (
                    <rect key={i} x={Math.min(xs(0), xs(v))} y={i} width={Math.abs(xs(v) - xs(0))} height={1} fill={v < 0 ? K.red : PALETTE[s.cluster % PALETTE.length]} />
                  ))}
                  <line x1={xs(0)} x2={xs(0)} y1={0} y2={s.values.length} stroke="#71717a" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                  {isNum(d.metrics.silhouette) && <line x1={xs(d.metrics.silhouette)} x2={xs(d.metrics.silhouette)} y1={0} y2={s.values.length} stroke={K.yellow} strokeDasharray="3 2" vectorEffect="non-scaling-stroke" />}
                </svg>
              </div>
            )
          })}
          <div className="flex gap-1 text-[9px] text-zinc-500"><span className="w-24 shrink-0" /><span className="flex-1 flex justify-between"><span>-0.5</span><span>0</span><span>0.5</span><span>1</span></span></div>
          <div className="flex gap-1 text-[10px] text-zinc-400"><span className="w-24 shrink-0" /><span className="flex-1 text-center">silhouette value</span></div>
        </div>
        </div>
        <Stats items={[['OVERALL', f3(d.metrics.silhouette), K.yellow], ['', 'red bars = probably in the wrong cluster']]} />
      </Card>
      <Card testid="card-profiles" wide title="Cluster profiles" help={HELP.profiles}>
        <div className="overflow-x-auto">
          <table className="text-[10px] border-collapse">
            <thead>
              <tr>
                <th className="px-1 text-left text-zinc-500 font-normal">FEATURE</th>
                {Array.from({ length: k }, (_, i) => <th key={i} className="px-1 font-normal" style={{ color: PALETTE[i % PALETTE.length] }}>C{i}</th>)}
              </tr>
            </thead>
            <tbody>
              {P.features.map((f, j) => {
                const col = P.centers.map((row) => row[j])
                const span = Math.max(...col.map(Math.abs)) || 1
                return (
                  <tr key={f}>
                    <td className="px-1 text-zinc-300 truncate max-w-[160px]" title={f}>{f} <span className="text-zinc-600">{P.units[j]}</span></td>
                    {col.map((v, i) => {
                      const t = v / span
                      const rawIdx = P.raw_columns.indexOf(f)
                      const raw = rawIdx >= 0 ? P.raw_means[i][rawIdx] : null
                      return (
                        <td key={i} className="w-16 h-6 text-center text-white border border-zinc-950" title={raw !== null ? `mean ${f} in cluster ${i}: ${fSig(raw)}` : undefined}
                          style={{ background: P.units[j] === 'share' ? `rgba(0,213,255,${0.1 + 0.8 * Math.max(0, v)})` : t >= 0 ? `rgba(251,146,60,${0.1 + 0.8 * t})` : `rgba(0,213,255,${0.1 + 0.8 * -t})` }}>
                          {P.units[j] === 'share' ? fPct(v, 0) : fSig(v)}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
      {c.target && (
        <Card testid="card-cluster-target" title="Clusters vs target" help={HELP.target}>
          {c.target.kind === 'classes' ? (
            <>
              <table className="text-[10px] border-collapse">
                <thead><tr><th className="px-1 text-zinc-500 font-normal text-left">CLUSTER</th>{c.target.labels.map((l) => <th key={l} className="px-1 text-zinc-400 font-normal">{l}</th>)}</tr></thead>
                <tbody>
                  {c.target.counts.map((row, i) => {
                    const tot = row.reduce((a, b) => a + b, 0) || 1
                    return (
                      <tr key={i}>
                        <td className="px-1" style={{ color: PALETTE[i % PALETTE.length] }}>C{i}</td>
                        {row.map((v, j) => <td key={j} className="w-16 h-6 text-center text-white border border-zinc-950" style={{ background: `rgba(184,0,230,${0.08 + 0.7 * (v / tot)})` }}>{fInt(v)} <span className="text-zinc-400">{fPct(v / tot, 0)}</span></td>)}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              <Stats items={[['PURITY', fPct(c.target.purity, 0)]]} />
            </>
          ) : (
            <Box h="h-56">
              <BarChart data={c.target.mean.map((m, i) => ({ c: `C${i}`, mean: m, i }))} margin={M}>
                <XAxis {...xAx('cluster')} dataKey="c" {...axis} />
                <YAxis {...yAx('mean target')} {...axis} width={64} tickFormatter={fSig} />
                <Tooltip {...barTip} cursor={{ fill: '#ffffff10' }} formatter={(v) => [fSig(v), 'mean target']} />
                <Bar dataKey="mean" isAnimationActive={false}>{c.target.mean.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}</Bar>
              </BarChart>
            </Box>
          )}
        </Card>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ page */
const ZIP_FILES = [
  ['model.joblib', 'the fitted pipeline (preprocessing + model) and its input schema'],
  ['predict.py', 'runs the model on new rows: python predict.py model.joblib new.csv out.csv'],
  ['MODEL_CARD.md', 'what the model is, these settings, how it scored, what it expects, its limits'],
  ['requirements.txt', 'the library versions it was trained with'],
  ['example_input.csv', 'the first rows of your file, limited to the columns the model reads'],
]
const showValue = (v) => (typeof v === 'number' ? fSig(v) : v === null || v === undefined ? '—' : String(v))

function ExportButton({ exp, onExport, disabled, testid }) {
  return (
    <Btn testid={testid} bg="#fff" fg="#000" disabled={disabled || exp.status === 'loading'} onClick={onExport}
      title="Download this model, trained with the settings on screen, as a zip">
      {exp.status === 'loading' ? 'EXPORTING...' : 'DOWNLOAD .ZIP'}
    </Btn>
  )
}

function ExportCard({ d, req, exp, onExport, stale }) {
  const reg = d.task === 'regression'
  const scores = reg ? [['R2', d.test.r2, f3], ['RMSE', d.test.rmse, fSig], ['MAE', d.test.mae, fSig]]
    : [['ACC', d.test.accuracy, f3], ['F1', d.test.f1, f3], ['AUC', d.test.roc_auc, f3]]
  const settings = [
    ['TEST SPLIT', `${Math.round(req.test_size * 100)}%`], ['SCALING', req.scale ? 'ON' : 'OFF'],
    ['FEATURES', `${d.data.features_used.length}/${d.data.features_available.length}`], ['CV', req.cv ? 'ON' : 'OFF'],
  ]
  return (
    <Card wide testid="card-export" title="Export model"
      help="Downloads this model refit with exactly the settings below (same seed, same split), so it matches the scores on screen. The zip predicts on new rows without the dashboard."
      right={<ExportButton testid="lab-export-card" exp={exp} onExport={onExport} />}>
      {stale && <p className="mb-1 text-[10px] font-bold uppercase" style={{ color: K.yellow }}>Settings changed since this fit // the zip uses the fit shown; press FIT to export the new settings</p>}
      {exp.error && <p data-testid="lab-export-error" className="mb-1 text-[10px] font-bold uppercase" style={{ color: K.red }}>{exp.error}</p>}
      {exp.status === 'ok' && <p data-testid="lab-export-done" className="mb-1 text-[10px] uppercase" style={{ color: K.green }}>SAVED {exp.name}</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2 text-[10px]">
        <div>
          <div className="uppercase text-zinc-400 mb-[2px]">Hyper-parameters</div>
          <div data-testid="lab-export-params" className="grid grid-cols-[auto_1fr] gap-x-3 text-zinc-200">
            {Object.entries(d.params).map(([k, v]) => (
              <React.Fragment key={k}><span className="text-zinc-500">{k}</span><span className="font-bold break-all">{showValue(v)}</span></React.Fragment>
            ))}
          </div>
          <div className="uppercase text-zinc-400 mt-2 mb-[2px]">Training</div>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {settings.map(([k, v]) => <span key={k}><span className="text-zinc-500">{k} </span><span className="font-bold text-white">{v}</span></span>)}
          </div>
          <div className="uppercase text-zinc-400 mt-2 mb-[2px]">Test scores</div>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {scores.map(([k, v, f]) => <span key={k}><span className="text-zinc-500">{k} </span><span className="font-bold text-white">{isNum(v) ? f(v) : '—'}</span></span>)}
            {d.cv && <span><span className="text-zinc-500">CV </span><span className="font-bold text-white">{f3(d.cv.mean)} +/- {f3(d.cv.std)}</span></span>}
          </div>
        </div>
        <div>
          <div className="uppercase text-zinc-400 mb-[2px]">In the zip</div>
          <div className="grid grid-cols-[auto_1fr] gap-x-3">
            {ZIP_FILES.map(([f, what]) => (
              <React.Fragment key={f}><span className="text-white font-mono">{f}</span><span className="text-zinc-500">{what}</span></React.Fragment>
            ))}
          </div>
          <p className="mt-2 text-zinc-500 leading-snug">New rows need the original column names of the features used. predict.py repeats your cleaning steps, fills missing cells with the training median or most common value, and keeps extra columns in the output.</p>
          {d.data.capped && <p className="mt-1 leading-snug" style={{ color: K.yellow }}>Trained on the lab's sample of {fInt(d.data.rows_used)} of {fInt(d.data.rows_total)} rows.</p>}
        </div>
      </div>
    </Card>
  )
}

const taskFor = (m, task) => (m.tasks.includes('clustering') ? 'clustering' : m.tasks.includes(task) ? task : null)

export default function ModelLab({ sessionId, target, task, dataKey, columns, onOpenDashboard }) {
  const [catalog, setCatalog] = useState({ data: null, error: null })
  const [modelKey, setModelKeyState] = useState(() => {
    try {
      return localStorage.getItem(MODEL_KEY) || 'xgboost'
    } catch {
      return 'xgboost'
    }
  })
  const [paramsBy, setParamsBy] = useState({})
  const [opts, setOpts] = useState({ testSize: 0.2, scale: true, cv: true, auto: true })
  const [features, setFeatures] = useState(null)
  const [showFeatures, setShowFeatures] = useState(false)
  const [extras, setExtrasState] = useState({ row: 0, treeIndex: 0, treeDepth: 3 })
  const [fit, setFit] = useState({ status: 'idle', data: null, error: null })
  const [exp, setExp] = useState({ status: 'idle', error: null, name: null })
  const [pins, setPins] = useState({})
  const [sweep, setSweep] = useState({ status: 'idle', data: null, error: null, sig: null })
  const [lc, setLc] = useState({ status: 'idle', data: null, error: null, sig: null })
  const [sweepSel, setSweepSel] = useState({ param: '', param2: '' })
  const seq = useRef({ fit: 0, sweep: 0, lc: 0 })
  const last = useRef({ base: null, param: null, view: null })
  const autoSweepBase = useRef(null)

  useEffect(() => {
    let alive = true
    api.get('/api/v1/lab/models').then((d) => alive && setCatalog({ data: d, error: null })).catch((e) => alive && setCatalog({ data: null, error: messageOf(e) }))
    return () => { alive = false }
  }, [])

  const models = catalog.data ? catalog.data.models : []
  const model = models.find((m) => m.key === modelKey) || models[0]
  const lt = model ? taskFor(model, task) : null
  const params = useMemo(() => (model && lt ? model.params.filter((p) => !p.tasks.length || p.tasks.includes(lt)) : []), [model, lt])
  const defaults = useMemo(() => Object.fromEntries(params.map((p) => [p.name, p.default])), [params])
  const pkey = model ? `${model.key}:${lt}` : ''
  const values = useMemo(() => ({ ...defaults, ...(paramsBy[pkey] || {}) }), [defaults, paramsBy, pkey])
  const needsTarget = model && lt !== 'clustering' && !target
  const unsupported = model && target && !lt
  const canRun = Boolean(sessionId && model && lt && !needsTarget)

  const setModelKey = (k) => {
    setModelKeyState(k)
    setExtrasState((e) => ({ ...e, row: 0, treeIndex: 0 }))
    try {
      localStorage.setItem(MODEL_KEY, k)
    } catch {
      /* ignore */
    }
  }
  const setParams = useCallback((patch) => setParamsBy((all) => ({ ...all, [pkey]: { ...(all[pkey] || {}), ...patch } })), [pkey])
  const setExtras = (patch) => setExtrasState((e) => ({ ...e, ...patch }))

  // sweep selection follows the model
  useEffect(() => {
    if (model) setSweepSel({ param: model.default_sweep[0], param2: model.default_sweep[1] || '' })
    setSweep({ status: 'idle', data: null, error: null, sig: null })
    setLc({ status: 'idle', data: null, error: null, sig: null })
    autoSweepBase.current = null // coming back to a model re-runs its sweeps
  }, [pkey]) // eslint-disable-line react-hooks/exhaustive-deps

  const body = useMemo(() => ({ params: values, test_size: opts.testSize, scale: opts.scale, features }), [values, opts.testSize, opts.scale, features])
  const paramSig = JSON.stringify([body, opts.cv])
  const sweepSig = JSON.stringify(body)
  const viewSig = JSON.stringify(extras)
  const baseSig = `${sessionId}|${dataKey}|${pkey}`

  const runFit = useCallback(async () => {
    if (!canRun) return
    const id = ++seq.current.fit
    last.current = { base: baseSig, param: paramSig, view: viewSig }
    setFit((f) => ({ ...f, status: 'loading', error: null }))
    const req = { ...body, cv: opts.cv }
    try {
      const data = await api.post(`${sx(sessionId)}/lab/${model.key}/fit`, {
        ...req, row: extras.row, tree_index: extras.treeIndex, tree_depth: extras.treeDepth,
      })
      if (id !== seq.current.fit) return
      setFit({ status: 'ok', data, error: null, base: baseSig, req })
    } catch (e) {
      if (id !== seq.current.fit) return
      setFit((f) => ({ status: 'error', data: f.base === baseSig ? f.data : null, error: messageOf(e), base: f.base }))
    }
  }, [canRun, baseSig, paramSig, viewSig, sessionId, model, body, opts.cv, extras])

  const runSweep = useCallback(async (sel) => {
    if (!canRun || !sel.param) return
    const id = ++seq.current.sweep
    setSweep((s) => ({ ...s, status: 'loading', error: null }))
    try {
      const data = await api.post(`${sx(sessionId)}/lab/${model.key}/sweep`, { ...body, param: sel.param, param2: sel.param2 || null })
      if (id !== seq.current.sweep) return
      setSweep({ status: 'ok', data, error: null, sig: sweepSig })
    } catch (e) {
      if (id !== seq.current.sweep) return
      setSweep((s) => ({ ...s, status: 'error', error: messageOf(e) }))
    }
  }, [canRun, sessionId, model, body, sweepSig])

  const runLc = useCallback(async () => {
    if (!canRun || lt === 'clustering') return
    const id = ++seq.current.lc
    setLc((s) => ({ ...s, status: 'loading', error: null }))
    try {
      const data = await api.post(`${sx(sessionId)}/lab/${model.key}/sweep`, { ...body, param: 'train_size' })
      if (id !== seq.current.lc) return
      setLc({ status: 'ok', data, error: null, sig: sweepSig })
    } catch (e) {
      if (id !== seq.current.lc) return
      setLc((s) => ({ ...s, status: 'error', error: messageOf(e) }))
    }
  }, [canRun, lt, sessionId, model, body, sweepSig])

  // fit on model / data change; on parameter change when auto-refit is on; always on view changes (row, tree)
  useEffect(() => {
    if (!canRun) return undefined
    const l = last.current
    const base = l.base !== baseSig
    const param = l.param !== paramSig
    const view = l.view !== viewSig
    if (!base && !param && !view) return undefined
    if (!base && param && !view && !opts.auto) return undefined
    const t = setTimeout(runFit, base ? 0 : 450)
    return () => clearTimeout(t)
  }, [canRun, baseSig, paramSig, viewSig, opts.auto, runFit])

  // the first successful fit of a model / dataset also runs its default sweep and the learning curve
  useEffect(() => {
    if (fit.status !== 'ok' || !fit.data || fit.base !== baseSig || autoSweepBase.current === baseSig || !model) return
    autoSweepBase.current = baseSig
    runSweep({ param: model.default_sweep[0], param2: model.default_sweep[1] || '' })
    runLc()
  }, [fit, baseSig, model, runSweep, runLc])

  const d = fit.data && fit.base === baseSig ? fit.data : null
  const exportable = Boolean(d && d.task !== 'clustering' && fit.req)
  useEffect(() => { setExp({ status: 'idle', error: null, name: null }) }, [d])
  const runExport = async () => {
    if (!exportable) return
    setExp({ status: 'loading', error: null, name: null })
    try {
      const name = await download(`${sx(sessionId)}/lab/${d.model}/export`, fit.req)
      setExp({ status: 'ok', error: null, name })
    } catch (e) {
      setExp({ status: 'error', error: messageOf(e), name: null })
    }
  }
  const dirty = last.current.param !== paramSig
  const pin = pins[pkey]
  const available = d ? d.data.features_available : (columns || []).filter((c) => c !== target)
  const chosen = features || available
  const labels = d && d.data.class_labels ? d.data.class_labels : []

  const header = (
    <div className="shrink-0 border-b border-zinc-800 bg-zinc-950 px-2 py-1 flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="flex items-center gap-1">
        <span className={lbl}>MODEL</span>
        <select data-testid="lab-model" className={`${ctl} min-w-[200px] font-bold`} value={model ? model.key : ''} onChange={(e) => setModelKey(e.target.value)}>
          {models.map((m) => {
            const mt = taskFor(m, task)
            const reason = target && !mt ? ` (${m.tasks.includes('regression') ? 'numeric target' : 'categorical target'} only)` : ''
            return <option key={m.key} value={m.key}>{m.abbr} // {m.name}{m.tasks.includes('clustering') ? ' (unsupervised)' : ''}{reason}</option>
          })}
        </select>
      </span>
      {lt && <span className="inline-flex items-center h-5 px-1 text-[10px] font-bold uppercase" style={{ background: lt === 'regression' ? K.purple : lt === 'clustering' ? K.cyan : '#e4e4e7', color: lt === 'regression' ? '#fff' : '#000' }}>{lt}</span>}
      {lt !== 'clustering' && (
        <span className="flex items-center gap-1">
          <span className={lbl}>TEST SPLIT</span>
          <select data-testid="lab-split" className={ctl} value={opts.testSize} onChange={(e) => setOpts({ ...opts, testSize: Number(e.target.value) })}>
            {[0.1, 0.2, 0.3, 0.4].map((v) => <option key={v} value={v}>{Math.round(v * 100)}%</option>)}
          </select>
        </span>
      )}
      <Toggle testid="lab-scale" on={opts.scale} onClick={() => setOpts({ ...opts, scale: !opts.scale })} title="Standardise numeric features (fit on training rows only)">SCALE</Toggle>
      {lt !== 'clustering' && <Toggle testid="lab-cv" on={opts.cv} onClick={() => setOpts({ ...opts, cv: !opts.cv })} title="5-fold cross-validation on training rows">CV</Toggle>}
      <Toggle testid="lab-auto" on={opts.auto} onClick={() => setOpts({ ...opts, auto: !opts.auto })} title="Refit automatically when a setting changes">AUTO REFIT</Toggle>
      <Btn testid="lab-fit" bg={dirty && !opts.auto ? K.yellow : K.green} fg="#000" disabled={!canRun || fit.status === 'loading'} onClick={runFit}>
        {fit.status === 'loading' ? 'FITTING...' : dirty && !opts.auto ? 'FIT (CHANGED)' : 'FIT'}
      </Btn>
      <Btn testid="lab-pin" disabled={!d} title="Keep this result as A; later fits show their difference from it" onClick={() => setPins({ ...pins, [pkey]: d })}>PIN AS A</Btn>
      {pin && <Btn testid="lab-unpin" onClick={() => { const next = { ...pins }; delete next[pkey]; setPins(next) }}>CLEAR A</Btn>}
      {pin && <button type="button" className="text-[10px] text-zinc-400 underline" onClick={() => setParams(pin.params)}>RESTORE A SETTINGS</button>}
      {exportable && <ExportButton testid="lab-export" exp={exp} onExport={runExport} disabled={fit.status === 'loading'} />}
      {d && <span className="ml-auto text-[10px] uppercase text-zinc-500">{fInt(d.data.rows_used)} ROWS{d.data.capped ? ` (SAMPLE OF ${fInt(d.data.rows_total)})` : ''} // {d.data.features_used.length} FEATURES // FIT {d.fit_seconds}s</span>}
    </div>
  )

  let content
  if (catalog.error) content = <Msg color={K.red}>{catalog.error}</Msg>
  else if (!catalog.data) content = <Msg>LOADING MODEL CATALOG...</Msg>
  else if (!sessionId) content = <Msg testid="lab-empty">NO DATASET // UPLOAD A FILE ON THE TOOLBAR TO OPEN THE MODEL LAB</Msg>
  else if (needsTarget) content = <Msg testid="lab-need-target">{model.name} NEEDS A TARGET // SET ONE ON THE TOOLBAR, OR PICK K-MEANS (NO TARGET NEEDED)</Msg>
  else if (unsupported) content = <Msg testid="lab-unsupported">{model.name} CANNOT RUN ON A {String(task).toUpperCase()} TARGET // {model.task_note}</Msg>
  else {
    const c = d ? d.charts : null
    const reg = d && d.task === 'regression'
    content = (
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row">
        <aside data-testid="lab-controls" className="lg:w-[270px] shrink-0 border-b lg:border-b-0 lg:border-r border-zinc-800 bg-zinc-900 overflow-auto p-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase font-bold text-white">Hyper-parameters</span>
            <button type="button" data-testid="lab-reset" className="text-[10px] uppercase text-zinc-400 underline" onClick={() => setParamsBy((all) => ({ ...all, [pkey]: {} }))}>DEFAULTS</button>
          </div>
          {params.filter((p) => visible(p, values)).map((p) => (
            <ParamControl key={`${p.name}:${p.tasks.join()}`} p={p} value={values[p.name]} onChange={(v) => setParams({ [p.name]: v })} />
          ))}
          <div className="border-t border-zinc-800 pt-1 mt-1">
            <button type="button" data-testid="lab-features-toggle" className="w-full flex justify-between text-[10px] uppercase font-bold text-white" onClick={() => setShowFeatures(!showFeatures)}>
              <span>Features ({chosen.length}/{available.length})</span><span>{showFeatures ? '-' : '+'}</span>
            </button>
            {showFeatures && (
              <div className="mt-1">
                <div className="flex gap-2 mb-1">
                  <button type="button" className="text-[10px] uppercase text-zinc-400 underline" onClick={() => setFeatures(null)}>ALL</button>
                  {features && <span className="text-[10px] text-zinc-500">custom selection</span>}
                </div>
                {available.map((col) => (
                  <label key={col} className="flex items-center gap-1 text-[10px] text-zinc-200 truncate">
                    <input type="checkbox" checked={chosen.includes(col)} onChange={() => {
                      const next = chosen.includes(col) ? chosen.filter((x) => x !== col) : [...chosen, col]
                      if (next.length) setFeatures(next.length === available.length ? null : next)
                    }} />
                    <span className="truncate" title={col}>{col}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
          <div className="border-t border-zinc-800 pt-1 mt-2">
            <div className="text-[10px] uppercase font-bold text-white mb-[2px]">How {model.name} works</div>
            <p className="text-[10px] leading-snug text-zinc-400">{model.summary}</p>
            {!model.scaled && <p className="mt-1 text-[10px] leading-snug text-zinc-500">Trees split on thresholds, so scaling does not change this model.</p>}
          </div>
        </aside>
        <div data-testid="lab-main" className="flex-1 min-w-0 overflow-auto p-2 space-y-2">
          {fit.error && (
            <div data-testid="lab-error" className="flex flex-wrap items-center gap-2 px-2 py-1 text-[10px] font-bold uppercase" style={{ background: K.red, color: '#fff' }}>
              <span>{fit.error}</span>
              {/impute|missing/i.test(fit.error) && onOpenDashboard && <Btn bg="#000" onClick={onOpenDashboard}>OPEN DASHBOARD</Btn>}
            </div>
          )}
          {!d ? <Msg>{fit.status === 'loading' ? `FITTING ${model.name.toUpperCase()}...` : 'NOT FITTED YET'}</Msg> : (
            <>
              <MetricsStrip d={d} pin={pin && pin.model === d.model ? pin : null} />
              <Notes notes={d.notes} />
              <div className={`grid grid-cols-1 xl:grid-cols-2 gap-2 ${fit.status === 'loading' ? 'opacity-60' : ''}`}>
                {d.model === 'linear_regression' && <LinearCards c={c} onParams={setParams} values={values} />}
                {d.model === 'logistic_regression' && <LogisticCards c={c} onParams={setParams} values={values} />}
                {d.model === 'knn' && <KnnCards c={c} labels={labels} task={d.task} row={extras.row} setRow={(r) => setExtras({ row: r })} />}
                {d.model === 'xgboost' && <XgbCards c={c} labels={labels} task={d.task} extras={extras} setExtras={setExtras} />}
                {d.model === 'decision_tree' && <DtCards c={c} labels={labels} task={d.task} extras={extras} setExtras={setExtras} onParams={setParams} values={values} />}
                {d.model === 'random_forest' && <RfCards c={c} labels={labels} task={d.task} extras={extras} setExtras={setExtras} />}
                {d.model === 'svm' && <SvmCards c={c} labels={labels} task={d.task} />}
                {d.model === 'kmeans' && <KmeansCards d={d} />}
                {c && c.boundary && <BoundaryCard b={c.boundary} labels={labels} task={d.task} />}
                <SweepCard model={model} task={lt} values={values} sweep={sweep} sel={sweepSel} setSel={setSweepSel}
                  onRun={runSweep} onApply={(patch) => setParams(patch)} stale={Boolean(sweep.data) && sweep.sig !== sweepSig} />
                {lt !== 'clustering' && <LearningCard lc={lc} onRun={runLc} stale={Boolean(lc.data) && lc.sig !== sweepSig} />}
                {d.task === 'classification' && (
                  <>
                    <ThresholdCard key={`${d.model}`} perf={d.performance} />
                    <RocCard perf={d.performance} />
                    <PrCard perf={d.performance} />
                    <SeparationCard perf={d.performance} />
                    <CalibrationCard perf={d.performance} />
                    <ConfusionCard perf={d.performance} />
                  </>
                )}
                {reg && <RegressionPerf perf={d.performance} />}
                {exportable && <ExportCard d={d} req={fit.req} exp={exp} onExport={runExport} stale={dirty} />}
              </div>
            </>
          )}
        </div>
      </div>
    )
  }
  return (
    <div data-testid="model-lab" className="flex-1 min-h-0 flex flex-col">
      {header}
      {content}
    </div>
  )
}

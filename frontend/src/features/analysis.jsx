/**
 * Panel extensions for the single-screen layout.
 * Each export is a tab body for one of the six panels (data preview, ROC/PR/threshold, residuals,
 * tuning, importance, SHAP, saved-run comparison).
 * Every number shown here is a backend response; styling matches App.jsx (zinc / mono / 10px).
 */
import React, { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { api, messageOf, sx } from '../lib/api'
import { useApi, useDebounced } from '../lib/hooks'
import { f3, fCell, fDate, fInt, fPct, fSig, fSigned, isNum } from '../lib/fmt'

export const K = {
  purple: '#b800e6', green: '#00e600', yellow: '#e6e600', red: '#ff0000', cyan: '#00d5ff', orange: '#fb923c',
  dim: '#a1a1aa', grid: '#27272a', pending: '#3f3f46', text: '#e4e4e7',
}
const PALETTE = ['#00d5ff', '#fb923c', '#a3e635', '#f472b6', '#facc15', '#94a3b8', '#2dd4bf', '#818cf8']
const axis = { stroke: '#a1a1aa', tick: { fill: '#a1a1aa', fontSize: 10 }, tickLine: { stroke: '#a1a1aa' }, axisLine: { stroke: '#52525b' } }
const tip = { contentStyle: { background: '#09090b', border: '1px solid #52525b', fontSize: 10, borderRadius: 0 }, labelStyle: { color: '#a1a1aa' }, itemStyle: { padding: 0 } }
const ctl = 'h-6 bg-zinc-950 border border-zinc-700 text-white text-[10px] px-1 rounded-none disabled:opacity-40 disabled:cursor-not-allowed'
const lbl = 'text-[10px] uppercase text-zinc-400'
const grid = <CartesianGrid stroke={K.grid} />

/* ------------------------------------------------------------------ atoms */
export function Msg({ children, color, testid }) {
  return (
    <div data-testid={testid} className="min-h-[80px] flex items-center justify-center text-center text-[10px] uppercase text-zinc-500 px-2 py-4" style={color ? { color } : undefined}>
      {children}
    </div>
  )
}

function Load({ q, children, empty }) {
  if (q.error) return <Msg color={K.red}>{q.error}</Msg>
  if (!q.data) return <Msg>{q.loading ? 'LOADING...' : empty || 'NO DATA'}</Msg>
  return children(q.data)
}

export function TabBar({ tabs, value, onChange, testid }) {
  return (
    <div className="flex">
      {tabs.map(([v, label]) => (
        <button key={v} type="button" data-testid={`${testid}-${v}`} onClick={() => onChange(v)}
          className={`px-2 py-[2px] text-[10px] uppercase whitespace-nowrap border-b-2 ${value === v ? 'text-white border-[#b800e6] font-bold' : 'text-zinc-500 border-transparent hover:text-zinc-300'}`}>
          {label}
        </button>
      ))}
    </div>
  )
}

function Seg({ value, onChange, options, testid }) {
  return (
    <span className="inline-flex" data-testid={testid}>
      {options.map(([v, label]) => (
        <button key={String(v)} type="button" onClick={() => onChange(v)}
          className={`h-5 px-1.5 text-[10px] uppercase border ${value === v ? 'bg-white text-black border-white font-bold' : 'bg-zinc-950 text-zinc-300 border-zinc-700'}`}>
          {label}
        </button>
      ))}
    </span>
  )
}

function Btn({ children, onClick, disabled, title, testid, bg = '#27272a', fg = '#fff' }) {
  return (
    <button type="button" data-testid={testid} title={title} disabled={disabled} onClick={onClick}
      className="h-5 px-1.5 text-[10px] font-bold uppercase border border-zinc-700 rounded-none whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed"
      style={disabled ? { background: '#27272a', color: '#a1a1aa' } : { background: bg, color: fg }}>
      {children}
    </button>
  )
}

function Stats({ items }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase text-white mb-1">
      {items.filter(Boolean).map(([k, v, color]) => (
        <span key={k}><span className="text-zinc-500">{k} </span><span style={color ? { color } : undefined} className="font-bold">{v}</span></span>
      ))}
    </div>
  )
}

const Sub = ({ children }) => <div className="text-[10px] uppercase text-zinc-400 mb-[2px]">{children}</div>
const Note = ({ children }) => <div className="mt-1 text-[10px] normal-case text-zinc-500 leading-snug">{children}</div>
const axisLabel = (value) => ({ value, position: 'insideBottom', offset: -8, fill: '#a1a1aa', fontSize: 10 })

/* ------------------------------------------------------------------ 01 // DATA PREVIEW */
const KIND_COLOR = { numeric: K.cyan, categorical: K.orange, boolean: K.yellow, datetime: K.green }

function ProfileCard({ col }) {
  const hist = col.histogram && col.histogram.counts.map((c, i) => ({ i, c, lo: col.histogram.edges[i], hi: col.histogram.edges[i + 1] }))
  return (
    <div className="border border-zinc-800 bg-zinc-950 p-1 min-w-0" data-testid={`profile-${col.name}`}>
      <div className="flex items-center gap-1 mb-[2px] text-[10px]">
        <span className="text-white font-bold truncate" title={col.name}>{col.name}</span>
        <span className="uppercase text-[8px]" style={{ color: KIND_COLOR[col.kind] || K.dim }}>{col.kind}</span>
        {col.missing > 0 && <span className="px-[2px] text-[8px] font-bold" style={{ background: K.yellow, color: '#000' }}>{col.missing} NULL</span>}
      </div>
      {hist ? (
        <>
          <div className="h-10">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hist} margin={{ top: 2, right: 0, bottom: 0, left: 0 }} barCategoryGap={1}>
                <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v) => [v, 'rows']} labelFormatter={(_, p) => (p && p[0] ? `${fSig(p[0].payload.lo)} - ${fSig(p[0].payload.hi)}` : '')} />
                <Bar dataKey="c" fill={K.cyan} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-x-1 text-[9px] text-zinc-400">
            <span>MEAN {fSig(col.mean)}</span><span>STD {fSig(col.std)}</span><span>MED {fSig(col.median)}</span>
            <span>MIN {fSig(col.min)}</span><span>MAX {fSig(col.max)}</span>
            <span style={col.outliers ? { color: K.yellow } : undefined}>{col.outliers} OUTL</span>
          </div>
        </>
      ) : col.top ? (
        <div>
          {col.top.slice(0, 4).map((t) => (
            <div key={t.value} className="flex items-center gap-1 text-[9px]">
              <span className="w-14 truncate text-zinc-300" title={t.value}>{t.value}</span>
              <span className="h-[5px]" style={{ width: Math.max(2, t.ratio * 60), background: K.orange }} />
              <span className="text-zinc-500">{fPct(t.ratio, 0)}</span>
            </div>
          ))}
          <div className="text-[9px] text-zinc-500">{col.unique} UNIQUE</div>
        </div>
      ) : <div className="text-[9px] text-zinc-400">{col.min ? `${String(col.min).slice(0, 10)} > ${String(col.max).slice(0, 10)}` : 'NO DATA'}</div>}
    </div>
  )
}

export function PreviewTab({ id, ver }) {
  const [tab, setTab] = useState('rows')
  const [source, setSource] = useState('working')
  const [page, setPage] = useState(0)
  const [sort, setSort] = useState({ col: null, desc: false })
  const [search, setSearch] = useState('')
  const q = useDebounced(search, 300)
  const limit = 12
  useEffect(() => setPage(0), [q, source, sort, id])
  const qs = `offset=${page * limit}&limit=${limit}&source=${source}${sort.col ? `&sort=${encodeURIComponent(sort.col)}&descending=${sort.desc}` : ''}${q ? `&search=${encodeURIComponent(q)}` : ''}&v=${ver}`
  const rows = useApi(id && tab === 'rows' ? `${sx(id)}/preview?${qs}` : null)
  const prof = useApi(id && tab === 'profile' ? `${sx(id)}/profile?source=${source}&v=${ver}` : null)
  const d = rows.data
  const pages = d ? Math.max(1, Math.ceil(d.filtered / limit)) : 1
  return (
    <div data-testid="preview">
      <div className="flex flex-wrap items-center gap-1 mb-1">
        <Seg value={tab} onChange={setTab} testid="preview-tab" options={[['rows', 'ROWS'], ['profile', 'COLUMN PROFILES']]} />
        <Seg value={source} onChange={setSource} testid="preview-source" options={[['working', 'WORKING'], ['raw', 'RAW']]} />
        {tab === 'rows' && <input data-testid="preview-search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="SEARCH..." className={`${ctl} w-28`} />}
        {tab === 'rows' && d && (
          <span className="ml-auto flex items-center gap-1 text-[10px] text-zinc-400 uppercase">
            {fInt(d.filtered)} / {fInt(d.total)}
            <Btn disabled={page === 0} onClick={() => setPage((p) => p - 1)}>&lt;</Btn>
            {page + 1}/{pages}
            <Btn disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>&gt;</Btn>
          </span>
        )}
      </div>
      {tab === 'rows' ? (
        <Load q={rows}>
          {(data) => (
            <div className="overflow-auto border border-zinc-800">
              <table className="w-full text-[10px] border-collapse" data-testid="preview-table">
                <thead className="bg-zinc-950">
                  <tr>
                    <th className="text-left px-1 py-[2px] text-zinc-500 font-normal">#</th>
                    {data.columns.map((c) => (
                      <th key={c.name} onClick={() => setSort((p) => (p.col === c.name ? { col: c.name, desc: !p.desc } : { col: c.name, desc: false }))}
                        className="text-left px-1 py-[2px] font-normal whitespace-nowrap cursor-pointer select-none hover:bg-zinc-800">
                        <span className="text-white font-bold">{c.name}</span>
                        <span className="ml-1 text-[8px]" style={{ color: KIND_COLOR[c.kind] }}>{c.kind.slice(0, 3).toUpperCase()}</span>
                        {sort.col === c.name && <span className="ml-1" style={{ color: K.purple }}>{sort.desc ? 'v' : '^'}</span>}
                        {c.missing > 0 && <span className="ml-1" style={{ color: K.yellow }}>{c.missing}</span>}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r, i) => (
                    <tr key={data.index[i]} className="border-t border-zinc-800 hover:bg-zinc-800">
                      <td className="px-1 py-[2px] text-zinc-500">{data.index[i]}</td>
                      {data.columns.map((c) => (
                        <td key={c.name} className="px-1 py-[2px] whitespace-nowrap" style={r[c.name] === null ? { color: K.yellow } : { color: '#e4e4e7' }}>
                          {r[c.name] === null ? 'NULL' : fCell(r[c.name])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Load>
      ) : (
        <Load q={prof}>
          {(data) => (
            <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(170px,1fr))' }}>
              {data.columns.map((c) => <ProfileCard key={c.name} col={c} />)}
            </div>
          )}
        </Load>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 04 // CURVES, RESIDUALS, TUNING */
function Cell2({ v, good }) {
  return <div className="py-1 text-xs font-bold text-center" style={{ background: good ? `${K.green}22` : `${K.red}22`, color: good ? K.green : K.red }}>{v}</div>
}

function CurvesBody({ d, thr, setThr }) {
  const t = d.thresholds
  const i = t ? Math.max(0, Math.min(t.grid.length - 1, Math.round(thr * (t.grid.length - 1)))) : 0
  const at = t && { tp: t.tp[i], fp: t.fp[i], tn: t.tn[i], fn: t.fn[i], precision: t.precision[i], recall: t.recall[i], f1: t.f1[i], accuracy: t.accuracy[i], bal: t.balanced_accuracy[i], spec: t.specificity[i] }
  const sweep = t ? t.grid.map((g, k) => ({ t: g, P: t.precision[k], R: t.recall[k], F1: t.f1[k] })) : []
  const roc = d.roc.classes.map((c, k) => ({ ...c, color: d.binary ? K.cyan : PALETTE[k % PALETTE.length], pts: c.fpr.map((x, j) => ({ x, y: c.tpr[j] })) }))
  const pr = d.pr.classes.map((c, k) => ({ ...c, color: d.binary ? K.orange : PALETTE[k % PALETTE.length], pts: c.recall.map((x, j) => ({ x, y: c.precision[j] })) }))
  const m = { top: 4, right: 8, bottom: 18, left: 0 }
  return (
    <div data-testid="curves">
      <Stats items={[[d.binary ? 'ROC AUC' : 'ROC AUC (MACRO)', f3(d.roc.macro_auc), K.cyan], [d.binary ? 'AVG PREC' : 'AP (MACRO)', f3(d.pr.macro_ap), K.orange], ['TEST ROWS', fInt(d.n_test)], ['SCORE', d.score_kind === 'probability' ? 'PROBABILITY' : 'MARGIN (RESCALED 0-1)']]} />
      <div className="grid grid-cols-2 gap-1">
        <div>
          <Sub>ROC CURVE</Sub>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={m}>
                {grid}
                <XAxis type="number" dataKey="x" domain={[0, 1]} label={axisLabel('FPR')} {...axis} />
                <YAxis type="number" dataKey="y" domain={[0, 1]} width={30} {...axis} />
                <Tooltip {...tip} formatter={(v) => v.toFixed(3)} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#52525b" strokeDasharray="3 3" />
                {roc.map((l) => <Line key={l.label} data={l.pts} dataKey="y" name={`${l.label} (AUC ${f3(l.auc)})`} stroke={l.color} dot={false} strokeWidth={2} isAnimationActive={false} />)}
                {at && <ReferenceDot x={1 - at.spec} y={at.recall} r={4} fill={K.purple} stroke="#fff" />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <Sub>PRECISION-RECALL</Sub>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={m}>
                {grid}
                <XAxis type="number" dataKey="x" domain={[0, 1]} label={axisLabel('RECALL')} {...axis} />
                <YAxis type="number" dataKey="y" domain={[0, 1]} width={30} {...axis} />
                <Tooltip {...tip} formatter={(v) => v.toFixed(3)} />
                {d.binary && <ReferenceLine y={d.pr.classes[0].prevalence} stroke="#52525b" strokeDasharray="3 3" />}
                {pr.map((l) => <Line key={l.label} data={l.pts} dataKey="y" name={`${l.label} (AP ${f3(l.ap)})`} stroke={l.color} dot={false} strokeWidth={2} isAnimationActive={false} />)}
                {at && <ReferenceDot x={at.recall} y={at.precision} r={4} fill={K.purple} stroke="#fff" />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      {!d.binary && (
        <div className="flex flex-wrap gap-x-3 text-[10px] uppercase text-zinc-300">
          {roc.map((l) => <span key={l.label} className="flex items-center gap-1"><span className="inline-block w-2 h-2" style={{ background: l.color }} />{l.label}</span>)}
        </div>
      )}
      {t ? (
        <div className="mt-1 border border-zinc-800 p-1" data-testid="threshold-box">
          <div className="flex flex-wrap items-center gap-1">
            <span className={lbl}>DECISION THRESHOLD</span>
            <input data-testid="dthreshold-slider" type="range" min={0} max={1} step={0.01} value={thr} onChange={(e) => setThr(Number(e.target.value))} className="w-32 accent-zinc-300" />
            <b data-testid="dthreshold-value" className="text-white w-8">{thr.toFixed(2)}</b>
            <Btn onClick={() => setThr(0.5)}>0.50</Btn>
            <Btn testid="btn-best-f1" onClick={() => setThr(t.best_f1.threshold)}>BEST F1 {t.best_f1.threshold.toFixed(2)}</Btn>
            <Btn onClick={() => setThr(t.youden.threshold)}>YOUDEN {t.youden.threshold.toFixed(2)}</Btn>
          </div>
          <div className="mt-1 grid grid-cols-[110px_1fr] gap-2">
            <div>
              <div className="grid grid-cols-[auto_1fr_1fr] gap-px text-center text-[9px] uppercase" data-testid="confusion">
                <span /><span className="text-zinc-500">PRED -</span><span className="text-zinc-500">PRED +</span>
                <span className="text-zinc-500 self-center pr-1">ACT -</span><Cell2 v={at.tn} good /><Cell2 v={at.fp} />
                <span className="text-zinc-500 self-center pr-1">ACT +</span><Cell2 v={at.fn} /><Cell2 v={at.tp} good />
              </div>
              <div className="mt-1 text-[10px] uppercase leading-tight">
                <div>PREC <b style={{ color: K.orange }}>{f3(at.precision)}</b></div>
                <div>REC <b style={{ color: K.cyan }}>{f3(at.recall)}</b></div>
                <div>F1 <b data-testid="stat-f1-at" style={{ color: K.purple }}>{f3(at.f1)}</b></div>
                <div>BAL ACC <b>{f3(at.bal)}</b></div>
              </div>
            </div>
            <div className="h-36">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sweep} margin={{ top: 4, right: 8, bottom: 18, left: 0 }}>
                  {grid}
                  <XAxis dataKey="t" type="number" domain={[0, 1]} label={axisLabel('THRESHOLD')} {...axis} />
                  <YAxis domain={[0, 1]} width={30} {...axis} />
                  <Tooltip {...tip} formatter={(v) => v.toFixed(3)} labelFormatter={(v) => `THRESHOLD ${Number(v).toFixed(2)}`} />
                  <Line dataKey="P" name="Precision" stroke={K.orange} dot={false} strokeWidth={2} isAnimationActive={false} />
                  <Line dataKey="R" name="Recall" stroke={K.cyan} dot={false} strokeWidth={2} isAnimationActive={false} />
                  <Line dataKey="F1" stroke={K.purple} dot={false} strokeWidth={2.5} isAnimationActive={false} />
                  <ReferenceLine x={thr} stroke="#fff" strokeDasharray="3 3" />
                  <ReferenceDot x={t.best_f1.threshold} y={t.best_f1.f1} r={3} fill={K.green} stroke="none" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <Note>Flags &quot;{d.positive_class}&quot; when the score is at or above the threshold. Raising it trades recall for precision; the green dot marks the best F1. Pick the point that matches the cost of a false alarm versus a miss.</Note>
        </div>
      ) : <Note>Threshold tuning applies to binary targets. Multiclass shows one-vs-rest curves per class.</Note>}
    </div>
  )
}

export function CurvesTab({ base, mkey }) {
  const q = useApi(base && mkey ? `${base}/curves/${mkey}` : null)
  const [thr, setThr] = useState(0.5)
  return <Load q={q}>{(d) => <CurvesBody d={d} thr={thr} setThr={setThr} />}</Load>
}

export function ResidualsTab({ base, mkey }) {
  const q = useApi(base && mkey ? `${base}/residuals/${mkey}` : null)
  return (
    <Load q={q}>
      {(d) => {
        const pts = d.scatter.actual.map((a, i) => ({ a, p: d.scatter.predicted[i] }))
        const lo = Math.min(...d.scatter.actual, ...d.scatter.predicted)
        const hi = Math.max(...d.scatter.actual, ...d.scatter.predicted)
        const hist = d.histogram.counts.map((c, i) => ({ c, x: (d.histogram.edges[i] + d.histogram.edges[i + 1]) / 2 }))
        return (
          <div data-testid="residuals">
            <Stats items={[['R2', f3(d.r2), K.purple], ['BIAS', fSigned(d.bias, 3)], ['RESIDUAL STD', fSig(d.residual_std)], ['TEST ROWS', fInt(d.n_test)]]} />
            <div className="grid grid-cols-2 gap-1">
              <div>
                <Sub>ACTUAL VS PREDICTED</Sub>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 4, right: 8, bottom: 18, left: 0 }}>
                      {grid}
                      <XAxis type="number" dataKey="a" name="actual" domain={[lo, hi]} tickFormatter={fSig} label={axisLabel('ACTUAL')} {...axis} />
                      <YAxis type="number" dataKey="p" name="predicted" domain={[lo, hi]} tickFormatter={fSig} width={40} {...axis} />
                      <ZAxis range={[10, 10]} />
                      <Tooltip {...tip} cursor={{ strokeDasharray: '3 3' }} formatter={(v) => fSig(v)} />
                      <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={K.green} strokeDasharray="4 4" />
                      <Scatter data={pts} fill={K.cyan} fillOpacity={0.6} isAnimationActive={false} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <Sub>RESIDUALS (ACTUAL - PREDICTED)</Sub>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hist} margin={{ top: 4, right: 8, bottom: 18, left: 0 }} barCategoryGap={1}>
                      {grid}
                      <XAxis dataKey="x" tickFormatter={fSig} {...axis} />
                      <YAxis width={30} {...axis} />
                      <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v) => [v, 'rows']} labelFormatter={(v) => `~ ${fSig(v)}`} />
                      <Bar dataKey="c" fill={K.orange} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <Note>Points on the green line are perfect predictions. A residual histogram centred on zero with no skew means the errors are unbiased.</Note>
          </div>
        )
      }}
    </Load>
  )
}

export function TuningTab({ base, mkey, row }) {
  const tuned = Boolean(row && (row.tuning || row.tuned))
  const q = useApi(base && mkey && tuned ? `${base}/tuning/${mkey}` : null)
  if (!tuned) return <Msg testid="tune-none">TRAINED WITH DEFAULT HYPER-PARAMETERS // TURN TUNE ON IN THE TOP BAR AND RUN AGAIN</Msg>
  return (
    <Load q={q}>
      {(d) => {
        let best = -Infinity
        const hist = d.history.map((h) => { best = Math.max(best, h.score); return { trial: h.trial, score: h.score, best } })
        const imp = Object.entries(d.param_importance || {}).map(([k, v]) => ({ k, v })).sort((a, b) => b.v - a.v)
        return (
          <div data-testid="tuning">
            <Stats items={[
              [`DEFAULT CV (${String(d.metric).toUpperCase()})`, f3(d.default_score)], ['BEST CV', f3(d.best_score), d.applied ? K.green : K.dim],
              ['GAIN', fSigned(d.improvement, 3), d.improvement > 0 ? K.green : K.dim], ['TRIALS', `${d.n_trials} (${d.cv_folds}-FOLD, ${fInt(d.rows_used)} ROWS)`],
              ['OUTCOME', d.applied ? 'TUNED PARAMS USED' : 'DEFAULTS KEPT', d.applied ? K.purple : K.yellow],
            ]} />
            {d.note && <Note>{d.note}</Note>}
            <Sub>OPTUNA SEARCH HISTORY</Sub>
            <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={hist} margin={{ top: 4, right: 8, bottom: 18, left: 0 }}>
                  {grid}
                  <XAxis dataKey="trial" label={axisLabel('TRIAL')} {...axis} />
                  <YAxis domain={['auto', 'auto']} width={40} tickFormatter={(v) => v.toFixed(2)} {...axis} />
                  <Tooltip {...tip} formatter={(v) => v.toFixed(4)} />
                  {isNum(d.default_score) && <ReferenceLine y={d.default_score} stroke={K.yellow} strokeDasharray="4 4" />}
                  <Scatter dataKey="score" name="trial" fill={K.cyan} isAnimationActive={false} />
                  <Line type="stepAfter" dataKey="best" name="best so far" stroke={K.purple} strokeWidth={2} dot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="text-[10px] uppercase text-zinc-500">YELLOW DASH = DEFAULT PARAMETERS // PURPLE = BEST SO FAR</div>
            <div className="mt-1 grid grid-cols-2 gap-2">
              <div>
                <Sub>BEST PARAMETERS</Sub>
                <table className="w-full text-[10px]"><tbody>
                  {Object.entries(d.best_params).map(([k, v]) => (
                    <tr key={k} className="border-b border-zinc-800"><td className="py-[1px] text-zinc-400">{k}</td><td className="text-right text-white">{typeof v === 'number' ? fSig(v) : String(v)}</td></tr>
                  ))}
                </tbody></table>
              </div>
              {imp.length > 0 && (
                <div>
                  <Sub>WHICH PARAMETERS MATTERED</Sub>
                  {imp.map((p) => (
                    <div key={p.k} className="flex items-center gap-1 text-[10px]">
                      <span className="w-20 truncate text-zinc-300">{p.k}</span>
                      <span className="h-[6px]" style={{ width: Math.max(2, p.v * 70), background: K.purple }} />
                      <span className="text-zinc-500">{fPct(p.v, 0)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )
      }}
    </Load>
  )
}

/* ------------------------------------------------------------------ 06 // IMPORTANCE, SHAP */
export function ImportanceTab({ base, mkey }) {
  const q = useApi(base && mkey ? `${base}/importance/${mkey}` : null)
  const [mode, setMode] = useState('permutation')
  return (
    <Load q={q}>
      {(d) => {
        const native = d.native
        const m = mode === 'native' && native ? 'native' : 'permutation'
        const data = (m === 'native' ? native : d.features).slice(0, 12).map((f) => ({ ...f, err: f.std }))
        return (
          <div data-testid="importance">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <Seg value={m} onChange={setMode} testid="imp-mode" options={[['permutation', 'PERMUTATION'], ...(native ? [['native', 'MODEL-NATIVE']] : [])]} />
              <span className="text-[10px] normal-case text-zinc-500">
                {m === 'permutation' ? `Score drop when a column is shuffled (${String(d.metric).replace('_', ' ')}, ${d.repeats} repeats, ${fInt(d.n_rows)} test rows).` : 'Importance reported by the model itself, normalised to 1.'}
              </span>
            </div>
            <div style={{ height: Math.max(140, data.length * 20 + 24) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data} layout="vertical" margin={{ top: 2, right: 24, bottom: 2, left: 0 }}>
                  {grid}
                  <XAxis type="number" tickFormatter={(v) => fSig(v)} {...axis} />
                  <YAxis type="category" dataKey="feature" width={90} {...axis} />
                  <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v) => fSig(v)} />
                  <Bar dataKey="importance" isAnimationActive={false}>
                    {data.map((f, i) => <Cell key={f.feature} fill={f.importance < 0 ? K.red : i === 0 ? K.purple : K.cyan} />)}
                    {m === 'permutation' && <ErrorBar dataKey="err" stroke={K.dim} width={3} />}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            {m === 'permutation' && <Note>Bars near zero (or red, negative) mean the model barely uses that column, so dropping it will likely not hurt.</Note>}
          </div>
        )
      }}
    </Load>
  )
}

const lerp = (a, b, t) => Math.round(a + (b - a) * t)
const swarmColor = (n) => (n === null || n === undefined ? '#97a1b0' : `rgb(${lerp(62, 255, n)},${lerp(198, 77, n)},${lerp(255, 120, n)})`)

function Swarm({ swarm }) {
  const W = 520, rowH = 22, left = 100
  const max = Math.max(1e-9, ...swarm.flatMap((f) => f.points.map((p) => Math.abs(p.shap))))
  const H = swarm.length * rowH + 20
  const x = (v) => left + ((v / max + 1) / 2) * (W - left - 8)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" data-testid="swarm" role="img" aria-label="SHAP beeswarm">
      <line x1={x(0)} x2={x(0)} y1={0} y2={H - 14} stroke="#52525b" />
      {swarm.map((f, r) => (
        <g key={f.feature}>
          <text x={left - 6} y={r * rowH + rowH / 2 + 3} textAnchor="end" fontSize="10" fill="#e4e4e7">{f.feature.slice(0, 14)}</text>
          {f.points.map((p, i) => (
            <circle key={i} cx={x(p.shap)} cy={r * rowH + rowH / 2 + (((i * 2654435761) % 1000) / 1000 - 0.5) * (rowH - 7)} r="2.2" fill={swarmColor(p.norm)} fillOpacity="0.8">
              <title>{`${f.feature} = ${p.value}\nSHAP ${p.shap}`}</title>
            </circle>
          ))}
        </g>
      ))}
      <text x={x(0)} y={H - 3} textAnchor="middle" fontSize="9" fill="#a1a1aa">SHAP VALUE (IMPACT ON OUTPUT)</text>
    </svg>
  )
}

function ShapBody({ d, reg, setCls, row, setRow }) {
  const top = d.global.slice(0, 10)
  const loc = d.local
  const maxLoc = Math.max(1e-9, ...loc.contributions.map((c) => Math.abs(c.shap)), Math.abs(loc.other.shap))
  return (
    <div data-testid="shap">
      <div className="flex flex-wrap items-center gap-2 mb-1 text-[10px] uppercase">
        <span className="px-1 font-bold" style={{ background: K.purple, color: '#fff' }}>{d.explainer}</span>
        <span className="text-zinc-500 normal-case">values in {d.output_space}; {d.n_samples} test rows explained</span>
        {!reg && d.classes.length > 2 && (
          <select className={ctl} value={String(d.class_index)} onChange={(e) => setCls(Number(e.target.value))}>
            {d.classes.map((c, i) => <option key={c} value={i}>CLASS {c}</option>)}
          </select>
        )}
        {!reg && d.classes.length === 2 && <span className="text-zinc-300">EXPLAINING &quot;{d.class_label}&quot;</span>}
      </div>
      <Sub>GLOBAL IMPORTANCE // MEAN |SHAP|</Sub>
      <div style={{ height: top.length * 18 + 22 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={top} layout="vertical" margin={{ top: 2, right: 20, bottom: 2, left: 0 }}>
            <XAxis type="number" tickFormatter={fSig} {...axis} />
            <YAxis type="category" dataKey="feature" width={90} {...axis} />
            <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v) => fSig(v)} />
            <Bar dataKey="mean_abs" isAnimationActive={false}>{top.map((f, i) => <Cell key={f.feature} fill={i === 0 ? K.purple : K.cyan} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center gap-2 mt-1">
        <Sub>BEESWARM</Sub>
        <span className="inline-block w-14 h-[6px]" style={{ background: 'linear-gradient(90deg,#3ec6ff,#ff4d78)' }} />
        <span className="text-[9px] uppercase text-zinc-500">LOW &gt; HIGH FEATURE VALUE</span>
      </div>
      <Swarm swarm={d.swarm} />
      <div className="mt-1 border border-zinc-800 p-1" data-testid="shap-local">
        <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase">
          <span className={lbl}>EXPLAIN ONE ROW</span>
          <select data-testid="shap-row" className={`${ctl} max-w-[200px]`} value={String(row)} onChange={(e) => setRow(Number(e.target.value))}>
            {d.rows.slice(0, 60).map((r) => <option key={r.i} value={r.i}>#{r.i} ACT {String(r.actual)} PRED {String(r.predicted)}</option>)}
          </select>
          <span className="text-zinc-300">BASE {fSig(loc.base)} &gt; OUT <b className="text-white">{fSig(loc.output)}</b> // PRED <b style={{ color: K.purple }}>{String(loc.prediction)}</b></span>
        </div>
        <div className="mt-1">
          {loc.contributions.map((c) => (
            <div key={c.feature} className="grid grid-cols-[110px_1fr_52px] items-center gap-1 text-[10px]">
              <span className="truncate text-zinc-300" title={`${c.feature} = ${c.value}`}>{c.feature} <span className="text-zinc-600">= {String(c.value).slice(0, 7)}</span></span>
              <span className="relative h-[10px] bg-zinc-800">
                <span className="absolute top-0 h-full" style={{ left: c.shap >= 0 ? '50%' : `${50 - (Math.abs(c.shap) / maxLoc) * 50}%`, width: `${(Math.abs(c.shap) / maxLoc) * 50}%`, background: c.shap >= 0 ? K.red : K.cyan }} />
                <span className="absolute left-1/2 top-0 h-full w-px bg-zinc-400" />
              </span>
              <span className="text-right" style={{ color: c.shap >= 0 ? K.red : K.cyan }}>{fSigned(c.shap, 3)}</span>
            </div>
          ))}
          {loc.other.count > 0 && <div className="text-[9px] uppercase text-zinc-600">+ {loc.other.count} OTHER FEATURES {fSigned(loc.other.shap, 3)}</div>}
        </div>
        <Note>Red pushes the output up, cyan pushes it down. The bars add up to the gap between the base value and the model output.</Note>
      </div>
    </div>
  )
}

export function ShapTab({ base, mkey, reg }) {
  const [cls, setCls] = useState(null)
  const [row, setRow] = useState(0)
  const q = useApi(base && mkey ? `${base}/shap/${mkey}?row=${row}${cls !== null ? `&class_index=${cls}` : ''}` : null)
  return <Load q={q}>{(d) => <ShapBody d={d} reg={reg} setCls={setCls} row={row} setRow={setRow} />}</Load>
}

/* ------------------------------------------------------------------ 05 // ROC OVERLAY, SAVED RUNS */
export function RocOverlayTab({ base, rows, colorOf }) {
  const sig = rows.map((r) => r.key).join(',')
  const [data, setData] = useState({})
  const [err, setErr] = useState(null)
  useEffect(() => {
    let live = true
    setErr(null)
    Promise.all(rows.map((r) => api.get(`${base}/curves/${r.key}`).then((d) => [r.abbr, d])))
      .then((pairs) => live && setData(Object.fromEntries(pairs)))
      .catch((e) => live && setErr(messageOf(e)))
    return () => { live = false }
  }, [sig, base])
  if (err) return <Msg color={K.red}>{err}</Msg>
  return (
    <div data-testid="roc-overlay">
      <Sub>ROC // ALL TRAINED MODELS</Sub>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 4, right: 8, bottom: 18, left: 0 }}>
            {grid}
            <XAxis type="number" dataKey="x" domain={[0, 1]} label={axisLabel('FPR')} {...axis} />
            <YAxis type="number" dataKey="y" domain={[0, 1]} width={30} {...axis} />
            <Tooltip {...tip} formatter={(v) => v.toFixed(3)} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#52525b" strokeDasharray="3 3" />
            {rows.map((r) => {
              const c = data[r.abbr]
              if (!c || !c.roc.classes.length) return null
              const cl = c.roc.classes[0]
              return <Line key={r.abbr} data={cl.fpr.map((x, i) => ({ x, y: cl.tpr[i] }))} dataKey="y" name={`${r.abbr} ${f3(cl.auc)}`} stroke={colorOf(r.abbr)} dot={false} strokeWidth={2} isAnimationActive={false} />
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase text-white">
        {rows.map((r) => {
          const c = data[r.abbr]
          const auc = c && c.roc.classes.length ? c.roc.classes[0].auc : null
          return <span key={r.abbr} className="flex items-center gap-1"><span className="inline-block w-[10px] h-[10px]" style={{ background: colorOf(r.abbr) }} />{r.abbr} {isNum(auc) ? f3(auc) : '--'}</span>
        })}
      </div>
      <Note>Higher and further left means the model ranks positives above negatives more reliably. Multiclass shows the first class only.</Note>
    </div>
  )
}

const LOWER_BETTER = new Set(['rmse', 'mae'])
const CLS_METRICS = [['score', 'F1'], ['accuracy', 'ACC'], ['precision', 'PREC'], ['recall', 'REC'], ['roc_auc', 'AUC'], ['avg_precision', 'AP']]
const REG_METRICS = [['score', 'R2'], ['rmse', 'RMSE'], ['mae', 'MAE']]

function ModelAcrossRuns({ cmp, order }) {
  const reg = cmp.task === 'regression'
  const metrics = reg ? REG_METRICS : CLS_METRICS
  const [metric, setMetric] = useState('score')
  const lower = LOWER_BETTER.has(metric)
  const val = (row) => (row ? (metric === 'score' ? row.score : row[metric]) : null)
  const models = [...cmp.models].sort((a, b) => Math.max(...order.map((i) => val(b.by_run[i]) ?? -Infinity)) - Math.max(...order.map((i) => val(a.by_run[i]) ?? -Infinity)))
  const data = models.map((m) => ({ abbr: m.abbr, ...Object.fromEntries(order.map((id) => [id, val(m.by_run[id])])) }))
  const label = (id) => cmp.runs.find((r) => r.id === id).label
  const first = order[0]
  const lastId = order[order.length - 1]
  return (
    <div data-testid="cmp-models">
      <div className="flex items-center gap-2 mb-1"><Sub>MODELS ACROSS RUNS</Sub><Seg value={metric} onChange={setMetric} testid="cmp-metric" options={metrics} /></div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            {grid}
            <XAxis dataKey="abbr" {...axis} />
            <YAxis width={40} tickFormatter={fSig} domain={[0, 'auto']} {...axis} />
            <Tooltip {...tip} cursor={{ fill: '#ffffff10' }} formatter={(v, id) => [isNum(v) ? fSig(v) : 'NOT RUN', label(id)]} />
            {order.map((id, i) => <Bar key={id} dataKey={id} fill={PALETTE[i % PALETTE.length]} isAnimationActive={false} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="overflow-auto">
        <table className="w-full text-[10px]" data-testid="cmp-table">
          <thead>
            <tr className="text-zinc-500 uppercase">
              <th className="text-left py-[2px] font-normal">MDL</th>
              {order.map((id, i) => <th key={id} className="text-right px-1 font-normal" style={{ color: PALETTE[i % PALETTE.length] }}>{label(id).slice(0, 14)}</th>)}
              {order.length > 1 && <th className="text-right px-1 font-normal">D LAST/FIRST</th>}
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const vals = order.map((i) => val(m.by_run[i])).filter(isNum)
              const best = lower ? Math.min(...vals) : Math.max(...vals)
              const a = val(m.by_run[first]); const b = val(m.by_run[lastId])
              const good = isNum(a) && isNum(b) && Math.abs(b - a) > 1e-9 ? ((lower ? b < a : b > a) ? K.green : K.red) : K.dim
              return (
                <tr key={m.abbr} className="border-t border-zinc-800">
                  <td className="py-[2px] text-white font-bold">{m.abbr}</td>
                  {order.map((id) => { const v = val(m.by_run[id]); return <td key={id} className="text-right px-1" style={{ color: v === best && order.length > 1 ? K.green : '#d4d4d8', fontWeight: v === best ? 700 : 400 }}>{isNum(v) ? fSig(v) : '--'}</td> })}
                  {order.length > 1 && <td className="text-right px-1" style={{ color: good }}>{isNum(a) && isNum(b) ? fSigned(b - a, 3) : '--'}</td>}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <Note>Green marks each model&apos;s best run. The delta compares the last ticked run with the first, so tick the baseline first.</Note>
    </div>
  )
}

function DatasetCompare({ cmp, order }) {
  const runs = order.map((id) => cmp.runs.find((r) => r.id === id))
  const ds = cmp.datasets
  const dsOf = (r) => ds.find((d) => d.id === r.dataset_id).profile
  const tp = (r) => r.target_profile
  const row = (label, get, fmt = (v) => v) => (
    <tr className="border-t border-zinc-800">
      <td className="py-[2px] pr-2 text-zinc-500 whitespace-nowrap">{label}</td>
      {runs.map((r) => { const v = get(r); return <td key={r.id} className="px-1 text-white">{v === undefined || v === null ? '--' : fmt(v)}</td> })}
    </tr>
  )
  return (
    <div data-testid="cmp-datasets" className="mt-2">
      <div className="flex items-center gap-2 mb-1">
        <Sub>DATASET COMPARISON</Sub>
        <span className="px-1 text-[10px] font-bold" style={{ background: ds.length > 1 ? K.yellow : K.green, color: '#000' }}>{ds.length} DATASET{ds.length > 1 ? 'S' : ''}</span>
      </div>
      <div className="overflow-auto">
        <table className="w-full text-[10px] uppercase">
          <thead><tr><th />{runs.map((r, i) => <th key={r.id} className="text-left px-1 font-bold" style={{ color: PALETTE[i % PALETTE.length] }}>{r.label.slice(0, 16)}</th>)}</tr></thead>
          <tbody>
            {row('DATASET', (r) => r.dataset_name)}
            {row('ROWS', (r) => dsOf(r).rows, fInt)}
            {row('COLUMNS', (r) => dsOf(r).columns)}
            {row('NUM / CAT', (r) => `${dsOf(r).numeric_columns} / ${dsOf(r).categorical_columns}`)}
            {row('MISSING', (r) => dsOf(r).missing_pct, (v) => `${v}%`)}
            {row('DUPLICATES', (r) => dsOf(r).duplicate_rows, fInt)}
            {row('TOP CORR', (r) => dsOf(r).top_correlation, (c) => `${c.a} x ${c.b} (${c.r})`)}
            {row('TRAINED ON', (r) => r.rows_used, fInt)}
            {row('TARGET', (r) => r.target)}
            {row('BALANCE', (r) => (tp(r) ? (tp(r).n_classes ? `${tp(r).n_classes} CLS, MIN ${fPct(tp(r).minority_ratio)}` : `MEAN ${fSig(tp(r).mean)} STD ${fSig(tp(r).std)}`) : null))}
            {row('TEST SIZE', (r) => r.settings.test_size, (v) => `${Math.round(v * 100)}%`)}
            {row('IMBALANCE', (r) => `${r.settings.handle_imbalance} / SCALE ${r.settings.scale_features ? 'ON' : 'OFF'}`)}
            {row('TUNING', (r) => (r.settings.tune ? `${r.settings.tune_trials} TRIALS` : 'OFF'))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function CurveOverlay({ cmp, order }) {
  const withCurves = order.filter((id) => cmp.curves[id])
  const keys = [...new Set(withCurves.flatMap((id) => Object.keys(cmp.curves[id])))]
  const nameOf = (key) => { for (const m of cmp.models) for (const r of Object.values(m.by_run)) if (r.model === key) return r.abbr; return key }
  const [key, setKey] = useState(keys[0])
  const [kind, setKind] = useState('roc')
  const k = keys.includes(key) ? key : keys[0]
  if (!keys.length) return null
  const label = (id) => cmp.runs.find((r) => r.id === id).label
  const lines = withCurves.filter((id) => cmp.curves[id][k]).map((id) => {
    const c = cmp.curves[id][k]
    const cl = (kind === 'roc' ? c.roc : c.pr).classes[0]
    if (!cl) return null
    return { id, auc: kind === 'roc' ? cl.auc : cl.ap, pts: kind === 'roc' ? cl.fpr.map((x, i) => ({ x, y: cl.tpr[i] })) : cl.recall.map((x, i) => ({ x, y: cl.precision[i] })) }
  }).filter(Boolean)
  return (
    <div className="mt-2" data-testid="cmp-curves">
      <div className="flex items-center gap-2 mb-1">
        <Sub>CURVE OVERLAY</Sub>
        <Seg value={kind} onChange={setKind} options={[['roc', 'ROC'], ['pr', 'PR']]} />
        <select className={ctl} value={k} onChange={(e) => setKey(e.target.value)}>{keys.map((x) => <option key={x} value={x}>{nameOf(x)}</option>)}</select>
      </div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 4, right: 8, bottom: 18, left: 0 }}>
            {grid}
            <XAxis type="number" dataKey="x" domain={[0, 1]} label={axisLabel(kind === 'roc' ? 'FPR' : 'RECALL')} {...axis} />
            <YAxis type="number" dataKey="y" domain={[0, 1]} width={30} {...axis} />
            <Tooltip {...tip} formatter={(v) => v.toFixed(3)} />
            <Legend verticalAlign="top" height={18} wrapperStyle={{ fontSize: 10 }} />
            {kind === 'roc' && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#52525b" strokeDasharray="3 3" />}
            {lines.map((l) => <Line key={l.id} data={l.pts} dataKey="y" name={`${label(l.id).slice(0, 14)} (${f3(l.auc)})`} stroke={PALETTE[order.indexOf(l.id) % PALETTE.length]} dot={false} strokeWidth={2} isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function RunsTab({ refreshKey }) {
  const [tick, setTick] = useState(0)
  const runsQ = useApi(`/api/v1/runs?limit=200&t=${tick}&k=${refreshKey}`)
  const dsQ = useApi(`/api/v1/datasets?t=${tick}&k=${refreshKey}`)
  const [sel, setSel] = useState([])
  const [cmp, setCmp] = useState(null)
  const [err, setErr] = useState(null)
  const [dsFilter, setDsFilter] = useState('all')
  const runs = runsQ.data ? runsQ.data.runs : []
  const shown = runs.filter((r) => dsFilter === 'all' || r.dataset_id === dsFilter)
  useEffect(() => setSel((s) => s.filter((id) => runs.some((r) => r.id === id))), [runsQ.data])
  useEffect(() => {
    let live = true
    setErr(null)
    if (sel.length < 2) { setCmp(null); return undefined }
    api.post('/api/v1/compare', { run_ids: sel }).then((d) => live && setCmp(d)).catch((e) => live && setErr(messageOf(e)))
    return () => { live = false }
  }, [sel.join(',')])
  const toggle = (id) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-8)))
  const del = async (id) => {
    try { setErr(null); await api.del(`/api/v1/runs/${id}`); setTick((t) => t + 1) } catch (e) { setErr(messageOf(e)) }
  }
  return (
    <div data-testid="runs">
      <div className="flex flex-wrap items-center gap-1 mb-1">
        <Sub>SAVED RUNS // EVERY TRAINING JOB IS SAVED AUTOMATICALLY</Sub>
        {dsQ.data && dsQ.data.datasets.length > 1 && (
          <select data-testid="ds-filter" className={ctl} value={dsFilter} onChange={(e) => setDsFilter(e.target.value)}>
            <option value="all">ALL DATASETS</option>
            {dsQ.data.datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.n_runs})</option>)}
          </select>
        )}
        <Btn disabled={!sel.length} onClick={() => setSel([])}>CLEAR</Btn>
      </div>
      {runsQ.error ? <Msg color={K.red}>{runsQ.error}</Msg> : shown.length === 0 ? <Msg testid="no-runs">NO SAVED RUNS YET // RUN THE PIPELINE</Msg> : (
        <div className="overflow-auto max-h-48 border border-zinc-800">
          <table className="w-full text-[10px] uppercase" data-testid="runs-table">
            <thead className="bg-zinc-950 text-zinc-500">
              <tr><th className="w-5" /><th className="text-left font-normal px-1">RUN</th><th className="text-left font-normal px-1">TASK</th><th className="text-left font-normal px-1">BEST</th><th className="text-right font-normal px-1">MDL</th><th className="text-left font-normal px-1">SETUP</th><th className="text-left font-normal px-1">WHEN</th><th className="w-8" /></tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const idx = sel.indexOf(r.id)
                return (
                  <tr key={r.id} data-testid={`run-${r.id}`} className={`border-t border-zinc-800 ${idx >= 0 ? 'bg-zinc-800' : 'hover:bg-zinc-800'}`}>
                    <td className="px-1"><input type="checkbox" data-testid={`pick-${r.id}`} checked={idx >= 0} onChange={() => toggle(r.id)} /></td>
                    <td className="px-1 py-[2px] text-white font-bold max-w-[140px] truncate" title={`${r.label} (${r.dataset_name} > ${r.target})`}>
                      {idx >= 0 && <span className="inline-block w-2 h-2 mr-1" style={{ background: PALETTE[idx % PALETTE.length] }} />}{r.label}
                    </td>
                    <td className="px-1 text-zinc-300">{r.task === 'classification' ? 'CLS' : 'REG'}</td>
                    <td className="px-1">{r.top ? <><b style={{ color: K.purple }}>{r.top.abbr}</b> <span className="text-white">{f3(r.top.score)}</span></> : '--'}</td>
                    <td className="px-1 text-right text-zinc-300">{r.n_models}</td>
                    <td className="px-1 text-zinc-400">{Math.round((r.settings.test_size || 0) * 100)}%{r.tuned ? ' TUN' : ''}{r.settings.handle_imbalance && r.settings.handle_imbalance !== 'none' ? ` ${r.settings.handle_imbalance}` : ''}</td>
                    <td className="px-1 text-zinc-500 whitespace-nowrap">{fDate(r.created_at)}</td>
                    <td className="px-1 text-right"><button type="button" title="Delete this run" onClick={() => del(r.id)} className="text-[9px] font-bold px-1" style={{ background: K.red, color: '#fff' }}>DEL</button></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-1 text-[10px] uppercase text-zinc-500">{sel.length < 2 ? 'TICK TWO OR MORE RUNS TO COMPARE THEM (DIFFERENT MODELS, TUNED VS DEFAULT, OR DIFFERENT DATASETS)' : `${sel.length} RUNS SELECTED (MAX 8) // ORDER OF TICKING = ORDER OF COMPARISON`}</div>
      {err && <Msg color={K.red}>{err}</Msg>}
      {cmp && (
        <div data-testid="comparison" className="mt-2">
          {cmp.warnings.map((w) => <div key={w} className="mb-1 px-1 text-[10px] font-bold uppercase" style={{ background: K.yellow, color: '#000' }}>{w}</div>)}
          <ModelAcrossRuns cmp={cmp} order={sel} />
          {cmp.task === 'classification' && <CurveOverlay cmp={cmp} order={sel} />}
          <DatasetCompare cmp={cmp} order={sel} />
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 05 // SUMMARY (best model and why) */
const trainOf = (r) => (isNum(r.train_score) ? r.train_score : r.train_balanced_accuracy)
const testOf = (r) => (isNum(r.test_score) ? r.test_score : r.test_balanced_accuracy)
const secs = (t) => (isNum(t) ? (t < 10 ? `${t.toFixed(2)}s` : `${t.toFixed(1)}s`) : '--')
const cvOf = (r) => (r.cv && isNum(r.cv.score) ? r.cv : null)
const cvMetric = (r, name) => {
  const m = r.cv && r.cv.metrics && r.cv.metrics[name]
  return m && isNum(m.mean) ? m : null
}
const pm = (m, f = f3) => (m ? `${f(m.mean)} +/- ${f(m.std)}` : '--')

/**
 * Pick the model to recommend. Cross-validated score (average over folds) decides when every candidate has one, because a
 * single hold-out split can be lucky; otherwise the hold-out leaderboard order is used. A model flagged overfit/underfit is
 * never recommended while a clean one exists.
 */
export function recommend(items) {
  const clean = items.filter((m) => !m.flags.overfit && !m.flags.underfit)
  const pool = clean.length ? clean : items
  const useCv = pool.every((m) => cvOf(m.row))
  const rec = useCv ? [...pool].sort((a, b) => cvOf(b.row).score - cvOf(a.row).score)[0] : pool[0]
  // flagged models that out-scored the recommendation on the hold-out split (so the user knows why they were passed over)
  const beats = (m) => (m.row.score ?? 0) > (rec.row.score ?? 0) || (useCv && cvOf(m.row) && cvOf(m.row).score > cvOf(rec.row).score)
  const skipped = items.filter((m) => !pool.includes(m) && beats(m))
  const sem = (m) => cvOf(m.row).score_sem ?? cvOf(m.row).score_std ?? 0
  // paired fold-by-fold test when both models share folds; otherwise the one-standard-error rule
  const tiedWith = (m) => {
    const clear = clearlyBetter(rec.row, m.row)
    return clear === null ? cvOf(m.row).score >= cvOf(rec.row).score - sem(rec) : !clear
  }
  const tied = useCv ? pool.filter((m) => m !== rec && tiedWith(m)) : []
  const paired = tied.length > 0 && tied.every((m) => clearlyBetter(rec.row, m.row) !== null)
  return { rec, skipped, useCv, tied, paired, holdoutLeader: pool[0] }
}

// two-sided 5% critical values of Student's t for 1..9 degrees of freedom
const T_CRIT = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262]
const headFolds = (r) => {
  const c = cvOf(r)
  const f = c && c.metrics && c.metrics[c.metric] && c.metrics[c.metric].folds
  return Array.isArray(f) && f.length >= 3 && f.length <= 10 ? f : null
}

/**
 * Is model `a` clearly different from `b`? Paired t-test on the two models' scores for the same CV folds, with the Nadeau-Bengio
 * correction because folds share training rows (a plain paired test would call too many differences real). Returns null when
 * the folds cannot be paired (a model used fewer folds, or no fold scores), so callers fall back to the standard-error rule.
 */
export function clearlyBetter(a, b) {
  const fa = headFolds(a)
  const fb = headFolds(b)
  if (!fa || !fb || fa.length !== fb.length) return null
  const k = fa.length
  const d = fa.map((v, i) => v - fb[i])
  const mean = d.reduce((s, v) => s + v, 0) / k
  const variance = d.reduce((s, v) => s + (v - mean) ** 2, 0) / (k - 1)
  if (variance === 0) return mean !== 0
  const t = mean / Math.sqrt((1 / k + 1 / (k - 1)) * variance)
  return Math.abs(t) > T_CRIT[k - 2]
}

function reasons({ items, pick, reg, ctx }) {
  const { rec, useCv, tied, holdoutLeader } = pick
  const r = rec.row
  const cv = cvOf(r)
  const rest = items.filter((m) => m !== rec)
  const lines = []
  const sn = r.score_name || (reg ? 'R2' : 'F1')
  const second = rest.filter((m) => !m.flags.overfit && !m.flags.underfit).sort((a, b) => (useCv ? cvOf(b.row).score - cvOf(a.row).score : (b.row.score ?? 0) - (a.row.score ?? 0)))[0] || rest[0]
  if (useCv && cv) {
    let line = `CROSS-VALIDATED ${sn} ${f3(cv.score)} +/- ${f3(cv.score_std)} OVER ${cv.folds} FOLDS (HOLD-OUT ${f3(r.score)})`
    if (second) {
      const d = cv.score - cvOf(second.row).score
      const noise = cv.score_sem ?? cv.score_std ?? 0
      const clear = clearlyBetter(r, second.row)
      const isClear = clear === null ? Math.abs(d) > noise : clear
      line += `; ${d >= 0 ? '+' : ''}${f3(d)} AHEAD OF ${second.abbr} (${f3(cvOf(second.row).score)}) - ${isClear ? (clear === null ? 'OUTSIDE ONE STANDARD ERROR, A REAL LEAD' : 'A REAL LEAD, CONSISTENT ACROSS THE SAME FOLDS') : 'WITHIN NOISE, SO THIS LEAD IS NOT CLEAR-CUT'}`
    }
    lines.push([K.green, `${line}.`])
    if (holdoutLeader && holdoutLeader !== rec) {
      lines.push([K.yellow, `THE SINGLE HOLD-OUT SPLIT RANKS ${holdoutLeader.abbr} FIRST (${sn} ${f3(holdoutLeader.row.score)}) BUT ${cv.folds}-FOLD CROSS-VALIDATION PREFERS ${rec.abbr}. ONE SPLIT CAN BE LUCKY, SO THE AVERAGE OVER FOLDS IS THE MORE RELIABLE GUIDE.`])
    } else if (rest.length) {
      lines.push([K.green, 'THE HOLD-OUT SPLIT AND CROSS-VALIDATION AGREE ON THE LEADER.'])
    }
    const rel = cv.score !== 0 ? cv.score_std / Math.abs(cv.score) : null
    const head = cv.metrics[cv.metric]
    lines.push([rel !== null && rel > 0.15 ? K.yellow : K.green, `FOLD-TO-FOLD RANGE ${f3(head.min)} TO ${f3(head.max)}: ${rel !== null && rel > 0.15 ? 'UNSTABLE, THE SCORE DEPENDS A LOT ON WHICH ROWS IT SEES' : 'STABLE ACROSS DIFFERENT SPLITS'}.`])
    if (tied.length) {
      const fastest = [rec, ...tied].sort((a, b) => (a.row.train_time_sec ?? 1e9) - (b.row.train_time_sec ?? 1e9))[0]
      lines.push([K.yellow, `${tied.map((m) => m.abbr).join(', ')} ${tied.length > 1 ? 'ARE' : 'IS'} STATISTICALLY TIED WITH ${rec.abbr} (${pick.paired ? 'NO CLEAR DIFFERENCE FOLD BY FOLD ON THE SAME SPLITS' : 'WITHIN ONE STANDARD ERROR'}).${fastest !== rec ? ` IF YOU PREFER SPEED OR SIMPLICITY, ${fastest.abbr} (${secs(fastest.row.train_time_sec)}) IS A FAIR CHOICE.` : ''}`])
    }
  } else if (second) {
    const d = (r.score ?? 0) - (second.row.score ?? 0)
    lines.push([K.green, `HIGHEST ${sn} AMONG THE MODELS THAT FIT CLEANLY: ${f3(r.score)} VS ${f3(second.row.score)} FOR ${second.abbr} (${d >= 0 ? '+' : ''}${f3(d)})${Math.abs(d) < 0.01 ? ' - A CLOSE CALL, CONSIDER THE RUNNER-UP IF YOU NEED SPEED OR SIMPLICITY' : ''}. NO CROSS-VALIDATION IS AVAILABLE FOR THIS RUN, SO THIS RESTS ON ONE SPLIT.`])
  } else lines.push([K.green, `ONLY ONE MODEL TRAINED: ${sn} ${f3(r.score)}. TRAIN MORE MODELS TO COMPARE.`])
  if (reg) {
    const r2 = cvMetric(r, 'r2')
    const rmse = cvMetric(r, 'rmse')
    const mae = cvMetric(r, 'mae')
    const rmseV = rmse ? rmse.mean : r.rmse
    const rel = isNum(rmseV) && isNum(ctx.targetStd) && ctx.targetStd > 0 ? rmseV / ctx.targetStd : null
    lines.push([K.cyan, `EXPLAINS ${fPct(r2 ? r2.mean : r.r2, 1)} OF THE TARGET'S VARIANCE (R2 ${pm(r2) !== '--' ? pm(r2) : f3(r.r2)}). TYPICAL ERROR: RMSE ${rmse ? `${fSig(rmse.mean)} +/- ${fSig(rmse.std)}` : fSig(r.rmse)}, MAE ${mae ? `${fSig(mae.mean)} +/- ${fSig(mae.std)}` : fSig(r.mae)}${rel !== null ? ` - RMSE IS ${fPct(rel, 0)} OF THE TARGET'S STANDARD DEVIATION (LOWER IS BETTER, 100% = NO BETTER THAN PREDICTING THE MEAN)` : ''}.`])
  } else {
    const cm = r.confusion_matrix
    const total = Array.isArray(cm) ? cm.flat().reduce((a, b) => a + b, 0) : 0
    const major = Array.isArray(cm) && total ? Math.max(...cm.map((row) => row.reduce((a, b) => a + b, 0))) / total : null
    if (major !== null && major > 0.6) lines.push([K.yellow, `THE TEST SET IS ${fPct(major, 0)} ONE CLASS, SO ACCURACY (${f3(r.accuracy)}) ALONE FLATTERS: ALWAYS GUESSING THE MAJORITY CLASS SCORES ${f3(major)}. THE RANKING USES ${sn} AND BALANCED ACCURACY (${f3(testOf(r))}) INSTEAD.`])
    const p = cvMetric(r, 'precision')
    const rc = cvMetric(r, 'recall')
    const pv = p ? p.mean : r.precision
    const rv = rc ? rc.mean : r.recall
    lines.push([K.cyan, `PRECISION ${p ? pm(p) : f3(r.precision)} (OF ITS POSITIVE CALLS, ${fPct(pv, 0)} ARE RIGHT) AND RECALL ${rc ? pm(rc) : f3(r.recall)} (IT CATCHES ${fPct(rv, 0)} OF TRUE POSITIVES).${isNum(rv) && isNum(pv) && rv + 0.15 < pv ? ' IT IS CAUTIOUS: LOWER THE DECISION THRESHOLD IN PANEL 04 TO TRADE PRECISION FOR RECALL.' : ''}`])
    const auc = cvMetric(r, 'roc_auc')
    const aucV = auc ? auc.mean : r.roc_auc
    if (isNum(aucV)) {
      const aucOf = (m) => (cvMetric(m.row, 'roc_auc') ? cvMetric(m.row, 'roc_auc').mean : m.row.roc_auc ?? -1)
      const bestAuc = [...items].sort((a, b) => aucOf(b) - aucOf(a))[0]
      lines.push([K.cyan, `ROC AUC ${auc ? pm(auc) : f3(aucV)}, AVG PRECISION ${f3(r.avg_precision)}: ${aucV >= 0.9 ? 'EXCELLENT' : aucV >= 0.8 ? 'GOOD' : aucV >= 0.7 ? 'FAIR' : 'WEAK'} AT RANKING POSITIVES ABOVE NEGATIVES${bestAuc !== rec && aucOf(bestAuc) > aucV + 0.005 ? `. NOTE ${bestAuc.abbr} HAS THE HIGHEST AUC (${f3(aucOf(bestAuc))}), SO IT MAY SORT BETTER EVEN THOUGH IT SCORES LOWER ON ${sn} AT THE DEFAULT 0.5 THRESHOLD` : ''}.`])
    }
  }
  const gap = r.train_test_gap
  lines.push([rec.flags.overfit || rec.flags.underfit ? K.yellow : K.green, r.gap_check === false
    ? `TRAIN ${f3(trainOf(r))} IS ~PERFECT BY DESIGN FOR THIS MODEL TYPE (IT MEMORISES), SO JUDGE IT BY ITS TEST SCORE ${f3(testOf(r))}.`
    : `GENERALISES: TRAIN ${f3(trainOf(r))} VS TEST ${f3(testOf(r))} (GAP ${f3(gap)}, LIMIT ${f3(ctx.thresholds.overfitGap)})${rec.flags.overfit ? ' - OVER THE LIMIT, SO TREAT IT AS OVERFIT' : rec.flags.underfit ? ' - BUT THE MODEL IS BARELY LEARNING (UNDERFIT)' : ' - NO OVERFIT OR UNDERFIT DETECTED'}.`])
  const times = items.map((m) => m.row.train_time_sec).filter(isNum)
  if (isNum(r.train_time_sec) && times.length > 1) {
    const fastest = items.find((m) => m.row.train_time_sec === Math.min(...times))
    lines.push([K.dim, `TRAINED IN ${secs(r.train_time_sec)}${fastest === rec ? ' (THE FASTEST)' : `; THE FASTEST WAS ${fastest.abbr} AT ${secs(fastest.row.train_time_sec)}`}.${r.tuned ? ' HYPER-PARAMETERS WERE TUNED WITH OPTUNA.' : ''}`])
  }
  return lines
}

/** Mean +/- std per model (bar + whisker) against its single hold-out score (white tick). */
function CvChart({ items, rec, name }) {
  const rows = items.filter((m) => cvOf(m.row)).sort((a, b) => cvOf(b.row).score - cvOf(a.row).score)
  if (!rows.length) return null
  const hi = Math.max(...rows.flatMap((m) => [cvOf(m.row).score + cvOf(m.row).score_std, m.row.score ?? 0]), 0.0001) * 1.05
  const lo = Math.min(0, ...rows.map((m) => cvOf(m.row).score - cvOf(m.row).score_std))
  const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`
  return (
    <div className="mb-1" data-testid="summary-cv">
      <Sub>CROSS-VALIDATED {name} // BAR = MEAN OVER FOLDS, WHISKER = +/- 1 STD, WHITE TICK = SINGLE HOLD-OUT</Sub>
      {rows.map((m) => {
        const c = cvOf(m.row)
        const color = m === rec ? K.purple : '#71717a'
        return (
          <div key={m.abbr} data-testid={`cv-row-${m.abbr}`} className="grid grid-cols-[34px_1fr_104px] items-center gap-1 text-[10px] mb-[2px]">
            <span className="font-bold" style={{ color: m === rec ? K.purple : '#fff' }}>{m.abbr}</span>
            <span className="relative h-3 bg-zinc-900 block">
              <span className="absolute top-0 left-0 h-full" style={{ width: pos(c.score), background: color, opacity: 0.85 }} />
              <span className="absolute top-[5px] h-[2px] bg-white" style={{ left: pos(c.score - c.score_std), width: `calc(${pos(c.score + c.score_std)} - ${pos(c.score - c.score_std)})` }} />
              <span className="absolute top-0 h-full w-[2px]" style={{ left: pos(m.row.score ?? 0), background: '#fff' }} title={`hold-out ${f3(m.row.score)}`} />
            </span>
            <span className="text-right text-zinc-300">{f3(c.score)} <span className="text-zinc-500">+/- {f3(c.score_std)}</span></span>
          </div>
        )
      })}
    </div>
  )
}

export function SummaryTab({ rows, ctx }) {
  if (!rows.length) return <Msg>NO RESULTS // RUN THE PIPELINE</Msg>
  const reg = ctx.reg
  const pick = recommend(rows)
  const { rec, skipped, useCv } = pick
  const leader = rows[0]
  const r = rec.row
  const cv = cvOf(r)
  const why = reasons({ items: rows, pick, reg, ctx })
  const sn = r.score_name || (reg ? 'R2' : 'F1')
  const cols = reg
    ? [['CV R2', (x) => cvOf(x)?.score, 1, (x) => (cvOf(x) ? `${f3(cvOf(x).score)}+/-${f3(cvOf(x).score_std)}` : null)], ['R2', (x) => x.score, 1], ['RMSE', (x) => x.rmse, -1], ['MAE', (x) => x.mae, -1], ['TRAIN', trainOf, 0], ['TEST', testOf, 1], ['GAP', (x) => x.train_test_gap, -1], ['TIME', (x) => x.train_time_sec, -1]]
    : [['CV F1', (x) => cvOf(x)?.score, 1, (x) => (cvOf(x) ? `${f3(cvOf(x).score)}+/-${f3(cvOf(x).score_std)}` : null)], ['F1', (x) => x.score, 1], ['ACC', (x) => x.accuracy, 1], ['PREC', (x) => x.precision, 1], ['REC', (x) => x.recall, 1], ['AUC', (x) => x.roc_auc, 1], ['TEST BAL', testOf, 1], ['GAP', (x) => x.train_test_gap, -1], ['TIME', (x) => x.train_time_sec, -1]]
  const best = cols.map(([, get, dir]) => {
    const vals = rows.map((m) => get(m.row)).filter(isNum)
    return dir === 0 || !vals.length ? null : dir > 0 ? Math.max(...vals) : Math.min(...vals)
  })
  const fmt = (name, v) => (!isNum(v) ? '--' : name === 'TIME' ? secs(v) : name === 'RMSE' || name === 'MAE' ? fSig(v) : f3(v))
  return (
    <div data-testid="summary">
      <div className="border p-1 mb-1" style={{ borderColor: K.purple }} data-testid="summary-verdict">
        <div className="text-[10px] uppercase text-zinc-400">RECOMMENDED MODEL FOR {ctx.target ? `[${ctx.target}]` : 'THIS DATASET'} // {reg ? 'REGRESSION' : 'CLASSIFICATION'} // {rows.length} MODELS COMPARED{useCv ? ` // ${cv.folds}-FOLD CROSS-VALIDATION` : ''}</div>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-base font-bold" style={{ color: K.purple }}>{rec.abbr}</span>
          <span className="text-white text-xs">{r.name}</span>
          {useCv && cv ? <span className="text-white font-bold">CV {sn} {f3(cv.score)} <span className="text-zinc-400 font-normal">+/- {f3(cv.score_std)}</span></span> : null}
          <span className={useCv ? 'text-zinc-400' : 'text-white font-bold'}>{useCv ? 'HOLD-OUT ' : ''}{sn} {f3(r.score)}</span>
          {r.tuned && <span className="px-1 text-[10px] font-bold" style={{ background: K.purple, color: '#fff' }}>TUNED</span>}
          {rec.flags.overfit && <span className="px-1 text-[10px] font-bold" style={{ background: K.red, color: '#fff' }}>OVF</span>}
          {rec.flags.underfit && <span className="px-1 text-[10px] font-bold" style={{ background: K.yellow, color: '#000' }}>UNF</span>}
        </div>
        {skipped.length > 0 && (
          <div className="mt-1 text-[10px] uppercase" style={{ color: K.yellow }} data-testid="summary-skipped">
            {skipped.map((m) => `${m.abbr} (${(m.row.score_name || 'F1')} ${useCv && cvOf(m.row) ? `CV ${f3(cvOf(m.row).score)}` : f3(m.row.score)})`).join(', ')} RANKED HIGHER BUT {skipped.length > 1 ? 'ARE' : 'IS'} FLAGGED {skipped.some((m) => m.flags.overfit) ? 'OVERFIT' : 'UNDERFIT'}, SO {rec.abbr} IS RECOMMENDED INSTEAD.
          </div>
        )}
      </div>
      <Sub>WHY</Sub>
      <ul className="mb-1" data-testid="summary-why">
        {why.map(([color, text], i) => (
          <li key={i} className="flex gap-1 text-[10px] leading-snug mb-[2px]"><span className="shrink-0 inline-block w-[6px] h-[6px] mt-[3px]" style={{ background: color }} /><span className="text-zinc-200">{text}</span></li>
        ))}
      </ul>
      <CvChart items={rows} rec={rec} name={sn} />
      <Sub>ALL MODELS // GREEN = BEST IN COLUMN</Sub>
      <div className="overflow-auto border border-zinc-800">
        <table className="w-full text-[10px] uppercase" data-testid="summary-table">
          <thead className="bg-zinc-950 text-zinc-500">
            <tr>
              <th className="text-left font-normal px-1">#</th><th className="text-left font-normal px-1">MDL</th>
              {cols.map(([n]) => <th key={n} className="text-right font-normal px-1 whitespace-nowrap">{n}</th>)}
              <th className="text-left font-normal px-1">FLAGS</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.abbr} data-testid={`summary-row-${m.abbr}`} className={`border-t border-zinc-800 ${m === rec ? 'bg-zinc-800' : ''}`}>
                <td className="px-1 text-zinc-500">{m.row.rank}</td>
                <td className="px-1 font-bold" style={{ color: m === rec ? K.purple : '#fff' }}>{m.abbr}{m === rec ? ' *' : ''}</td>
                {cols.map(([n, get, , show], i) => {
                  const v = get(m.row)
                  const isBest = best[i] !== null && v === best[i]
                  return <td key={n} className="px-1 text-right whitespace-nowrap" style={{ color: isBest ? K.green : '#d4d4d8', fontWeight: isBest ? 700 : 400 }}>{show ? show(m.row) || '--' : fmt(n, v)}</td>
                })}
                <td className="px-1 whitespace-nowrap">
                  {m.flags.overfit && <span className="px-[2px] mr-[2px] text-[8px] font-bold" style={{ background: K.red, color: '#fff' }}>OVF</span>}
                  {m.flags.underfit && <span className="px-[2px] mr-[2px] text-[8px] font-bold" style={{ background: K.yellow, color: '#000' }}>UNF</span>}
                  {m.row.tuned && <span className="px-[2px] text-[8px] font-bold" style={{ background: K.purple, color: '#fff' }}>TUN</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-1 text-[10px] uppercase text-zinc-500" data-testid="summary-caveats">
        {ctx.split && <div>HOLD-OUT: {fInt(ctx.split.test_rows)} ROWS (TRAINED ON {fInt(ctx.split.train_rows)}){ctx.split.stratified ? ', STRATIFIED' : ''}{ctx.average ? `, ${String(ctx.average).toUpperCase()} AVERAGING` : ''}. {ctx.split.test_rows < 300 ? 'SMALL TEST SET: DIFFERENCES OF A FEW POINTS MAY BE NOISE.' : ''}</div>}
        {useCv && cv && <div>CROSS-VALIDATION: {cv.folds} FOLDS ON {fInt(cv.rows_used)} TRAINING ROWS ONLY (THE HOLD-OUT ROWS ARE NEVER USED). {cv.note || ''}</div>}
        {!useCv && <div style={{ color: K.yellow }}>NO CROSS-VALIDATION FOR AT LEAST ONE MODEL: THE RECOMMENDATION RESTS ON A SINGLE SPLIT.</div>}
        {leader !== rec && !useCv && <div>LEADERBOARD #1 IS {leader.abbr}; SEE THE NOTE ABOVE.</div>}
        {(ctx.failed || []).length > 0 && <div style={{ color: K.red }}>FAILED: {ctx.failed.map((f) => f.model).join(', ')}</div>}
        {(ctx.warnings || []).map((w) => <div key={w} style={{ color: K.yellow }}>{w}</div>)}
        <div className="normal-case text-zinc-600">A recommendation, not a guarantee: the data you see later may differ from this sample. For a decision that matters, re-run with a different test split and check the result holds.</div>
      </div>
    </div>
  )
}


/* ------------------------------------------------------------------ 01 // FEATURES (selection) */
const SEV = { high: K.red, medium: K.yellow, low: K.dim }

function RelevanceBar({ v }) {
  return <span className="inline-block h-[6px] align-middle" style={{ width: Math.max(2, v * 56), background: v > 0.35 ? K.green : v > 0.1 ? K.cyan : '#52525b' }} />
}

export function FeaturesTab({ id, ver, target, kinds, missingTotal, onApply, onUndo }) {
  const q = useApi(id && target ? `${sx(id)}/features/suggest?v=${ver}` : null)
  const [picked, setPicked] = useState([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [inplace, setInplace] = useState({ op: 'log', column: '' })
  const cols = Object.keys(kinds || {}).filter((c) => c !== target)
  const numeric = cols.filter((c) => kinds[c] === 'numeric')
  useEffect(() => { setPicked((p) => p.filter((c) => cols.includes(c))) }, [Object.keys(kinds || {}).join(',')])

  const run = async (ops, label) => {
    setBusy(true)
    setMsg(null)
    const r = await onApply(ops)
    setBusy(false)
    if (r.ok) {
      const steps = r.result.steps.map((st) => st.summary).join('; ')
      setMsg({ ok: true, text: `${label ? `${label}: ` : ''}${steps} // ROWS ${fInt(r.result.rows_before)} > ${fInt(r.result.rows_after)} // COLUMNS ${r.result.columns_before} > ${r.result.columns_after}${r.result.missing_total > 0 ? ` // ${fInt(r.result.missing_total)} MISSING VALUES NOW: CLEAN THEM IN STATUS BEFORE TRAINING` : ''}` })
      setPicked([])
    } else setMsg({ ok: false, text: r.error })
  }
  if (!id) return <Msg>NO DATASET // UPLOAD A FILE TO BEGIN</Msg>
  if (!target) return <Msg testid="features-needs-target">SET A TARGET FIRST // FEATURE RELEVANCE IS MEASURED AGAINST IT</Msg>
  const d = q.data
  const toggle = (c) => setPicked((p) => (p.includes(c) ? p.filter((x) => x !== c) : [...p, c]))
  const weak = d ? d.features.filter((f) => !f.constant && !f.id_like && f.mi_rel < 0.03 && f.rank > 3).map((f) => f.name) : []
  const auto = d ? [...d.auto_excluded.id_like, ...d.auto_excluded.constant] : []
  return (
    <div data-testid="features">
      {msg && <div data-testid="features-msg" className="mb-1 px-1 py-[2px] text-[10px] font-bold uppercase break-words" style={{ background: msg.ok ? K.green : K.red, color: msg.ok ? '#000' : '#fff' }}>{msg.text}</div>}
      <Sub>FEATURE SELECTION // HOW MUCH EACH COLUMN TELLS US ABOUT [{target}]</Sub>
      {q.error ? <Msg color={K.red}>{q.error}</Msg> : !d ? <Msg>{q.loading ? 'SCORING FEATURES...' : 'NO DATA'}</Msg> : (
        <>
          <div className="flex flex-wrap items-center gap-1 mb-1">
            <Btn testid="feat-drop" bg={K.red} fg="#fff" disabled={!picked.length || busy} onClick={() => run([{ op: 'drop_columns', columns: picked }], 'DROPPED')}>DROP SELECTED ({picked.length})</Btn>
            <Btn testid="feat-pick-weak" disabled={!weak.length || busy} onClick={() => setPicked(weak)}>SELECT WEAK ({weak.length})</Btn>
            <Btn disabled={!picked.length} onClick={() => setPicked([])}>CLEAR</Btn>
            <Btn testid="feat-undo" disabled={busy} onClick={async () => { setBusy(true); setMsg(null); await onUndo(); setBusy(false); setMsg({ ok: true, text: 'RESTORED THE RAW FILE (ALL FEATURE CHANGES AND IMPUTATION UNDONE)' }) }}>UNDO ALL</Btn>
          </div>
          <div className="overflow-auto border border-zinc-800 max-h-56 mb-1">
            <table className="w-full text-[10px] uppercase" data-testid="features-table">
              <thead className="bg-zinc-950 text-zinc-500">
                <tr><th className="w-5" /><th className="text-left font-normal px-1">#</th><th className="text-left font-normal px-1">FEATURE</th><th className="text-left font-normal px-1">RELEVANCE</th><th className="text-right font-normal px-1">MISS</th><th className="text-left font-normal px-1">NOTES</th></tr>
              </thead>
              <tbody>
                {d.features.map((f) => (
                  <tr key={f.name} data-testid={`feat-${f.name}`} className={`border-t border-zinc-800 ${picked.includes(f.name) ? 'bg-zinc-800' : 'hover:bg-zinc-900'}`}>
                    <td className="px-1"><input type="checkbox" data-testid={`feat-pick-${f.name}`} checked={picked.includes(f.name)} onChange={() => toggle(f.name)} /></td>
                    <td className="px-1 text-zinc-500">{f.rank ?? '-'}</td>
                    <td className="px-1 text-white max-w-[120px] truncate" title={`${f.name} (${f.kind}, ${f.unique} unique)`}>{f.name} <span className="text-[8px] text-zinc-500">{f.kind.slice(0, 3)}</span></td>
                    <td className="px-1 whitespace-nowrap">{f.id_like || f.constant ? <span className="text-zinc-600">NOT SCORED</span> : <><RelevanceBar v={f.mi_rel} /> <span className="text-zinc-400">{f.mi_rel.toFixed(2)}</span></>}</td>
                    <td className="px-1 text-right" style={f.missing_pct > 0 ? { color: K.yellow } : { color: '#71717a' }}>{f.missing_pct > 0 ? `${f.missing_pct}%` : '0'}</td>
                    <td className="px-1 whitespace-nowrap">
                      {f.id_like && <span className="px-[2px] mr-[2px] text-[8px] font-bold" style={{ background: '#52525b', color: '#fff' }}>ID</span>}
                      {f.constant && <span className="px-[2px] mr-[2px] text-[8px] font-bold" style={{ background: '#52525b', color: '#fff' }}>CONST</span>}
                      {f.skew !== null && Math.abs(f.skew) > 1.5 && <span className="px-[2px] mr-[2px] text-[8px] font-bold" style={{ background: K.yellow, color: '#000' }} title={`skew ${f.skew}`}>SKEW</span>}
                      {f.outlier_pct !== null && f.outlier_pct > 5 && <span className="px-[2px] mr-[2px] text-[8px] font-bold" style={{ background: K.yellow, color: '#000' }} title={`${f.outlier_pct}% outliers`}>OUT</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {auto.length > 0 && <div className="mb-1 text-[10px] uppercase text-zinc-500">AUTO-EXCLUDED FROM TRAINING: {auto.join(', ')} (ID-LIKE OR CONSTANT)</div>}
          <Sub>SUGGESTIONS // EACH ONE SHOWS THE CHANGE IT WOULD MAKE</Sub>
          {d.suggestions.length === 0 ? <div className="mb-1 text-[10px] uppercase text-zinc-500">NOTHING TO SUGGEST: NO REDUNDANT, MOSTLY-EMPTY OR SKEWED COLUMNS FOUND.</div> : (
            <div className="mb-1" data-testid="features-suggestions">
              <div className="mb-1"><Btn testid="feat-apply-main" disabled={busy || !d.suggestions.some((x) => x.severity !== 'low')} onClick={() => run(d.suggestions.filter((x) => x.severity !== 'low').map((x) => x.op), 'APPLIED')}>APPLY ALL HIGH + MEDIUM ({d.suggestions.filter((x) => x.severity !== 'low').length})</Btn></div>
              {d.suggestions.map((sg) => (
                <div key={sg.id} data-testid={`sugg-${sg.id}`} className="flex items-start gap-1 mb-[3px] border-l-2 pl-1" style={{ borderLeftColor: SEV[sg.severity] }}>
                  <div className="flex-1 min-w-0 text-[10px] leading-snug">
                    <span className="font-bold uppercase" style={{ color: SEV[sg.severity] }}>{sg.action} </span>
                    <span className="text-white break-words">{sg.columns.join(', ')}</span>
                    <div className="normal-case text-zinc-400">{sg.reason}</div>
                  </div>
                  <Btn disabled={busy} onClick={() => run([sg.op], sg.action.toUpperCase())}>APPLY</Btn>
                </div>
              ))}
            </div>
          )}
          {d.redundant_pairs.length > 0 && <div className="mb-1 text-[10px] uppercase text-zinc-500">REDUNDANT PAIRS: {d.redundant_pairs.map((p) => `${p.a} / ${p.b} (${p.r > 0 ? '+' : ''}${p.r})`).join(', ')}</div>}
        </>
      )}
      <Sub>TRANSFORM IN PLACE</Sub>
      <div className="flex flex-wrap items-center gap-1 mb-1" data-testid="inplace-form">
        <select data-testid="inplace-op" className={ctl} value={inplace.op} onChange={(e) => setInplace((p) => ({ ...p, op: e.target.value }))}>
          <option value="log">LOG1P (FOR SKEWED, NON-NEGATIVE)</option>
          <option value="clip">CLIP OUTLIERS (IQR x 3)</option>
        </select>
        <select data-testid="inplace-col" className={`${ctl} max-w-[120px]`} value={numeric.includes(inplace.column) ? inplace.column : numeric[0] || ''} onChange={(e) => setInplace((p) => ({ ...p, column: e.target.value }))}>{numeric.map((c) => <option key={c} value={c}>{c}</option>)}</select>
        <Btn testid="inplace-apply" disabled={busy || !numeric.length} onClick={() => {
          const col = numeric.includes(inplace.column) ? inplace.column : numeric[0]
          run([inplace.op === 'log' ? { op: 'log_transform', columns: [col] } : { op: 'outliers', columns: [col], method: 'iqr', factor: 3.0, action: 'clip' }], inplace.op === 'log' ? 'LOG' : 'CLIP')
        }}>APPLY</Btn>
      </div>
      {missingTotal > 0 && <div className="text-[10px] uppercase" style={{ color: K.yellow }}>{fInt(missingTotal)} MISSING VALUES IN THE DATA // CLEAN THEM IN THE STATUS TAB BEFORE TRAINING</div>}
    </div>
  )
}

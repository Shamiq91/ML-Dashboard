/**
 * Panel 05 // EXPORT: the selected model's card and a download of the model as a zip
 * (model.joblib + predict.py + MODEL_CARD.md + requirements.txt + example_input.csv).
 * Every number shown is the backend's model card.
 */
import React from 'react'
import { API_BASE_URL } from '../lib/api'
import { useApi } from '../lib/hooks'
import { f3, fInt, isNum } from '../lib/fmt'
import { K, Msg } from './analysis'

const Sub = ({ children }) => <div className="text-[10px] uppercase text-zinc-400 mt-2 mb-[2px]">{children}</div>
const Note = ({ children }) => <div className="text-[10px] normal-case text-zinc-500 leading-snug">{children}</div>
const METRIC_LABEL = {
  accuracy: 'ACC', balanced_accuracy: 'BAL ACC', precision: 'PREC', recall: 'REC', f1: 'F1', roc_auc: 'AUC', avg_precision: 'AP',
  r2: 'R2', rmse: 'RMSE', mae: 'MAE',
}
const FILES = [
  ['model.joblib', 'the fitted pipeline (preprocessing + model) and its input schema'],
  ['predict.py', 'runs the model on new rows: python predict.py model.joblib new.csv out.csv'],
  ['MODEL_CARD.md', 'what the model is, how it scored, what it expects, its limits'],
  ['requirements.txt', 'the library versions it was trained with'],
  ['example_input.csv', 'the first rows of your file, limited to the columns the model reads'],
]

function step(s) {
  const cols = (s.columns || [s.column]).filter(Boolean).join(', ')
  if (s.op === 'rename') return `rename ${Object.entries(s.mapping).map(([a, b]) => `${a} -> ${b}`).join(', ')}`
  if (s.op === 'log_transform') return `log1p ${cols}`
  if (s.op === 'clip') return `clip ${cols} to [${isNum(s.lower) ? f3(s.lower) : '-inf'}, ${isNum(s.upper) ? f3(s.upper) : 'inf'}]`
  if (s.op === 'cast') return `convert ${cols} to ${s.dtype}`
  if (s.op === 'strip_whitespace') return `trim whitespace in ${cols}`
  if (s.op === 'fill_constant') return `fill missing ${cols} with ${JSON.stringify(s.value)}`
  return s.op
}

function Card({ d, href }) {
  const reg = d.task === 'regression'
  return (
    <div data-testid="export-card" className="text-[10px] uppercase text-white">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <span className="font-bold">{d.name} ({d.abbr})</span>
        {d.flags.map((f) => (
          <span key={f} className="px-[3px] text-[9px] font-bold" style={{ background: f === 'TUNED' ? K.purple : f === 'OVERFIT' ? K.red : K.yellow, color: f === 'UNDERFIT' ? '#000' : '#fff' }}>{f}</span>
        ))}
        <a data-testid="export-download" href={href} download
          className="ml-auto h-6 px-2 inline-flex items-center text-[10px] font-bold uppercase border border-white bg-white text-black hover:bg-zinc-200">
          DOWNLOAD .ZIP
        </a>
      </div>
      <div className="text-zinc-400 normal-case">
        {reg ? 'Regression' : `Classification (${d.classes.join(', ')})`} of <span className="text-white">{d.target}</span> // {fInt(d.rows_trained)} rows trained, {fInt(d.rows_tested)} tested
      </div>

      <Sub>TEST SCORES</Sub>
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {Object.entries(d.metrics).map(([k, v]) => (
          <span key={k}><span className="text-zinc-500">{METRIC_LABEL[k] || k} </span><span className="font-bold">{f3(v)}</span></span>
        ))}
        {d.cv && <span><span className="text-zinc-500">CV </span><span className="font-bold">{f3(d.cv.score)} +/- {f3(d.cv.std)}</span></span>}
      </div>

      <Sub>INPUT // {d.input.columns.length} COLUMNS FROM YOUR ORIGINAL FILE</Sub>
      <div data-testid="export-columns" className="normal-case text-zinc-300 break-words">{d.input.columns.join(', ')}</div>
      {d.input.recipe.length > 0 && (
        <>
          <Note>predict.py first repeats these steps, as they ran before training:</Note>
          <ol className="list-decimal pl-5 normal-case text-zinc-300">{d.input.recipe.map((s, i) => <li key={i}>{step(s)}</li>)}</ol>
        </>
      )}
      <Note>Missing cells are filled with the training median (numbers) or most common value (text). Extra columns are kept in the output.</Note>

      <Sub>IN THE ZIP</Sub>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 normal-case">
        {FILES.map(([f, what]) => (
          <React.Fragment key={f}><span className="text-white font-mono">{f}</span><span className="text-zinc-500">{what}</span></React.Fragment>
        ))}
      </div>
      <Note>Needs: {d.requirements.join(', ')}</Note>

      {d.limitations.length > 0 && (
        <>
          <Sub>LIMITATIONS</Sub>
          <ul data-testid="export-limits" className="list-disc pl-5 normal-case" style={{ color: K.yellow }}>
            {d.limitations.map((x) => <li key={x}>{x}</li>)}
          </ul>
        </>
      )}
    </div>
  )
}

export function ExportTab({ base, mkey, abbr }) {
  const q = useApi(base && mkey ? `${base}/export/${mkey}/card` : null)
  if (!mkey) return <Msg>SELECT A TRAINED MODEL IN THE LEADERBOARD{abbr ? ` (${abbr} FAILED)` : ''}</Msg>
  if (q.error) return <Msg color={K.red}>{q.error}</Msg>
  if (!q.data) return <Msg>{q.loading ? 'LOADING...' : 'NO DATA'}</Msg>
  return <Card d={q.data} href={`${API_BASE_URL}${base}/export/${mkey}`} />
}

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
export const f3 = (v) => (isNum(v) ? v.toFixed(3) : '—')
export const fInt = (v) => (isNum(v) ? v.toLocaleString('en-US') : '—')
export const fPct = (v, d = 1) => (isNum(v) ? `${(v * 100).toFixed(d)}%` : '—')
export const fSig = (v) => {
  if (!isNum(v)) return '—'
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 })
  if (a >= 1) return String(Number(v.toFixed(2)))
  if (a >= 0.01) return String(Number(v.toFixed(3)))
  return v.toExponential(1)
}
export const fSigned = (v, d = 3) => (isNum(v) ? `${v > 0 ? '+' : ''}${v.toFixed(d)}` : '—')
export const fCell = (v) => (v === null || v === undefined ? '' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)))) : String(v))
export const fDate = (iso) => {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
export const downsample = (arr, n) => {
  if (arr.length <= n) return arr
  const step = (arr.length - 1) / (n - 1)
  return Array.from({ length: n }, (_, i) => arr[Math.round(i * step)])
}

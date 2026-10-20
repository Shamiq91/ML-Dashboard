const env = typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env : {}
const origin = typeof window !== 'undefined' && window.location ? window.location : null
/** Dev server (5173) talks to the API on 8000; when the API serves the build, same origin. */
export const API_BASE_URL =
  env.VITE_API_BASE_URL || (origin && origin.port && origin.port !== '5173' && origin.protocol.startsWith('http') ? origin.origin : 'http://localhost:8000')
export const WS_BASE_URL = API_BASE_URL.replace(/^http/i, 'ws')

export class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

function describe(body, status) {
  const base = body && typeof body.message === 'string' && body.message ? body.message : `Request failed (HTTP ${status})`
  if (body && Array.isArray(body.details) && body.error_code === 'VALIDATION_ERROR') {
    const extra = body.details.slice(0, 2).map((d) => `${(d.loc || []).slice(-1)[0] || ''} ${d.msg || ''}`.trim()).join('; ')
    return extra ? `${base} ${extra}` : base
  }
  return base
}

async function request(path, { method = 'GET', json, form, signal } = {}) {
  let res
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      signal,
      headers: json !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: json !== undefined ? JSON.stringify(json) : form,
    })
  } catch (e) {
    if (e && e.name === 'AbortError') throw e
    throw new ApiError(0, 'NETWORK_ERROR', `Cannot reach the API at ${API_BASE_URL}`)
  }
  const text = await res.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  if (!res.ok) throw new ApiError(res.status, (body && body.error_code) || `HTTP_${res.status}`, describe(body, res.status), body && body.details)
  return body
}

export const api = {
  get: (path, signal) => request(path, { signal }),
  post: (path, json) => request(path, { method: 'POST', json: json === undefined ? {} : json }),
  patch: (path, json) => request(path, { method: 'PATCH', json }),
  del: (path) => request(path, { method: 'DELETE' }),
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/api/v1/sessions/upload', { method: 'POST', form })
  },
}
export const sx = (id) => `/api/v1/sessions/${id}`
export const messageOf = (e) => (e instanceof ApiError ? e.message : e && e.message ? e.message : 'Unexpected error')
export const isExpired = (e) => e instanceof ApiError && e.status === 404 && e.code === 'SESSION_NOT_FOUND'

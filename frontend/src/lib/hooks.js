import { useCallback, useEffect, useRef, useState } from 'react'
import { api, messageOf } from './api'

/** GET ``path`` whenever it changes (null disables). Aborts stale requests; keeps previous data while reloading. */
export function useApi(path, { enabled = true } = {}) {
  const [st, setSt] = useState({ data: null, error: null, loading: false, path: null })
  const [tick, setTick] = useState(0)
  const reload = useCallback(() => setTick((t) => t + 1), [])
  const pathRef = useRef(null)
  useEffect(() => {
    if (!path || !enabled) {
      setSt({ data: null, error: null, loading: false, path: null })
      return undefined
    }
    const ctl = new AbortController()
    const same = pathRef.current === path
    pathRef.current = path
    setSt((s) => ({ data: same ? s.data : null, error: null, loading: true, path }))
    api.get(path, ctl.signal)
      .then((data) => setSt({ data, error: null, loading: false, path }))
      .catch((e) => {
        if (e && e.name === 'AbortError') return
        setSt({ data: null, error: messageOf(e), loading: false, path })
      })
    return () => ctl.abort()
  }, [path, enabled, tick])
  return { ...st, reload }
}

/** Debounce a fast-changing value (slider -> request). */
export function useDebounced(value, ms = 250) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

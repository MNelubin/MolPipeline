import { useCallback, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export function useResearchSearch() {
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const searchResearch = useCallback(async (query, mode = 'literature') => {
    setStatus('running')
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/research/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          mode,
          max_sources: 8,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      setResult(data)
      setStatus(data.status === 'empty' ? 'empty' : 'done')
      return data
    } catch (e) {
      setError(e.message)
      setStatus('error')
      throw e
    }
  }, [])

  const reset = useCallback(() => {
    setStatus('idle')
    setResult(null)
    setError(null)
  }, [])

  return {
    status,
    result,
    error,
    searchResearch,
    reset,
  }
}

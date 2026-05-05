import { useCallback, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export function useAdmetAnalysis() {
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const analyzeAdmet = useCallback(async (query) => {
    setStatus('running')
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/admet/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      setResult(data)
      setStatus('done')
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
    analyzeAdmet,
    reset,
  }
}

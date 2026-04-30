import { useCallback, useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

const FALLBACK_SOURCE_MODES = [
  { id: 'auto', label: 'Авто', description: 'Стандартный режим как в основном UI', enabled: true },
  { id: 'ord', label: 'ORD', description: 'Только Open Reaction Database', enabled: true },
  { id: 'retro_model', label: 'ASKCOS-derived model', description: 'Только локальная template-модель', enabled: true },
  { id: 'web', label: 'Web Search', description: 'Только веб-поиск ретросинтеза', enabled: true },
  { id: 'aizynthfinder', label: 'AiZynthFinder', description: 'Только внешний multi-step planner', enabled: true },
  { id: 'all', label: 'Все источники', description: 'Явный additive-пул всех источников', enabled: true },
]

export function useRetrosynthesisSearch() {
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [sourceModes, setSourceModes] = useState(FALLBACK_SOURCE_MODES)
  const [sourcesSnapshot, setSourcesSnapshot] = useState(null)

  const loadSources = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/retro/sources`)
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      setSourceModes(data.source_modes || FALLBACK_SOURCE_MODES)
      setSourcesSnapshot(data)
    } catch (e) {
      setSourceModes(FALLBACK_SOURCE_MODES)
      setSourcesSnapshot(null)
      setError(e.message)
    }
  }, [])

  useEffect(() => {
    loadSources()
  }, [loadSources])

  const searchRetrosynthesis = useCallback(async (query, sourceMode, model = 'openai/gpt-4o') => {
    setStatus('running')
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/retro/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          source_mode: sourceMode,
          model,
          top_n: 5,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      setResult(data)
      setStatus(data.status === 'blocked' ? 'blocked' : 'done')
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
    sourceModes,
    sourcesSnapshot,
    loadSources,
    searchRetrosynthesis,
    reset,
  }
}

import { useState, useCallback, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export function useInteractivePipeline() {
  const [status, setStatus] = useState('idle')   // idle | running | paused | done | error
  const [phase, setPhase] = useState(null)        // card_ready | select_pathway | completed
  const [pipelineState, setPipelineState] = useState(null)
  const [error, setError] = useState(null)
  const threadIdRef = useRef(null)

  const reset = useCallback(() => {
    setStatus('idle')
    setPhase(null)
    setPipelineState(null)
    setError(null)
    threadIdRef.current = null
  }, [])

  const startAnalysis = useCallback(async (query, model = 'openai/gpt-4o') => {
    reset()
    setStatus('running')
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode: 'interactive' }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      threadIdRef.current = data.thread_id
      setPipelineState(data.state)
      setPhase(data.phase)
      setStatus(data.phase === 'completed' ? 'done' : 'paused')
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }, [reset])

  const resume = useCallback(async (resumeData) => {
    if (!threadIdRef.current) return
    setStatus('running')
    try {
      const res = await fetch(`${API_BASE}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadIdRef.current, resume_data: resumeData }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      setPipelineState(data.state)
      setPhase(data.phase)
      setStatus(data.phase === 'completed' ? 'done' : 'paused')
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }, [])

  const confirmSynthesis = useCallback(() => resume(true), [resume])

  const selectPathway = useCallback((pathwayIndex, targetMassG = 1.0) => {
    return resume({
      selected_pathway: pathwayIndex,
      target_amount: { value: targetMassG, unit: 'g', amount_type: 'product_mass' },
    })
  }, [resume])

  return {
    status,
    phase,
    pipelineState,
    error,
    threadId: threadIdRef.current,
    startAnalysis,
    confirmSynthesis,
    selectPathway,
    reset,
  }
}

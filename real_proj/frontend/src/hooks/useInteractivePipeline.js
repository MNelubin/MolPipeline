import { useState, useCallback, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export function useInteractivePipeline() {
  const [status, setStatus] = useState('idle')   // idle | running | paused | done | error
  const [phase, setPhase] = useState(null)        // card_ready | select_pathway | completed
  const [pipelineState, setPipelineState] = useState(null)
  const [error, setError] = useState(null)
  const [threadId, setThreadId] = useState(null)
  const threadIdRef = useRef(null)

  const reset = useCallback(() => {
    setStatus('idle')
    setPhase(null)
    setPipelineState(null)
    setError(null)
    setThreadId(null)
    threadIdRef.current = null
  }, [])

  const restore = useCallback((saved) => {
    setPipelineState(saved.pipelineState)
    setPhase(saved.phase)
    setThreadId(saved.threadId)
    threadIdRef.current = saved.threadId
    setError(saved.error || null)
    setStatus(saved.phase === 'completed' ? 'done' : saved.error ? 'error' : 'paused')
  }, [])

  const _applyResponse = useCallback((data) => {
    setPipelineState(data.state)
    setPhase(data.phase)
    // API returns status: "ok" | "pending" | "banned" | "invalid" | "error"
    const apiStatus = data.status
    if (apiStatus && apiStatus !== 'ok' && apiStatus !== 'pending') {
      // pipeline stopped with an error/ban — show the output as error message
      setError(data.output || data.state?.error || 'Ошибка пайплайна')
      setStatus('error')
    } else if (data.phase === 'completed') {
      setStatus('done')
    } else if (data.phase && data.phase !== 'unknown') {
      setStatus('paused')
    } else if (data.state?.error) {
      // phase is unknown but error present
      setError(data.state.error)
      setStatus('error')
    } else {
      setStatus('paused')
    }
  }, [])

  const startAnalysis = useCallback(async (query, model = 'openai/gpt-4o') => {
    reset()
    setStatus('running')
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode: 'interactive', model }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      threadIdRef.current = data.thread_id
      setThreadId(data.thread_id)
      _applyResponse(data)
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }, [reset, _applyResponse])

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
      _applyResponse(data)
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }, [_applyResponse])

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
    threadId,
    startAnalysis,
    confirmSynthesis,
    selectPathway,
    reset,
    restore,
  }
}

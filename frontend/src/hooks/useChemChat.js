import { useCallback, useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export function useChemChat() {
  const [status, setStatus] = useState('idle')
  const [messages, setMessages] = useState([])
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [error, setError] = useState(null)

  const normalizeStoredMessage = item => {
    const payload = item.payload || {}
    return {
      id: `stored-${item.id}`,
      role: item.role,
      content: item.content || '',
      result: payload.result,
      progress: payload.progress || [],
      error: Boolean(payload.error),
      streaming: false,
      ts: item.created_at ? Date.parse(item.created_at) : Date.now(),
    }
  }

  const refreshSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/chat/sessions?limit=50`)
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      setSessions(data.sessions || [])
      return data.sessions || []
    } catch (e) {
      console.warn('Failed to load ChemChat sessions', e)
      return []
    }
  }, [])

  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])

  const loadSession = useCallback(async sessionId => {
    if (!sessionId || status === 'running') return null
    const res = await fetch(`${API_BASE}/chat/sessions/${encodeURIComponent(sessionId)}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
    const data = await res.json()
    setActiveSessionId(data.id)
    setMessages((data.messages || []).map(normalizeStoredMessage))
    setStatus('done')
    setError(null)
    return data
  }, [status])

  const startNewSession = useCallback(() => {
    if (status === 'running') return
    setActiveSessionId(null)
    setMessages([])
    setStatus('idle')
    setError(null)
  }, [status])

  const deleteSession = useCallback(async sessionId => {
    if (!sessionId || status === 'running') return
    const res = await fetch(`${API_BASE}/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
    })
    if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
    if (activeSessionId === sessionId) {
      setActiveSessionId(null)
      setMessages([])
      setStatus('idle')
    }
    await refreshSessions()
  }, [activeSessionId, refreshSessions, status])

  const sendMessage = useCallback(async (message, options = {}) => {
    const text = message.trim()
    if (!text) return null

    const userMessage = { role: 'user', content: text, ts: Date.now() }
    const history = messages
      .filter(item => item.content && !item.streaming && (item.role === 'user' || item.role === 'assistant'))
      .slice(-8)
      .map(item => ({ role: item.role, content: item.content }))
    const assistantId = `assistant-${Date.now()}`
    const assistantDraft = {
      id: assistantId,
      role: 'assistant',
      content: '',
      progress: [],
      streaming: true,
      ts: Date.now() + 1,
    }
    setMessages(prev => [...prev, userMessage, assistantDraft])
    setStatus('running')
    setError(null)

    const updateAssistant = updater => {
      setMessages(prev => prev.map(item => {
        if (item.id !== assistantId) return item
        return typeof updater === 'function' ? updater(item) : { ...item, ...updater }
      }))
    }

    const applyEvent = event => {
      if (!event) return
      if (event.type === 'final') {
        const data = event.result
        if (data?.session_id) setActiveSessionId(data.session_id)
        updateAssistant({
          content: data?.answer || 'Результат получен, но текстовый ответ пуст.',
          result: data,
          streaming: false,
        })
        setStatus('done')
        return
      }
      if (event.type === 'error') {
        updateAssistant({
          content: `Ошибка ChemChat: ${event.message || 'неизвестная ошибка'}`,
          error: true,
          streaming: false,
        })
        setError(event.message || 'ChemChat stream failed')
        setStatus('error')
        return
      }
      if (event.session_id) setActiveSessionId(event.session_id)
      updateAssistant(item => ({
        ...item,
        progress: [...(item.progress || []), event],
      }))
    }

    const parseSseBlock = block => {
      const dataLines = block
        .split('\n')
        .filter(line => line.startsWith('data:'))
        .map(line => line.slice(5).trimStart())
      if (dataLines.length === 0) return null
      return JSON.parse(dataLines.join('\n'))
    }

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: options.sessionId || activeSessionId,
          message: text,
          history,
          source_mode: options.sourceMode || 'auto',
          top_n: options.topN || 5,
          research_mode: options.researchMode || 'literature',
          max_sources: options.maxSources || 6,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)

      if (!res.body) {
        throw new Error('Streaming response body is not available')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalResult = null

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''
        for (const block of blocks) {
          if (!block.trim()) continue
          const event = parseSseBlock(block)
          if (event?.type === 'final') finalResult = event.result
          if (event?.type === 'error') {
            applyEvent(event)
            throw new Error(event.message || 'ChemChat stream failed')
          }
          applyEvent(event)
        }
      }

      if (buffer.trim()) {
        const event = parseSseBlock(buffer)
        if (event?.type === 'final') finalResult = event.result
        if (event?.type === 'error') {
          applyEvent(event)
          throw new Error(event.message || 'ChemChat stream failed')
        }
        applyEvent(event)
      }

      updateAssistant(item => ({ ...item, streaming: false }))
      setStatus(finalResult ? 'done' : 'done')
      await refreshSessions()
      return finalResult
    } catch (e) {
      setError(e.message)
      updateAssistant({
        content: `Ошибка ChemChat: ${e.message}`,
        error: true,
        streaming: false,
      })
      setStatus('error')
      throw e
    }
  }, [activeSessionId, messages, refreshSessions])

  const reset = useCallback(() => {
    startNewSession()
  }, [startNewSession])

  return {
    status,
    messages,
    sessions,
    activeSessionId,
    error,
    sendMessage,
    loadSession,
    startNewSession,
    deleteSession,
    refreshSessions,
    reset,
  }
}

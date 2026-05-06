import { useCallback, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export function useChemChat() {
  const [status, setStatus] = useState('idle')
  const [messages, setMessages] = useState([])
  const [error, setError] = useState(null)

  const sendMessage = useCallback(async (message, options = {}) => {
    const text = message.trim()
    if (!text) return null

    const userMessage = { role: 'user', content: text, ts: Date.now() }
    setMessages(prev => [...prev, userMessage])
    setStatus('running')
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          source_mode: options.sourceMode || 'auto',
          top_n: options.topN || 5,
          research_mode: options.researchMode || 'literature',
          max_sources: options.maxSources || 6,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      const data = await res.json()
      const assistantMessage = {
        role: 'assistant',
        content: data.answer || 'Результат получен, но текстовый ответ пуст.',
        result: data,
        ts: Date.now(),
      }
      setMessages(prev => [...prev, assistantMessage])
      setStatus('done')
      return data
    } catch (e) {
      setError(e.message)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Ошибка ChemChat: ${e.message}`,
        error: true,
        ts: Date.now(),
      }])
      setStatus('error')
      throw e
    }
  }, [])

  const reset = useCallback(() => {
    setStatus('idle')
    setMessages([])
    setError(null)
  }, [])

  return {
    status,
    messages,
    error,
    sendMessage,
    reset,
  }
}

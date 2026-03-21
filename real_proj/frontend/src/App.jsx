import { useState, useRef, useEffect, useCallback } from 'react'
import ChatMessage from './components/ChatMessage'
import ModelSelector from './components/ModelSelector'
import CalculatorCard from './components/CalculatorCard'
import { useSSEPipeline } from './hooks/useSSEPipeline'

const EXAMPLES = ['aspirin', 'caffeine', 'CC(=O)Oc1ccccc1C(O)=O', 'dopamine', 'ethanol']

function createBotMessage(id) {
  return {
    id,
    role: 'bot',
    nodes: { validate: 'idle', guard: 'idle', molecule_info: 'idle', retrosynthesis: 'idle' },
    streamText: '',
    moleculeInfo: null,
    guardResult: null,
    retroResult: null,
    error: null,
    done: false,
  }
}

// ── Sidebar nav items ─────────────────────────────────────────────────────────
const NAV_ITEMS = [
  {
    id: 'chat',
    label: 'Анализ молекул',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <circle cx="7.5" cy="5.5" r="3.5" />
        <path d="M2 13.5c0-3 2.5-5 5.5-5s5.5 2 5.5 5" />
      </svg>
    ),
  },
  {
    id: 'calculator',
    label: 'Калькулятор',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="11" height="11" rx="2" />
        <path d="M5 5h1M9 5h1M5 7.5h1M9 7.5h1M5 10h1M9 10h1M7 10v0" />
        <path d="M7 5h1" strokeWidth="2" />
      </svg>
    ),
  },
]

export default function App() {
  const [page, setPage] = useState('chat')        // 'chat' | 'calculator'
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [model, setModel] = useState('openai/gpt-4o')
  const [history, setHistory] = useState([])

  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const { run, cancel, isStreaming } = useSSEPipeline()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const updateBotMessage = useCallback((id, updater) => {
    setMessages(prev =>
      prev.map(m => m.id === id
        ? (typeof updater === 'function' ? updater(m) : { ...m, ...updater })
        : m)
    )
  }, [])

  const handleSubmit = useCallback(async () => {
    const query = input.trim()
    if (!query || isStreaming) return

    setInput('')
    textareaRef.current?.focus()

    const userMsg = { id: `u-${Date.now()}`, role: 'user', query }
    const botId   = `b-${Date.now()}`
    setMessages(prev => [...prev, userMsg, createBotMessage(botId)])
    setHistory(prev => [query, ...prev.filter(h => h !== query)].slice(0, 20))

    await run(query, model, ({ type, data }) => {
      switch (type) {
        case 'node_start':
          updateBotMessage(botId, m => ({ ...m, nodes: { ...m.nodes, [data.node]: 'running' } }))
          break
        case 'node_complete':
          updateBotMessage(botId, m => {
            const nodes = { ...m.nodes, [data.node]: 'done' }
            const update = { nodes }
            if (data.node === 'validate') {
              const val = data.output?.validation || {}
              if (!val.is_valid && val.error) update.error = val.error
            }
            if (data.node === 'guard')          update.guardResult = data.output?.guard_result   || null
            if (data.node === 'molecule_info')  update.moleculeInfo = data.output?.molecule_info  || null
            if (data.node === 'retrosynthesis') update.retroResult  = data.output?.retro_result   || null
            return { ...m, ...update }
          })
          break
        case 'token':
          updateBotMessage(botId, m => ({ ...m, streamText: m.streamText + data.text }))
          break
        case 'pipeline_done':
          updateBotMessage(botId, m => ({
            ...m, done: true,
            nodes: Object.fromEntries(Object.entries(m.nodes).map(([k, v]) => [k, v === 'running' ? 'done' : v])),
          }))
          break
        case 'error':
          updateBotMessage(botId, m => ({
            ...m, done: true, error: data.message,
            nodes: Object.fromEntries(Object.entries(m.nodes).map(([k, v]) => [k, v === 'running' ? 'error' : v])),
          }))
          break
        default: break
      }
    })
  }, [input, isStreaming, model, run, updateBotMessage])

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() }
  }

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-mark">MolPipeline</div>
          <div className="logo-sub">Molecule Analysis</div>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`nav-item${page === item.id ? ' active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>

        {/* History — only visible on chat page */}
        {page === 'chat' && (
          <>
            <div className="sidebar-section-title">История</div>
            <div className="sidebar-history">
              {history.length === 0 ? (
                <div style={{ padding: '8px 10px', fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                  Нет запросов
                </div>
              ) : (
                history.map((q, i) => (
                  <div key={i} className="history-item" onClick={() => setInput(q)} title={q}>{q}</div>
                ))
              )}
            </div>
          </>
        )}

        <div className="sidebar-footer">
          <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', lineHeight: 1.8 }}>
            validate → guard<br />
            → molecule_info<br />
            → retrosynthesis
          </div>
        </div>
      </aside>

      {/* ── Main area ── */}
      <main className="main">

        {/* ════════════════ CHAT PAGE ════════════════ */}
        {page === 'chat' && (
          <>
            <div className="topbar">
              <span className="topbar-title">
                {isStreaming ? (
                  <span style={{ color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div className="spinner" style={{ width: 12, height: 12 }} />
                    Обработка...
                  </span>
                ) : 'Введите название или SMILES молекулы'}
              </span>
              <ModelSelector value={model} onChange={setModel} disabled={isStreaming} />
            </div>

            <div className="messages">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">⬡</div>
                  <div className="empty-title">MolPipeline</div>
                  <div className="empty-sub">Введите название или SMILES</div>
                  <div className="empty-examples">
                    {EXAMPLES.map(ex => (
                      <button key={ex} className="example-chip" onClick={() => setInput(ex)}>{ex}</button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map(msg => <ChatMessage key={msg.id} message={msg} />)
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
              <div className="input-row">
                <textarea
                  ref={textareaRef}
                  className="input-box"
                  rows={1}
                  placeholder="Аспирин, caffeine, CC(=O)O, ..."
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isStreaming}
                  style={{ height: 44 }}
                  onInput={e => {
                    e.target.style.height = '44px'
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                  }}
                />
                <button
                  className="send-btn"
                  onClick={isStreaming ? cancel : handleSubmit}
                  disabled={!input.trim() && !isStreaming}
                >
                  {isStreaming ? (
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                      <rect x="3" y="3" width="10" height="10" rx="2" />
                    </svg>
                  ) : (
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="8" y1="13" x2="8" y2="3" />
                      <polyline points="4 7 8 3 12 7" />
                    </svg>
                  )}
                </button>
              </div>
              <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                Enter — отправить · Shift+Enter — новая строка
              </div>
            </div>
          </>
        )}

        {/* ════════════════ CALCULATOR PAGE ════════════════ */}
        {page === 'calculator' && (
          <>
            <div className="topbar">
              <span className="topbar-title">Калькулятор реагентов</span>
            </div>
            <div className="calculator-page">
              <div className="calculator-page-inner">
                <div style={{ marginBottom: 20 }}>
                  <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-1)', marginBottom: 6 }}>
                    Калькулятор стехиометрии
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                    Рассчитайте массы и объёмы реагентов для любой реакции
                  </div>
                </div>
                <CalculatorCard smiles="" />
              </div>
            </div>
          </>
        )}

      </main>
    </div>
  )
}

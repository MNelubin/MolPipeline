import { useState, useRef, useEffect, useCallback } from 'react'
import ChatMessage from './components/ChatMessage'
import ModelSelector from './components/ModelSelector'
import CalculatorCard from './components/CalculatorCard'
import MoleculeCard from './components/MoleculeCard'
import PathwaySelector from './components/PathwaySelector'
import ExperimentProtocol from './components/ExperimentProtocol'
import ProtocolGraph from './components/ProtocolGraph'
import { useInteractivePipeline } from './hooks/useInteractivePipeline'

const EXAMPLES = ['aspirin', 'caffeine', 'CC(=O)Oc1ccccc1C(O)=O', 'dopamine', 'ethanol']

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
  const [page, setPage] = useState('chat')
  const [input, setInput] = useState('')
  const [model, setModel] = useState('openai/gpt-4o')
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('mol_sessions_index') || '[]')
    } catch { return [] }
  })

  const textareaRef = useRef(null)
  const currentQueryRef = useRef('')

  const {
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
  } = useInteractivePipeline()

  const isRunning = status === 'running'

  // Save session to localStorage whenever pipeline state updates
  useEffect(() => {
    if (!pipelineState || !threadId) return
    const query = currentQueryRef.current || pipelineState?.query || ''
    if (!query) return
    try {
      const sessionData = { pipelineState, phase, threadId, error, query, model, ts: Date.now() }
      localStorage.setItem('mol_session_' + threadId, JSON.stringify(sessionData))
      setHistory(prev => {
        const entry = { query, threadId, ts: Date.now() }
        const next = [entry, ...prev.filter(h => h.threadId !== threadId)].slice(0, 20)
        localStorage.setItem('mol_sessions_index', JSON.stringify(next))
        return next
      })
    } catch { /* localStorage full — ignore */ }
  }, [pipelineState, phase, threadId, error, model])

  const deleteSession = useCallback((entry) => {
    try { localStorage.removeItem('mol_session_' + entry.threadId) } catch {}
    setHistory(prev => {
      const next = prev.filter(h => h.threadId !== entry.threadId)
      try { localStorage.setItem('mol_sessions_index', JSON.stringify(next)) } catch {}
      return next
    })
    if (threadId === entry.threadId) reset()
  }, [threadId, reset])

  const restoreSession = useCallback((entry) => {
    try {
      const saved = JSON.parse(localStorage.getItem('mol_session_' + entry.threadId))
      if (saved) restore(saved)
    } catch { /* corrupt data — ignore */ }
  }, [restore])

  const handleSubmit = useCallback(async () => {
    const query = input.trim()
    if (!query || isRunning) return

    setInput('')
    textareaRef.current?.focus()
    currentQueryRef.current = query

    await startAnalysis(query, model)
  }, [input, isRunning, model, startAnalysis])

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() }
  }

  const moleculeInfo = pipelineState?.molecule_info || null
  const guardResult = pipelineState?.guard_result || null
  const synthesisPaths = pipelineState?.synthesis_pathways || []
  const experimentProtocol = pipelineState?.experiment_protocol || null

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-mark">MolPipeline</div>
          <div className="logo-sub">Molecule Analysis</div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`nav-item${page === item.id ? ' active' : ''}`}
              onClick={() => { setPage(item.id); if (item.id === 'chat' && status !== 'idle') reset() }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>

        {page === 'chat' && (
          <>
            <div className="sidebar-section-title">История</div>
            <div className="sidebar-history">
              {history.length === 0 ? (
                <div className="sidebar-empty">Нет запросов</div>
              ) : (
                history.map((entry, i) => (
                  <div key={entry.threadId || i} className="history-item" onClick={() => restoreSession(entry)} title={entry.query}>
                    <span className="history-item-text">{entry.query}</span>
                    <span className="history-item-delete" onClick={e => { e.stopPropagation(); deleteSession(entry) }}>×</span>
                  </div>
                ))
              )}
            </div>
          </>
        )}

        <div className="sidebar-footer">
          <div className="sidebar-footer-text">
            classify → validate<br />
            → molecule_info → retro<br />
            → stoichiometry → plan
          </div>
        </div>
      </aside>

      {/* Main area */}
      <main className="main">

        {/* CHAT PAGE */}
        {page === 'chat' && (
          <>
            <div className="topbar">
              <span className="topbar-title">
                {isRunning ? (
                  <span className="topbar-status">
                    <div className="spinner spinner-sm" />
                    Обработка...
                  </span>
                ) : 'Введите название или SMILES молекулы'}
              </span>
              <ModelSelector value={model} onChange={setModel} disabled={isRunning} />
            </div>

            <div className="messages">

              {/* Empty state */}
              {status === 'idle' && (
                <div className="empty-state">
                  <div className="empty-icon">
                    <svg width="52" height="52" viewBox="0 0 52 52" fill="none" stroke="currentColor" strokeWidth="1.2" opacity="0.5">
                      <polygon points="26,2 49,14.5 49,37.5 26,50 3,37.5 3,14.5" stroke="var(--cyan-dim)" />
                      <circle cx="26" cy="26" r="6" stroke="var(--cyan)" strokeWidth="1.5" />
                      <line x1="26" y1="2" x2="26" y2="20" stroke="var(--border-hi)" />
                      <line x1="49" y1="14.5" x2="32" y2="24" stroke="var(--border-hi)" />
                      <line x1="49" y1="37.5" x2="32" y2="28" stroke="var(--border-hi)" />
                      <line x1="26" y1="50" x2="26" y2="32" stroke="var(--border-hi)" />
                      <line x1="3" y1="37.5" x2="20" y2="28" stroke="var(--border-hi)" />
                      <line x1="3" y1="14.5" x2="20" y2="24" stroke="var(--border-hi)" />
                    </svg>
                  </div>
                  <div className="empty-title">MolPipeline</div>
                  <div className="empty-sub">Введите название или SMILES</div>
                  <div className="empty-examples">
                    {EXAMPLES.map(ex => (
                      <button key={ex} className="example-chip" onClick={() => setInput(ex)}>{ex}</button>
                    ))}
                  </div>
                </div>
              )}

              {/* Running spinner */}
              {isRunning && (
                <div className="loading-row">
                  <div className="spinner spinner-md" />
                  Выполняется анализ...
                </div>
              )}

              {/* Error state */}
              {status === 'error' && (
                <div className="error-block">
                  {error || 'Произошла ошибка'}
                  <button className="reset-link" onClick={reset}>
                    Сбросить
                  </button>
                </div>
              )}

              {/* Phase: card_ready — molecule card + confirm button */}
              {!isRunning && moleculeInfo && (
                <div style={{ marginBottom: 16 }}>
                  <MoleculeCard
                    moleculeInfo={moleculeInfo}
                    guardResult={guardResult}
                    retroResult={pipelineState?.retro_result || null}
                  />

                  {phase === 'card_ready' && (
                    <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
                      <button className="action-btn" onClick={confirmSynthesis} disabled={isRunning}>
                        Продолжить синтез
                      </button>
                      <button className="action-btn-ghost" onClick={reset}>
                        Сбросить
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Phase: select_pathway */}
              {!isRunning && phase === 'select_pathway' && synthesisPaths.length > 0 && (
                <PathwaySelector
                  pathways={synthesisPaths}
                  onSelect={selectPathway}
                />
              )}

              {/* Phase: completed — graph + protocol */}
              {!isRunning && phase === 'completed' && experimentProtocol && (
                <>
                  <ProtocolGraph protocol={experimentProtocol} />
                  <ExperimentProtocol protocol={experimentProtocol} moleculeInfo={moleculeInfo} sessionId={threadId} />
                </>
              )}

              {/* Completed but no protocol */}
              {!isRunning && phase === 'completed' && !experimentProtocol && pipelineState?.error && (
                <div className="error-block">
                  {pipelineState.error}
                </div>
              )}

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
                  disabled={isRunning}
                  style={{ height: 44 }}
                  onInput={e => {
                    e.target.style.height = '44px'
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                  }}
                />
                <button
                  className="send-btn"
                  onClick={handleSubmit}
                  disabled={!input.trim() || isRunning}
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="8" y1="13" x2="8" y2="3" />
                    <polyline points="4 7 8 3 12 7" />
                  </svg>
                </button>
              </div>
              <div className="input-hint">
                Enter — отправить · Shift+Enter — новая строка
              </div>
            </div>
          </>
        )}

        {/* CALCULATOR PAGE */}
        {page === 'calculator' && (
          <>
            <div className="topbar">
              <span className="topbar-title">Калькулятор реагентов</span>
            </div>
            <div className="calculator-page">
              <div className="calculator-page-inner">
                <div style={{ marginBottom: 20 }}>
                  <div className="calc-page-title">Калькулятор стехиометрии</div>
                  <div className="calc-page-desc">
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

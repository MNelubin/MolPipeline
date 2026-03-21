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
  const [input, setInput] = useState('')
  const [model, setModel] = useState('openai/gpt-4o')
  const [history, setHistory] = useState([])

  const textareaRef = useRef(null)

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
  } = useInteractivePipeline()

  const isRunning = status === 'running'

  const handleSubmit = useCallback(async () => {
    const query = input.trim()
    if (!query || isRunning) return

    setInput('')
    textareaRef.current?.focus()
    setHistory(prev => [query, ...prev.filter(h => h !== query)].slice(0, 20))

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
            classify → validate<br />
            → molecule_info → retro<br />
            → stoichiometry → plan
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
                {isRunning ? (
                  <span style={{ color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div className="spinner" style={{ width: 12, height: 12 }} />
                    Обработка...
                  </span>
                ) : 'Введите название или SMILES молекулы'}
              </span>
              <ModelSelector value={model} onChange={setModel} disabled={isRunning} />
            </div>

            <div className="messages">

              {/* ── Empty state ── */}
              {status === 'idle' && (
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
              )}

              {/* ── Running spinner ── */}
              {isRunning && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '24px 0', color: 'var(--text-2)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                  <div className="spinner" style={{ width: 18, height: 18 }} />
                  Выполняется анализ...
                </div>
              )}

              {/* ── Error state ── */}
              {status === 'error' && (
                <div style={{
                  margin: '16px 0',
                  padding: '12px 16px',
                  background: 'var(--red)12',
                  border: '1px solid var(--red)40',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--red)',
                  fontSize: 13,
                  fontFamily: 'var(--font-mono)',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}>
                  {error || 'Произошла ошибка'}
                  <button
                    style={{ display: 'block', marginTop: 10, fontSize: 11, cursor: 'pointer', color: 'var(--text-3)', background: 'none', border: 'none' }}
                    onClick={reset}
                  >
                    Сбросить
                  </button>
                </div>
              )}

              {/* ── Phase: card_ready — show molecule card + confirm button ── */}
              {!isRunning && (phase === 'card_ready' || phase === 'select_pathway' || phase === 'completed') && moleculeInfo && (
                <div style={{ marginBottom: 16 }}>
                  <MoleculeCard
                    moleculeInfo={moleculeInfo}
                    guardResult={guardResult}
                    retroResult={pipelineState?.retro_result || null}
                  />

                  {phase === 'card_ready' && (
                    <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
                      <button
                        className="send-btn"
                        style={{ padding: '10px 24px', fontSize: 13, width: 'auto', borderRadius: 'var(--r-md)' }}
                        onClick={confirmSynthesis}
                        disabled={isRunning}
                      >
                        Продолжить синтез
                      </button>
                      <button
                        style={{
                          padding: '10px 24px', fontSize: 13, borderRadius: 'var(--r-md)',
                          background: 'none', border: '1px solid var(--border)',
                          color: 'var(--text-2)', cursor: 'pointer',
                        }}
                        onClick={reset}
                      >
                        Сбросить
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* ── Phase: select_pathway — show pathway selector ── */}
              {!isRunning && phase === 'select_pathway' && synthesisPaths.length > 0 && (
                <PathwaySelector
                  pathways={synthesisPaths}
                  onSelect={selectPathway}
                />
              )}

              {/* ── Phase: completed — show synthesis graph + experiment protocol ── */}
              {!isRunning && phase === 'completed' && experimentProtocol && (
                <>
                  <ProtocolGraph protocol={experimentProtocol} />
                  <ExperimentProtocol protocol={experimentProtocol} moleculeInfo={moleculeInfo} sessionId={threadId} />
                </>
              )}

              {/* ── Completed but no protocol (error/banned) ── */}
              {!isRunning && phase === 'completed' && !experimentProtocol && pipelineState?.error && (
                <div style={{
                  padding: '12px 16px',
                  background: 'var(--red)12',
                  border: '1px solid var(--red)40',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--red)',
                  fontSize: 13,
                  fontFamily: 'var(--font-mono)',
                }}>
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

import { useState, useRef, useEffect, useCallback, Suspense, lazy } from 'react'
import ChatMessage from './components/ChatMessage'
import ModelSelector from './components/ModelSelector'
import CalculatorCard from './components/CalculatorCard'
import MoleculeCard from './components/MoleculeCard'
import PathwaySelector from './components/PathwaySelector'
import ExperimentProtocol from './components/ExperimentProtocol'
import ProtocolGraph from './components/ProtocolGraph'
import { useInteractivePipeline } from './hooks/useInteractivePipeline'
import { useRetrosynthesisSearch } from './hooks/useRetrosynthesisSearch'
import { useResearchSearch } from './hooks/useResearchSearch'
import { useAdmetAnalysis } from './hooks/useAdmetAnalysis'
import { useAvailabilityCheck } from './hooks/useAvailabilityCheck'

const EXAMPLES = ['aspirin', 'caffeine', 'CC(=O)Oc1ccccc1C(O)=O', 'dopamine', 'ethanol']
const MoleculeEditor = lazy(() => import('./components/MoleculeEditor'))

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
  {
    id: 'retrosynthesis',
    label: 'Ретросинтез',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 4.5h5" />
        <path d="M8 4.5 6.2 2.8" />
        <path d="M8 4.5 6.2 6.2" />
        <path d="M13 10.5H8" />
        <path d="M7 10.5 8.8 8.8" />
        <path d="M7 10.5 8.8 12.2" />
        <circle cx="3" cy="10.5" r="1.2" />
        <circle cx="12" cy="4.5" r="1.2" />
      </svg>
    ),
  },
  {
    id: 'research',
    label: 'Исследования',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 2.5h6.5L12 5v7.5H3z" />
        <path d="M9.5 2.5V5H12" />
        <path d="M5 7h5" />
        <path d="M5 9.5h4" />
      </svg>
    ),
  },
  {
    id: 'admet',
    label: 'ADMET',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5.5 2.5h4" />
        <path d="M6.5 2.5v3.2L3.4 11a1.7 1.7 0 0 0 1.5 2.5h5.2a1.7 1.7 0 0 0 1.5-2.5L8.5 5.7V2.5" />
        <path d="M5 9.5h5" />
      </svg>
    ),
  },
  {
    id: 'availability',
    label: 'Поставщики',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 4.5h7.5v6H2z" />
        <path d="M9.5 6h2l1.5 2v2.5H9.5" />
        <circle cx="4.5" cy="11.5" r="1" />
        <circle cx="11.5" cy="11.5" r="1" />
        <path d="M3.5 2.5h4" />
      </svg>
    ),
  },
]

const ADMET_SECTION_LABELS = {
  absorption: 'Всасывание',
  distribution: 'Распределение',
  metabolism: 'Метаболизм',
  excretion: 'Выведение',
  toxicity: 'Токсичность',
}

const ADMET_METHOD_LABELS = {
  rdkit_descriptor_heuristics_v1: 'RDKit-дескрипторы',
  rdkit_descriptor_heuristics_v2_with_safety_overlay: 'RDKit-дескрипторы + проверка безопасности',
}

const ADMET_DESCRIPTOR_LABELS = {
  molecular_weight: 'Молекулярная масса',
  logp: 'LogP',
  tpsa: 'TPSA',
  h_bond_donors: 'Доноры H-связей',
  h_bond_acceptors: 'Акцепторы H-связей',
  rotatable_bonds: 'Вращаемые связи',
  ring_count: 'Кольца',
  aromatic_rings: 'Ароматические кольца',
  heavy_atoms: 'Тяжелые атомы',
  formal_charge: 'Формальный заряд',
  lipinski_violations: 'Нарушения Lipinski',
  veber_violations: 'Нарушения Veber',
  solubility_band: 'Растворимость',
  permeability_band: 'Проницаемость',
}

const RISK_LEVEL_LABELS = {
  low: 'низкий',
  medium: 'средний',
  high: 'высокий',
}

const SAFETY_STATUS_LABELS = {
  SAFE: 'без критичных ограничений',
  WARNING: 'требуется внимание',
  CRITICAL_STOP: 'критический стоп',
  UNKNOWN: 'нет данных',
}

const SEVERITY_LABELS = {
  low: 'низкая',
  medium: 'средняя',
  high: 'высокая',
}

const AVAILABILITY_LEVEL_LABELS = {
  catalog: 'В каталоге',
  common_lab_reagent: 'Обычный реагент',
  heuristic_likely: 'Вероятно доступен',
  not_found: 'Не найден',
  invalid: 'Ошибка',
}

export default function App() {
  const [page, setPage] = useState('chat')
  const [input, setInput] = useState('')
  const [retroInput, setRetroInput] = useState('')
  const [retroSourceMode, setRetroSourceMode] = useState('auto')
  const [researchInput, setResearchInput] = useState('')
  const [researchMode, setResearchMode] = useState('literature')
  const [admetInput, setAdmetInput] = useState('')
  const [availabilityInput, setAvailabilityInput] = useState('')
  const [model, setModel] = useState('openai/gpt-4o')
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('mol_sessions_index') || '[]')
    } catch { return [] }
  })

  const textareaRef = useRef(null)
  const retroTextareaRef = useRef(null)
  const researchTextareaRef = useRef(null)
  const admetTextareaRef = useRef(null)
  const availabilityTextareaRef = useRef(null)
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
  const {
    status: retroStatus,
    result: retroAnalysis,
    error: retroError,
    sourceModes,
    searchRetrosynthesis,
    reset: resetRetro,
  } = useRetrosynthesisSearch()
  const {
    status: researchStatus,
    result: researchResult,
    error: researchError,
    searchResearch,
    reset: resetResearch,
  } = useResearchSearch()
  const {
    status: admetStatus,
    result: admetResult,
    error: admetError,
    analyzeAdmet,
    reset: resetAdmet,
  } = useAdmetAnalysis()
  const {
    status: availabilityStatus,
    result: availabilityResult,
    error: availabilityError,
    checkAvailability,
    reset: resetAvailability,
  } = useAvailabilityCheck()

  const isRunning = status === 'running'
  const isRetroRunning = retroStatus === 'running'
  const isResearchRunning = researchStatus === 'running'
  const isAdmetRunning = admetStatus === 'running'
  const isAvailabilityRunning = availabilityStatus === 'running'

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
      if (saved) {
        currentQueryRef.current = saved.query || entry.query || ''
        restore(saved)
      }
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

  const handleRetroSubmit = useCallback(async () => {
    const query = retroInput.trim()
    if (!query || isRetroRunning) return

    setRetroInput('')
    retroTextareaRef.current?.focus()
    await searchRetrosynthesis(query, retroSourceMode, model)
  }, [isRetroRunning, model, retroInput, retroSourceMode, searchRetrosynthesis])

  const handleEditorSmiles = useCallback((smiles) => {
    setRetroInput(smiles)
    retroTextareaRef.current?.focus()
  }, [])

  const handleEditorRetrosynthesis = useCallback(async (smiles) => {
    const query = smiles.trim()
    if (!query || isRetroRunning) return

    setRetroInput(query)
    await searchRetrosynthesis(query, retroSourceMode, model)
  }, [isRetroRunning, model, retroSourceMode, searchRetrosynthesis])

  const handleRetroKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRetroSubmit() }
  }

  const handleResearchSubmit = useCallback(async () => {
    const query = researchInput.trim()
    if (!query || isResearchRunning) return

    await searchResearch(query, researchMode)
    researchTextareaRef.current?.focus()
  }, [isResearchRunning, researchInput, researchMode, searchResearch])

  const handleResearchKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleResearchSubmit() }
  }

  const handleAdmetSubmit = useCallback(async () => {
    const query = admetInput.trim()
    if (!query || isAdmetRunning) return

    await analyzeAdmet(query)
    admetTextareaRef.current?.focus()
  }, [admetInput, analyzeAdmet, isAdmetRunning])

  const handleAdmetKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAdmetSubmit() }
  }

  const handleAvailabilitySubmit = useCallback(async () => {
    const query = availabilityInput.trim()
    if (!query || isAvailabilityRunning) return

    await checkAvailability(query)
    availabilityTextareaRef.current?.focus()
  }, [availabilityInput, checkAvailability, isAvailabilityRunning])

  const handleAvailabilityKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAvailabilitySubmit() }
  }

  const moleculeInfo = pipelineState?.molecule_info || null
  const guardResult = pipelineState?.guard_result || null
  const synthesisPaths = pipelineState?.synthesis_pathways || []
  const experimentProtocol = pipelineState?.experiment_protocol || null
  const retroMoleculeInfo = retroAnalysis?.molecule_info || null
  const retroGuardResult = retroAnalysis?.guard_result || null
  const retroResult = retroAnalysis?.retro_result || null

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
              onClick={() => { setPage(item.id); if (item.id === 'chat' && status !== 'idle') { currentQueryRef.current = ''; reset() } }}
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

        {page === 'retrosynthesis' && (
          <>
            <div className="topbar">
              <span className="topbar-title">
                {isRetroRunning ? (
                  <span className="topbar-status">
                    <div className="spinner spinner-sm" />
                    Выполняется ретросинтез...
                  </span>
                ) : 'Отдельное рабочее пространство ретросинтеза'}
              </span>
              <div className="retro-topbar-controls">
                <select
                  className="retro-source-select"
                  value={retroSourceMode}
                  onChange={e => setRetroSourceMode(e.target.value)}
                  disabled={isRetroRunning}
                >
                  {sourceModes.map(mode => (
                    <option key={mode.id} value={mode.id} disabled={!mode.enabled}>
                      {mode.label}{mode.enabled ? '' : ' (offline)'}
                    </option>
                  ))}
                </select>
                <ModelSelector value={model} onChange={setModel} disabled={isRetroRunning} />
              </div>
            </div>

            <div className="messages">
              {retroStatus === 'idle' && (
                <div className="retro-empty-state">
                  <div className="empty-title">Ретросинтез</div>
                  <div className="empty-sub">Тот же сценарий карточки молекулы, но с явным выбором источника ретросинтеза.</div>
                  <div className="empty-examples">
                    {EXAMPLES.map(ex => (
                      <button key={ex} className="example-chip" onClick={() => setRetroInput(ex)}>{ex}</button>
                    ))}
                  </div>
                  <Suspense fallback={<div className="molecule-editor-loading">Загрузка редактора молекул...</div>}>
                    <MoleculeEditor
                      initialSmiles={retroInput}
                      disabled={isRetroRunning}
                      onUseSmiles={handleEditorSmiles}
                      onRunRetrosynthesis={handleEditorRetrosynthesis}
                    />
                  </Suspense>
                  <div className="retro-source-grid">
                    {sourceModes.map(mode => (
                      <button
                        key={mode.id}
                        type="button"
                        className={`retro-source-card${retroSourceMode === mode.id ? ' active' : ''}${mode.enabled ? '' : ' disabled'}`}
                        onClick={() => mode.enabled && setRetroSourceMode(mode.id)}
                        disabled={!mode.enabled}
                      >
                        <span className="retro-source-card-title">{mode.label}</span>
                        <span className="retro-source-card-desc">{mode.description}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {isRetroRunning && (
                <div className="loading-row">
                  <div className="spinner spinner-md" />
                  Идёт поиск ретросинтетических маршрутов...
                </div>
              )}

              {retroStatus === 'error' && (
                <div className="error-block">
                  {retroError || 'Не удалось выполнить запрос ретросинтеза'}
                  <button className="reset-link" onClick={resetRetro}>
                    Сбросить
                  </button>
                </div>
              )}

              {!isRetroRunning && retroMoleculeInfo && (
                <div style={{ marginBottom: 16 }}>
                  <div className="retro-result-meta">
                    <div className="retro-result-pill">
                      Источник: {sourceModes.find(mode => mode.id === (retroAnalysis?.source_mode || retroSourceMode))?.label || retroAnalysis?.source_mode || retroSourceMode}
                    </div>
                    {retroAnalysis?.status === 'blocked' && <div className="retro-result-pill warning">Блок по безопасности</div>}
                  </div>
                  <MoleculeCard
                    moleculeInfo={retroMoleculeInfo}
                    guardResult={retroGuardResult}
                    retroResult={retroResult}
                    defaultTab="synthesis"
                  />
                  {retroAnalysis?.error && (
                    <div className="error-block" style={{ marginTop: 16 }}>
                      {retroAnalysis.error}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="input-area">
              <div className="input-row">
                <textarea
                  ref={retroTextareaRef}
                  className="input-box"
                  rows={1}
                  placeholder="аспирин, caffeine, CC(=O)O, ..."
                  value={retroInput}
                  onChange={e => setRetroInput(e.target.value)}
                  onKeyDown={handleRetroKeyDown}
                  disabled={isRetroRunning}
                  style={{ height: 44 }}
                  onInput={e => {
                    e.target.style.height = '44px'
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                  }}
                />
                <button
                  className="send-btn"
                  onClick={handleRetroSubmit}
                  disabled={!retroInput.trim() || isRetroRunning}
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="8" y1="13" x2="8" y2="3" />
                    <polyline points="4 7 8 3 12 7" />
                  </svg>
                </button>
              </div>
              <div className="input-hint">
                Enter — запустить ретросинтез · Shift+Enter — новая строка
              </div>
            </div>
          </>
        )}

        {page === 'research' && (
          <>
            <div className="topbar">
              <span className="topbar-title">
                {isResearchRunning ? (
                  <span className="topbar-status">
                    <div className="spinner spinner-sm" />
                    Выполняется поиск...
                  </span>
                ) : 'Литература, патенты и подбор молекул'}
              </span>
              <div className="retro-topbar-controls">
                <select
                  className="retro-source-select"
                  value={researchMode}
                  onChange={e => setResearchMode(e.target.value)}
                  disabled={isResearchRunning}
                >
                  <option value="literature">Литература</option>
                  <option value="patent">Патенты</option>
                  <option value="molecule">Подбор молекул</option>
                </select>
              </div>
            </div>

            <div className="messages">
              {researchStatus === 'idle' && (
                <div className="research-empty-state">
                  <div className="empty-title">Исследовательский режим</div>
                  <div className="empty-sub">Отдельный сценарий для literature overview, patent-oriented search и подбора молекул. Основной синтез не меняется.</div>
                  <div className="empty-examples">
                    {[
                      'aspirin synthesis literature',
                      'patents for acetylsalicylic acid preparation',
                      'найди ингибиторы EGFR',
                    ].map(ex => (
                      <button key={ex} className="example-chip" onClick={() => setResearchInput(ex)}>{ex}</button>
                    ))}
                  </div>
                  <div className="research-mode-grid">
                    <button type="button" className={`research-mode-card${researchMode === 'literature' ? ' active' : ''}`} onClick={() => setResearchMode('literature')}>
                      <span>Литература</span>
                      <small>PubMed/web/RAG evidence для обзора темы и условий.</small>
                    </button>
                    <button type="button" className={`research-mode-card${researchMode === 'patent' ? ' active' : ''}`} onClick={() => setResearchMode('patent')}>
                      <span>Патенты</span>
                      <small>Поиск preparation examples, claims и patent-oriented источников.</small>
                    </button>
                    <button type="button" className={`research-mode-card${researchMode === 'molecule' ? ' active' : ''}`} onClick={() => setResearchMode('molecule')}>
                      <span>Подбор молекул</span>
                      <small>Извлечение кандидатов и проверка через PubChem.</small>
                    </button>
                  </div>
                </div>
              )}

              {isResearchRunning && (
                <div className="loading-row">
                  <div className="spinner spinner-md" />
                  Идёт поиск и извлечение источников...
                </div>
              )}

              {researchStatus === 'error' && (
                <div className="error-block">
                  {researchError || 'Не удалось выполнить исследовательский запрос'}
                  <button className="reset-link" onClick={resetResearch}>
                    Сбросить
                  </button>
                </div>
              )}

              {!isResearchRunning && researchResult && (
                <div className="research-results">
                  <div className="research-summary-card">
                    <div className="research-summary-kicker">{researchResult.mode} · {researchResult.status}</div>
                    <h2>{researchResult.interpreted_intent || researchResult.query}</h2>
                    <p>{researchResult.summary}</p>
                  </div>

                  {researchResult.analysis && (
                    <div className="research-analysis-card">
                      <div className="research-section-title">Анализ агента</div>
                      {researchResult.analysis.answer && <p className="research-analysis-answer">{researchResult.analysis.answer}</p>}
                      {researchResult.analysis.key_findings?.length > 0 && (
                        <div className="research-analysis-grid">
                          {researchResult.analysis.key_findings.map((finding, index) => (
                            <div key={`${finding.claim}-${index}`} className="research-finding-card">
                              <span className={`research-confidence ${finding.confidence || 'low'}`}>{finding.confidence || 'low'}</span>
                              <strong>{finding.claim}</strong>
                              {finding.evidence?.length > 0 && <small>Evidence: {finding.evidence.join(', ')}</small>}
                            </div>
                          ))}
                        </div>
                      )}
                      {researchResult.analysis.candidate_assessment?.length > 0 && (
                        <div className="research-agent-list">
                          <div className="research-agent-list-title">Оценка кандидатов</div>
                          {researchResult.analysis.candidate_assessment.map((item, index) => (
                            <div key={`${item.name}-${index}`} className="research-agent-row">
                              <strong>{item.name}</strong>
                              <span>{item.assessment}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {researchResult.analysis.limitations?.length > 0 && (
                        <div className="research-agent-list">
                          <div className="research-agent-list-title">Ограничения</div>
                          {researchResult.analysis.limitations.map((item, index) => (
                            <div key={`${item}-${index}`} className="research-agent-row muted">
                              <span>{item}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {researchResult.analysis.recommended_next_steps?.length > 0 && (
                        <div className="research-agent-list">
                          <div className="research-agent-list-title">Следующие шаги</div>
                          {researchResult.analysis.recommended_next_steps.map((item, index) => (
                            <div key={`${item}-${index}`} className="research-agent-row">
                              <span>{item}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="research-analysis-footnote">
                        Engine: {researchResult.analysis.analysis_engine || 'unknown'}
                        {researchResult.analysis.source_quality ? ` · ${researchResult.analysis.source_quality}` : ''}
                      </div>
                    </div>
                  )}

                  {researchResult.candidates?.length > 0 && (
                    <div className="research-section">
                      <div className="research-section-title">PubChem-кандидаты</div>
                      <div className="research-candidate-grid">
                        {researchResult.candidates.map(candidate => (
                          <div key={`${candidate.pubchem_cid}-${candidate.name}`} className="research-candidate-card">
                            <strong>{candidate.name}</strong>
                            {candidate.pubchem_cid && <span>CID {candidate.pubchem_cid}</span>}
                            {candidate.canonical_smiles && <code>{candidate.canonical_smiles}</code>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {researchResult.evidence?.length > 0 && (
                    <div className="research-section">
                      <div className="research-section-title">Источники и выдержки</div>
                      <div className="research-source-list">
                        {researchResult.evidence.map((source, index) => (
                          <a key={`${source.url}-${index}`} className="research-source-card" href={source.url} target="_blank" rel="noreferrer">
                            <span className="research-source-type">{source.source_type || 'web'}</span>
                            <strong>{source.title || source.url}</strong>
                            <p>{source.excerpt || source.snippet || 'Без извлечённого текста'}</p>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  {researchResult.rag_results?.length > 0 && (
                    <div className="research-section">
                      <div className="research-section-title">Локальный RAG</div>
                      <div className="research-source-list">
                        {researchResult.rag_results.map(result => (
                          <div key={`${result.rank}-${result.title}`} className="research-source-card">
                            <span className="research-source-type">score {result.score}</span>
                            <strong>{result.title}</strong>
                            <p>{result.child_text || result.parent_text}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="input-area">
              <div className="input-row">
                <textarea
                  ref={researchTextareaRef}
                  className="input-box"
                  rows={1}
                  placeholder="например: aspirin synthesis literature или patents for ibuprofen preparation"
                  value={researchInput}
                  onChange={e => setResearchInput(e.target.value)}
                  onKeyDown={handleResearchKeyDown}
                  disabled={isResearchRunning}
                  style={{ height: 44 }}
                  onInput={e => {
                    e.target.style.height = '44px'
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                  }}
                />
                <button
                  className="send-btn"
                  onClick={handleResearchSubmit}
                  disabled={!researchInput.trim() || isResearchRunning}
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="8" y1="13" x2="8" y2="3" />
                    <polyline points="4 7 8 3 12 7" />
                  </svg>
                </button>
              </div>
              <div className="input-hint">
                Enter — запустить поиск · Shift+Enter — новая строка
              </div>
            </div>
          </>
        )}

        {page === 'availability' && (
          <>
            <div className="topbar">
              <span className="topbar-title">
                {isAvailabilityRunning ? (
                  <span className="topbar-status">
                    <div className="spinner spinner-sm" />
                    Проверяю доступность реагентов...
                  </span>
                ) : 'Поставщики и доступность реагентов'}
              </span>
            </div>

            <div className="messages">
              {availabilityStatus === 'idle' && (
                <div className="availability-empty-state">
                  <div className="empty-title">Поставщики</div>
                  <div className="empty-sub">
                    Проверка стартовых веществ через локальную buyables DB, список обычных лабораторных реагентов и fallback-эвристику.
                  </div>
                  <div className="empty-examples">
                    {[
                      'CC(=O)OC(C)=O.O=C(O)c1ccccc1O',
                      'ethanol, acetic acid',
                      'salicylic acid',
                    ].map(ex => (
                      <button key={ex} className="example-chip" onClick={() => setAvailabilityInput(ex)}>{ex}</button>
                    ))}
                  </div>
                </div>
              )}

              {isAvailabilityRunning && (
                <div className="loading-row">
                  <div className="spinner spinner-md" />
                  Сверяю реагенты с локальными каталогами и готовлю ссылки на поставщиков...
                </div>
              )}

              {availabilityStatus === 'error' && (
                <div className="error-block">
                  {availabilityError || 'Не удалось проверить доступность реагентов'}
                  <button className="reset-link" onClick={resetAvailability}>
                    Сбросить
                  </button>
                </div>
              )}

              {!isAvailabilityRunning && availabilityResult && (
                <div className="availability-results">
                  <div className="availability-hero">
                    <div>
                      <div className="research-summary-kicker">Доступность · локальный каталог</div>
                      <h2>{availabilityResult.summary?.available_count || 0}/{availabilityResult.summary?.total || 0} доступны</h2>
                      <p>{availabilityResult.query}</p>
                    </div>
                    <div className="availability-score">
                      <strong>{Math.round((availabilityResult.summary?.availability_ratio || 0) * 100)}</strong>
                      <span>%</span>
                    </div>
                  </div>

                  <div className="availability-summary-grid">
                    <div><span>Каталог</span><strong>{availabilityResult.summary?.catalog_count || 0}</strong></div>
                    <div><span>Обычные</span><strong>{availabilityResult.summary?.common_count || 0}</strong></div>
                    <div><span>Эвристика</span><strong>{availabilityResult.summary?.heuristic_count || 0}</strong></div>
                    <div><span>Не найдено</span><strong>{availabilityResult.summary?.not_found_count || 0}</strong></div>
                    <div><span>С ценой</span><strong>{availabilityResult.summary?.priced_count || 0}</strong></div>
                  </div>

                  <div className="availability-list">
                    {availabilityResult.items?.map((item, index) => (
                      <div key={`${item.input}-${index}`} className={`availability-card level-${item.availability_level}`}>
                        <div className="availability-card-head">
                          <div>
                            <strong>{item.label || item.input}</strong>
                            {item.canonical_smiles && <code>{item.canonical_smiles}</code>}
                          </div>
                          <span>{AVAILABILITY_LEVEL_LABELS[item.availability_level] || item.availability_level}</span>
                        </div>

                        <div className="availability-meta">
                          <div><small>Основание</small><b>{item.basis}</b></div>
                          <div><small>Источник</small><b>{item.source_label || item.source || '—'}</b></div>
                          <div><small>Цена $/g</small><b>{item.ppg != null ? item.ppg : '—'}</b></div>
                          <div><small>Уверенность</small><b>{item.confidence}</b></div>
                        </div>

                        {item.estimated_pack_prices?.length > 0 && (
                          <div className="availability-pack-row">
                            {item.estimated_pack_prices.map(pack => (
                              <span key={pack.size_g}>{pack.size_g} g ~ ${pack.estimated_usd}</span>
                            ))}
                          </div>
                        )}

                        {item.supplier_search_links?.length > 0 && (
                          <div className="availability-links">
                            {item.supplier_search_links.map(link => (
                              <a key={link.label} href={link.url} target="_blank" rel="noreferrer">{link.label}</a>
                            ))}
                          </div>
                        )}

                        {item.warnings?.length > 0 && (
                          <div className="availability-warning">{item.warnings[0]}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="input-area">
              <div className="input-row">
                <textarea
                  ref={availabilityTextareaRef}
                  className="input-box"
                  rows={1}
                  placeholder="ethanol, acetic acid или CC(=O)OC(C)=O.O=C(O)c1ccccc1O"
                  value={availabilityInput}
                  onChange={e => setAvailabilityInput(e.target.value)}
                  onKeyDown={handleAvailabilityKeyDown}
                  disabled={isAvailabilityRunning}
                  style={{ height: 44 }}
                  onInput={e => {
                    e.target.style.height = '44px'
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                  }}
                />
                <button
                  className="send-btn"
                  onClick={handleAvailabilitySubmit}
                  disabled={!availabilityInput.trim() || isAvailabilityRunning}
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="8" y1="13" x2="8" y2="3" />
                    <polyline points="4 7 8 3 12 7" />
                  </svg>
                </button>
              </div>
              <div className="input-hint">
                Enter - проверить поставщиков · Shift+Enter - новая строка
              </div>
            </div>
          </>
        )}

        {page === 'admet' && (
          <>
            <div className="topbar">
              <span className="topbar-title">
                {isAdmetRunning ? (
                  <span className="topbar-status">
                    <div className="spinner spinner-sm" />
                    Выполняется ADMET-оценка...
                  </span>
                ) : 'ADMET-оценка'}
              </span>
            </div>

            <div className="messages">
              {admetStatus === 'idle' && (
                <div className="admet-empty-state">
                  <div className="empty-title">ADMET</div>
                  <div className="empty-sub">Быстрая интерпретируемая оценка всасывания, распределения, метаболизма, выведения и токсичности по RDKit-дескрипторам и данным безопасности.</div>
                  <div className="empty-examples">
                    {EXAMPLES.map(ex => (
                      <button key={ex} className="example-chip" onClick={() => setAdmetInput(ex)}>{ex}</button>
                    ))}
                  </div>
                </div>
              )}

              {isAdmetRunning && (
                <div className="loading-row">
                  <div className="spinner spinner-md" />
                  Считаю дескрипторы и ADMET-флаги...
                </div>
              )}

              {admetStatus === 'error' && (
                <div className="error-block">
                  {admetError || 'Не удалось выполнить ADMET анализ'}
                  <button className="reset-link" onClick={resetAdmet}>
                    Сбросить
                  </button>
                </div>
              )}

              {!isAdmetRunning && admetResult && (
                <div className="admet-results">
                  <div className={`admet-hero risk-${admetResult.admet?.overall?.risk_level || 'medium'}`}>
                    <div>
                      <div className="research-summary-kicker">ADMET · метод: {ADMET_METHOD_LABELS[admetResult.admet?.method] || admetResult.admet?.method}</div>
                      <h2>{admetResult.query}</h2>
                      <p>{admetResult.smiles}</p>
                    </div>
                    <div className="admet-score">
                      <strong>{admetResult.admet?.overall?.score}</strong>
                      <span>{RISK_LEVEL_LABELS[admetResult.admet?.overall?.risk_level] || admetResult.admet?.overall?.risk_level}</span>
                    </div>
                  </div>

                  {admetResult.admet?.safety_overlay?.available && (
                    <div className={`admet-safety-banner status-${admetResult.admet.safety_overlay.overall_status || 'SAFE'}`}>
                      <div>
                        <strong>Проверка безопасности: {SAFETY_STATUS_LABELS[admetResult.admet.safety_overlay.overall_status] || admetResult.admet.safety_overlay.overall_status}</strong>
                        <span>{admetResult.admet.safety_overlay.molecule_reason || 'Данные GHS и рекомендации по СИЗ учтены вместе с ADMET-дескрипторами.'}</span>
                      </div>
                      {admetResult.admet.safety_overlay.h_codes?.length > 0 && (
                        <code>{admetResult.admet.safety_overlay.h_codes.slice(0, 8).join(', ')}</code>
                      )}
                    </div>
                  )}

                  <div className="admet-descriptor-grid">
                    {Object.entries(admetResult.admet?.descriptors || {}).map(([key, value]) => (
                      <div key={key} className="admet-descriptor">
                        <span>{ADMET_DESCRIPTOR_LABELS[key] || key.replaceAll('_', ' ')}</span>
                        <strong>{String(value)}</strong>
                      </div>
                    ))}
                  </div>

                  <div className="admet-section-grid">
                    {Object.entries(admetResult.admet?.sections || {}).map(([key, section]) => (
                      <div key={key} className="admet-section-card">
                        <div className="admet-section-head">
                          <strong>{ADMET_SECTION_LABELS[key] || key}</strong>
                          <span>{section.score}/100</span>
                        </div>
                        <p>{section.interpretation}</p>
                        {section.flags?.length > 0 ? (
                          <div className="admet-flag-list">
                            {section.flags.map((flag, index) => (
                              <div key={`${flag.message}-${index}`} className={`admet-flag severity-${flag.severity}`}>
                                <strong>{SEVERITY_LABELS[flag.severity] || flag.severity}</strong>
                                <span>{flag.message}</span>
                                <small>{flag.evidence}</small>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="admet-clear">Критичных флагов не найдено</div>
                        )}
                      </div>
                    ))}
                  </div>

                  {admetResult.admet?.recommendations?.length > 0 && (
                    <div className="research-section">
                      <div className="research-section-title">Рекомендации</div>
                      <div className="admet-recommendations">
                        {admetResult.admet.recommendations.map((item, index) => (
                          <div key={`${item}-${index}`}>{item}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="input-area">
              <div className="input-row">
                <textarea
                  ref={admetTextareaRef}
                  className="input-box"
                  rows={1}
                  placeholder="aspirin, caffeine, CC(=O)O, ..."
                  value={admetInput}
                  onChange={e => setAdmetInput(e.target.value)}
                  onKeyDown={handleAdmetKeyDown}
                  disabled={isAdmetRunning}
                  style={{ height: 44 }}
                  onInput={e => {
                    e.target.style.height = '44px'
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
                  }}
                />
                <button
                  className="send-btn"
                  onClick={handleAdmetSubmit}
                  disabled={!admetInput.trim() || isAdmetRunning}
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="8" y1="13" x2="8" y2="3" />
                    <polyline points="4 7 8 3 12 7" />
                  </svg>
                </button>
              </div>
              <div className="input-hint">
                Enter — запустить ADMET · Shift+Enter — новая строка
              </div>
            </div>
          </>
        )}

      </main>
    </div>
  )
}

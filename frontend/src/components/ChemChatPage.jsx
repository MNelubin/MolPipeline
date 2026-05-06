import { useCallback, useRef, useState } from 'react'
import RetroCard from './RetroCard'
import { useChemChat } from '../hooks/useChemChat'

const EXAMPLES = [
  'Найди путь синтеза аспирина и оцени доступность реагентов',
  'Чем SN1 отличается от SN2?',
  'Проверь ADMET и безопасность кофеина',
  'Можно ли купить бензальдегид и этанол?',
]

const SOURCE_OPTIONS = [
  { id: 'auto', label: 'Авто' },
  { id: 'all', label: 'Все источники' },
  { id: 'ord', label: 'ORD' },
  { id: 'retro_model', label: 'Локальная модель' },
  { id: 'web', label: 'Web' },
  { id: 'aizynthfinder', label: 'AiZynthFinder' },
]

function ArtifactBlock({ result }) {
  const artifacts = result?.artifacts || {}
  const molecule = artifacts.molecule
  const safety = artifacts.safety
  const retro = artifacts.retrosynthesis
  const admet = artifacts.admet
  const availability = artifacts.availability
  const research = artifacts.research

  return (
    <div className="chemchat-artifacts">
      {molecule?.status === 'ok' && (
        <div className="chemchat-artifact-card">
          <div className="chemchat-artifact-title">Молекула</div>
          <div className="chemchat-kv">
            <span>SMILES</span>
            <code>{molecule.smiles}</code>
          </div>
          <div className="chemchat-kv">
            <span>Формула</span>
            <strong>{molecule.validation?.molecular_formula || 'n/a'}</strong>
          </div>
          <div className="chemchat-kv">
            <span>Масса</span>
            <strong>{molecule.validation?.molecular_weight || 'n/a'}</strong>
          </div>
        </div>
      )}

      {safety && (
        <div className={`chemchat-artifact-card safety-${safety.overall_status || 'SAFE'}`}>
          <div className="chemchat-artifact-title">Safety gate</div>
          <div className="chemchat-status-line">{safety.overall_status || 'UNKNOWN'}</div>
          <p>{safety.molecule_check?.reason || safety.reaction_check?.reason || 'Критичных флагов не найдено.'}</p>
        </div>
      )}

      {retro && (
        <div className="chemchat-wide-artifact">
          <RetroCard retroResult={retro} />
        </div>
      )}

      {admet && (
        <div className="chemchat-artifact-card">
          <div className="chemchat-artifact-title">ADMET</div>
          <div className="chemchat-score">{admet.overall?.score ?? 'n/a'}<span>/100</span></div>
          <p>Риск: {admet.overall?.risk_level || 'unknown'}</p>
        </div>
      )}

      {availability && (
        <div className="chemchat-artifact-card">
          <div className="chemchat-artifact-title">Поставщики</div>
          <div className="chemchat-kv">
            <span>Доступно</span>
            <strong>{availability.summary?.available_count || 0}/{availability.summary?.total || 0}</strong>
          </div>
          <div className="chemchat-kv">
            <span>С ценами</span>
            <strong>{availability.summary?.priced_count || 0}</strong>
          </div>
          <div className="chemchat-mini-list">
            {(availability.items || []).slice(0, 4).map((item, index) => (
              <div key={`${item.input || item.smiles}-${index}`}>
                <code>{item.label || item.input || item.smiles}</code>
                <span>{item.source_label || item.availability_level || 'не найдено'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {research && (
        <div className="chemchat-wide-artifact">
          <div className="chemchat-artifact-card">
            <div className="chemchat-artifact-title">Research</div>
            <p>{research.summary || 'Исследовательский отчет собран.'}</p>
            <div className="chemchat-mini-list">
              {(research.sources || []).slice(0, 5).map((source, index) => (
                <div key={`${source.url || source.title}-${index}`}>
                  <strong>{source.title || source.name || 'Источник'}</strong>
                  <span>{source.source_type || source.type || 'source'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function AssistantMessage({ message }) {
  const result = message.result
  return (
    <div className={`chemchat-message assistant${message.error ? ' error' : ''}`}>
      <div className="chemchat-bubble">
        <div className="chemchat-answer">
          {message.content.split('\n').map((line, index) => (
            <p key={index}>{line}</p>
          ))}
        </div>
        {result?.tools_used?.length > 0 && (
          <div className="chemchat-tools">
            {result.tools_used.map(tool => <span key={tool}>{tool}</span>)}
          </div>
        )}
        {result && <ArtifactBlock result={result} />}
        {result?.suggested_next_actions?.length > 0 && (
          <div className="chemchat-suggestions">
            {result.suggested_next_actions.map(action => <span key={action}>{action}</span>)}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChemChatPage() {
  const [input, setInput] = useState('')
  const [sourceMode, setSourceMode] = useState('auto')
  const textareaRef = useRef(null)
  const { status, messages, sendMessage, reset } = useChemChat()
  const isRunning = status === 'running'

  const handleSubmit = useCallback(async () => {
    const text = input.trim()
    if (!text || isRunning) return
    setInput('')
    textareaRef.current?.focus()
    await sendMessage(text, { sourceMode })
  }, [input, isRunning, sendMessage, sourceMode])

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <>
      <div className="topbar">
        <span className="topbar-title">
          {isRunning ? (
            <span className="topbar-status">
              <div className="spinner spinner-sm" />
              ChemChat вызывает инструменты...
            </span>
          ) : 'Общий химический чат с инструментами MolPipeline'}
        </span>
        <select
          className="chemchat-source-select"
          value={sourceMode}
          onChange={e => setSourceMode(e.target.value)}
          disabled={isRunning}
        >
          {SOURCE_OPTIONS.map(option => (
            <option key={option.id} value={option.id}>{option.label}</option>
          ))}
        </select>
      </div>

      <div className="messages chemchat-messages">
        {messages.length === 0 && (
          <div className="chemchat-empty">
            <div className="empty-title">ChemChat</div>
            <div className="empty-sub">
              Задавайте химические вопросы свободно: ретросинтез, безопасность, ADMET, поставщики, литература или общая химия.
            </div>
            <div className="empty-examples">
              {EXAMPLES.map(example => (
                <button key={example} className="example-chip" onClick={() => setInput(example)}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          message.role === 'user' ? (
            <div key={`${message.ts}-${index}`} className="chemchat-message user">
              <div className="chemchat-bubble">{message.content}</div>
            </div>
          ) : (
            <AssistantMessage key={`${message.ts}-${index}`} message={message} />
          )
        ))}

        {isRunning && (
          <div className="loading-row">
            <div className="spinner spinner-md" />
            Оркестратор выбирает tools и собирает ответ...
          </div>
        )}
      </div>

      <div className="input-area">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            className="input-box"
            rows={1}
            placeholder="Например: найди безопасный путь синтеза аспирина и проверь поставщиков"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isRunning}
            style={{ height: 44 }}
            onInput={e => {
              e.target.style.height = '44px'
              e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
            }}
          />
          <button className="send-btn" onClick={handleSubmit} disabled={!input.trim() || isRunning}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="8" y1="13" x2="8" y2="3" />
              <polyline points="4 7 8 3 12 7" />
            </svg>
          </button>
        </div>
        <div className="input-hint">
          Enter - отправить · Shift+Enter - новая строка · ретросинтез идет через выбранный источник
          <button className="chemchat-reset" onClick={reset} disabled={isRunning || messages.length === 0}>Очистить</button>
        </div>
      </div>
    </>
  )
}

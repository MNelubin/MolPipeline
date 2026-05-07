import { useCallback, useEffect, useRef, useState } from 'react'
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

function formatSessionTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function renderInlineMarkdown(text) {
  const parts = []
  const pattern = /(\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|(https?:\/\/[^\s<]+))/g
  let lastIndex = 0
  let match
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index))
    if (match[2] && match[3]) {
      parts.push(
        <a key={match.index} href={match[3]} target="_blank" rel="noreferrer">
          {match[2]}
        </a>
      )
    } else if (match[4]) {
      parts.push(<code key={match.index}>{match[4]}</code>)
    } else if (match[5]) {
      parts.push(<strong key={match.index}>{match[5]}</strong>)
    } else if (match[6]) {
      const trailing = match[6].match(/[),.;:!?]+$/)?.[0] || ''
      const href = trailing ? match[6].slice(0, -trailing.length) : match[6]
      parts.push(
        <a key={match.index} href={href} target="_blank" rel="noreferrer">
          {href}
        </a>
      )
      if (trailing) parts.push(trailing)
    }
    lastIndex = pattern.lastIndex
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex))
  return parts
}

function MarkdownText({ text }) {
  const lines = String(text || '').split('\n')
  const blocks = []
  let listItems = []
  let tableRows = []

  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push(
        <ul key={`list-${blocks.length}`}>
          {listItems.map((item, index) => <li key={index}>{renderInlineMarkdown(item)}</li>)}
        </ul>
      )
      listItems = []
    }
  }

  const flushTable = () => {
    if (tableRows.length === 0) return
    const rows = tableRows
      .map(row => row.replace(/^\||\|$/g, '').split('|').map(cell => cell.trim()))
      .filter(cells => !cells.every(cell => /^:?-{3,}:?$/.test(cell) || cell === ''))
    if (rows.length > 0) {
      const [head, ...body] = rows
      blocks.push(
        <div key={`table-${blocks.length}`} className="chemchat-table-wrap">
          <table className="chemchat-table">
            <thead>
              <tr>{head.map((cell, index) => <th key={index}>{renderInlineMarkdown(cell)}</th>)}</tr>
            </thead>
            <tbody>
              {body.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => <td key={cellIndex}>{renderInlineMarkdown(cell)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    }
    tableRows = []
  }

  const isTableLine = (line, index) => {
    if (!line.includes('|')) return false
    if (/^[-*]\s+/.test(line)) return false
    if (/\[[^\]]+\]\(https?:\/\/[^)]+\)/.test(line)) return false
    const currentLooksTable = /^\|.+\|$/.test(line) || line.split('|').length >= 4
    const next = lines[index + 1]?.trim() || ''
    const nextIsSeparator = /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(next)
    return currentLooksTable && (nextIsSeparator || tableRows.length > 0)
  }

  lines.forEach((line, index) => {
    const trimmed = line.trim()
    if (!trimmed) {
      flushList()
      flushTable()
      return
    }
    if (isTableLine(trimmed, index)) {
      flushList()
      tableRows.push(trimmed)
      return
    }
    flushTable()
    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      flushList()
      const Tag = heading[1].length === 1 ? 'h3' : 'h4'
      blocks.push(<Tag key={index}>{renderInlineMarkdown(heading[2])}</Tag>)
      return
    }
    const list = trimmed.match(/^[-*]\s+(.+)$/)
    if (list) {
      listItems.push(list[1])
      return
    }
    flushList()
    blocks.push(<p key={index}>{renderInlineMarkdown(trimmed)}</p>)
  })
  flushList()
  flushTable()

  return <div className="chemchat-md">{blocks}</div>
}

function ProgressTimeline({ events = [] }) {
  if (!events.length) return null
  return (
    <div className="chemchat-progress">
      {events.slice(-12).map((event, index) => (
        <div key={`${event.type}-${event.tool || event.stage}-${index}`} className={`chemchat-progress-row event-${event.type}`}>
          <span className="chemchat-progress-dot" />
          <div>
            <strong>{event.label || event.tool || event.stage || event.type}</strong>
            <small>
              {event.tool || event.stage || event.intent || ''}
              {event.routes !== undefined ? ` · маршрутов: ${event.routes}` : ''}
              {event.sources !== undefined ? ` · источников: ${event.sources}` : ''}
              {event.status ? ` · ${event.status}` : ''}
            </small>
          </div>
        </div>
      ))}
    </div>
  )
}

function ArtifactBlock({ result }) {
  const artifacts = result?.artifacts || {}
  const molecule = artifacts.molecule
  const safety = artifacts.safety
  const retro = artifacts.retrosynthesis
  const admet = artifacts.admet
  const availability = artifacts.availability
  const research = artifacts.research
  const safetyCategories = safety?.safety_taxonomy?.categories || []
  const safetyPrimaryReason = safety?.safety_taxonomy?.blocked_categories?.[0]?.reason
    || safety?.explosive_check?.reason
    || safety?.safety_taxonomy?.warning_categories?.[0]?.reason
    || safety?.molecule_check?.reason
    || safety?.reaction_check?.reason

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
          <p>{safetyPrimaryReason || 'Критичных флагов не найдено.'}</p>
          {safety.explosive_check?.status && safety.explosive_check.status !== 'clear' && (
            <div className="chemchat-kv">
              <span>Тип риска</span>
              <strong>{safety.explosive_check.hazard_type || 'explosive'}</strong>
            </div>
          )}
          {safetyCategories.length > 0 && (
            <div className="chemchat-safety-taxonomy">
              {safetyCategories.slice(0, 5).map((item, index) => (
                <div key={`${item.hazard_type || 'hazard'}-${index}`} className={`chemchat-safety-chip ${item.status || 'warning'}`}>
                  <strong>{item.label_ru || item.hazard_type || 'risk'}</strong>
                  <span>{item.status || 'warning'} · {item.danger_level || 'unknown'}</span>
                </div>
              ))}
            </div>
          )}
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
                  {source.url ? (
                    <a href={source.url} target="_blank" rel="noreferrer">
                      {source.citation_id ? `[${source.citation_id}] ` : ''}{source.title || source.name || source.url}
                    </a>
                  ) : (
                    <strong>{source.title || source.name || 'Источник'}</strong>
                  )}
                  <span>{source.domain || source.source_type || source.type || 'source'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function AssistantMessage({ message, onSuggestionClick }) {
  const result = message.result
  return (
    <div className={`chemchat-message assistant${message.error ? ' error' : ''}`}>
      <div className="chemchat-bubble">
        {message.progress?.length > 0 && <ProgressTimeline events={message.progress} />}
        {message.streaming && !message.content && (
          <div className="chemchat-stream-note">
            <div className="spinner spinner-sm" />
            Жду следующий шаг...
          </div>
        )}
        {message.content && (
          <div className="chemchat-answer">
            <MarkdownText text={message.content} />
          </div>
        )}
        {result?.tools_used?.length > 0 && (
          <div className="chemchat-tools">
            {result.model && <span>model: {result.model}</span>}
            {result.tools_used.map(tool => <span key={tool}>{tool}</span>)}
          </div>
        )}
        {result && <ArtifactBlock result={result} />}
        {result?.suggested_next_actions?.length > 0 && (
          <div className="chemchat-suggestions">
            {result.suggested_next_actions.map(action => (
              <button type="button" key={action} onClick={() => onSuggestionClick(action)}>
                {action}
              </button>
            ))}
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
  const messagesEndRef = useRef(null)
  const {
    status,
    messages,
    sessions,
    activeSessionId,
    sendMessage,
    loadSession,
    startNewSession,
    deleteSession,
    reset,
  } = useChemChat()
  const isRunning = status === 'running'

  useEffect(() => {
    if (messages.length === 0) return
    messagesEndRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [messages, status])

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

  const handleSuggestionClick = useCallback(action => {
    setInput(action)
    textareaRef.current?.focus()
  }, [])

  const handleLoadSession = useCallback(async sessionId => {
    if (isRunning) return
    await loadSession(sessionId)
    textareaRef.current?.focus()
  }, [isRunning, loadSession])

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

      <div className="chemchat-session-panel">
        <button
          type="button"
          className="chemchat-session-new"
          onClick={startNewSession}
          disabled={isRunning}
        >
          Новый чат
        </button>
        <div className="chemchat-session-list">
          {sessions.length === 0 ? (
            <span className="chemchat-session-empty">История чатов пока пустая</span>
          ) : (
            sessions.map(session => (
              <div
                key={session.id}
                className={`chemchat-session-item${session.id === activeSessionId ? ' active' : ''}`}
              >
                <button
                  type="button"
                  className="chemchat-session-open"
                  onClick={() => handleLoadSession(session.id)}
                  disabled={isRunning}
                  title={session.title}
                >
                  <span className="chemchat-session-title">{session.title}</span>
                  <span className="chemchat-session-meta">
                    {formatSessionTime(session.updated_at)} · {session.message_count || 0} сообщ.
                  </span>
                </button>
                <button
                  type="button"
                  className="chemchat-session-delete"
                  onClick={() => deleteSession(session.id)}
                  disabled={isRunning}
                  aria-label="Удалить чат"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
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
                <button key={example} className="example-chip" onClick={() => handleSuggestionClick(example)}>
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
            <AssistantMessage key={`${message.ts}-${index}`} message={message} onSuggestionClick={handleSuggestionClick} />
          )
        ))}

        {isRunning && (
          <div className="loading-row">
            <div className="spinner spinner-md" />
            Оркестратор выбирает tools и собирает ответ...
          </div>
        )}
        <div ref={messagesEndRef} className="chemchat-scroll-anchor" aria-hidden="true" />
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

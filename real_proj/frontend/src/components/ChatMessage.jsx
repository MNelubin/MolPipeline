import PipelineProgress from './PipelineProgress'
import MoleculeCard from './MoleculeCard'

export default function ChatMessage({ message }) {
  if (message.role === 'user') {
    return (
      <div className="message-row user">
        <div className="message-avatar user">U</div>
        <div className="message-content">
          <div className="user-bubble">{message.query}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="message-row bot">
      <div className="message-avatar bot">⬡</div>
      <div className="message-content">
        <PipelineProgress nodes={message.nodes} />

        {message.error && (
          <div className="error-card">⚠ {message.error}</div>
        )}

        {!message.done && message.streamText && !message.moleculeInfo && (
          <div className="streaming-text">
            {message.streamText}
            <span className="cursor-blink" />
          </div>
        )}

        {/* Передаём retroResult в MoleculeCard */}
        {message.moleculeInfo && (
          <MoleculeCard
            moleculeInfo={message.moleculeInfo}
            guardResult={message.guardResult}
            retroResult={message.retroResult}
          />
        )}

        {!message.error && !message.streamText && !message.moleculeInfo && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-3)', fontSize: 13 }}>
            <div className="spinner" />
            <span>Обработка...</span>
          </div>
        )}
      </div>
    </div>
  )
}

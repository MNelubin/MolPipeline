/**
 * PathwaySelector — displays synthesis pathways for user selection.
 *
 * pathways: synthesis_pathways array from state, each with:
 *   { reactants, source, final_score, viable, buyable_leaves, unresolved_leaves, ... }
 *
 * onSelect(pathwayIndex, targetMassG) — called when user confirms selection
 */

import { useState } from 'react'

const SOURCE_LABEL = {
  ord:         { text: 'ORD',   color: 'var(--green)' },
  retro_model: { text: 'MODEL', color: 'var(--purple)' },
  web:         { text: 'WEB', color: 'var(--cyan)' },
  aizynthfinder: { text: 'AIZYNTH', color: 'var(--amber)' },
}

function ScoreBar({ value, max = 1 }) {
  const pct = Math.min(Math.round((value / max) * 100), 100)
  const level = pct > 70 ? 'high' : pct > 40 ? 'medium' : 'low'
  return (
    <div className="score-bar">
      <div className="score-track">
        <div className={`score-fill ${level}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="score-value">{value.toFixed(3)}</span>
    </div>
  )
}

export default function PathwaySelector({ pathways, onSelect }) {
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [targetMass, setTargetMass] = useState('1.0')

  if (!pathways || pathways.length === 0) return null

  const handleConfirm = () => {
    const mass = parseFloat(targetMass)
    if (isNaN(mass) || mass <= 0) return
    onSelect(selectedIdx, mass)
  }

  return (
    <div style={{ marginTop: 8, marginBottom: 16 }}>
      <div className="pathway-heading">Выберите маршрут синтеза</div>

      <div className="pathway-list">
        {pathways.map((pathway, i) => {
          const src = SOURCE_LABEL[pathway.source] || { text: (pathway.source || 'UNKNOWN').toUpperCase(), color: 'var(--text-3)' }
          const score = pathway.final_score ?? 0
          const viable = pathway.viable !== false
          const buyable = pathway.buyable_leaves ?? 0
          const unresolved = pathway.unresolved_leaves ?? 0
          const isSelected = selectedIdx === i
          const availabilitySummary = pathway.availability_summary || null

          return (
            <div
              key={i}
              onClick={() => setSelectedIdx(i)}
              className={`pathway-card${isSelected ? ' selected' : ''}`}
            >
              <div className="pathway-card-header">
                <div className="pathway-radio">
                  {isSelected && <div className="pathway-radio-dot" />}
                </div>

                <span
                  className="source-badge"
                  style={{
                    background: `color-mix(in srgb, ${src.color} 10%, transparent)`,
                    color: src.color,
                    border: `1px solid color-mix(in srgb, ${src.color} 25%, transparent)`,
                  }}
                >
                  {src.text}
                </span>

                {viable ? (
                  <span className="viable-text" style={{ color: 'var(--green)' }}>✓ Выполним</span>
                ) : (
                  <span className="viable-text" style={{ color: 'var(--amber)' }}>⚠ Возможны проблемы</span>
                )}

                <span className="leaf-counts">{buyable} куп. / {unresolved} нераз.</span>
              </div>

              {availabilitySummary && (
                <div className="pathway-availability">
                  <span>{availabilitySummary.available_count}/{availabilitySummary.total} реагента доступны</span>
                  {availabilitySummary.estimated_total_1g_usd != null && (
                    <span>1 г ~ ${availabilitySummary.estimated_total_1g_usd}</span>
                  )}
                </div>
              )}

              <div className="pathway-reactants">{pathway.reactants || '—'}</div>

              <div className="pathway-score-area">
                <ScoreBar value={score} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Target mass + confirm */}
      <div className="pathway-controls">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label className="pathway-label">Целевая масса:</label>
          <input
            type="number"
            min="0.001"
            step="0.1"
            value={targetMass}
            onChange={e => setTargetMass(e.target.value)}
            className="pathway-input"
          />
          <span className="pathway-unit">г</span>
        </div>

        <button className="action-btn" onClick={handleConfirm}>
          Запустить расчёт
        </button>
      </div>
    </div>
  )
}

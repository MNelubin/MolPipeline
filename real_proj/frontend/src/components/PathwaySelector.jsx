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
}

function ScoreBar({ value, max = 1 }) {
  const pct = Math.min(Math.round((value / max) * 100), 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        flex: 1, height: 4, background: 'var(--bg-3)',
        borderRadius: 2, overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: pct > 70 ? 'var(--green)' : pct > 40 ? 'var(--amber)' : 'var(--red)',
          borderRadius: 2, transition: 'width 0.4s',
        }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-2)', minWidth: 32 }}>
        {value.toFixed(3)}
      </span>
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
      <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-1)', marginBottom: 14 }}>
        Выберите маршрут синтеза
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 20 }}>
        {pathways.map((pathway, i) => {
          const src = SOURCE_LABEL[pathway.source] || { text: (pathway.source || 'UNKNOWN').toUpperCase(), color: 'var(--text-3)' }
          const score = pathway.final_score ?? 0
          const viable = pathway.viable !== false
          const buyable = pathway.buyable_leaves ?? 0
          const unresolved = pathway.unresolved_leaves ?? 0
          const isSelected = selectedIdx === i

          return (
            <div
              key={i}
              onClick={() => setSelectedIdx(i)}
              style={{
                background: isSelected ? 'var(--bg-2)' : 'var(--bg-card)',
                border: `1px solid ${isSelected ? 'var(--cyan)' : 'var(--border)'}`,
                borderRadius: 'var(--r-md)',
                padding: '12px 14px',
                cursor: 'pointer',
                transition: 'border-color 0.15s, background 0.15s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                {/* Radio */}
                <div style={{
                  width: 16, height: 16, borderRadius: '50%',
                  border: `2px solid ${isSelected ? 'var(--cyan)' : 'var(--border)'}`,
                  background: isSelected ? 'var(--cyan)' : 'transparent',
                  flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {isSelected && (
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--bg-1)' }} />
                  )}
                </div>

                {/* Source badge */}
                <span style={{
                  fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
                  padding: '2px 7px', borderRadius: 4,
                  background: `${src.color}18`, color: src.color,
                  border: `1px solid ${src.color}40`, flexShrink: 0,
                }}>
                  {src.text}
                </span>

                {/* Viability */}
                {viable ? (
                  <span style={{ fontSize: 11, color: 'var(--green)', fontFamily: 'var(--font-mono)' }}>
                    ✓ Выполним
                  </span>
                ) : (
                  <span style={{ fontSize: 11, color: 'var(--amber)', fontFamily: 'var(--font-mono)' }}>
                    ⚠ Возможны проблемы
                  </span>
                )}

                {/* Leaf counts */}
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                  {buyable} куп. / {unresolved} нераз.
                </span>
              </div>

              {/* Reactants SMILES */}
              <div style={{
                fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-2)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                marginBottom: 8, paddingLeft: 26,
              }}>
                {pathway.reactants || '—'}
              </div>

              {/* Score bar */}
              <div style={{ paddingLeft: 26 }}>
                <ScoreBar value={score} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Target mass input + confirm */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ fontSize: 13, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
            Целевая масса:
          </label>
          <input
            type="number"
            min="0.001"
            step="0.1"
            value={targetMass}
            onChange={e => setTargetMass(e.target.value)}
            style={{
              width: 80,
              padding: '6px 10px',
              background: 'var(--bg-2)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--r-sm)',
              color: 'var(--text-1)',
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
            }}
          />
          <span style={{ fontSize: 13, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>г</span>
        </div>

        <button
          className="send-btn"
          style={{ padding: '10px 24px', fontSize: 13, width: 'auto', borderRadius: 'var(--r-md)' }}
          onClick={handleConfirm}
        >
          Запустить расчёт
        </button>
      </div>
    </div>
  )
}

/**
 * RetroCard — отображает результаты ретросинтеза.
 *
 * retroResult shape (из state.retro_result):
 * {
 *   routes: [
 *     {
 *       reaction_id, reaction_smiles, reactants,
 *       expected_yield, temperature, solvent, catalyst,
 *       procedure_details, procedure_steps_ru,
 *       final_score, scoring: { model_score, plausibility, buyability, ... },
 *       source: "ord" | "retro_model"
 *     }
 *   ],
 *   best_route, sources_used, total_found
 * }
 */

import { useState } from 'react'
import SynthesisGraph from './SynthesisGraph'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

const SOURCE_LABEL = {
  ord:          { text: 'ORD',   color: 'var(--green)' },
  retro_model:  { text: 'MODEL', color: 'var(--purple)' },
}

function ScoreBar({ value, max = 1 }) {
  const pct = Math.round((value / max) * 100)
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
        {value.toFixed(2)}
      </span>
    </div>
  )
}

function RouteCard({ route, index, smiles }) {
  const [open, setOpen] = useState(index === 0)
  const [tree, setTree] = useState(null)
  const [treeLoading, setTreeLoading] = useState(false)
  const [treeError, setTreeError] = useState(null)
  const [graphOpen, setGraphOpen] = useState(false)
  const src = SOURCE_LABEL[route.source] || { text: route.source?.toUpperCase(), color: 'var(--text-3)' }
  const scoring = route.scoring || {}
  const steps = route.procedure_steps_ru || []

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-md)',
      overflow: 'hidden',
      marginBottom: 10,
    }}>
      {/* Route header */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 14px', cursor: 'pointer',
          borderBottom: open ? '1px solid var(--border)' : 'none',
        }}
        onClick={() => setOpen(o => !o)}
      >
        <span style={{
          fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
          padding: '2px 7px', borderRadius: 4,
          background: `${src.color}18`, color: src.color,
          border: `1px solid ${src.color}40`, flexShrink: 0,
        }}>
          {src.text}
        </span>

        <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-2)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {route.reactants || '—'}
        </span>

        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--cyan)', flexShrink: 0 }}>
          {route.final_score?.toFixed(3)}
        </span>

        <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div style={{ padding: '12px 14px' }}>

          {/* Reaction equation */}
          {route.reaction_smiles && (() => {
            const parts = route.reaction_smiles.split('>>')
            const reactantStr = (parts[0] || '').split('.').filter(Boolean).join(' + ')
            const productStr = (parts[1] || '').split('.').filter(Boolean).join(' + ')
            return (
              <>
                <div className="section-title">Реакция</div>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-1)',
                  background: 'var(--bg-card)', padding: '10px 12px', borderRadius: 'var(--r-sm)',
                  border: '1px solid var(--border)', marginBottom: 12,
                  display: 'flex', flexDirection: 'column', gap: 4,
                }}>
                  <div style={{ color: 'var(--text-2)', wordBreak: 'break-all' }}>{reactantStr}</div>
                  <div style={{ color: 'var(--cyan)', fontSize: 14, fontWeight: 700 }}>↓</div>
                  <div style={{ color: 'var(--green)', wordBreak: 'break-all' }}>{productStr}</div>
                </div>
              </>
            )
          })()}

          {/* Reactants full */}
          <div className="section-title">Реагенты (SMILES)</div>
          <div className="smiles-box" style={{ wordBreak: 'break-all', marginBottom: 12 }}>
            {route.reactants || '—'}
          </div>

          {/* Conditions row */}
          {(route.temperature || route.solvent || route.catalyst || route.expected_yield != null) && (
            <>
              <div className="section-title">Условия</div>
              <div className="props-grid" style={{ marginBottom: 12 }}>
                {route.temperature && (
                  <div className="prop-item">
                    <div className="prop-label">Температура</div>
                    <div className="prop-value">{route.temperature}</div>
                  </div>
                )}
                {route.solvent && (
                  <div className="prop-item">
                    <div className="prop-label">Растворитель</div>
                    <div className="prop-value" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{route.solvent}</div>
                  </div>
                )}
                {route.catalyst && (
                  <div className="prop-item">
                    <div className="prop-label">Катализатор</div>
                    <div className="prop-value" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{route.catalyst}</div>
                  </div>
                )}
                {route.expected_yield != null && (
                  <div className="prop-item">
                    <div className="prop-label">Выход</div>
                    <div className="prop-value">{(route.expected_yield * 100).toFixed(0)}%</div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Scoring breakdown */}
          <div className="section-title">Оценка маршрута</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 16px', marginBottom: 12 }}>
            {[
              ['Итого', route.final_score],
              ['Достоверность', scoring.plausibility],
              ['Доступность', scoring.buyability],
              ['Простота', scoring.simplicity],
            ].map(([label, val]) => val != null && (
              <div key={label}>
                <div style={{ fontSize: 10, color: 'var(--text-3)', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.6px' }}>{label}</div>
                <ScoreBar value={val} />
              </div>
            ))}
          </div>

          {/* Step-by-step procedure */}
          {steps.length > 0 && (
            <>
              <div className="section-title">Процедура синтеза</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {steps.map((step, i) => (
                  <div key={i} style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    borderLeft: '3px solid var(--cyan-dim)',
                    borderRadius: '0 var(--r-sm) var(--r-sm) 0',
                    padding: '8px 12px',
                  }}>
                    <div style={{ fontSize: 11, color: 'var(--cyan)', fontFamily: 'var(--font-mono)', marginBottom: 3 }}>
                      Шаг {step.step}
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--text-1)', lineHeight: 1.5 }}>{step.description}</div>
                    {step.reason && step.reason !== 'ORD процедура' && (
                      <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>↳ {step.reason}</div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Raw procedure if no steps */}
          {steps.length === 0 && route.procedure_details && (
            <>
              <div className="section-title">Описание процедуры</div>
              <div className="description-text">{route.procedure_details}</div>
            </>
          )}

          {/* ORD ID */}
          {route.reaction_id && (
            <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
              ORD ID: {route.reaction_id}
            </div>
          )}

          {/* Tree expand button */}
          {smiles && route.reactants && (
            <div style={{ marginTop: 14 }}>
              {!tree && !treeLoading && (
                <button
                  style={{
                    background: 'var(--cyan)18',
                    border: '1px solid var(--cyan)40',
                    color: 'var(--cyan)',
                    padding: '8px 16px',
                    borderRadius: 'var(--r-sm)',
                    fontSize: 12,
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}
                  onClick={async () => {
                    setTreeLoading(true)
                    setTreeError(null)
                    try {
                      const res = await fetch(`${API_BASE}/tree/expand`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          smiles,
                          reactants: route.reactants,
                          max_depth: 6,
                          timeout_sec: 120,
                        }),
                      })
                      if (!res.ok) {
                        const err = await res.text()
                        throw new Error(`HTTP ${res.status}: ${err}`)
                      }
                      const data = await res.json()
                      setTree(data)
                    } catch (e) {
                      setTreeError(e.message)
                    } finally {
                      setTreeLoading(false)
                    }
                  }}
                >
                  ⬡ Построить дерево синтеза
                </button>
              )}

              {treeLoading && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--cyan)',
                }}>
                  <div className="spinner" style={{ width: 14, height: 14 }} />
                  Строим дерево синтеза...
                </div>
              )}

              {treeError && (
                <div style={{
                  fontSize: 12, color: 'var(--red)',
                  fontFamily: 'var(--font-mono)',
                  padding: '6px 10px',
                  background: 'var(--red)10',
                  border: '1px solid var(--red)30',
                  borderRadius: 'var(--r-sm)',
                }}>
                  Ошибка: {treeError}
                </div>
              )}

              {tree && (
                <>
                  <button
                    onClick={() => setGraphOpen(true)}
                    style={{
                      marginTop: 10,
                      background: 'var(--cyan)18',
                      border: '1px solid var(--cyan)40',
                      color: 'var(--cyan)',
                      padding: '8px 16px',
                      borderRadius: 'var(--r-sm)',
                      fontSize: 12,
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 6,
                    }}
                  >
                    ⬡ Открыть граф синтеза ({tree.stats?.total_nodes} узлов)
                  </button>
                  {graphOpen && (
                    <SynthesisGraph
                      tree={tree.tree}
                      stats={tree.stats}
                      onClose={() => setGraphOpen(false)}
                    />
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function RetroCard({ retroResult, smiles }) {
  if (!retroResult) {
    return (
      <div style={{ color: 'var(--text-3)', fontSize: 13, fontFamily: 'var(--font-mono)', padding: '8px 0' }}>
        Данные ретросинтеза недоступны
      </div>
    )
  }

  const routes = retroResult.routes || []
  const sources = retroResult.sources_used || []
  const total = retroResult.total_found || 0

  const SOURCE_LABEL_FULL = {
    ord:         'Open Reaction Database',
    retro_model: 'Template-relevance модель',
  }

  return (
    <div>
      {/* Meta */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div className="prop-item" style={{ minWidth: 120 }}>
          <div className="prop-label">Найдено маршрутов</div>
          <div className="prop-value">{total}</div>
        </div>
        <div className="prop-item" style={{ flex: 1, minWidth: 180 }}>
          <div className="prop-label">Источники</div>
          <div className="prop-value" style={{ fontSize: 12 }}>
            {sources.map(s => SOURCE_LABEL_FULL[s] || s).join(', ') || '—'}
          </div>
        </div>
      </div>

      {routes.length === 0 ? (
        <div style={{ color: 'var(--text-3)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>
          Маршруты синтеза не найдены
        </div>
      ) : (
        routes.map((route, i) => (
          <RouteCard key={i} route={route} index={i} smiles={smiles} />
        ))
      )}
    </div>
  )
}

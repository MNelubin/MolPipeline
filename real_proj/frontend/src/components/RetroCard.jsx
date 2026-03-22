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
  const level = pct > 70 ? 'high' : pct > 40 ? 'medium' : 'low'
  return (
    <div className="score-bar">
      <div className="score-track">
        <div className={`score-fill ${level}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="score-value">{value.toFixed(2)}</span>
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
    <div className="route-card">
      {/* Route header */}
      <div className="route-header" style={{ borderBottom: open ? '1px solid var(--border)' : 'none' }} onClick={() => setOpen(o => !o)}>
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

        <span className="route-reactants-text">{route.reactants || '—'}</span>
        <span className="route-score-text">{route.final_score?.toFixed(3)}</span>
        <span className="route-toggle">{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div className="route-body">

          {/* Reaction equation */}
          {route.reaction_smiles && (() => {
            const parts = route.reaction_smiles.split('>>')
            const reactantStr = (parts[0] || '').split('.').filter(Boolean).join(' + ')
            const productStr = (parts[1] || '').split('.').filter(Boolean).join(' + ')
            return (
              <>
                <div className="section-title">Реакция</div>
                <div className="route-reaction-box">
                  <div className="route-reaction-reactants">{reactantStr}</div>
                  <div className="route-reaction-arrow">↓</div>
                  <div className="route-reaction-products">{productStr}</div>
                </div>
              </>
            )
          })()}

          {/* Reactants SMILES */}
          <div className="section-title">Реагенты (SMILES)</div>
          <div className="smiles-box" style={{ wordBreak: 'break-all', marginBottom: 12 }}>
            {route.reactants || '—'}
          </div>

          {/* Conditions */}
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
          <div className="score-grid">
            {[
              ['Итого', route.final_score],
              ['Достоверность', scoring.plausibility],
              ['Доступность', scoring.buyability],
              ['Простота', scoring.simplicity],
            ].map(([label, val]) => val != null && (
              <div key={label}>
                <div className="score-label">{label}</div>
                <ScoreBar value={val} />
              </div>
            ))}
          </div>

          {/* Procedure steps */}
          {steps.length > 0 && (
            <>
              <div className="section-title">Процедура синтеза</div>
              <div className="procedure-list">
                {steps.map((step, i) => (
                  <div key={i} className="procedure-step">
                    <div className="procedure-step-num">Шаг {step.step}</div>
                    <div className="procedure-step-text">{step.description}</div>
                    {step.reason && step.reason !== 'ORD процедура' && (
                      <div className="procedure-step-reason">↳ {step.reason}</div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Raw procedure */}
          {steps.length === 0 && route.procedure_details && (
            <>
              <div className="section-title">Описание процедуры</div>
              <div className="description-text">{route.procedure_details}</div>
            </>
          )}

          {/* ORD ID */}
          {route.reaction_id && (
            <div className="ord-id-text">ORD ID: {route.reaction_id}</div>
          )}

          {/* Tree expand button */}
          {smiles && route.reactants && (
            <div style={{ marginTop: 14 }}>
              {!tree && !treeLoading && (
                <button
                  className="tree-btn"
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
                          max_depth: 20,
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
                <div className="tree-loading">
                  <div className="spinner" style={{ width: 14, height: 14 }} />
                  Строим дерево синтеза...
                </div>
              )}

              {treeError && (
                <div className="tree-error">Ошибка: {treeError}</div>
              )}

              {tree && (
                <>
                  <button className="tree-btn" style={{ marginTop: 10 }} onClick={() => setGraphOpen(true)}>
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
    return <div className="retro-empty">Данные ретросинтеза недоступны</div>
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
        <div className="retro-empty">Маршруты синтеза не найдены</div>
      ) : (
        routes.map((route, i) => (
          <RouteCard key={i} route={route} index={i} smiles={smiles} />
        ))
      )}
    </div>
  )
}

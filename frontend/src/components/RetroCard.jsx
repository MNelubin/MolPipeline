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

const SOURCE_LABEL = {
  ord:          { text: 'ORD',   color: 'var(--green)' },
  retro_model:  { text: 'MODEL', color: 'var(--purple)' },
  web:          { text: 'WEB',   color: 'var(--cyan)' },
  aizynthfinder:{ text: 'AIZYNTH', color: 'var(--amber)' },
}

const SOURCE_LABEL_FULL = {
  ord: 'Open Reaction Database',
  retro_model: 'Template-relevance model',
  web: 'Web Search (PubMed + DuckDuckGo)',
  aizynthfinder: 'AiZynthFinder',
}

const SEARCH_MODE_LABEL = {
  auto: 'Auto',
  ord: 'ORD only',
  retro_model: 'ASKCOS-derived model only',
  web: 'Web only',
  aizynthfinder: 'AiZynthFinder only',
  all: 'All enabled sources',
}

const AVAILABILITY_LEVEL_LABEL = {
  catalog: 'В каталоге',
  common_lab_reagent: 'Обычный реагент',
  heuristic_likely: 'Вероятно доступен',
  not_found: 'Не найден',
  invalid: 'Ошибка',
}

const TREE_STATUS_LABEL = {
  intermediate: 'Промежуточный узел',
  buyable: 'Доступен',
  restricted: 'Доступен с ограничениями',
  unresolved: 'Не раскрыт',
  banned: 'Заблокирован',
  depth_limit: 'Лимит глубины',
  timeout: 'Таймаут',
  circular: 'Цикл',
  invalid_smiles: 'Ошибка SMILES',
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

function formatTreeName(node) {
  return node.name || node.smiles || 'unknown'
}

function TreeNode({ node, isRoot = false }) {
  if (!node) return null
  const children = node.children || []
  const route = node.route || null
  const status = node.status || 'unknown'

  return (
    <div className={`retro-tree-node status-${status}${isRoot ? ' root' : ''}`}>
      <div className="retro-tree-node-main">
        <span className="retro-tree-depth">d{node.depth ?? 0}</span>
        <span className={`retro-tree-status status-${status}`}>
          {isRoot ? 'Цель' : (TREE_STATUS_LABEL[status] || status)}
        </span>
        <div className="retro-tree-title">
          <strong>{formatTreeName(node)}</strong>
          {node.smiles && node.name && <code>{node.smiles}</code>}
        </div>
      </div>

      {route?.reactants && (
        <div className="retro-tree-route">
          <span>Разбиение</span>
          <code>{route.reactants}</code>
        </div>
      )}

      {node.guard?.status && node.guard.status !== 'clear' && (
        <div className="retro-tree-warning">
          {node.guard.reason || node.guard.status}
        </div>
      )}

      {children.length > 0 && (
        <div className="retro-tree-children">
          {children.map((child, index) => (
            <TreeNode key={`${child.smiles || child.status}-${index}`} node={child} />
          ))}
        </div>
      )}
    </div>
  )
}

function MultiStepTreeBlock({ treeResult }) {
  const [showGraph, setShowGraph] = useState(false)
  const tree = treeResult?.tree
  const stats = treeResult?.stats || {}
  if (!tree) return null
  const isShallow = (stats.max_depth_reached ?? 0) <= 1 && (stats.unresolved_count ?? 0) > 0

  return (
    <div className="retro-tree-block">
      <div className="retro-tree-header">
        <div>
          <div className="section-title">Многостадийное дерево ретросинтеза</div>
          <p>Расширение лучшего маршрута до доступных или нераскрытых исходных веществ.</p>
        </div>
        <div className="retro-tree-stats">
          <span>узлов: {stats.total_nodes ?? 'n/a'}</span>
          <span>доступно: {stats.buyable_count ?? 0}</span>
          <span>не раскрыто: {stats.unresolved_count ?? 0}</span>
          <span>глубина: {stats.max_depth_reached ?? 0}</span>
        </div>
        <button type="button" className="retro-tree-graph-btn" onClick={() => setShowGraph(true)}>
          Открыть граф
        </button>
      </div>

      {isShallow && (
        <div className="retro-tree-warning">
          Дерево пока раскрыто только на один шаг: планировщик не нашел надежного продолжения для крупного интермедиата.
        </div>
      )}

      {treeResult.selected_route?.reactants && (
        <div className="retro-tree-selected">
          <span>Выбранный one-step маршрут</span>
          <code>{treeResult.selected_route.reactants}</code>
        </div>
      )}

      <div className="retro-tree-preview">
        <span>Граф использует общий компонент дерева синтеза из вкладки анализа молекул.</span>
        <code>{tree.name || tree.smiles}</code>
      </div>

      {showGraph && (
        <SynthesisGraph tree={tree} stats={stats} onClose={() => setShowGraph(false)} />
      )}
    </div>
  )
}

function RouteCard({ route, index }) {
  const [open, setOpen] = useState(index === 0)
  const src = SOURCE_LABEL[route.source] || { text: route.source?.toUpperCase(), color: 'var(--text-3)' }
  const scoring = route.scoring || {}
  const steps = route.procedure_steps_ru || []
  const availability = route.reactant_availability || []
  const availabilitySummary = route.availability_summary || null

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

          {availabilitySummary && (
            <>
              <div className="section-title">Доступность реагентов</div>
              <div className="route-availability-summary">
                <span>{availabilitySummary.available_count}/{availabilitySummary.total} доступны</span>
                <span>Каталог: {availabilitySummary.catalog_count}</span>
                <span>С ценой: {availabilitySummary.priced_count}</span>
                {availabilitySummary.estimated_total_1g_usd != null && (
                  <span>Оценка 1 г: ${availabilitySummary.estimated_total_1g_usd}</span>
                )}
              </div>
              {availability.length > 0 && (
                <div className="route-availability-list">
                  {availability.map((item, i) => (
                    <div key={`${item.input}-${i}`} className={`route-availability-item level-${item.availability_level}`}>
                      <code>{item.canonical_smiles || item.input}</code>
                      <span>{AVAILABILITY_LEVEL_LABEL[item.availability_level] || item.availability_level}</span>
                      <small>
                        {item.source_label || item.basis}
                        {item.ppg != null ? ` · $${item.ppg}/г` : ''}
                      </small>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

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

        </div>
      )}
    </div>
  )
}

export default function RetroCard({ retroResult }) {
  if (!retroResult) {
    return <div className="retro-empty">Данные ретросинтеза недоступны</div>
  }

  const routes = retroResult.routes || []
  const sources = retroResult.sources_used || []
  const total = retroResult.total_found || 0

  const totalUnique = retroResult.total_unique || routes.length
  const searchMode = retroResult.source_mode || 'auto'
  const multiStepTree = retroResult.multi_step_tree

  return (
    <div>
      {/* Meta */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div className="prop-item" style={{ minWidth: 120 }}>
          <div className="prop-label">Найдено маршрутов</div>
          <div className="prop-value">{total}</div>
        </div>
        <div className="prop-item" style={{ minWidth: 120 }}>
          <div className="prop-label">Unique routes</div>
          <div className="prop-value">{totalUnique}</div>
        </div>
        <div className="prop-item" style={{ minWidth: 160 }}>
          <div className="prop-label">Search mode</div>
          <div className="prop-value" style={{ fontSize: 12 }}>
            {SEARCH_MODE_LABEL[searchMode] || searchMode}
          </div>
        </div>
        <div className="prop-item" style={{ flex: 1, minWidth: 180 }}>
          <div className="prop-label">Источники</div>
          <div className="prop-value" style={{ fontSize: 12 }}>
            {sources.map(s => SOURCE_LABEL_FULL[s] || s).join(', ') || '—'}
          </div>
        </div>
      </div>

      {multiStepTree && <MultiStepTreeBlock treeResult={multiStepTree} />}

      {routes.length === 0 ? (
        <div className="retro-empty">Маршруты синтеза не найдены</div>
      ) : (
        routes.map((route, i) => (
          <RouteCard key={i} route={route} index={i} />
        ))
      )}
    </div>
  )
}

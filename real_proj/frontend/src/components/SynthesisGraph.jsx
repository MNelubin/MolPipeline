/**
 * SynthesisGraph — full-screen interactive graph of the retrosynthesis tree.
 *
 * Opens as a fixed overlay. Click a node → detail panel slides in with:
 *   - 2D structure, SMILES, name
 *   - Safety / guard status
 *   - Reaction conditions (temperature, solvent, yield, score)
 *   - Procedure steps
 *
 * Color coding:
 *   green  = buyable | red = banned | cyan = intermediate | amber = unresolved
 */

import { useState, useCallback, useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'

const SC = {
  buyable:      { color: '#22d3a0', label: 'Покупаемый',    icon: '✓' },
  restricted:   { color: '#ffe033', label: 'Предупреждение', icon: '⚠' },
  banned:       { color: '#f05050', label: 'Запрещён',      icon: '✕' },
  intermediate: { color: '#06d6f0', label: 'Промежуточный', icon: '◆' },
  unresolved:   { color: '#ffe033', label: 'Не найден',     icon: '?' },
  depth_limit:  { color: '#ffe033', label: 'Лимит глубины', icon: '↓' },
  timeout:      { color: '#ffe033', label: 'Таймаут',       icon: '⏱' },
  circular:     { color: '#f4a522', label: 'Цикл',          icon: '↻' },
  invalid_smiles:{ color: '#f05050', label: 'Невалидный',   icon: '!' },
}

function MolNode({ data, selected }) {
  const cfg = SC[data.status] || SC.unresolved
  return (
    <div
      className={`graph-node${selected ? ' selected' : ''}`}
      style={{
        borderColor: selected ? cfg.color : cfg.color + '55',
        borderLeft: `4px solid ${cfg.color}`,
        width: 210,
        boxShadow: selected ? `0 0 16px ${cfg.color}35` : 'var(--shadow-card)',
      }}
    >
      <Handle type="target" position={Position.Top}
        style={{ background: cfg.color, border: 'none', width: 8, height: 8 }} />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
        <span
          className="graph-node-badge"
          style={{
            background: cfg.color + '18',
            color: cfg.color,
            border: `1px solid ${cfg.color}40`,
          }}
        >
          {cfg.icon} {cfg.label}
        </span>
        <span style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>d{data.depth}</span>
      </div>

      <div className="graph-node-name" style={{ fontSize: 12, marginBottom: 2 }}>
        {data.name || data.smiles?.slice(0, 26) || '—'}
      </div>

      {data.smiles && (
        <div className="graph-node-smiles">{data.smiles}</div>
      )}

      {data.route?.final_score != null && (
        <div style={{ marginTop: 6 }}>
          <div className="score-bar">
            <div className="score-track" style={{ height: 3 }}>
              <div
                className={`score-fill ${data.route.final_score > 0.7 ? 'high' : data.route.final_score > 0.4 ? 'medium' : 'low'}`}
                style={{ width: `${Math.round(data.route.final_score * 100)}%` }}
              />
            </div>
            <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
              {data.route.final_score.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      <Handle type="source" position={Position.Bottom}
        style={{ background: cfg.color, border: 'none', width: 8, height: 8 }} />
    </div>
  )
}

const NODE_TYPES = { mol: MolNode }

const NW = 210, NH = 100, HGAP = 50, VGAP = 90

function subtreeWidth(node) {
  if (!node.children?.length) return NW
  return Math.max(NW,
    node.children.reduce((s, c, i) => s + subtreeWidth(c) + (i > 0 ? HGAP : 0), 0))
}

function buildGraph(tree) {
  const nodes = [], edges = []
  let uid = 0

  function place(node, cx, y, parentId) {
    const id = `n${uid++}`
    nodes.push({ id, type: 'mol', position: { x: cx - NW / 2, y }, data: node })

    if (parentId) {
      const cfg = SC[node.status] || SC.unresolved
      edges.push({
        id: `e${parentId}-${id}`,
        source: parentId, target: id,
        type: 'smoothstep',
        style: { stroke: cfg.color + '70', strokeWidth: 1.5 },
        markerEnd: { type: 'arrowclosed', color: cfg.color + '70', width: 10, height: 10 },
      })
    }

    const children = node.children || []
    if (children.length) {
      const widths = children.map(subtreeWidth)
      const total = widths.reduce((s, w) => s + w, 0) + HGAP * (children.length - 1)
      let x = cx - total / 2
      children.forEach((child, i) => {
        place(child, x + widths[i] / 2, y + NH + VGAP, id)
        x += widths[i] + HGAP
      })
    }
  }

  place(tree, 0, 0, null)
  return { nodes, edges }
}

function Prop({ label, value }) {
  return (
    <div className="synth-detail-prop">
      <div className="synth-detail-prop-label">{label}</div>
      <div className="synth-detail-prop-value">{value}</div>
    </div>
  )
}

function DetailPanel({ node, onClose }) {
  if (!node) return null
  const d = node.data
  const cfg = SC[d.status] || SC.unresolved
  const route = d.route || {}
  const guard = d.guard || {}

  const img2dUrl = d.smiles
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(d.smiles)}/PNG?image_size=280x180`
    : null

  return (
    <div className="synth-detail-panel" style={{ borderLeft: `2px solid ${cfg.color}40` }}>
      <div className="synth-detail-content">

        <div className="synth-detail-header">
          <div style={{ flex: 1 }}>
            <span
              className="graph-node-badge"
              style={{
                background: cfg.color + '18',
                color: cfg.color,
                border: `1px solid ${cfg.color}40`,
              }}
            >
              {cfg.icon} {cfg.label}
            </span>
            <div className="synth-detail-name">{d.name || '—'}</div>
            <div className="synth-detail-depth">Глубина: {d.depth}</div>
          </div>
          <button className="synth-detail-close" onClick={onClose}>✕</button>
        </div>

        {d.smiles && (
          <>
            <div className="synth-detail-title">SMILES</div>
            <div className="synth-detail-code">{d.smiles}</div>
          </>
        )}

        {img2dUrl && (
          <>
            <div className="synth-detail-title">Структура</div>
            <div className="synth-detail-img">
              <img src={img2dUrl} alt={d.name || d.smiles}
                onError={e => { e.target.parentElement.style.display = 'none' }} />
            </div>
          </>
        )}

        <div className="synth-detail-title">Безопасность</div>
        <div
          className="synth-detail-safety"
          style={{
            background: cfg.color + '10',
            border: `1px solid ${cfg.color}30`,
            color: cfg.color,
          }}
        >
          {guard.status === 'banned' || guard.status === 'restricted'
            ? `⚠ ${guard.reason || 'Запрещённое вещество'}`
            : '✓ Не найдено в списке запрещённых'}
        </div>

        {route.reactants && (
          <>
            <div className="synth-detail-title">Реакция</div>
            <div style={{ marginBottom: 4 }}>
              {route.source && (
                <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginBottom: 6 }}>
                  Источник:{' '}
                  <span style={{
                    color: route.source === 'ord' ? 'var(--green)' : 'var(--purple)',
                    fontWeight: 700,
                  }}>
                    {route.source === 'ord' ? 'ORD' : route.source === 'retro_model' ? 'MODEL' : route.source.toUpperCase()}
                  </span>
                  {route.final_score != null && (
                    <span style={{ color: 'var(--cyan)', marginLeft: 10 }}>
                      score: {route.final_score.toFixed(3)}
                    </span>
                  )}
                </div>
              )}
              <div className="synth-detail-prop-label">Реагенты</div>
              <div className="synth-detail-code">{route.reactants}</div>
            </div>
          </>
        )}

        {(route.temperature || route.solvent || route.catalyst || route.expected_yield != null) && (
          <>
            <div className="synth-detail-title">Условия реакции</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {route.temperature && <Prop label="Температура" value={route.temperature} />}
              {route.solvent && <Prop label="Растворитель" value={route.solvent} />}
              {route.catalyst && <Prop label="Катализатор" value={route.catalyst} />}
              {route.expected_yield != null && (
                <Prop label="Выход" value={`${(route.expected_yield * 100).toFixed(0)}%`} />
              )}
            </div>
          </>
        )}

        {route.procedure_steps_ru?.length > 0 && (
          <>
            <div className="synth-detail-title">Процедура синтеза</div>
            <div className="procedure-list">
              {route.procedure_steps_ru.map((step, i) => (
                <div key={i} className="procedure-step">
                  <div className="procedure-step-num">Шаг {step.step}</div>
                  <div className="procedure-step-text" style={{ fontSize: 12 }}>{step.description}</div>
                  {step.reason && step.reason !== 'ORD процедура' && (
                    <div className="procedure-step-reason">↳ {step.reason}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {!route.procedure_steps_ru?.length && route.procedure_details && (
          <>
            <div className="synth-detail-title">Описание процедуры</div>
            <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6 }}>
              {route.procedure_details}
            </div>
          </>
        )}

        {route.reaction_id && (
          <div className="ord-id-text" style={{ marginTop: 12 }}>
            ORD ID: {route.reaction_id}
          </div>
        )}
      </div>
    </div>
  )
}

export default function SynthesisGraph({ tree, stats, onClose }) {
  const { nodes: initNodes, edges: initEdges } = useMemo(() => buildGraph(tree), [tree])
  const [nodes, , onNodesChange] = useNodesState(initNodes)
  const [edges, , onEdgesChange] = useEdgesState(initEdges)
  const [selected, setSelected] = useState(null)

  const onNodeClick = useCallback((_, node) => {
    setSelected(prev => prev?.id === node.id ? null : node)
  }, [])

  const onPaneClick = useCallback(() => setSelected(null), [])

  return (
    <div className="synth-overlay">
      {/* Header */}
      <div className="synth-header">
        <span className="synth-title">⬡ Дерево синтеза</span>

        {stats && (
          <div className="synth-stats">
            <span>
              <span className="synth-stat-label">Узлов: </span>
              <span className="synth-stat-value">{stats.total_nodes}</span>
            </span>
            <span>
              <span style={{ color: 'var(--green)' }}>Покупаемых: </span>
              <span className="synth-stat-value">{stats.buyable_count}</span>
            </span>
            {stats.banned_count > 0 && (
              <span>
                <span style={{ color: 'var(--red)' }}>Запрещённых: </span>
                <span className="synth-stat-value">{stats.banned_count}</span>
              </span>
            )}
            {stats.unresolved_count > 0 && (
              <span>
                <span style={{ color: 'var(--amber)' }}>Не найдено: </span>
                <span className="synth-stat-value">{stats.unresolved_count}</span>
              </span>
            )}
            <span>
              <span className="synth-stat-label">Глубина: </span>
              <span className="synth-stat-value">{stats.max_depth_reached}</span>
            </span>
            <span>
              <span className="synth-stat-label">Время: </span>
              <span className="synth-stat-value">{stats.elapsed_sec}с</span>
            </span>
          </div>
        )}

        {selected && (
          <span className="synth-hint">← кликни по пустому месту чтобы сбросить выбор</span>
        )}

        <button className="synth-close-btn" onClick={onClose}>✕ Закрыть</button>
      </div>

      {/* Body: graph + side panel */}
      <div className="synth-body">
        <div className="synth-graph-area">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            minZoom={0.08}
            maxZoom={2.5}
            style={{ background: 'var(--bg-0)' }}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#1a2740" gap={24} size={1} />
            <Controls
              style={{
                background: 'var(--bg-1)', border: '1px solid var(--border)',
                borderRadius: 6, overflow: 'hidden',
              }}
            />
          </ReactFlow>
        </div>

        {selected && (
          <DetailPanel node={selected} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  )
}

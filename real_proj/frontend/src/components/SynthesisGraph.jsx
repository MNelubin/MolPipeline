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

// ── Config ────────────────────────────────────────────────────────────────────

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

// ── Custom node ───────────────────────────────────────────────────────────────

function MolNode({ data, selected }) {
  const cfg = SC[data.status] || SC.unresolved
  return (
    <div style={{
      background: selected ? '#1a2740' : '#0f1929',
      border: `2px solid ${selected ? cfg.color : cfg.color + '55'}`,
      borderLeft: `4px solid ${cfg.color}`,
      borderRadius: 8,
      padding: '8px 12px',
      width: 210,
      cursor: 'pointer',
      boxShadow: selected ? `0 0 16px ${cfg.color}35` : '0 2px 8px #00000060',
      transition: 'border-color 0.2s, box-shadow 0.2s',
    }}>
      <Handle type="target" position={Position.Top}
        style={{ background: cfg.color, border: 'none', width: 8, height: 8 }} />

      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
        <span style={{
          fontSize: 9, fontFamily: 'monospace', fontWeight: 700,
          padding: '1px 6px', borderRadius: 3,
          background: cfg.color + '18', color: cfg.color,
          border: `1px solid ${cfg.color}40`,
        }}>
          {cfg.icon} {cfg.label}
        </span>
        <span style={{ fontSize: 9, color: '#4d6585', fontFamily: 'monospace' }}>d{data.depth}</span>
      </div>

      {/* Name */}
      <div style={{
        fontSize: 12, fontWeight: 600, color: '#e8edf5',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        marginBottom: 2,
      }}>
        {data.name || data.smiles?.slice(0, 26) || '—'}
      </div>

      {/* SMILES */}
      {data.smiles && (
        <div style={{
          fontSize: 9, fontFamily: 'monospace', color: '#4d6585',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {data.smiles}
        </div>
      )}

      {/* Score bar */}
      {data.route?.final_score != null && (
        <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{
            flex: 1, height: 3, background: '#1e3050', borderRadius: 2, overflow: 'hidden',
          }}>
            <div style={{
              width: `${Math.round(data.route.final_score * 100)}%`,
              height: '100%', background: '#06d6f0', borderRadius: 2,
            }} />
          </div>
          <span style={{ fontSize: 9, fontFamily: 'monospace', color: '#06d6f0' }}>
            {data.route.final_score.toFixed(2)}
          </span>
        </div>
      )}

      <Handle type="source" position={Position.Bottom}
        style={{ background: cfg.color, border: 'none', width: 8, height: 8 }} />
    </div>
  )
}

const NODE_TYPES = { mol: MolNode }

// ── Tree layout ───────────────────────────────────────────────────────────────

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

// ── Detail panel ──────────────────────────────────────────────────────────────

const T = {
  title: {
    fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
    color: '#4d6585', textTransform: 'uppercase', letterSpacing: '0.8px',
    marginBottom: 6, marginTop: 12,
  },
  code: {
    fontFamily: 'monospace', fontSize: 11, color: '#8fa3bf',
    background: '#121d2e', padding: '6px 10px', borderRadius: 6,
    wordBreak: 'break-all', border: '1px solid #1e3050',
  },
}

function Prop({ label, value }) {
  return (
    <div style={{
      background: '#121d2e', border: '1px solid #1e3050',
      borderRadius: 6, padding: '5px 10px',
    }}>
      <div style={{ fontSize: 9, color: '#4d6585', fontFamily: 'monospace', marginBottom: 2, textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ fontSize: 12, color: '#e8edf5', fontFamily: 'monospace' }}>{value}</div>
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
    <div style={{
      width: 340, height: '100%', flexShrink: 0,
      background: '#0d1520',
      borderLeft: `2px solid ${cfg.color}40`,
      overflow: 'auto',
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ padding: '14px 16px', flex: 1 }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
          <div style={{ flex: 1 }}>
            <span style={{
              fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
              padding: '2px 8px', borderRadius: 4,
              background: cfg.color + '18', color: cfg.color,
              border: `1px solid ${cfg.color}40`,
            }}>
              {cfg.icon} {cfg.label}
            </span>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#e8edf5', marginTop: 6, lineHeight: 1.3 }}>
              {d.name || '—'}
            </div>
            <div style={{ fontSize: 11, color: '#4d6585', fontFamily: 'monospace', marginTop: 2 }}>
              Глубина: {d.depth}
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', color: '#4d6585',
            fontSize: 18, cursor: 'pointer', padding: '2px 6px', lineHeight: 1,
          }}>
            ✕
          </button>
        </div>

        {/* SMILES */}
        {d.smiles && (
          <>
            <div style={T.title}>SMILES</div>
            <div style={{ ...T.code, marginBottom: 0 }}>{d.smiles}</div>
          </>
        )}

        {/* 2D structure */}
        {img2dUrl && (
          <>
            <div style={T.title}>Структура</div>
            <div style={{
              background: '#fff', borderRadius: 8, overflow: 'hidden',
              border: '1px solid #1e3050', lineHeight: 0,
            }}>
              <img src={img2dUrl} alt={d.name || d.smiles}
                style={{ width: '100%', display: 'block' }}
                onError={e => { e.target.parentElement.style.display = 'none' }} />
            </div>
          </>
        )}

        {/* Safety */}
        <div style={T.title}>Безопасность</div>
        <div style={{
          padding: '8px 10px', borderRadius: 6,
          background: cfg.color + '10', border: `1px solid ${cfg.color}30`,
          fontSize: 12, color: cfg.color,
        }}>
          {guard.status === 'banned' || guard.status === 'restricted'
            ? `⚠ ${guard.reason || 'Запрещённое вещество'}`
            : '✓ Не найдено в списке запрещённых'}
        </div>

        {/* Reaction */}
        {route.reactants && (
          <>
            <div style={T.title}>Реакция</div>
            <div style={{ marginBottom: 4 }}>
              {route.source && (
                <div style={{ fontSize: 11, fontFamily: 'monospace', color: '#4d6585', marginBottom: 6 }}>
                  Источник:{' '}
                  <span style={{
                    color: route.source === 'ord' ? '#22d3a0' : '#a78bfa',
                    fontWeight: 700,
                  }}>
                    {route.source === 'ord' ? 'ORD' : route.source === 'retro_model' ? 'MODEL' : route.source.toUpperCase()}
                  </span>
                  {route.final_score != null && (
                    <span style={{ color: '#06d6f0', marginLeft: 10 }}>
                      score: {route.final_score.toFixed(3)}
                    </span>
                  )}
                </div>
              )}
              <div style={{ fontSize: 10, color: '#4d6585', fontFamily: 'monospace', marginBottom: 4, textTransform: 'uppercase' }}>
                Реагенты
              </div>
              <div style={T.code}>{route.reactants}</div>
            </div>
          </>
        )}

        {/* Conditions */}
        {(route.temperature || route.solvent || route.catalyst || route.expected_yield != null) && (
          <>
            <div style={T.title}>Условия реакции</div>
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

        {/* Procedure steps */}
        {route.procedure_steps_ru?.length > 0 && (
          <>
            <div style={T.title}>Процедура синтеза</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {route.procedure_steps_ru.map((step, i) => (
                <div key={i} style={{
                  background: '#121d2e', border: '1px solid #1e3050',
                  borderLeft: '3px solid #0a9fb8',
                  borderRadius: '0 6px 6px 0', padding: '7px 10px',
                }}>
                  <div style={{ fontSize: 10, color: '#06d6f0', fontFamily: 'monospace', marginBottom: 3 }}>
                    Шаг {step.step}
                  </div>
                  <div style={{ fontSize: 12, color: '#e8edf5', lineHeight: 1.5 }}>{step.description}</div>
                  {step.reason && step.reason !== 'ORD процедура' && (
                    <div style={{ fontSize: 11, color: '#4d6585', marginTop: 3 }}>↳ {step.reason}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {/* Raw procedure */}
        {!route.procedure_steps_ru?.length && route.procedure_details && (
          <>
            <div style={T.title}>Описание процедуры</div>
            <div style={{ fontSize: 12, color: '#8fa3bf', lineHeight: 1.6 }}>
              {route.procedure_details}
            </div>
          </>
        )}

        {/* ORD ID */}
        {route.reaction_id && (
          <div style={{ fontSize: 10, color: '#4d6585', fontFamily: 'monospace', marginTop: 12 }}>
            ORD ID: {route.reaction_id}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

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
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: '#070b12',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* ── Header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0,
        padding: '10px 20px',
        background: '#0d1520',
        borderBottom: '1px solid #1e3050',
      }}>
        <span style={{ fontSize: 15, fontWeight: 700, color: '#e8edf5', fontFamily: 'var(--font-ui, sans-serif)' }}>
          ⬡ Дерево синтеза
        </span>

        {stats && (
          <div style={{ display: 'flex', gap: 14, fontSize: 12, fontFamily: 'monospace', flexWrap: 'wrap' }}>
            <span>
              <span style={{ color: '#4d6585' }}>Узлов: </span>
              <span style={{ color: '#e8edf5' }}>{stats.total_nodes}</span>
            </span>
            <span>
              <span style={{ color: '#22d3a0' }}>Покупаемых: </span>
              <span style={{ color: '#e8edf5' }}>{stats.buyable_count}</span>
            </span>
            {stats.banned_count > 0 && (
              <span>
                <span style={{ color: '#f05050' }}>Запрещённых: </span>
                <span style={{ color: '#e8edf5' }}>{stats.banned_count}</span>
              </span>
            )}
            {stats.unresolved_count > 0 && (
              <span>
                <span style={{ color: '#f4a522' }}>Не найдено: </span>
                <span style={{ color: '#e8edf5' }}>{stats.unresolved_count}</span>
              </span>
            )}
            <span>
              <span style={{ color: '#4d6585' }}>Глубина: </span>
              <span style={{ color: '#e8edf5' }}>{stats.max_depth_reached}</span>
            </span>
            <span>
              <span style={{ color: '#4d6585' }}>Время: </span>
              <span style={{ color: '#e8edf5' }}>{stats.elapsed_sec}с</span>
            </span>
          </div>
        )}

        {selected && (
          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#06d6f0' }}>
            ← кликни по пустому месту чтобы сбросить выбор
          </span>
        )}

        <button
          onClick={onClose}
          style={{
            marginLeft: 'auto', background: '#1a2740',
            border: '1px solid #1e3050', color: '#8fa3bf',
            padding: '6px 16px', borderRadius: 6, fontSize: 13,
            cursor: 'pointer', fontFamily: 'monospace',
          }}
        >
          ✕ Закрыть
        </button>
      </div>

      {/* ── Body: graph + side panel ── */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>

        {/* Graph */}
        <div style={{ flex: 1, minWidth: 0 }}>
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
            style={{ background: '#070b12' }}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#1a2740" gap={24} size={1} />
            <Controls
              style={{
                background: '#0d1520', border: '1px solid #1e3050',
                borderRadius: 6, overflow: 'hidden',
              }}
            />
          </ReactFlow>
        </div>

        {/* Detail panel */}
        {selected && (
          <DetailPanel node={selected} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  )
}

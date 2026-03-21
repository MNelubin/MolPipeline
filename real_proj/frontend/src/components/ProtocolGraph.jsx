/**
 * ProtocolGraph — inline ReactFlow graph built from experiment_protocol.reaction_sections.
 *
 * Shows the synthesis route as a directed graph:
 *   buyable starting materials (green) → intermediates (purple) → target (cyan)
 *
 * Clicking a node shows its 2D structure and details below the graph.
 * Collapsed by default on mobile, open on desktop.
 */

import { useState, useMemo, useCallback } from 'react'
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'

// ── Status colour config ───────────────────────────────────────────────────────

const SC = {
  target:       { color: '#06d6f0', label: 'Целевой',       icon: '★' },
  intermediate: { color: '#a78bfa', label: 'Промежуточный', icon: '◆' },
  buyable:      { color: '#22d3a0', label: 'Коммерч.',      icon: '✓' },
}

// ── Custom node ───────────────────────────────────────────────────────────────

function MolNode({ data, selected }) {
  const cfg = SC[data.status] || SC.intermediate
  const imgUrl = data.smiles
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(data.smiles)}/PNG?image_size=160x100`
    : null

  return (
    <div style={{
      background: selected ? '#1a2740' : '#0f1929',
      border: `1.5px solid ${selected ? cfg.color : cfg.color + '50'}`,
      borderLeft: `3px solid ${cfg.color}`,
      borderRadius: 8,
      padding: '8px 10px',
      width: 175,
      cursor: 'pointer',
      boxShadow: selected ? `0 0 14px ${cfg.color}45` : '0 2px 8px #00000060',
      transition: 'border-color 0.15s, box-shadow 0.15s',
    }}>
      <Handle type="target" position={Position.Top}
        style={{ background: cfg.color, border: 'none', width: 7, height: 7 }} />

      {/* Molecule image */}
      {imgUrl && (
        <div style={{
          background: '#fff', borderRadius: 4, marginBottom: 6,
          lineHeight: 0, overflow: 'hidden',
        }}>
          <img
            src={imgUrl}
            alt={data.name || data.smiles}
            style={{ width: '100%', height: 65, objectFit: 'contain', display: 'block' }}
            onError={e => { e.target.parentElement.style.display = 'none' }}
          />
        </div>
      )}

      {/* Status badge */}
      <div style={{ marginBottom: 4 }}>
        <span style={{
          fontSize: 9, fontFamily: 'monospace', fontWeight: 700,
          padding: '1px 6px', borderRadius: 3,
          background: cfg.color + '18', color: cfg.color,
          border: `1px solid ${cfg.color}35`,
        }}>
          {cfg.icon} {cfg.label}
        </span>
      </div>

      {/* Name */}
      <div style={{
        fontSize: 11, fontWeight: 600, color: '#e8edf5',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>
        {data.name || data.smiles?.slice(0, 24) || '—'}
      </div>

      {/* SMILES */}
      {data.smiles && (
        <div style={{
          fontSize: 9, fontFamily: 'monospace', color: '#4d6585',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          marginTop: 2,
        }}>
          {data.smiles}
        </div>
      )}

      <Handle type="source" position={Position.Bottom}
        style={{ background: cfg.color, border: 'none', width: 7, height: 7 }} />
    </div>
  )
}

const NODE_TYPES = { mol: MolNode }

// ── Graph builder ─────────────────────────────────────────────────────────────

const NW = 175, NH = 145, HGAP = 45, VGAP = 170

function buildGraph(sections) {
  if (!sections?.length) return { nodes: [], edges: [] }

  const sorted = [...sections].sort((a, b) => a.step_number - b.step_number)
  const lastStep = sorted[sorted.length - 1].step_number

  // Collect all unique molecules with metadata
  const molMeta = new Map() // smiles → { name, status, producedAtStepIdx }

  const edgePairs = [] // { from, to, label }

  for (let si = 0; si < sorted.length; si++) {
    const sec = sorted[si]
    const isTarget = sec.step_number === lastStep
    const pSmi = sec.product_smiles

    if (pSmi && !molMeta.has(pSmi)) {
      molMeta.set(pSmi, {
        name: sec.product_name || pSmi.slice(0, 20),
        status: isTarget ? 'target' : 'intermediate',
        producedAtStepIdx: si,
      })
    }

    // Prefer reagent_table (has names + is_leaf); fallback to parsing reaction_smiles
    const reagentSmiles = []
    if (sec.reagent_table?.length) {
      for (const r of sec.reagent_table) {
        if (!r.smiles) continue
        if (!molMeta.has(r.smiles)) {
          molMeta.set(r.smiles, {
            name: r.name || r.smiles.slice(0, 20),
            status: r.is_leaf !== false ? 'buyable' : 'intermediate',
            producedAtStepIdx: null,
          })
        }
        reagentSmiles.push(r.smiles)
      }
    } else if (sec.reaction_smiles?.includes('>>')) {
      const reactantPart = sec.reaction_smiles.split('>>')[0]
      for (const smi of reactantPart.split('.').filter(Boolean)) {
        if (!molMeta.has(smi)) {
          molMeta.set(smi, {
            name: smi.slice(0, 20),
            status: 'buyable',
            producedAtStepIdx: null,
          })
        }
        reagentSmiles.push(smi)
      }
    }

    for (const smi of reagentSmiles) {
      if (pSmi) edgePairs.push({ from: smi, to: pSmi, label: `ст. ${sec.step_number}` })
    }
  }

  // ── Layout: row = synthesis order ──
  // Reagents used in step i → row i
  // Product of step i → row i+1
  // Target → last row

  const rowMap = new Map() // row → smiles[]

  for (const [smi, meta] of molMeta) {
    let row
    if (meta.status === 'target') {
      row = sorted.length  // bottom
    } else if (meta.producedAtStepIdx != null) {
      row = meta.producedAtStepIdx + 1
    } else {
      // Buyable: find earliest step that uses it
      let found = sorted.length - 1
      for (let si = 0; si < sorted.length; si++) {
        const sec = sorted[si]
        const inTable = (sec.reagent_table || []).some(r => r.smiles === smi)
        const inSmiles = sec.reaction_smiles?.split('>>')[0]?.split('.').includes(smi)
        if (inTable || inSmiles) { found = si; break }
      }
      row = found
    }
    if (!rowMap.has(row)) rowMap.set(row, [])
    rowMap.get(row).push(smi)
  }

  // Assign x/y positions
  const posMap = new Map()
  for (const [row, smilesList] of rowMap) {
    const total = smilesList.length * NW + (smilesList.length - 1) * HGAP
    smilesList.forEach((smi, i) => {
      posMap.set(smi, {
        x: i * (NW + HGAP) - total / 2 + NW / 2,
        y: row * VGAP,
      })
    })
  }

  const nodes = Array.from(molMeta.entries()).map(([smi, meta]) => ({
    id: smi,
    type: 'mol',
    position: posMap.get(smi) || { x: 0, y: 0 },
    data: { smiles: smi, name: meta.name, status: meta.status },
  }))

  const edges = edgePairs
    .filter(e => e.from && e.to && e.from !== e.to)
    .map((e, i) => ({
      id: `e${i}`,
      source: e.from,
      target: e.to,
      label: e.label,
      type: 'smoothstep',
      style: { stroke: '#06d6f065', strokeWidth: 1.5 },
      markerEnd: { type: 'arrowclosed', color: '#06d6f0', width: 10, height: 10 },
      labelStyle: { fontSize: 10, fill: '#5a7a9a', fontFamily: 'monospace' },
      labelBgStyle: { fill: '#0a1420', fillOpacity: 0.85 },
      labelBgPadding: [4, 3],
      labelBgBorderRadius: 4,
    }))

  return { nodes, edges }
}

// ── Detail panel (below graph) ────────────────────────────────────────────────

function NodeDetail({ nodeData, onClose }) {
  if (!nodeData) return null
  const cfg = SC[nodeData.status] || SC.intermediate
  const imgUrl = nodeData.smiles
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(nodeData.smiles)}/PNG?image_size=300x180`
    : null

  return (
    <div style={{
      marginTop: 10,
      padding: '12px 16px',
      background: '#0d1520',
      border: `1px solid ${cfg.color}40`,
      borderRadius: 8,
      display: 'flex', gap: 16, alignItems: 'flex-start',
    }}>
      {/* 2D image */}
      {imgUrl && (
        <div style={{
          flexShrink: 0, background: '#fff', borderRadius: 6,
          overflow: 'hidden', width: 150,
          border: `1px solid ${cfg.color}30`,
        }}>
          <img
            src={imgUrl}
            alt={nodeData.name || nodeData.smiles}
            style={{ width: '100%', display: 'block' }}
            onError={e => { e.target.parentElement.style.display = 'none' }}
          />
        </div>
      )}

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{
            fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
            padding: '2px 8px', borderRadius: 4,
            background: cfg.color + '18', color: cfg.color,
            border: `1px solid ${cfg.color}40`,
          }}>
            {cfg.icon} {cfg.label}
          </span>
          <button
            onClick={onClose}
            style={{
              marginLeft: 'auto', background: 'none', border: 'none',
              color: '#4d6585', fontSize: 16, cursor: 'pointer', lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ fontSize: 14, fontWeight: 700, color: '#e8edf5', marginBottom: 4 }}>
          {nodeData.name || '—'}
        </div>

        {nodeData.smiles && (
          <div style={{
            fontSize: 11, fontFamily: 'monospace', color: '#8fa3bf',
            background: '#121d2e', padding: '6px 10px', borderRadius: 6,
            border: '1px solid #1e3050', wordBreak: 'break-all',
          }}>
            {nodeData.smiles}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ProtocolGraph({ protocol }) {
  const [open, setOpen] = useState(true)
  const [selectedData, setSelectedData] = useState(null)

  const sections = protocol?.reaction_sections
  const { nodes: initNodes, edges: initEdges } = useMemo(
    () => buildGraph(sections),
    [sections],
  )
  const [nodes, , onNodesChange] = useNodesState(initNodes)
  const [edges, , onEdgesChange] = useEdgesState(initEdges)

  const onNodeClick = useCallback((_, node) => {
    setSelectedData(prev => prev?.smiles === node.data.smiles ? null : node.data)
  }, [])
  const onPaneClick = useCallback(() => setSelectedData(null), [])

  if (!sections?.length) return null

  // Dynamic height based on number of steps
  const numRows = new Set(sections.map(s => s.step_number)).size
  const graphHeight = Math.max(320, (numRows + 1) * VGAP + NH + 20)

  const stepCount = sections.length
  const stepLabel = stepCount === 1 ? 'стадия' : stepCount < 5 ? 'стадии' : 'стадий'

  return (
    <div style={{ marginBottom: 20 }}>

      {/* ── Section header ── */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          marginBottom: 10, cursor: 'pointer', userSelect: 'none',
        }}
        onClick={() => { setOpen(o => !o); setSelectedData(null) }}
      >
        <div style={{
          fontSize: 11, fontWeight: 700, color: 'var(--text-2)',
          fontFamily: 'var(--font-mono)', letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          Схема синтеза · {stepCount} {stepLabel}
        </div>
        <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
          {open ? 'свернуть ▴' : 'развернуть ▾'}
        </span>
      </div>

      {open && (
        <>
          {/* ── Graph ── */}
          <div style={{
            height: graphHeight,
            borderRadius: 8,
            overflow: 'hidden',
            border: '1px solid #1e3050',
          }}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              nodeTypes={NODE_TYPES}
              fitView
              fitViewOptions={{ padding: 0.18 }}
              minZoom={0.08}
              maxZoom={2.5}
              style={{ background: '#070b12' }}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#1a2740" gap={24} size={1} />
              <Controls style={{
                background: '#0d1520', border: '1px solid #1e3050',
                borderRadius: 6, overflow: 'hidden',
              }} />
            </ReactFlow>
          </div>

          {/* ── Legend ── */}
          <div style={{
            display: 'flex', gap: 18, marginTop: 7,
            fontSize: 10, fontFamily: 'var(--font-mono)',
          }}>
            {Object.entries(SC).map(([key, cfg]) => (
              <span key={key} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: cfg.color, flexShrink: 0,
                }} />
                <span style={{ color: 'var(--text-3)' }}>{cfg.label}</span>
              </span>
            ))}
            <span style={{ marginLeft: 'auto', color: 'var(--text-3)' }}>
              Кликни на узел для деталей
            </span>
          </div>

          {/* ── Node detail panel ── */}
          {selectedData && (
            <NodeDetail
              nodeData={selectedData}
              onClose={() => setSelectedData(null)}
            />
          )}
        </>
      )}
    </div>
  )
}

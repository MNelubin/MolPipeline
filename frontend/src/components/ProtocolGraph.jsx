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

const SC = {
  target:       { color: '#06d6f0', label: 'Целевой',       icon: '★' },
  intermediate: { color: '#a78bfa', label: 'Промежуточный', icon: '◆' },
  buyable:      { color: '#22d3a0', label: 'Коммерч.',      icon: '✓' },
}

function MolNode({ data, selected }) {
  const cfg = SC[data.status] || SC.intermediate
  const imgUrl = data.pubchem_cid > 0
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${data.pubchem_cid}/PNG?image_size=160x100`
    : data.smiles
      ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(data.smiles)}/PNG?image_size=160x100`
      : null

  return (
    <div
      className={`graph-node${selected ? ' selected' : ''}`}
      style={{
        borderColor: selected ? cfg.color : cfg.color + '50',
        borderLeft: `3px solid ${cfg.color}`,
        width: 175,
        boxShadow: selected ? `0 0 14px ${cfg.color}45` : 'var(--shadow-card)',
      }}
    >
      <Handle type="target" position={Position.Top}
        style={{ background: cfg.color, border: 'none', width: 7, height: 7 }} />

      {imgUrl && (
        <div style={{
          background: '#fff', borderRadius: 'var(--r-xs, 4px)', marginBottom: 6,
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

      <div style={{ marginBottom: 4 }}>
        <span
          className="graph-node-badge"
          style={{
            background: cfg.color + '18',
            color: cfg.color,
            border: `1px solid ${cfg.color}35`,
          }}
        >
          {cfg.icon} {cfg.label}
        </span>
      </div>

      <div className="graph-node-name">
        {data.name || data.smiles?.slice(0, 24) || '—'}
      </div>

      {data.smiles && (
        <div className="graph-node-smiles">{data.smiles}</div>
      )}

      <Handle type="source" position={Position.Bottom}
        style={{ background: cfg.color, border: 'none', width: 7, height: 7 }} />
    </div>
  )
}

const NODE_TYPES = { mol: MolNode }

const NW = 175, NH = 145, HGAP = 45, VGAP = 170

function buildGraph(sections) {
  if (!sections?.length) return { nodes: [], edges: [] }

  const sorted = [...sections].sort((a, b) => a.step_number - b.step_number)
  const lastStep = sorted[sorted.length - 1].step_number

  const molMeta = new Map()
  const edgePairs = []

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

  const rowMap = new Map()

  for (const [smi, meta] of molMeta) {
    let row
    if (meta.status === 'target') {
      row = sorted.length
    } else if (meta.producedAtStepIdx != null) {
      row = meta.producedAtStepIdx + 1
    } else {
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
      labelStyle: { fontSize: 10, fill: '#5a7a9a', fontFamily: 'var(--font-mono)' },
      labelBgStyle: { fill: '#0a1420', fillOpacity: 0.85 },
      labelBgPadding: [4, 3],
      labelBgBorderRadius: 4,
    }))

  return { nodes, edges }
}

function NodeDetail({ nodeData, onClose }) {
  if (!nodeData) return null
  const cfg = SC[nodeData.status] || SC.intermediate
  const imgUrl = nodeData.pubchem_cid > 0
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${nodeData.pubchem_cid}/PNG?image_size=300x180`
    : nodeData.smiles
      ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(nodeData.smiles)}/PNG?image_size=300x180`
      : null

  return (
    <div className="graph-detail-panel" style={{ borderColor: cfg.color + '40' }}>
      {imgUrl && (
        <div className="graph-detail-img" style={{ borderColor: cfg.color + '30' }}>
          <img
            src={imgUrl}
            alt={nodeData.name || nodeData.smiles}
            onError={e => { e.target.parentElement.style.display = 'none' }}
          />
        </div>
      )}

      <div className="graph-detail-info">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
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
          <button className="graph-detail-close" onClick={onClose}>✕</button>
        </div>

        <div className="graph-detail-name">{nodeData.name || '—'}</div>

        {nodeData.smiles && (
          <div className="graph-detail-smiles">{nodeData.smiles}</div>
        )}
      </div>
    </div>
  )
}

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

  const numRows = new Set(sections.map(s => s.step_number)).size
  const graphHeight = Math.max(320, (numRows + 1) * VGAP + NH + 20)

  const stepCount = sections.length
  const stepLabel = stepCount === 1 ? 'стадия' : stepCount < 5 ? 'стадии' : 'стадий'

  return (
    <div style={{ marginBottom: 20 }}>
      <div className="graph-section-header" onClick={() => { setOpen(o => !o); setSelectedData(null) }}>
        <div className="graph-section-title">
          Схема синтеза · {stepCount} {stepLabel}
        </div>
        <div className="graph-section-line" />
        <span className="graph-section-toggle">
          {open ? 'свернуть ▴' : 'развернуть ▾'}
        </span>
      </div>

      {open && (
        <>
          <div className="graph-container" style={{ height: graphHeight }}>
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
              style={{ background: 'var(--bg-0)' }}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#1a2740" gap={24} size={1} />
              <Controls style={{
                background: 'var(--bg-1)', border: '1px solid var(--border)',
                borderRadius: 6, overflow: 'hidden',
              }} />
            </ReactFlow>
          </div>

          <div className="graph-legend">
            {Object.entries(SC).map(([key, cfg]) => (
              <span key={key} className="graph-legend-item">
                <span className="graph-legend-dot" style={{ background: cfg.color }} />
                <span className="graph-legend-label">{cfg.label}</span>
              </span>
            ))}
            <span className="graph-legend-hint">Кликни на узел для деталей</span>
          </div>

          {selectedData && (
            <NodeDetail nodeData={selectedData} onClose={() => setSelectedData(null)} />
          )}
        </>
      )}
    </div>
  )
}

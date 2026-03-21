/**
 * SynthesisTree — recursive tree visualization for retrosynthesis.
 *
 * Color coding:
 *   green  = buyable (commercially available)
 *   red    = banned (controlled substance)
 *   cyan   = intermediate (has children)
 *   amber  = unresolved / depth_limit / timeout / circular
 */

import { useState } from 'react'

const STATUS_CONFIG = {
  buyable:       { color: 'var(--green)',  label: 'Покупаемый',     icon: '✓' },
  banned:        { color: 'var(--red)',    label: 'Запрещён',       icon: '✕' },
  intermediate:  { color: 'var(--cyan)',   label: 'Промежуточный',  icon: '◆' },
  unresolved:    { color: 'var(--amber)',  label: 'Не найден',      icon: '?' },
  depth_limit:   { color: 'var(--amber)',  label: 'Лимит глубины', icon: '↓' },
  timeout:       { color: 'var(--amber)',  label: 'Таймаут',        icon: '⏱' },
  circular:      { color: 'var(--amber)',  label: 'Цикл',           icon: '↻' },
  invalid_smiles:{ color: 'var(--red)',    label: 'Невалидный',     icon: '!' },
}

function TreeNode({ node, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen && node.depth < 3)
  const cfg = STATUS_CONFIG[node.status] || STATUS_CONFIG.unresolved
  const hasChildren = node.children?.length > 0

  const img2dUrl = node.smiles
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(node.smiles)}/PNG?image_size=200x150`
    : null

  return (
    <div style={{ marginLeft: node.depth > 0 ? 24 : 0, position: 'relative' }}>
      {/* Connector line */}
      {node.depth > 0 && (
        <div style={{
          position: 'absolute',
          left: -16,
          top: 0,
          bottom: hasChildren && open ? 0 : '50%',
          width: 2,
          background: `${cfg.color}30`,
        }} />
      )}
      {node.depth > 0 && (
        <div style={{
          position: 'absolute',
          left: -16,
          top: 20,
          width: 14,
          height: 2,
          background: `${cfg.color}30`,
        }} />
      )}

      {/* Node card */}
      <div
        style={{
          background: 'var(--bg-2)',
          border: `1px solid ${cfg.color}40`,
          borderLeft: `3px solid ${cfg.color}`,
          borderRadius: '0 var(--r-sm) var(--r-sm) 0',
          marginBottom: 8,
          overflow: 'hidden',
        }}
      >
        {/* Node header */}
        <div
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '8px 12px',
            cursor: hasChildren ? 'pointer' : 'default',
          }}
          onClick={() => hasChildren && setOpen(o => !o)}
        >
          {/* Status icon */}
          <span style={{
            fontSize: 10, fontWeight: 700,
            width: 20, height: 20,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            borderRadius: '50%',
            background: `${cfg.color}18`,
            color: cfg.color,
            border: `1px solid ${cfg.color}40`,
            flexShrink: 0,
          }}>
            {cfg.icon}
          </span>

          {/* Name + SMILES */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: 13, fontWeight: 600, color: 'var(--text-1)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {node.name || node.smiles?.slice(0, 40) || '—'}
            </div>
            {node.name && node.smiles && (
              <div style={{
                fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {node.smiles}
              </div>
            )}
          </div>

          {/* Status badge */}
          <span style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
            padding: '2px 8px', borderRadius: 4,
            background: `${cfg.color}18`, color: cfg.color,
            border: `1px solid ${cfg.color}40`,
            flexShrink: 0, whiteSpace: 'nowrap',
          }}>
            {cfg.label}
          </span>

          {/* Depth */}
          <span style={{
            fontSize: 10, fontFamily: 'var(--font-mono)',
            color: 'var(--text-3)', flexShrink: 0,
          }}>
            d{node.depth}
          </span>

          {/* Expand toggle */}
          {hasChildren && (
            <span style={{ color: 'var(--text-3)', fontSize: 11, flexShrink: 0 }}>
              {open ? '▲' : '▼'}
            </span>
          )}
        </div>

        {/* Expanded details */}
        {open && (
          <div style={{ padding: '0 12px 10px 12px' }}>
            {/* 2D image */}
            {img2dUrl && (
              <div style={{
                marginBottom: 8, borderRadius: 'var(--r-sm)',
                overflow: 'hidden', background: '#fff',
                maxWidth: 200,
              }}>
                <img
                  src={img2dUrl}
                  alt={node.name || node.smiles}
                  style={{ width: '100%', display: 'block' }}
                  onError={e => { e.target.style.display = 'none' }}
                />
              </div>
            )}

            {/* Guard info for banned */}
            {node.status === 'banned' && node.guard && (
              <div style={{
                background: 'var(--red)10',
                border: '1px solid var(--red)30',
                borderRadius: 'var(--r-sm)',
                padding: '6px 10px',
                marginBottom: 8,
                fontSize: 12,
                color: 'var(--red)',
              }}>
                {node.guard.reason || 'Вещество находится в списке запрещённых'}
              </div>
            )}

            {/* Route info for intermediates */}
            {node.route && (
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginBottom: 4 }}>
                <span style={{ color: 'var(--text-2)' }}>Маршрут:</span>{' '}
                {node.route.source || '—'}
                {node.route.final_score != null && (
                  <span style={{ color: 'var(--cyan)', marginLeft: 8 }}>
                    score: {node.route.final_score.toFixed(3)}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Children */}
      {hasChildren && open && (
        <div style={{ paddingLeft: 2 }}>
          {node.children.map((child, i) => (
            <TreeNode key={`${child.smiles}-${i}`} node={child} defaultOpen={child.depth < 3} />
          ))}
        </div>
      )}
    </div>
  )
}

export default function SynthesisTree({ tree, stats }) {
  if (!tree) return null

  return (
    <div style={{ marginTop: 16 }}>
      {/* Stats bar */}
      {stats && (
        <div style={{
          display: 'flex', gap: 16, flexWrap: 'wrap',
          marginBottom: 14,
          padding: '10px 14px',
          background: 'var(--bg-2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--r-md)',
          fontSize: 12, fontFamily: 'var(--font-mono)',
        }}>
          <div>
            <span style={{ color: 'var(--text-3)' }}>Узлов: </span>
            <span style={{ color: 'var(--text-1)' }}>{stats.total_nodes}</span>
          </div>
          <div>
            <span style={{ color: 'var(--green)' }}>Покупаемых: </span>
            <span style={{ color: 'var(--text-1)' }}>{stats.buyable_count}</span>
          </div>
          {stats.banned_count > 0 && (
            <div>
              <span style={{ color: 'var(--red)' }}>Запрещённых: </span>
              <span style={{ color: 'var(--text-1)' }}>{stats.banned_count}</span>
            </div>
          )}
          {stats.unresolved_count > 0 && (
            <div>
              <span style={{ color: 'var(--amber)' }}>Не найдено: </span>
              <span style={{ color: 'var(--text-1)' }}>{stats.unresolved_count}</span>
            </div>
          )}
          <div>
            <span style={{ color: 'var(--text-3)' }}>Глубина: </span>
            <span style={{ color: 'var(--text-1)' }}>{stats.max_depth_reached}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-3)' }}>Время: </span>
            <span style={{ color: 'var(--text-1)' }}>{stats.elapsed_sec}с</span>
          </div>
        </div>
      )}

      {/* Tree */}
      <TreeNode node={tree} defaultOpen={true} />
    </div>
  )
}

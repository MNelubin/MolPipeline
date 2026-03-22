import { useState } from 'react'
import Viewer3D from './Viewer3D'
import RetroCard from './RetroCard'
const TABS = ['overview', 'properties', 'structure', 'safety', 'synthesis']
const TAB_LABELS = {
  overview:   'Обзор',
  properties: 'Свойства',
  structure:  'Структура',
  safety:     'Безопасность',
  synthesis:  'Синтез',
}

function PropItem({ label, value }) {
  if (value === null || value === undefined || value === '' || value === 'Н/Д') return null
  return (
    <div className="prop-item">
      <div className="prop-label">{label}</div>
      <div className="prop-value">{value}</div>
    </div>
  )
}

export default function MoleculeCard({ moleculeInfo, guardResult, retroResult }) {
  const [tab, setTab] = useState('overview')
  const [viewMode, setViewMode] = useState('2d')
  const [copied, setCopied] = useState(false)

  if (!moleculeInfo) return null

  const m = moleculeInfo
  const p = m.properties || {}
  const guard = guardResult || {}
  const safety = guard.safety_data || {}
  const status = guard.overall_status || 'SAFE'

  const handleCopy = () => {
    navigator.clipboard.writeText(m.smiles || '').then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  // 2D — prefer CID (stable URL), fall back to SMILES
  const img2dUrl = m.pubchem_cid > 0
    ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${m.pubchem_cid}/PNG`
    : m.smiles
      ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(m.smiles)}/PNG`
      : null

  return (
    <div className="molecule-card">
      {/* Header */}
      <div className="card-header">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="card-name">{m.name || '—'}</div>
          <div className="card-formula">
            {m.molecular_formula}
            {m.molecular_weight ? ` · ${m.molecular_weight.toFixed(2)} g/mol` : ''}
          </div>
          <div className="card-badges">
            {m.cas_number && <span className="badge badge-cid">CAS {m.cas_number}</span>}
            {m.pubchem_cid > 0 && <span className="badge badge-cid">CID {m.pubchem_cid}</span>}
          </div>
        </div>
        <span className={`status-badge status-${status}`}>{status}</span>
      </div>

      {/* Tabs */}
      <div className="card-tabs">
        {TABS.map(t => (
          <button
            key={t}
            className={`card-tab${tab === t ? ' active' : ''}`}
            onClick={() => setTab(t)}
          >
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>

      <div className="card-body">

        {/* ── OVERVIEW ── */}
        {tab === 'overview' && (
          <div>
            {m.smiles && (
              <>
                <div className="section-title">SMILES</div>
                <div className="smiles-box">
                  {m.smiles}
                  <button className="copy-btn" onClick={handleCopy}>{copied ? '✓' : 'copy'}</button>
                </div>
              </>
            )}
            {m.description && (
              <>
                <div className="section-title">Описание</div>
                <div className="description-text">{m.description}</div>
              </>
            )}
            {m.physical_description && m.physical_description !== 'Нет данных' && (
              <>
                <div className="section-title">Физическое описание</div>
                <div className="description-text">{m.physical_description}</div>
              </>
            )}
            {m.synonyms?.length > 0 && (
              <>
                <div className="section-title">Синонимы</div>
                <div className="synonyms-list">
                  {m.synonyms.slice(0, 8).map((s, i) => <span key={i} className="synonym-tag">{s}</span>)}
                </div>
              </>
            )}
            {m.pubchem_cid > 0 && (
              <div style={{ marginTop: 14 }}>
                <a className="ext-link"
                  href={m.pubchem_url || `https://pubchem.ncbi.nlm.nih.gov/compound/${m.pubchem_cid}`}
                  target="_blank" rel="noopener noreferrer">
                  ↗ PubChem CID {m.pubchem_cid}
                </a>
              </div>
            )}
          </div>
        )}

        {/* ── PROPERTIES ── */}
        {tab === 'properties' && (
          <div>
            <div className="section-title">Физические свойства</div>
            <div className="props-grid">
              <PropItem label="Т. плавления"   value={p.melting_point !== 'Н/Д' ? `${p.melting_point} °C` : null} />
              <PropItem label="Т. кипения"     value={p.boiling_point !== 'Н/Д' ? `${p.boiling_point} °C` : null} />
              <PropItem label="Плотность"      value={p.density !== 'Н/Д' ? `${p.density} г/мл` : null} />
              <PropItem label="Т. вспышки"     value={p.flash_point ? `${p.flash_point} °C` : null} />
              <PropItem label="Давление паров" value={p.vapor_pressure} />
              <PropItem label="Состояние"      value={p.physical_state !== 'Н/Д' ? p.physical_state : null} />
            </div>
            <div className="section-title">Растворимость</div>
            <div className="description-text">{p.solubility || 'Нет данных'}</div>
            <div className="section-title">Молекулярные дескрипторы</div>
            <div className="props-grid">
              <PropItem label="LogP"          value={p.logP} />
              <PropItem label="TPSA (Å²)"     value={p.tpsa} />
              <PropItem label="H-доноры"      value={p.h_bond_donors} />
              <PropItem label="H-акцепторы"   value={p.h_bond_acceptors} />
              <PropItem label="Вращ. связи"   value={p.rotatable_bonds} />
              <PropItem label="Кольца"        value={p.ring_count} />
            </div>
            {(m.toxicity?.ld50_oral || m.toxicity?.ld50_dermal || m.toxicity?.ld50_inhalation) && (
              <>
                <div className="section-title">Токсичность (LD50)</div>
                <div className="tox-row">
                  {m.toxicity.ld50_oral && (
                    <div className="tox-item"><div className="tox-route">Перорально</div><div className="tox-value">{m.toxicity.ld50_oral}</div></div>
                  )}
                  {m.toxicity.ld50_dermal && (
                    <div className="tox-item"><div className="tox-route">Дермально</div><div className="tox-value">{m.toxicity.ld50_dermal}</div></div>
                  )}
                  {m.toxicity.ld50_inhalation && (
                    <div className="tox-item"><div className="tox-route">Ингаляционно</div><div className="tox-value">{m.toxicity.ld50_inhalation}</div></div>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── STRUCTURE ── */}
        {tab === 'structure' && (
          <div>
            <div className="viewer-tabs">
              <button className={`viewer-tab${viewMode === '2d' ? ' active' : ''}`} onClick={() => setViewMode('2d')}>2D</button>
              <button className={`viewer-tab${viewMode === '3d' ? ' active' : ''}`} onClick={() => setViewMode('3d')}>3D</button>
            </div>
            {viewMode === '2d' && img2dUrl && (
              <div className="viewer-2d">
                <img src={img2dUrl} alt="2D structure" style={{ maxWidth: '100%', maxHeight: 200, display: 'block', margin: '0 auto' }} />
              </div>
            )}
            {viewMode === '3d' && (
              <Viewer3D smiles={m.smiles} cid={m.pubchem_cid > 0 ? m.pubchem_cid : null} />
            )}
            {m.smiles && (
              <div style={{ marginTop: 12 }}>
                <div className="section-title">SMILES</div>
                <div className="smiles-box" style={{ paddingRight: 60 }}>
                  {m.smiles}
                  <button className="copy-btn" onClick={handleCopy}>{copied ? '✓' : 'copy'}</button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── SAFETY ── */}
        {tab === 'safety' && (
          <div>
            <div className="section-title">Статус</div>
            <span className={`status-badge status-${status}`} style={{ display: 'inline-flex' }}>{status}</span>
            {m.ghs_classification?.length > 0 && (
              <>
                <div className="section-title">Классификация GHS</div>
                <div className="ghs-grid">
                  {m.ghs_classification.map((cls, i) => <div key={i} className="ghs-item">{cls}</div>)}
                </div>
              </>
            )}
            {safety.h_phrases?.length > 0 && (
              <>
                <div className="section-title">H-фразы</div>
                <div className="h-phrases">
                  {safety.h_phrases.map((h, i) => <span key={i} className="h-phrase-tag">{h}</span>)}
                </div>
              </>
            )}
            {guard.ppe_recommendations?.length > 0 && (
              <>
                <div className="section-title">Средства защиты (СИЗ)</div>
                <div className="ppe-list">
                  {guard.ppe_recommendations.map((p, i) => <span key={i} className="ppe-tag">{p}</span>)}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── SYNTHESIS ── */}
        {tab === 'synthesis' && (
          <RetroCard retroResult={retroResult} smiles={m.smiles} />
        )}

      </div>
    </div>
  )
}

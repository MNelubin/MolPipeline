/**
 * CalculatorCard — интерактивный калькулятор стехиометрии.
 *
 * Два режима:
 *   1. Stoichiometry — reaction_smiles + target_mass_g
 *   2. Equivalents   — reference_smiles + reagents с эквивалентами
 *
 * Вызывает POST /api/calculate (локальный бэкенд).
 * Поле smiles целевой молекулы подставляется автоматически из пропса.
 */

import { useState } from 'react'

const API = '/api/calculate'

const STATE_COLORS = {
  solid:   { color: 'var(--cyan)',   label: 'тв.' },
  liquid:  { color: 'var(--green)',  label: 'жидк.' },
  gas:     { color: 'var(--amber)',  label: 'газ' },
  unknown: { color: 'var(--text-3)', label: '?' },
}

function StateDot({ state }) {
  const s = STATE_COLORS[state] || STATE_COLORS.unknown
  return (
    <span style={{
      display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
      background: s.color, marginRight: 5,
    }} title={s.label} />
  )
}

function ReagentRow({ reagent }) {
  const isLiquid = reagent.state === 'liquid'
  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-sm)',
      padding: '10px 14px',
      display: 'grid',
      gridTemplateColumns: '1fr auto',
      gap: '4px 16px',
      alignItems: 'start',
    }}>
      {/* Left: name + smiles */}
      <div>
        <div style={{ fontSize: 13, color: 'var(--text-1)', fontWeight: 500, marginBottom: 2 }}>
          <StateDot state={reagent.state} />
          {reagent.name || reagent.smiles}
        </div>
        {reagent.name && (
          <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
            {reagent.smiles}
          </div>
        )}
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 3 }}>
          MW: {reagent.molecular_weight} g/mol · {reagent.equivalents} экв
        </div>
      </div>

      {/* Right: amounts */}
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: 15, fontFamily: 'var(--font-mono)', color: 'var(--cyan)', fontWeight: 700 }}>
          {reagent.mass_g} г
        </div>
        {isLiquid && reagent.volume_ml != null && (
          <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--green)' }}>
            {reagent.volume_ml} мл
          </div>
        )}
        {reagent.density != null && (
          <div style={{ fontSize: 11, color: 'var(--text-3)' }}>
            ρ = {reagent.density} г/мл
          </div>
        )}
        {reagent.notes && (
          <div style={{ fontSize: 11, color: 'var(--amber)', marginTop: 2 }}>
            {reagent.notes}
          </div>
        )}
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
          {(reagent.moles * 1000).toFixed(3)} ммоль
        </div>
      </div>
    </div>
  )
}

function Input({ label, value, onChange, placeholder, type = 'text', disabled }) {
  return (
    <div>
      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-3)', marginBottom: 5 }}>
        {label}
      </div>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        style={{
          width: '100%', background: 'var(--bg-2)', border: '1px solid var(--border-hi)',
          borderRadius: 'var(--r-sm)', padding: '8px 12px',
          fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-1)',
          outline: 'none', boxSizing: 'border-box',
          opacity: disabled ? 0.5 : 1,
        }}
        onFocus={e => e.target.style.borderColor = 'var(--cyan-dim)'}
        onBlur={e => e.target.style.borderColor = 'var(--border-hi)'}
      />
    </div>
  )
}

// ── Equivalents mode: dynamic reagent list ────────────────────────────────────

function EquivReagentEditor({ rows, onChange }) {
  const add = () => onChange([...rows, { smiles: '', name: '', equivalents: '1.0' }])
  const remove = i => onChange(rows.filter((_, idx) => idx !== i))
  const update = (i, field, val) => onChange(rows.map((r, idx) => idx === i ? { ...r, [field]: val } : r))

  return (
    <div>
      {rows.map((r, i) => (
        <div key={i} style={{ display: 'grid', gridTemplateColumns: '2fr 1.5fr 0.7fr auto', gap: 6, marginBottom: 6 }}>
          <input
            placeholder="SMILES"
            value={r.smiles}
            onChange={e => update(i, 'smiles', e.target.value)}
            style={{ background: 'var(--bg-2)', border: '1px solid var(--border-hi)', borderRadius: 'var(--r-sm)', padding: '7px 10px', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-1)', outline: 'none' }}
          />
          <input
            placeholder="Название"
            value={r.name}
            onChange={e => update(i, 'name', e.target.value)}
            style={{ background: 'var(--bg-2)', border: '1px solid var(--border-hi)', borderRadius: 'var(--r-sm)', padding: '7px 10px', fontSize: 12, color: 'var(--text-1)', outline: 'none' }}
          />
          <input
            placeholder="экв"
            type="number"
            step="0.1"
            value={r.equivalents}
            onChange={e => update(i, 'equivalents', e.target.value)}
            style={{ background: 'var(--bg-2)', border: '1px solid var(--border-hi)', borderRadius: 'var(--r-sm)', padding: '7px 10px', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-1)', outline: 'none', textAlign: 'center' }}
          />
          <button onClick={() => remove(i)} style={{ background: 'transparent', border: '1px solid var(--border)', borderRadius: 'var(--r-sm)', color: 'var(--text-3)', cursor: 'pointer', padding: '0 8px', fontSize: 14 }}>×</button>
        </div>
      ))}
      <button onClick={add} style={{
        background: 'rgba(6,214,240,0.07)', border: '1px dashed var(--cyan-dim)',
        borderRadius: 'var(--r-sm)', color: 'var(--cyan)', cursor: 'pointer',
        padding: '6px 14px', fontSize: 12, fontFamily: 'var(--font-mono)', width: '100%', marginTop: 4,
      }}>
        + добавить реагент
      </button>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function CalculatorCard({ smiles: targetSmiles }) {
  const [mode, setMode] = useState('stoichio')      // 'stoichio' | 'equiv'
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError]   = useState(null)

  // Stoichiometry fields
  const [reactionSmiles, setReactionSmiles] = useState('')
  const [targetMass, setTargetMass]         = useState('1.0')
  const [targetProduct, setTargetProduct]   = useState(targetSmiles || '')

  // Equivalents fields
  const [refSmiles, setRefSmiles]       = useState(targetSmiles || '')
  const [refAmount, setRefAmount]       = useState('1.0')
  const [amountType, setAmountType]     = useState('reagent_moles')
  const [equivRows, setEquivRows]       = useState([
    { smiles: '', name: '', equivalents: '1.0' },
  ])

  const calc = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    let body
    if (mode === 'stoichio') {
      body = {
        reaction_smiles: reactionSmiles,
        target_mass_g:   parseFloat(targetMass),
        ...(targetProduct ? { target_product_smiles: targetProduct } : {}),
      }
    } else {
      body = {
        reference_smiles:  refSmiles,
        reference_amount:  parseFloat(refAmount),
        amount_type:       amountType,
        reagents: equivRows
          .filter(r => r.smiles.trim())
          .map(r => ({ smiles: r.smiles.trim(), name: r.name.trim(), equivalents: parseFloat(r.equivalents) || 1.0 })),
      }
    }

    try {
      const resp = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || JSON.stringify(data))
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const btnStyle = {
    background: 'rgba(6,214,240,0.12)', border: '1px solid var(--cyan-dim)',
    borderRadius: 'var(--r-sm)', color: 'var(--cyan)', cursor: 'pointer',
    padding: '9px 20px', fontSize: 13, fontFamily: 'var(--font-mono)',
    transition: 'all 0.15s', opacity: loading ? 0.5 : 1,
  }

  return (
    <div>
      {/* Mode switcher */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
        {[['stoichio', 'Стехиометрия'], ['equiv', 'По эквивалентам']].map(([m, label]) => (
          <button key={m} onClick={() => { setMode(m); setResult(null); setError(null) }} style={{
            padding: '5px 14px', fontSize: 12, fontFamily: 'var(--font-mono)',
            borderRadius: '999px', border: '1px solid',
            cursor: 'pointer', transition: 'all 0.15s',
            borderColor: mode === m ? 'var(--cyan-dim)' : 'var(--border)',
            color:       mode === m ? 'var(--cyan)'    : 'var(--text-3)',
            background:  mode === m ? 'rgba(6,214,240,0.08)' : 'transparent',
          }}>{label}</button>
        ))}
      </div>

      {/* ── Stoichiometry form ── */}
      {mode === 'stoichio' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Input label="Реакция (SMILES)" value={reactionSmiles} onChange={setReactionSmiles}
            placeholder="CC(=O)O.CCO>>CC(=O)OCC.O" />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Input label="Масса продукта (г)" type="number" value={targetMass} onChange={setTargetMass} placeholder="1.0" />
            <Input label="Целевой продукт (SMILES, опц.)" value={targetProduct} onChange={setTargetProduct} placeholder="автоматически" />
          </div>
          <button style={btnStyle} onClick={calc} disabled={loading || !reactionSmiles}>
            {loading ? 'Считаем...' : 'Рассчитать →'}
          </button>
        </div>
      )}

      {/* ── Equivalents form ── */}
      {mode === 'equiv' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Input label="Референсный SMILES" value={refSmiles} onChange={setRefSmiles} placeholder="CCO" />
            <Input label="Количество" type="number" value={refAmount} onChange={setRefAmount} placeholder="1.0" />
          </div>
          <div>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-3)', marginBottom: 5 }}>
              Тип количества
            </div>
            <select value={amountType} onChange={e => setAmountType(e.target.value)} style={{
              background: 'var(--bg-2)', border: '1px solid var(--border-hi)', borderRadius: 'var(--r-sm)',
              padding: '8px 12px', fontSize: 13, fontFamily: 'var(--font-mono)',
              color: 'var(--text-1)', outline: 'none', width: '100%',
            }}>
              <option value="reagent_moles">Моли реагента</option>
              <option value="reagent_mass">Масса реагента (г)</option>
              <option value="product_mass">Масса продукта (г)</option>
            </select>
          </div>
          <div>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-3)', marginBottom: 8 }}>
              Реагенты
            </div>
            <EquivReagentEditor rows={equivRows} onChange={setEquivRows} />
          </div>
          <button style={btnStyle} onClick={calc} disabled={loading || !refSmiles}>
            {loading ? 'Считаем...' : 'Рассчитать →'}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="error-card" style={{ marginTop: 14 }}>⚠ {error}</div>
      )}

      {/* Result */}
      {result && (
        <div style={{ marginTop: 20 }}>
          <div className="divider" />

          {/* Summary */}
          <div style={{ display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
            <div className="prop-item">
              <div className="prop-label">Целевой продукт</div>
              <div className="prop-value" style={{ fontSize: 11, wordBreak: 'break-all' }}>{result.target_product_smiles}</div>
            </div>
            <div className="prop-item">
              <div className="prop-label">Масса</div>
              <div className="prop-value">{result.target_mass_g} г</div>
            </div>
            <div className="prop-item">
              <div className="prop-label">Моли</div>
              <div className="prop-value">{(result.target_moles * 1000).toFixed(3)} ммоль</div>
            </div>
          </div>

          {/* Reagents */}
          <div className="section-title">Реагенты</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {result.reagents.map((r, i) => <ReagentRow key={i} reagent={r} />)}
          </div>

          {/* Warnings */}
          {result.warnings?.length > 0 && (
            <div style={{ marginTop: 12 }}>
              {result.warnings.map((w, i) => (
                <div key={i} style={{ fontSize: 12, color: 'var(--amber)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                  ⚠ {w}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

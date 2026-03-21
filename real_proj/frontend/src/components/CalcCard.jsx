/**
 * CalcCard — клиентский калькулятор стехиометрии и эквивалентов.
 *
 * Все вычисления в браузере. PubChem REST вызывается напрямую (CORS разрешён).
 *
 * Режимы:
 *   stoichiometry — реакционный SMILES + целевая масса → таблица реагентов
 *   equivalents   — референсный SMILES + список реагентов → таблица
 */

import { useState, useCallback } from 'react'

// ─── PubChem API helpers ─────────────────────────────────────────────────────

const PUBCHEM = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
const PUBCHEM_VIEW = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view'

const _cache = {}
async function _get(url) {
  if (_cache[url] !== undefined) return _cache[url]
  try {
    const r = await fetch(url)
    if (!r.ok) { _cache[url] = null; return null }
    const d = await r.json()
    _cache[url] = d
    return d
  } catch { _cache[url] = null; return null }
}

async function getMW(smiles) {
  const url = `${PUBCHEM}/compound/smiles/${encodeURIComponent(smiles)}/property/MolecularWeight/JSON`
  const d = await _get(url)
  return parseFloat(d?.PropertyTable?.Properties?.[0]?.MolecularWeight) || null
}

async function getName(smiles) {
  const url = `${PUBCHEM}/compound/smiles/${encodeURIComponent(smiles)}/property/IUPACName/JSON`
  const d = await _get(url)
  return d?.PropertyTable?.Properties?.[0]?.IUPACName || ''
}

async function getCID(smiles) {
  const url = `${PUBCHEM}/compound/smiles/${encodeURIComponent(smiles)}/cids/JSON`
  const d = await _get(url)
  return d?.IdentifierList?.CID?.[0] || null
}

async function getDensity(cid) {
  if (!cid) return null
  const url = `${PUBCHEM_VIEW}/data/compound/${cid}/JSON?heading=Density`
  const d = await _get(url)
  const sections = d?.Record?.Section || []
  const text = walkSections(sections, 'density')
  if (!text) return null
  const m = text.match(/-?\d+\.?\d*/)
  return m ? parseFloat(m[0]) : null
}

async function getPhysicalState(cid) {
  if (!cid) return 'unknown'
  // Try melting point
  const mpUrl = `${PUBCHEM_VIEW}/data/compound/${cid}/JSON?heading=Melting+Point`
  const bpUrl = `${PUBCHEM_VIEW}/data/compound/${cid}/JSON?heading=Boiling+Point`
  const [mpD, bpD] = await Promise.all([_get(mpUrl), _get(bpUrl)])

  const mp = extractTemp(mpD)
  const bp = extractTemp(bpD)

  if (mp !== null && mp > 25) return 'solid'
  if (bp !== null && bp < 25) return 'gas'
  if (mp !== null && mp <= 25) return 'liquid'
  if (bp !== null && bp >= 25) return 'liquid'
  return 'unknown'
}

function extractTemp(data) {
  const sections = data?.Record?.Section || []
  const sec = walkSections(sections, 'melting point') || walkSections(sections, 'boiling point')
  if (!sec) return null
  for (const info of sec.Information || []) {
    const swm = info.Value?.StringWithMarkup || []
    for (const s of swm) {
      const text = s.String || ''
      const m = text.match(/-?\d+\.?\d*/)
      if (!m) continue
      const num = parseFloat(m[0])
      if (text.toLowerCase().includes('°f') || text.toLowerCase().includes('deg f')) {
        return (num - 32) * 5 / 9
      }
      return num
    }
  }
  return null
}

function walkSections(sections, heading) {
  for (const sec of sections) {
    if (sec.TOCHeading?.toLowerCase() === heading.toLowerCase()) {
      // return string value
      for (const info of sec.Information || []) {
        for (const s of info.Value?.StringWithMarkup || []) {
          if (s.String) return s.String
        }
      }
      return sec
    }
    const found = walkSections(sec.Section || [], heading)
    if (found) return found
  }
  return null
}

// ─── SMILES parsing (no RDKit, JS only) ──────────────────────────────────────

function parseReactionSmiles(rxn) {
  if (!rxn.includes('>>')) throw new Error('Нет разделителя >> в реакционном SMILES')
  const [left, right] = rxn.split('>>')
  const reactants = left.split('.').map(s => s.trim()).filter(Boolean)
  const products  = right.split('.').map(s => s.trim()).filter(Boolean)
  if (!reactants.length) throw new Error('Не найдены реагенты')
  if (!products.length)  throw new Error('Не найдены продукты')
  return { reactants, products }
}

function countCoefficients(smilesList) {
  const counts = {}
  for (const s of smilesList) {
    counts[s] = (counts[s] || 0) + 1
  }
  return counts
}

// ─── Core calculation ─────────────────────────────────────────────────────────

async function buildReagentResult(smiles, { equivalents, moles }) {
  const [mw, name, cid] = await Promise.all([getMW(smiles), getName(smiles), getCID(smiles)])
  const resolvedMW = mw || 100 // fallback
  const mass_g = moles * resolvedMW

  const [density, state] = await Promise.all([getDensity(cid), getPhysicalState(cid)])

  let volume_ml = null
  let notes = ''
  if (state === 'liquid' && density && density > 0) {
    volume_ml = mass_g / density
    if (volume_ml < 0.1) {
      const drops = Math.round(volume_ml / 0.05 * 10) / 10
      notes = `~${drops} кап.`
    }
  }

  return {
    smiles,
    name: name || smiles,
    mw: resolvedMW,
    equivalents,
    moles,
    mass_g,
    density,
    volume_ml,
    state,
    notes,
  }
}

async function calcStoichiometry({ reactionSmiles, targetMassG, targetProductSmiles }) {
  const { reactants, products } = parseReactionSmiles(reactionSmiles)
  const targetSmiles = targetProductSmiles || products[0]
  const warnings = []
  if (!targetProductSmiles && products.length > 1) {
    warnings.push('Несколько продуктов — расчёт по первому. Укажите целевой продукт при необходимости.')
  }

  const targetMW = await getMW(targetSmiles)
  if (!targetMW) throw new Error(`Не удалось получить MW для продукта: ${targetSmiles}`)

  const targetMoles = targetMassG / targetMW
  const coeff = countCoefficients(reactants)
  const productCoeff = countCoefficients(products)[targetSmiles] || 1

  const results = await Promise.all(
    Object.entries(coeff).map(([smiles, c]) => {
      const equiv = c / productCoeff
      return buildReagentResult(smiles, { equivalents: equiv, moles: targetMoles * equiv })
    })
  )

  return {
    targetSmiles,
    targetMW,
    targetMoles,
    targetMassG,
    results,
    warnings,
  }
}

async function calcEquivalents({ referenceSmiles, referenceAmount, amountType, reagents }) {
  const refMW = await getMW(referenceSmiles)
  if (!refMW) throw new Error(`Не удалось получить MW для: ${referenceSmiles}`)

  let referenceMoles
  if (amountType === 'reagent_moles') {
    referenceMoles = referenceAmount
  } else {
    referenceMoles = referenceAmount / refMW  // product_mass or reagent_mass
  }

  const results = await Promise.all(
    reagents.map(r =>
      buildReagentResult(r.smiles, {
        equivalents: r.equivalents,
        moles: referenceMoles * r.equivalents,
      })
    )
  )

  return { referenceMoles, referenceAmount, results, warnings: [] }
}

// ─── UI helpers ───────────────────────────────────────────────────────────────

const STATE_ICON = { solid: '⬡', liquid: '💧', gas: '☁', unknown: '?' }
const STATE_COLOR = { solid: 'var(--text-2)', liquid: 'var(--cyan)', gas: 'var(--purple)', unknown: 'var(--text-3)' }

function ResultsTable({ results }) {
  if (!results.length) return null
  return (
    <div style={{ overflowX: 'auto', marginTop: 14 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Реагент', 'MW', 'Экв.', 'Моль', 'Масса (г)', 'Объём (мл)', 'Состояние'].map(h => (
              <th key={h} style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.7px', color: 'var(--text-3)', fontWeight: 400 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--border)', background: i % 2 === 0 ? 'transparent' : 'var(--bg-2)' }}>
              <td style={{ padding: '8px 10px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--cyan)', marginBottom: 2 }}>{r.name !== r.smiles ? r.name : ''}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)', wordBreak: 'break-all' }}>{r.smiles}</div>
              </td>
              <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>{r.mw?.toFixed(2)}</td>
              <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontWeight: 600 }}>{r.equivalents?.toFixed(3)}</td>
              <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>{r.moles?.toExponential(3)}</td>
              <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: 'var(--amber)', fontWeight: 600 }}>{r.mass_g?.toFixed(4)}</td>
              <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: r.volume_ml ? 'var(--cyan)' : 'var(--text-3)' }}>
                {r.volume_ml ? r.volume_ml.toFixed(4) : '—'}
                {r.notes && <span style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 4 }}>{r.notes}</span>}
              </td>
              <td style={{ padding: '8px 10px', color: STATE_COLOR[r.state] }}>
                {STATE_ICON[r.state]} {r.state}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Input({ label, value, onChange, placeholder, mono, hint }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-3)', marginBottom: 5 }}>{label}</div>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%', background: 'var(--bg-2)', border: '1px solid var(--border-hi)',
          borderRadius: 'var(--r-sm)', padding: '8px 12px', color: 'var(--text-1)',
          fontFamily: mono ? 'var(--font-mono)' : 'var(--font-ui)', fontSize: 13, outline: 'none',
        }}
        onFocus={e => e.target.style.borderColor = 'var(--cyan-dim)'}
        onBlur={e => e.target.style.borderColor = 'var(--border-hi)'}
      />
      {hint && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>{hint}</div>}
    </div>
  )
}

// ─── Stoichiometry mode ───────────────────────────────────────────────────────

function StoichiometryCalc({ defaultSmiles }) {
  const [rxn, setRxn] = useState(defaultSmiles ? `>>  ${defaultSmiles}` : '')
  const [targetMass, setTargetMass] = useState('1.0')
  const [targetProduct, setTargetProduct] = useState(defaultSmiles || '')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const run = useCallback(async () => {
    setLoading(true); setError(''); setResult(null)
    try {
      const r = await calcStoichiometry({
        reactionSmiles: rxn.trim(),
        targetMassG: parseFloat(targetMass),
        targetProductSmiles: targetProduct.trim() || null,
      })
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [rxn, targetMass, targetProduct])

  return (
    <div>
      <Input label="Реакционный SMILES (реагенты>>продукты)" value={rxn} onChange={setRxn}
        placeholder="CC(=O)O.CCO>>CC(=O)OCC.O" mono
        hint='Повторяющиеся компоненты = стехиометрические коэффициенты (A.A.B>>C = 2 экв. A)'
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Input label="Целевая масса продукта (г)" value={targetMass} onChange={setTargetMass} placeholder="1.0" />
        <Input label="SMILES продукта (если несколько)" value={targetProduct} onChange={setTargetProduct} placeholder="автоматически" mono />
      </div>

      <button
        onClick={run}
        disabled={loading || !rxn.trim()}
        style={{
          background: loading ? 'var(--bg-3)' : 'rgba(6,214,240,0.12)',
          border: '1px solid var(--cyan-dim)', borderRadius: 'var(--r-sm)',
          color: 'var(--cyan)', padding: '8px 20px', cursor: loading ? 'not-allowed' : 'pointer',
          fontFamily: 'var(--font-mono)', fontSize: 13, transition: 'all 0.15s',
        }}
      >
        {loading ? '⟳ Загрузка данных PubChem...' : 'Рассчитать'}
      </button>

      {error && <div className="error-card" style={{ marginTop: 12 }}>⚠ {error}</div>}

      {result && (
        <div style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
            <div className="prop-item">
              <div className="prop-label">Целевой продукт</div>
              <div className="prop-value" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{result.targetSmiles}</div>
            </div>
            <div className="prop-item">
              <div className="prop-label">MW продукта</div>
              <div className="prop-value">{result.targetMW?.toFixed(2)} г/моль</div>
            </div>
            <div className="prop-item">
              <div className="prop-label">Молей продукта</div>
              <div className="prop-value">{result.targetMoles?.toExponential(3)}</div>
            </div>
          </div>
          {result.warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 12, color: 'var(--amber)', marginBottom: 6, fontFamily: 'var(--font-mono)' }}>⚠ {w}</div>
          ))}
          <ResultsTable results={result.results} />
        </div>
      )}
    </div>
  )
}

// ─── Equivalents mode ─────────────────────────────────────────────────────────

function EquivalentsCalc({ defaultSmiles }) {
  const [refSmiles, setRefSmiles] = useState(defaultSmiles || '')
  const [refAmount, setRefAmount] = useState('1.0')
  const [amountType, setAmountType] = useState('product_mass')
  const [rows, setRows] = useState([
    { smiles: '', equivalents: '1.0' },
    { smiles: '', equivalents: '1.0' },
  ])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const updateRow = (i, field, val) => {
    setRows(prev => prev.map((r, idx) => idx === i ? { ...r, [field]: val } : r))
  }

  const addRow = () => setRows(prev => [...prev, { smiles: '', equivalents: '1.0' }])
  const removeRow = i => setRows(prev => prev.filter((_, idx) => idx !== i))

  const run = useCallback(async () => {
    const validRows = rows.filter(r => r.smiles.trim())
    if (!validRows.length) { setError('Добавьте хотя бы один реагент'); return }
    setLoading(true); setError(''); setResult(null)
    try {
      const r = await calcEquivalents({
        referenceSmiles: refSmiles.trim(),
        referenceAmount: parseFloat(refAmount),
        amountType,
        reagents: validRows.map(r => ({ smiles: r.smiles.trim(), equivalents: parseFloat(r.equivalents) || 1 })),
      })
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [refSmiles, refAmount, amountType, rows])

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12 }}>
        <Input label="Референсный SMILES (продукт или лим. реагент)" value={refSmiles} onChange={setRefSmiles}
          placeholder="CC(=O)Oc1ccccc1C(=O)O" mono />
        <Input label="Количество" value={refAmount} onChange={setRefAmount} placeholder="1.0" />
      </div>

      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-3)', marginBottom: 6 }}>Тип количества</div>
        <div style={{ display: 'flex', gap: 6 }}>
          {[
            ['product_mass', 'г продукта'],
            ['reagent_mass', 'г реагента'],
            ['reagent_moles', 'моль'],
          ].map(([val, label]) => (
            <button key={val} onClick={() => setAmountType(val)} style={{
              padding: '4px 12px', borderRadius: 999, border: '1px solid',
              borderColor: amountType === val ? 'var(--cyan-dim)' : 'var(--border)',
              background: amountType === val ? 'rgba(6,214,240,0.08)' : 'transparent',
              color: amountType === val ? 'var(--cyan)' : 'var(--text-3)',
              fontSize: 12, fontFamily: 'var(--font-mono)', cursor: 'pointer',
            }}>{label}</button>
          ))}
        </div>
      </div>

      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-3)', marginBottom: 8 }}>Реагенты</div>
      {rows.map((row, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
          <input value={row.smiles} onChange={e => updateRow(i, 'smiles', e.target.value)}
            placeholder="SMILES реагента"
            style={{ flex: 3, background: 'var(--bg-2)', border: '1px solid var(--border-hi)', borderRadius: 'var(--r-sm)', padding: '7px 10px', color: 'var(--text-1)', fontFamily: 'var(--font-mono)', fontSize: 12, outline: 'none' }}
          />
          <input value={row.equivalents} onChange={e => updateRow(i, 'equivalents', e.target.value)}
            placeholder="экв."
            style={{ flex: 1, background: 'var(--bg-2)', border: '1px solid var(--border-hi)', borderRadius: 'var(--r-sm)', padding: '7px 10px', color: 'var(--text-1)', fontFamily: 'var(--font-mono)', fontSize: 12, outline: 'none', maxWidth: 80 }}
          />
          <button onClick={() => removeRow(i)} style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', fontSize: 16, padding: '0 4px' }}>×</button>
        </div>
      ))}
      <button onClick={addRow} style={{ background: 'none', border: '1px dashed var(--border)', borderRadius: 'var(--r-sm)', color: 'var(--text-3)', padding: '5px 14px', cursor: 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)', marginBottom: 14 }}>
        + Добавить реагент
      </button>

      <div>
        <button onClick={run} disabled={loading || !refSmiles.trim()}
          style={{ background: loading ? 'var(--bg-3)' : 'rgba(6,214,240,0.12)', border: '1px solid var(--cyan-dim)', borderRadius: 'var(--r-sm)', color: 'var(--cyan)', padding: '8px 20px', cursor: loading ? 'not-allowed' : 'pointer', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          {loading ? '⟳ Загрузка данных PubChem...' : 'Рассчитать'}
        </button>
      </div>

      {error && <div className="error-card" style={{ marginTop: 12 }}>⚠ {error}</div>}
      {result && (
        <div style={{ marginTop: 16 }}>
          {result.warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 12, color: 'var(--amber)', marginBottom: 6, fontFamily: 'var(--font-mono)' }}>⚠ {w}</div>
          ))}
          <ResultsTable results={result.results} />
        </div>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function CalcCard({ smiles }) {
  const [mode, setMode] = useState('stoichiometry')

  return (
    <div>
      {/* Mode switcher */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 18 }}>
        {[
          ['stoichiometry', 'Стехиометрия'],
          ['equivalents',   'Эквиваленты'],
        ].map(([val, label]) => (
          <button key={val} onClick={() => setMode(val)} style={{
            padding: '5px 16px', borderRadius: 999, border: '1px solid',
            borderColor: mode === val ? 'var(--cyan-dim)' : 'var(--border)',
            background: mode === val ? 'rgba(6,214,240,0.08)' : 'transparent',
            color: mode === val ? 'var(--cyan)' : 'var(--text-3)',
            fontSize: 12, fontFamily: 'var(--font-mono)', cursor: 'pointer', transition: 'all 0.15s',
          }}>{label}</button>
        ))}
      </div>

      <div style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginBottom: 14, lineHeight: 1.5 }}>
        {mode === 'stoichiometry'
          ? 'Введите реакцию в формате SMILES и целевую массу продукта → получите массы/объёмы всех реагентов'
          : 'Введите референсный SMILES, количество и список реагентов с эквивалентами → получите массы/объёмы'}
      </div>

      {mode === 'stoichiometry'
        ? <StoichiometryCalc defaultSmiles={smiles} />
        : <EquivalentsCalc defaultSmiles={smiles} />}
    </div>
  )
}

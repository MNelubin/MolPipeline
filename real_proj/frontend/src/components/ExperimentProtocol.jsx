/**
 * ExperimentProtocol — displays the experiment_protocol from Phase 3 state.
 *
 * protocol shape:
 * {
 *   target_mass_g: number,
 *   is_multistep: boolean,
 *   buyable_reagent_table: [{ name, smiles, mass_g, volume_ml, moles, equivalents }],
 *   reaction_sections: [
 *     {
 *       step_number: number,
 *       product_name: string,
 *       product_smiles: string,
 *       product_mass_g: number,
 *       reaction_smiles: string,   // "A.B>>C"
 *       procedure_steps: [string | { step, description, reason }],
 *       reagent_table: [{ name, smiles, mass_g, volume_ml, moles, equivalents }],
 *     }
 *   ],
 *   calculations: { target_mass_g, target_moles, warnings }
 * }
 */

function ReactionFormula({ reactionSmiles, reagentTable }) {
  if (!reactionSmiles || !reactionSmiles.includes('>>')) return null

  const [lhs, rhs] = reactionSmiles.split('>>')
  const reactantSmiles = lhs.split('.').filter(Boolean)
  const productSmiles = (rhs || '').split('.').filter(Boolean)

  // Build name map from reagent table
  const nameMap = {}
  reagentTable?.forEach(r => {
    if (r.smiles) nameMap[r.smiles] = r.name
  })

  const fmt = (smi) => nameMap[smi] || smi.slice(0, 30) + (smi.length > 30 ? '…' : '')

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-sm)',
      padding: '10px 14px',
      marginBottom: 12,
      fontFamily: 'var(--font-mono)',
      fontSize: 12,
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-3)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        Уравнение реакции
      </div>
      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
        {reactantSmiles.map((smi, i) => (
          <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {i > 0 && <span style={{ color: 'var(--text-3)' }}>+</span>}
            <span style={{
              color: 'var(--text-2)',
              background: 'var(--bg-2)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '2px 8px',
              fontSize: 11,
              wordBreak: 'break-all',
            }}>{fmt(smi)}</span>
          </span>
        ))}
        <span style={{ color: 'var(--cyan)', fontWeight: 700, fontSize: 16, margin: '0 4px' }}>→</span>
        {productSmiles.map((smi, i) => (
          <span key={i} style={{
            color: 'var(--green)',
            background: 'var(--green-dim, #0a2a1a)',
            border: '1px solid var(--green)40',
            borderRadius: 4,
            padding: '2px 8px',
            fontSize: 11,
            wordBreak: 'break-all',
          }}>{fmt(smi)}</span>
        ))}
      </div>
      <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-3)', wordBreak: 'break-all' }}>
        SMILES: {reactionSmiles.slice(0, 120)}{reactionSmiles.length > 120 ? '…' : ''}
      </div>
    </div>
  )
}

function ReagentTable({ rows, compact }) {
  if (!rows?.length) return null
  return (
    <div style={{
      background: compact ? 'var(--bg-card)' : 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: compact ? 'var(--r-sm)' : 'var(--r-md)',
      overflow: 'auto',
      marginBottom: compact ? 0 : 0,
    }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Реагент', 'Масса, г', 'Объём, мл', 'Моль', 'Экв.'].map(h => (
              <th key={h} style={{
                padding: compact ? '6px 10px' : '8px 12px',
                textAlign: 'left',
                color: 'var(--text-3)',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                fontSize: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                whiteSpace: 'nowrap',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: i < rows.length - 1 ? '1px solid var(--border)' : 'none' }}>
              <td style={{ padding: compact ? '6px 10px' : '8px 12px', color: 'var(--text-1)', fontWeight: 500 }}>
                {row.name || row.smiles || '—'}
                {row.notes ? <span style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 6 }}>({row.notes})</span> : null}
              </td>
              <td style={{ padding: compact ? '6px 10px' : '8px 12px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                {row.mass_g != null ? row.mass_g.toFixed(4) : '—'}
              </td>
              <td style={{ padding: compact ? '6px 10px' : '8px 12px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                {row.volume_ml != null ? row.volume_ml.toFixed(3) : '—'}
              </td>
              <td style={{ padding: compact ? '6px 10px' : '8px 12px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                {row.moles != null ? row.moles.toExponential(3) : '—'}
              </td>
              <td style={{ padding: compact ? '6px 10px' : '8px 12px', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                {row.equivalents != null ? row.equivalents.toFixed(2) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function generatePrintHtml(protocol, moleculeInfo) {
  const molName = moleculeInfo?.name || 'Молекула'
  const molSmiles = moleculeInfo?.smiles || ''
  const targetMass = protocol.target_mass_g ?? protocol.calculations?.target_mass_g
  const sections = protocol.reaction_sections || []
  const buyable = protocol.buyable_reagent_table || []
  const isMulti = protocol.is_multistep && sections.length > 1
  const now = new Date().toLocaleString('ru-RU')

  const tableRow = (r) => `
    <tr>
      <td>${r.name || r.smiles || '—'}</td>
      <td>${r.mass_g != null ? r.mass_g.toFixed(4) : '—'}</td>
      <td>${r.volume_ml != null ? r.volume_ml.toFixed(3) : '—'}</td>
      <td>${r.moles != null ? Number(r.moles).toExponential(3) : '—'}</td>
      <td>${r.equivalents != null ? Number(r.equivalents).toFixed(2) : '—'}</td>
    </tr>`

  const formatFormula = (rxnSmi, reagentTable) => {
    if (!rxnSmi || !rxnSmi.includes('>>')) return ''
    const [lhs, rhs] = rxnSmi.split('>>')
    const nameMap = {}
    reagentTable?.forEach(r => { if (r.smiles) nameMap[r.smiles] = r.name })
    const fmt = s => nameMap[s] || s
    const reactants = lhs.split('.').filter(Boolean).map(fmt).join(' + ')
    const products = (rhs || '').split('.').filter(Boolean).map(fmt).join(' + ')
    return `
      <div class="formula-box">
        <div class="formula-label">Уравнение реакции</div>
        <div class="formula">${reactants} <span class="arrow">→</span> ${products}</div>
        <div class="formula-smiles">SMILES: ${rxnSmi.slice(0, 150)}${rxnSmi.length > 150 ? '…' : ''}</div>
      </div>`
  }

  const sectionsHtml = sections.map((s, si) => {
    const stepLabel = isMulti ? `Стадия ${s.step_number ?? si + 1}` : 'Синтез'
    const rxnFormula = formatFormula(s.reaction_smiles, s.reagent_table)
    const reagentRows = (s.reagent_table || []).map(tableRow).join('')
    const procSteps = (s.procedure_steps || []).map((step, pi) => {
      const desc = typeof step === 'string' ? step : (step.description || JSON.stringify(step))
      const reason = typeof step === 'object' && step.reason && step.reason !== 'inferred' ? `<div class="step-reason">${step.reason}</div>` : ''
      return `<div class="proc-step"><span class="step-num">${pi + 1}</span><div><div>${desc}</div>${reason}</div></div>`
    }).join('')

    return `
      <div class="section">
        <div class="section-header">
          <span class="step-badge">${stepLabel}</span>
          <span class="section-title">${s.product_name || s.product_smiles || ''}</span>
          ${s.product_mass_g ? `<span class="mass-badge">${Number(s.product_mass_g).toFixed(3)} г</span>` : ''}
        </div>
        ${rxnFormula}
        ${reagentRows ? `
          <div class="subsection-title">Реагенты стадии</div>
          <table class="data-table">
            <thead><tr><th>Реагент</th><th>Масса, г</th><th>Объём, мл</th><th>Моль</th><th>Экв.</th></tr></thead>
            <tbody>${reagentRows}</tbody>
          </table>` : ''}
        ${procSteps ? `
          <div class="subsection-title">Процедура</div>
          <div class="procedure">${procSteps}</div>` : ''}
      </div>`
  }).join('')

  const buyableRows = buyable.map(tableRow).join('')
  const warnings = (protocol.calculations?.warnings || []).map(w => `<li>${w}</li>`).join('')

  return `<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Протокол синтеза: ${molName}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; color: #1a2030; background: white; }
  .page { max-width: 800px; margin: 0 auto; padding: 20mm 15mm; }
  h1 { font-size: 18pt; color: #0a1628; margin-bottom: 4px; }
  .meta { font-size: 9pt; color: #666; margin-bottom: 6px; }
  .smiles { font-family: monospace; font-size: 9pt; color: #335; background: #f4f6fa; padding: 4px 8px; border-radius: 4px; word-break: break-all; margin-bottom: 16px; }
  .section { border: 1px solid #d0d8e8; border-radius: 8px; margin-bottom: 20px; overflow: hidden; page-break-inside: avoid; }
  .section-header { background: #f0f4fb; padding: 10px 14px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid #d0d8e8; }
  .step-badge { font-size: 9pt; font-weight: 700; color: #1a5fb4; background: #dce9f8; border: 1px solid #b0c8e8; padding: 2px 8px; border-radius: 4px; }
  .mass-badge { margin-left: auto; font-size: 9pt; color: #2d6a4f; background: #d8f3dc; border: 1px solid #95d5b2; padding: 2px 8px; border-radius: 4px; font-family: monospace; }
  .section-title { font-size: 12pt; font-weight: 600; color: #0a1628; }
  .formula-box { margin: 12px 14px; background: #f8faff; border: 1px solid #ccd8f0; border-radius: 6px; padding: 10px 14px; }
  .formula-label { font-size: 8pt; text-transform: uppercase; letter-spacing: 0.5px; color: #888; font-weight: 600; margin-bottom: 6px; }
  .formula { font-family: monospace; font-size: 11pt; color: #1a2030; word-break: break-all; }
  .formula .arrow { color: #1a5fb4; font-weight: 700; font-size: 14pt; margin: 0 6px; }
  .formula-smiles { margin-top: 6px; font-size: 8pt; color: #888; font-family: monospace; word-break: break-all; }
  .subsection-title { font-size: 9pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #555; padding: 8px 14px 4px; }
  table.data-table { width: 100%; border-collapse: collapse; margin: 0 0 12px; }
  table.data-table th { background: #f0f4fb; padding: 6px 10px; text-align: left; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.4px; color: #555; font-weight: 600; border-bottom: 1px solid #d0d8e8; }
  table.data-table td { padding: 6px 10px; border-bottom: 1px solid #eaecf0; font-size: 10pt; }
  table.data-table tr:last-child td { border-bottom: none; }
  .procedure { padding: 0 14px 12px; }
  .proc-step { display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start; }
  .step-num { min-width: 22px; height: 22px; background: #1a5fb4; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 9pt; font-weight: 700; flex-shrink: 0; margin-top: 1px; }
  .step-reason { font-size: 9pt; color: #888; margin-top: 3px; font-style: italic; }
  .summary-block { background: #f0f8f4; border: 1px solid #95d5b2; border-radius: 8px; padding: 14px; margin-bottom: 20px; }
  .summary-title { font-size: 11pt; font-weight: 700; color: #2d6a4f; margin-bottom: 10px; }
  .warnings { color: #b45309; font-size: 10pt; margin-top: 12px; padding-left: 20px; }
  .footer { margin-top: 24px; font-size: 8pt; color: #aaa; text-align: center; border-top: 1px solid #eaecf0; padding-top: 10px; }
  @media print {
    @page { margin: 15mm; }
    body { font-size: 10pt; }
    .section { page-break-inside: avoid; }
  }
</style>
</head>
<body>
<div class="page">
  <h1>Протокол синтеза</h1>
  <div class="meta">Молекула: <strong>${molName}</strong> · Целевая масса: <strong>${targetMass != null ? Number(targetMass).toFixed(3) + ' г' : '—'}</strong> · Дата: ${now}</div>
  ${molSmiles ? `<div class="smiles">SMILES: ${molSmiles}</div>` : ''}

  ${buyable.length > 1 ? `
  <div class="summary-block">
    <div class="summary-title">Сводная таблица закупок (коммерческие реагенты)</div>
    <table class="data-table">
      <thead><tr><th>Реагент</th><th>Масса, г</th><th>Объём, мл</th><th>Моль</th><th>Экв.</th></tr></thead>
      <tbody>${buyableRows}</tbody>
    </table>
  </div>` : ''}

  ${sectionsHtml}

  ${warnings ? `<ul class="warnings"><strong>Предупреждения:</strong>${warnings}</ul>` : ''}

  <div class="footer">Сгенерировано MolPipeline · ${now}</div>
</div>
</body>
</html>`
}

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

export default function ExperimentProtocol({ protocol, moleculeInfo, sessionId }) {
  if (!protocol) return null

  const reagentTable = protocol.buyable_reagent_table || []
  const sections = protocol.reaction_sections || []
  const targetMass = protocol.target_mass_g ?? protocol.calculations?.target_mass_g
  const isMulti = protocol.is_multistep && sections.length > 1

  const handleDownloadPdf = () => {
    const html = generatePrintHtml(protocol, moleculeInfo)
    const win = window.open('', '_blank', 'width=900,height=700')
    if (!win) { alert('Разрешите всплывающие окна для этого сайта'); return }
    win.document.write(html)
    win.document.close()
    win.focus()
    setTimeout(() => { win.print() }, 600)
  }

  const handleDownloadJournal = async () => {
    if (!sessionId) return
    try {
      const res = await fetch(`${API_BASE}/journal/${sessionId}/md`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `journal_${sessionId}.md`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('Не удалось скачать журнал: ' + e.message)
    }
  }

  return (
    <div style={{ marginTop: 8 }}>
      {/* ── Header ── */}
      <div style={{
        fontSize: 16, fontWeight: 700, color: 'var(--cyan)', marginBottom: 16,
        display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
      }}>
        Протокол эксперимента
        {targetMass != null && (
          <span style={{
            fontSize: 12, fontWeight: 500,
            background: 'var(--cyan)18', color: 'var(--cyan)',
            border: '1px solid var(--cyan)40',
            padding: '2px 10px', borderRadius: 20,
            fontFamily: 'var(--font-mono)',
          }}>
            {Number(targetMass).toFixed(3)} г
          </span>
        )}
        {isMulti && (
          <span style={{
            fontSize: 11, fontWeight: 500,
            background: 'var(--green)18', color: 'var(--green)',
            border: '1px solid var(--green)40',
            padding: '2px 10px', borderRadius: 20,
            fontFamily: 'var(--font-mono)',
          }}>
            {sections.length} стадии
          </span>
        )}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {sessionId && (
            <button
              onClick={handleDownloadJournal}
              title="Скачать журнал агента (Markdown)"
              style={{
                padding: '6px 14px', fontSize: 12,
                fontFamily: 'var(--font-mono)',
                background: 'var(--bg-2)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--r-sm)',
                color: 'var(--text-3)',
                cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 6,
              }}
            >
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 2h7l3 3v9H3z" /><path d="M10 2v3h3" /><path d="M6 7h4M6 10h4M6 13h2" />
              </svg>
              Журнал агента
            </button>
          )}
          <button
            onClick={handleDownloadPdf}
            style={{
              padding: '6px 14px', fontSize: 12,
              fontFamily: 'var(--font-mono)',
              background: 'var(--bg-2)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--r-sm)',
              color: 'var(--text-2)',
              cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 2v9M4 7l4 4 4-4M2 14h12" />
            </svg>
            Скачать PDF
          </button>
        </div>
      </div>

      {/* ── Summary reagent table (only for multi-step) ── */}
      {isMulti && reagentTable.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div className="section-title">Сводная таблица реагентов</div>
          <ReagentTable rows={reagentTable} />
        </div>
      )}

      {/* Single-step: summary at top */}
      {!isMulti && reagentTable.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div className="section-title">Сводная таблица реагентов</div>
          <ReagentTable rows={reagentTable} />
        </div>
      )}

      {/* ── Reaction sections ── */}
      {sections.map((section, si) => (
        <div key={si} style={{
          marginBottom: 20,
          background: 'var(--bg-2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--r-md)',
          overflow: 'hidden',
        }}>
          {/* Section header */}
          <div style={{
            padding: '10px 14px',
            borderBottom: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
          }}>
            <span style={{
              fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
              padding: '2px 8px', borderRadius: 4,
              background: 'var(--cyan)18', color: 'var(--cyan)',
              border: '1px solid var(--cyan)40',
            }}>
              {isMulti ? `Стадия ${section.step_number ?? si + 1}` : `Шаг ${section.step_number ?? si + 1}`}
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-1)' }}>
              {section.product_name || `Реакция ${si + 1}`}
            </span>
            {section.product_mass_g != null && (
              <span style={{
                marginLeft: 'auto', fontSize: 11,
                fontFamily: 'var(--font-mono)', color: 'var(--green)',
                background: 'var(--green)12', border: '1px solid var(--green)30',
                padding: '1px 8px', borderRadius: 10,
              }}>
                {Number(section.product_mass_g).toFixed(3)} г
              </span>
            )}
          </div>

          <div style={{ padding: '12px 14px' }}>

            {/* Reaction formula */}
            <ReactionFormula
              reactionSmiles={section.reaction_smiles}
              reagentTable={section.reagent_table}
            />

            {/* Reagent table for this step */}
            {section.reagent_table?.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <div className="section-title">Реагенты этого шага</div>
                <ReagentTable rows={section.reagent_table} compact />
              </div>
            )}

            {/* Procedure steps */}
            {section.procedure_steps?.length > 0 && (
              <div>
                <div className="section-title">Процедура</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {section.procedure_steps.map((step, pi) => (
                    <div key={pi} style={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      borderLeft: '3px solid var(--cyan-dim)',
                      borderRadius: '0 var(--r-sm) var(--r-sm) 0',
                      padding: '8px 12px',
                    }}>
                      <div style={{
                        fontSize: 10, color: 'var(--cyan)', fontFamily: 'var(--font-mono)',
                        marginBottom: 3, fontWeight: 600,
                      }}>
                        {pi + 1}
                      </div>
                      <div style={{ fontSize: 13, color: 'var(--text-1)', lineHeight: 1.55 }}>
                        {typeof step === 'string' ? step : (step.description || JSON.stringify(step))}
                      </div>
                      {typeof step === 'object' && step.reason && step.reason !== 'inferred' && step.reason !== 'ORD процедура' && (
                        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4, fontStyle: 'italic' }}>
                          {step.reason}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Warnings */}
      {protocol.calculations?.warnings?.length > 0 && (
        <div style={{
          padding: '10px 14px',
          background: 'var(--amber)12',
          border: '1px solid var(--amber)40',
          borderRadius: 'var(--r-md)',
          fontSize: 12,
          color: 'var(--amber)',
          fontFamily: 'var(--font-mono)',
        }}>
          {protocol.calculations.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
        </div>
      )}
    </div>
  )
}

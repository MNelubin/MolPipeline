/**
 * ExperimentProtocol — displays the experiment_protocol from Phase 3 state.
 *
 * protocol shape:
 * {
 *   target_mass_g: number,
 *   buyable_reagent_table: [{ name, mass_g, volume_ml, ... }],
 *   reaction_sections: [
 *     {
 *       step_number: number,
 *       product_name: string,
 *       procedure_steps: [string],
 *       reagent_table: [{ name, mass_g, volume_ml, ... }],
 *     }
 *   ]
 * }
 */

export default function ExperimentProtocol({ protocol }) {
  if (!protocol) return null

  const reagentTable = protocol.buyable_reagent_table || []
  const sections = protocol.reaction_sections || []
  const targetMass = protocol.target_mass_g

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{
        fontSize: 16, fontWeight: 700, color: 'var(--cyan)', marginBottom: 16,
        display: 'flex', alignItems: 'center', gap: 10,
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
            {targetMass.toFixed(3)} г
          </span>
        )}
      </div>

      {/* ── Summary reagent table ── */}
      {reagentTable.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <div className="section-title">Сводная таблица реагентов</div>
          <div style={{
            background: 'var(--bg-2)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--r-md)',
            overflow: 'auto',
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Реагент', 'Масса, г', 'Объём, мл'].map(h => (
                    <th key={h} style={{
                      padding: '8px 12px', textAlign: 'left',
                      color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
                      fontWeight: 600, fontSize: 10, textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {reagentTable.map((row, i) => (
                  <tr
                    key={i}
                    style={{
                      borderBottom: i < reagentTable.length - 1 ? '1px solid var(--border)' : 'none',
                    }}
                  >
                    <td style={{ padding: '8px 12px', color: 'var(--text-1)', fontWeight: 500 }}>
                      {row.name || row.smiles || '—'}
                    </td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                      {row.mass_g != null ? row.mass_g.toFixed(4) : '—'}
                    </td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                      {row.volume_ml != null ? row.volume_ml.toFixed(3) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
            display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <span style={{
              fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
              padding: '2px 8px', borderRadius: 4,
              background: 'var(--cyan)18', color: 'var(--cyan)',
              border: '1px solid var(--cyan)40',
            }}>
              Шаг {section.step_number ?? si + 1}
            </span>
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-1)' }}>
              {section.product_name || `Реакция ${si + 1}`}
            </span>
          </div>

          <div style={{ padding: '12px 14px' }}>

            {/* Reagent table for this step */}
            {section.reagent_table?.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <div className="section-title">Реагенты этого шага</div>
                <div style={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--r-sm)',
                  overflow: 'auto',
                }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Реагент', 'Масса, г', 'Объём, мл'].map(h => (
                          <th key={h} style={{
                            padding: '6px 10px', textAlign: 'left',
                            color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
                            fontWeight: 600, fontSize: 10, textTransform: 'uppercase',
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {section.reagent_table.map((row, ri) => (
                        <tr key={ri} style={{ borderBottom: ri < section.reagent_table.length - 1 ? '1px solid var(--border)' : 'none' }}>
                          <td style={{ padding: '6px 10px', color: 'var(--text-1)', fontWeight: 500 }}>
                            {row.name || row.smiles || '—'}
                          </td>
                          <td style={{ padding: '6px 10px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                            {row.mass_g != null ? row.mass_g.toFixed(4) : '—'}
                          </td>
                          <td style={{ padding: '6px 10px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                            {row.volume_ml != null ? row.volume_ml.toFixed(3) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
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
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>
      ))}
    </div>
  )
}

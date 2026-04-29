import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'

const STATUS_COLOR = {
  PASSED:  'var(--green, #4ade80)',
  FAILED:  'var(--red,   #f87171)',
  ERROR:   'var(--red,   #f87171)',
  SKIPPED: 'var(--amber, #fbbf24)',
  XFAIL:   'var(--amber, #fbbf24)',
  XPASS:   'var(--cyan,  #22d3ee)',
}

export default function TestPage() {
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [filter, setFilter] = useState('all')  // all | failed | passed

  async function runTests() {
    setRunning(true)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/tests/run`, { method: 'POST' })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setResult({ error: e.message })
    } finally {
      setRunning(false)
    }
  }

  const tests = result?.tests || []
  const filtered = filter === 'all' ? tests : tests.filter(t => t.status === filter.toUpperCase())
  const allPassed = result && result.failed === 0 && result.error === 0

  return (
    <div style={{ padding: '32px', maxWidth: 900, margin: '0 auto', fontFamily: 'monospace' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, color: 'var(--fg, #e2e8f0)' }}>
          pytest — Test Runner
        </h1>
        <button
          onClick={runTests}
          disabled={running}
          style={{
            padding: '8px 20px',
            background: running ? '#374151' : 'var(--cyan, #22d3ee)',
            color: running ? '#9ca3af' : '#0f172a',
            border: 'none',
            borderRadius: 6,
            cursor: running ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            fontSize: 14,
          }}
        >
          {running ? '⏳ Running...' : '▶ Run Tests'}
        </button>
      </div>

      {running && (
        <div style={{ color: '#94a3b8', marginBottom: 16 }}>
          Running <code>pytest mvp/tests/ -v</code> on server...
        </div>
      )}

      {result?.error && (
        <div style={{ background: '#450a0a', border: '1px solid #f87171', borderRadius: 8, padding: 16, color: '#f87171' }}>
          Error: {result.error}
        </div>
      )}

      {result && !result.error && (
        <>
          {/* Summary bar */}
          <div style={{
            background: allPassed ? '#052e16' : '#450a0a',
            border: `1px solid ${allPassed ? '#4ade80' : '#f87171'}`,
            borderRadius: 8, padding: '14px 20px', marginBottom: 20,
            display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap',
          }}>
            <span style={{ fontSize: 20, fontWeight: 700, color: allPassed ? '#4ade80' : '#f87171' }}>
              {allPassed ? '✓ ALL PASSED' : '✗ SOME FAILED'}
            </span>
            <span style={{ color: '#4ade80' }}>✓ {result.passed} passed</span>
            {result.failed > 0 && <span style={{ color: '#f87171' }}>✗ {result.failed} failed</span>}
            {result.error > 0 && <span style={{ color: '#f87171' }}>⚠ {result.error} errors</span>}
            {result.skipped > 0 && <span style={{ color: '#fbbf24' }}>— {result.skipped} skipped</span>}
            <span style={{ color: '#64748b', marginLeft: 'auto' }}>
              {result.total} total · {result.duration_sec}s
            </span>
          </div>

          {/* Filter buttons */}
          {tests.length > 0 && (
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              {['all', 'passed', 'failed'].map(f => (
                <button key={f} onClick={() => setFilter(f)} style={{
                  padding: '4px 12px', borderRadius: 4, border: '1px solid #334155',
                  background: filter === f ? '#1e293b' : 'transparent',
                  color: filter === f ? '#e2e8f0' : '#64748b',
                  cursor: 'pointer', fontSize: 12, textTransform: 'capitalize',
                }}>
                  {f === 'all' ? `All (${tests.length})`
                    : f === 'passed' ? `Passed (${result.passed})`
                    : `Failed (${result.failed + result.error})`}
                </button>
              ))}
            </div>
          )}

          {/* Test list */}
          {filtered.length > 0 && (
            <div style={{
              background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8,
              overflow: 'hidden', marginBottom: 20,
            }}>
              {filtered.map((t, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '6px 14px',
                  borderBottom: i < filtered.length - 1 ? '1px solid #1e293b' : 'none',
                  background: i % 2 === 0 ? 'transparent' : '#ffffff04',
                }}>
                  <span style={{
                    color: STATUS_COLOR[t.status] || '#94a3b8',
                    fontSize: 11, fontWeight: 700, minWidth: 52,
                  }}>
                    {t.status === 'PASSED' ? '✓' : t.status === 'FAILED' || t.status === 'ERROR' ? '✗' : '−'}{' '}
                    {t.status}
                  </span>
                  <span style={{ color: '#cbd5e1', fontSize: 12, wordBreak: 'break-all' }}>
                    {t.name}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Raw output */}
          {result.output && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', color: '#64748b', fontSize: 13, marginBottom: 8 }}>
                Raw pytest output
              </summary>
              <pre style={{
                background: '#0a0f1a', border: '1px solid #1e293b', borderRadius: 6,
                padding: 16, fontSize: 11, color: '#94a3b8', overflowX: 'auto',
                maxHeight: 400, overflowY: 'auto', whiteSpace: 'pre-wrap',
              }}>
                {result.output}
              </pre>
            </details>
          )}
        </>
      )}
    </div>
  )
}

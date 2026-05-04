import { useCallback, useMemo, useRef, useState } from 'react'
import { Editor } from 'ketcher-react'
import { StandaloneStructServiceProvider } from 'ketcher-standalone'
import 'ketcher-react/dist/index.css'

export default function MoleculeEditor({
  initialSmiles = '',
  disabled = false,
  onUseSmiles,
  onRunRetrosynthesis,
}) {
  const ketcherRef = useRef(null)
  const structServiceProvider = useMemo(() => new StandaloneStructServiceProvider(), [])
  const [smiles, setSmiles] = useState(initialSmiles)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')

  const readSmiles = useCallback(async () => {
    if (!ketcherRef.current) return ''
    const nextSmiles = (await ketcherRef.current.getSmiles()).trim()
    setSmiles(nextSmiles)
    return nextSmiles
  }, [])

  const handleInit = useCallback(async (ketcher) => {
    ketcherRef.current = ketcher
    if (initialSmiles.trim()) {
      try {
        await ketcher.setMolecule(initialSmiles.trim())
        setSmiles(initialSmiles.trim())
      } catch (err) {
        setError(err?.message || String(err))
      }
    }
  }, [initialSmiles])

  const handleLoadFromInput = useCallback(async () => {
    const value = smiles.trim() || initialSmiles.trim()
    if (!value || !ketcherRef.current) return
    setStatus('loading')
    setError('')
    try {
      await ketcherRef.current.setMolecule(value)
      setStatus('idle')
    } catch (err) {
      setError(err?.message || String(err))
      setStatus('idle')
    }
  }, [initialSmiles, smiles])

  const handleUse = useCallback(async () => {
    setStatus('reading')
    setError('')
    try {
      const value = await readSmiles()
      if (!value) throw new Error('Нарисуйте молекулу или вставьте SMILES')
      onUseSmiles?.(value)
    } catch (err) {
      setError(err?.message || String(err))
    } finally {
      setStatus('idle')
    }
  }, [onUseSmiles, readSmiles])

  const handleRun = useCallback(async () => {
    setStatus('running')
    setError('')
    try {
      const value = await readSmiles()
      if (!value) throw new Error('Нарисуйте молекулу или вставьте SMILES')
      await onRunRetrosynthesis?.(value)
    } catch (err) {
      setError(err?.message || String(err))
    } finally {
      setStatus('idle')
    }
  }, [onRunRetrosynthesis, readSmiles])

  return (
    <section className="molecule-editor-panel">
      <div className="molecule-editor-header">
        <div>
          <div className="molecule-editor-title">Редактор молекулы</div>
          <div className="molecule-editor-subtitle">Нарисуйте структуру и отправьте SMILES в выбранный источник ретросинтеза</div>
        </div>
        <div className="molecule-editor-actions">
          <button type="button" className="editor-btn ghost" onClick={handleLoadFromInput} disabled={disabled || status !== 'idle'}>
            Загрузить SMILES
          </button>
          <button type="button" className="editor-btn ghost" onClick={handleUse} disabled={disabled || status !== 'idle'}>
            Вставить в запрос
          </button>
          <button type="button" className="editor-btn primary" onClick={handleRun} disabled={disabled || status !== 'idle'}>
            Ретросинтез
          </button>
        </div>
      </div>

      <div className="molecule-editor-input-row">
        <input
          className="molecule-editor-smiles-input"
          value={smiles}
          onChange={event => setSmiles(event.target.value)}
          placeholder="SMILES для загрузки в редактор"
          disabled={disabled}
        />
      </div>

      <div className="molecule-editor-shell">
        <Editor
          staticResourcesUrl="/"
          structServiceProvider={structServiceProvider}
          onInit={handleInit}
          errorHandler={(message) => setError(String(message))}
        />
      </div>

      {error && <div className="molecule-editor-error">{error}</div>}
    </section>
  )
}

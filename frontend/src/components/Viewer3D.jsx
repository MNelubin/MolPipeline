import { useEffect, useRef } from 'react'

export default function Viewer3D({ smiles, cid }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || (!smiles && !cid)) return
    if (typeof window.$3Dmol === 'undefined') {
      containerRef.current.innerHTML =
        '<div style="color:#4d6585;font-size:12px;padding:20px;font-family:monospace">3Dmol.js не загружен</div>'
      return
    }

    // Напрямую PubChem — по CID если есть, иначе по SMILES
    const url = cid
      ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/SDF?record_type=3d`
      : `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(smiles)}/SDF?record_type=3d`

    const container = containerRef.current
    container.innerHTML = ''

    const viewer = window.$3Dmol.createViewer(container, { backgroundColor: '#060810' })

    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.text()
      })
      .then(sdf => {
        viewer.addModel(sdf, 'sdf')
        viewer.setStyle({}, {
          stick:  { radius: 0.12, colorscheme: 'Jmol' },
          sphere: { scale: 0.22, colorscheme: 'Jmol' },
        })
        viewer.setBackgroundColor('#060810')
        viewer.zoomTo()
        viewer.render()
        viewer.spin('y', 0.5)
      })
      .catch(() => {
        container.innerHTML =
          '<div style="color:#4d6585;font-size:12px;padding:20px;font-family:monospace;text-align:center">3D структура недоступна в PubChem</div>'
      })

    return () => { try { viewer.spin(false) } catch {} }
  }, [smiles, cid])

  return <div ref={containerRef} className="viewer-3d-container" />
}

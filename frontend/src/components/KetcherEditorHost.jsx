import { Editor } from 'ketcher-react'
import { StandaloneStructServiceProvider } from 'ketcher-standalone'
import 'ketcher-react/dist/index.css'

const structServiceProvider = new StandaloneStructServiceProvider()

export default function KetcherEditorHost({ onInit, onError }) {
  return (
    <Editor
      staticResourcesUrl=""
      disableMacromoleculesEditor
      structServiceProvider={structServiceProvider}
      onInit={onInit}
      errorHandler={(message) => onError?.(message)}
    />
  )
}

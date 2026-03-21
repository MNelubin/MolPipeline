import React from 'react'
import { NODE_ORDER } from '../hooks/useSSEPipeline'

const NODE_META = {
  validate:       { label: 'Validate',  short: 'VAL' },
  guard:          { label: 'Guard',     short: 'GRD' },
  molecule_info:  { label: 'Info',      short: 'INF' },
  retrosynthesis: { label: 'Retro',     short: 'RET' },
}

export default function PipelineProgress({ nodes }) {
  return (
    <div className="pipeline-progress">
      {NODE_ORDER.map((key, idx) => {
        const status = nodes[key] || 'idle'
        const meta = NODE_META[key]
        return (
          <React.Fragment key={key}>
            <div className={`pipeline-node ${status}`}>
              <div className="node-dot" />
              <span>{meta.label}</span>
            </div>
            {idx < NODE_ORDER.length - 1 && (
              <div className="pipeline-arrow" />
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}

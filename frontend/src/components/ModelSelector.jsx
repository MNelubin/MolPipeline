import { useState, useEffect } from 'react'

const FALLBACK = [
  { id: 'openai/gpt-5.4-nano',         name: 'GPT-5.4 Nano',      provider: 'OpenAI' },
  { id: 'openai/gpt-4o',               name: 'GPT-4o',            provider: 'OpenAI' },
  { id: 'openai/gpt-4o-mini',          name: 'GPT-4o Mini',       provider: 'OpenAI' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { id: 'anthropic/claude-3-haiku',    name: 'Claude 3 Haiku',    provider: 'Anthropic' },
  { id: 'google/gemini-2.0-flash-001', name: 'Gemini 2.0 Flash',  provider: 'Google' },
  { id: 'mistralai/mistral-small-3.1-24b-instruct', name: 'Mistral Small 3.1', provider: 'Mistral' },
]

export default function ModelSelector({ value, onChange, disabled }) {
  const [models, setModels] = useState(FALLBACK)

  useEffect(() => {
    fetch('/api/models')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data) && data.length) setModels(data) })
      .catch(() => {})
  }, [])

  // Group by provider
  const byProvider = models.reduce((acc, m) => {
    if (!acc[m.provider]) acc[m.provider] = []
    acc[m.provider].push(m)
    return acc
  }, {})

  return (
    <div className="model-selector">
      <span className="model-label">Model</span>
      <select
        className="model-select"
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      >
        {Object.entries(byProvider).map(([provider, list]) => (
          <optgroup key={provider} label={provider}>
            {list.map(m => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  )
}

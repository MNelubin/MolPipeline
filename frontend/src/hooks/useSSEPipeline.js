/**
 * useSSEPipeline — внешний REST API (hack.humaneconomy.ru).
 *
 * /analyze возвращает:
 * {
 *   "status": "ok" | "banned" | "invalid" | "error",
 *   "query": "...",
 *   "output": "полный текст",
 *   "state": { validation, guard_result, molecule_info, retro_result, ... }
 * }
 */

import { useState, useRef, useCallback } from 'react'

export const NODE_ORDER = ['validate', 'guard', 'molecule_info', 'retrosynthesis']

const API_BASE = import.meta.env.VITE_API_URL || 'https://hack.humaneconomy.ru'
const API_URL = `${API_BASE}/analyze`

export function useSSEPipeline() {
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef(null)

  const run = useCallback(async (query, model, onEvent) => {
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setIsStreaming(true)

    onEvent({ type: 'pipeline_start', data: { query, model } })

    // Все ноды сразу в running
    onEvent({ type: 'node_start', data: { node: 'validate',       label: 'Валидация' } })
    onEvent({ type: 'node_start', data: { node: 'guard',          label: 'Проверка безопасности' } })
    onEvent({ type: 'node_start', data: { node: 'molecule_info',  label: 'Сбор данных о молекуле' } })
    onEvent({ type: 'node_start', data: { node: 'retrosynthesis', label: 'Ретросинтез' } })

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const err = await response.text()
        onEvent({ type: 'error', data: { message: `HTTP ${response.status}: ${err}` } })
        return
      }

      const data = await response.json()
      const state = data.state || {}

      // validate
      onEvent({
        type: 'node_complete',
        data: {
          node: 'validate',
          output: {
            validation:  state.validation  || {},
            smiles:      state.smiles      || '',
            pubchem_cid: state.pubchem_cid || 0,
          },
        },
      })

      // guard
      onEvent({
        type: 'node_complete',
        data: {
          node: 'guard',
          output: { guard_result: state.guard_result || {} },
        },
      })

      // molecule_info
      if (data.status === 'ok' && state.molecule_info) {
        onEvent({
          type: 'node_complete',
          data: {
            node: 'molecule_info',
            output: {
              molecule_info: state.molecule_info,
              final_answer:  data.output || '',
            },
          },
        })
      }

      // retrosynthesis — берём retro_result из state
      onEvent({
        type: 'node_complete',
        data: {
          node: 'retrosynthesis',
          output: { retro_result: state.retro_result || null },
        },
      })

      if (data.status !== 'ok') {
        onEvent({
          type: 'error',
          data: { message: state.error || data.output || `Статус: ${data.status}` },
        })
      }

      onEvent({ type: 'pipeline_done', data: {} })

    } catch (err) {
      if (err.name !== 'AbortError') {
        onEvent({ type: 'error', data: { message: err.message } })
      }
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [])

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      setIsStreaming(false)
    }
  }, [])

  return { run, cancel, isStreaming }
}

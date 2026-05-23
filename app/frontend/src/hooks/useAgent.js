// hooks/useAgent.js — API calls to the FastAPI backend

const API = '/api'

/**
 * Extract a human-readable message from a failed HTTP response.
 * FastAPI wraps errors as {"detail": "..."}; falls back to raw text.
 */
async function _errorMessage(res) {
  const text = await res.text()
  try {
    const body = JSON.parse(text)
    if (body && typeof body.detail === 'string') return body.detail
    if (body && typeof body.detail === 'object') return JSON.stringify(body.detail)
  } catch (_) { /* not JSON */ }
  return text || `HTTP ${res.status}`
}

export function useAgent() {
  /**
   * Send caregiver text to the agent.
   * Returns { pictograms: [...], turn: N } or throws on error.
   */
  async function runAgent(text) {
    const res = await fetch(`${API}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
    if (!res.ok) throw new Error(await _errorMessage(res))
    return res.json()
  }

  /**
   * Record the subject's pictogram selection.
   */
  async function selectPictogram(pictogramId) {
    const res = await fetch(`${API}/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pictogram_id: pictogramId }),
    })
    if (!res.ok) throw new Error(await _errorMessage(res))
    return res.json()
  }

  /**
   * Reset the session.
   */
  async function resetSession() {
    const res = await fetch(`${API}/reset`, { method: 'POST' })
    if (!res.ok) throw new Error(await _errorMessage(res))
    return res.json()
  }

  /**
   * Fetch full session history.
   */
  async function getSession() {
    const res = await fetch(`${API}/session`)
    if (!res.ok) throw new Error(await _errorMessage(res))
    return res.json()
  }

  /**
   * Fetch backend health (model name + Ollama status).
   */
  async function getHealth() {
    const res = await fetch(`${API}/health`)
    if (!res.ok) return { ok: false, model: '?', ollama: false }
    return res.json()
  }

  return { runAgent, selectPictogram, resetSession, getSession, getHealth }
}

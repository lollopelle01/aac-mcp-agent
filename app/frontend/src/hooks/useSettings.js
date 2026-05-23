// hooks/useSettings.js — GET/PATCH /settings

const API = '/api'

export function useSettings() {
  async function getSettings() {
    const res = await fetch(`${API}/settings`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  async function patchSettings(updates) {
    const res = await fetch(`${API}/settings`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  return { getSettings, patchSettings }
}

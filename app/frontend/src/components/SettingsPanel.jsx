// components/SettingsPanel.jsx — modal overlay for editing user settings
import React, { useEffect, useState } from 'react'
import { useSettings } from '../hooks/useSettings.js'

const overlay = {
  position: 'fixed', inset: 0,
  background: 'rgba(0,0,0,0.45)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  zIndex: 100,
}
const panel = {
  background: '#fff',
  borderRadius: '16px',
  padding: '28px 32px',
  width: '480px',
  maxWidth: '95vw',
  maxHeight: '80vh',
  overflowY: 'auto',
  boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
}
const field = { marginBottom: '18px' }
const labelStyle = {
  display: 'block', fontSize: '12px', fontWeight: '700',
  color: '#718096', textTransform: 'uppercase', letterSpacing: '0.7px',
  marginBottom: '6px',
}
const inputStyle = {
  width: '100%', padding: '9px 12px', border: '1.5px solid #cbd5e0',
  borderRadius: '8px', fontSize: '14px', fontFamily: 'inherit',
  outline: 'none', background: '#f8f9fa',
}
const btnRow = { display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '24px' }
const btn = (primary) => ({
  padding: '9px 20px', borderRadius: '8px', border: 'none',
  fontSize: '14px', fontWeight: '600', cursor: 'pointer',
  background: primary ? '#4f7fe0' : '#e2e8f0',
  color: primary ? '#fff' : '#1a1a2e',
})

// Fields the settings panel exposes (credentials stay in .env)
const EDITABLE = [
  { key: 'agent_default_model',       label: 'Default model',           type: 'text' },
  { key: 'agent_max_results',         label: 'Window size (pictograms shown)',  type: 'number' },
  { key: 'agent_candidates_per_term', label: 'Candidates per keyword',          type: 'number' },
  { key: 'agent_memory_turns',        label: 'Memory turns',             type: 'number' },
  { key: 'timezone',                  label: 'Timezone',                 type: 'text' },
  { key: 'calendar_provider',         label: 'Calendar provider (apple|google)', type: 'text' },
  { key: 'agent_fetch_schedule',      label: 'Fetch schedule (true|false)', type: 'text' },
  { key: 'agent_synset_expand',       label: 'Synset expand (true|false)', type: 'text' },
]

export function SettingsPanel({ onClose }) {
  const { getSettings, patchSettings } = useSettings()
  const [values, setValues] = useState({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSettings().then(setValues).catch((e) => setError(String(e)))
  }, [])

  function handleChange(key, raw) {
    setValues((v) => ({ ...v, [key]: raw }))
  }

  async function save() {
    setSaving(true)
    setError(null)
    try {
      // Coerce number strings and booleans before sending
      const updates = {}
      for (const { key, type } of EDITABLE) {
        const raw = values[key]
        if (raw === undefined) continue
        if (type === 'number')        updates[key] = Number(raw)
        else if (raw === 'true')      updates[key] = true
        else if (raw === 'false')     updates[key] = false
        else                         updates[key] = raw
      }
      await patchSettings(updates)
      onClose()
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={panel}>
        <h2 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '20px' }}>
          Settings
        </h2>

        {error && (
          <p style={{ color: '#e53e3e', fontSize: '13px', marginBottom: '14px' }}>{error}</p>
        )}

        {EDITABLE.map(({ key, label }) => (
          <div key={key} style={field}>
            <label style={labelStyle}>{label}</label>
            <input
              style={inputStyle}
              value={values[key] !== undefined ? String(values[key]) : ''}
              onChange={(e) => handleChange(key, e.target.value)}
            />
          </div>
        ))}

        <p style={{ fontSize: '12px', color: '#a0aec0', marginTop: '8px' }}>
          Sensitive credentials (API keys, passwords) must be edited manually in{' '}
          <code>app/.env</code>.
        </p>

        <div style={btnRow}>
          <button style={btn(false)} onClick={onClose}>Cancel</button>
          <button style={btn(true)} onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

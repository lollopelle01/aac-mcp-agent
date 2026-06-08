// components/DatasetPanel.jsx — modal for local dataset status + update
import React, { useEffect, useState, useRef } from 'react'
import { useDatasets } from '../hooks/useDatasets.js'

// ── Styles ───────────────────────────────────────────────────────────────────
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
  width: '540px',
  maxWidth: '95vw',
  maxHeight: '85vh',
  overflowY: 'auto',
  boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
  display: 'flex',
  flexDirection: 'column',
  gap: '18px',
}
const title = {
  fontSize: '18px', fontWeight: '700', margin: 0,
}
const sectionLabel = {
  fontSize: '12px', fontWeight: '700',
  color: '#718096', textTransform: 'uppercase', letterSpacing: '0.7px',
  marginBottom: '10px',
}
const card = {
  border: '1.5px solid #e2e8f0',
  borderRadius: '10px',
  padding: '14px 16px',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: '12px',
}
const cardLabel = { fontSize: '14px', fontWeight: '700', color: '#1a1a2e' }
const cardMeta  = { fontSize: '12px', color: '#718096', marginTop: '3px' }
const badge = (ok) => ({
  padding: '3px 10px',
  borderRadius: '12px',
  fontSize: '11px',
  fontWeight: '700',
  background: ok ? 'rgba(72,187,120,0.12)' : 'rgba(160,174,192,0.15)',
  color: ok ? '#276749' : '#718096',
  border: `1px solid ${ok ? '#9ae6b4' : '#cbd5e0'}`,
  whiteSpace: 'nowrap',
})
const checkRow = {
  display: 'flex', alignItems: 'center', gap: '8px',
  fontSize: '14px', cursor: 'pointer', userSelect: 'none',
}
const logBox = {
  background: '#1a1a2e',
  borderRadius: '8px',
  padding: '12px 14px',
  fontFamily: 'monospace',
  fontSize: '12px',
  color: '#e2e8f0',
  maxHeight: '200px',
  overflowY: 'auto',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-all',
}
const btnRow = { display: 'flex', gap: '10px', justifyContent: 'flex-end' }
const btn = (variant) => ({
  padding: '9px 20px', borderRadius: '8px', border: 'none',
  fontSize: '14px', fontWeight: '600', cursor: 'pointer',
  ...(variant === 'primary'  ? { background: '#4f7fe0', color: '#fff' } :
      variant === 'danger'   ? { background: '#fc8181', color: '#fff' } :
                               { background: '#e2e8f0', color: '#1a1a2e' }),
})

function fmt(ts) {
  if (!ts) return '—'
  try { return new Date(ts).toLocaleString() } catch (_) { return ts }
}

// ── Component ─────────────────────────────────────────────────────────────────
export function DatasetPanel({ onClose }) {
  const { getStatus, startUpdate } = useDatasets()

  const [status,   setStatus]   = useState(null)   // { languages: {...} }
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)

  // Update job state
  const [running,        setRunning]        = useState(false)
  const [logs,           setLogs]           = useState([])
  const [force,          setForce]          = useState(false)
  const [downloadImages, setDownloadImages] = useState(false)
  const [jobDone,        setJobDone]        = useState(null)  // null | true | false

  const logRef = useRef(null)

  // Auto-scroll log box
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  // Fetch status on mount
  useEffect(() => {
    fetchStatus()
  }, [])

  async function fetchStatus() {
    setLoading(true)
    setError(null)
    try {
      const data = await getStatus()
      setStatus(data)
    } catch (e) {
      setError(`Could not load dataset status: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handleUpdate() {
    setRunning(true)
    setLogs([])
    setJobDone(null)
    setError(null)
    try {
      const result = await startUpdate({
        force,
        download_images: downloadImages,
        onLog: (msg) => setLogs((prev) => [...prev, msg]),
      })
      setJobDone(result.ok)
      // Refresh status after update
      await fetchStatus()
    } catch (e) {
      if (e.message?.includes('already running')) {
        setError('A dataset update job is already running. Try again shortly.')
      } else {
        setError(`Update failed: ${e.message}`)
      }
      setJobDone(false)
    } finally {
      setRunning(false)
    }
  }

  const langs = status?.languages ? Object.entries(status.languages) : []

  return (
    <div style={overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={panel}>
        <h2 style={title}>⬇ Local Datasets</h2>

        {/* ── Status cards ── */}
        <div>
          <p style={sectionLabel}>Current status</p>
          {loading && <p style={{ color: '#718096', fontSize: '14px' }}>Loading…</p>}
          {!loading && error && !langs.length && (
            <p style={{ color: '#e53e3e', fontSize: '13px' }}>{error}</p>
          )}
          {!loading && langs.length === 0 && !error && (
            <p style={{ color: '#718096', fontSize: '14px' }}>No languages configured.</p>
          )}
          {langs.map(([lang, info]) => (
            <div key={lang} style={{ ...card, marginBottom: '8px' }}>
              <div>
                <div style={cardLabel}>Language: {lang.toUpperCase()}</div>
                <div style={cardMeta}>
                  {info.records != null ? `${info.records.toLocaleString()} records` : 'unknown'}
                  {' · '}
                  {info.png_cached != null ? `${info.png_cached.toLocaleString()} PNG cached` : ''}
                </div>
                <div style={cardMeta}>
                  Updated: {fmt(info.last_updated)}
                </div>
              </div>
              <span style={badge(!!info.records)}>
                {info.records ? 'OK' : 'empty'}
              </span>
            </div>
          ))}

          {status?.png_total != null && (
            <p style={{ fontSize: '12px', color: '#a0aec0', marginTop: '4px' }}>
              Total PNG on disk: {status.png_total.toLocaleString()}
            </p>
          )}
        </div>

        {/* ── Options ── */}
        <div>
          <p style={sectionLabel}>Update options</p>
          <label style={checkRow}>
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
              disabled={running}
            />
            Force re-fetch (even if already up to date)
          </label>
          <label style={{ ...checkRow, marginTop: '10px' }}>
            <input
              type="checkbox"
              checked={downloadImages}
              onChange={(e) => setDownloadImages(e.target.checked)}
              disabled={running}
            />
            Download images (slow — ~50 k PNG files)
          </label>
          {downloadImages && (
            <p style={{ fontSize: '12px', color: '#dd6b20', marginTop: '6px' }}>
              ⚠ Downloading all images can take several minutes. Only needed before going offline.
            </p>
          )}
        </div>

        {/* ── Log stream ── */}
        {(running || logs.length > 0) && (
          <div>
            <p style={sectionLabel}>
              {running ? 'Update in progress…' : jobDone ? '✓ Update complete' : '✗ Update failed'}
            </p>
            <div style={logBox} ref={logRef}>
              {logs.length === 0
                ? 'Starting…'
                : logs.join('\n')}
            </div>
          </div>
        )}

        {/* ── Errors ── */}
        {error && (
          <p style={{ color: '#e53e3e', fontSize: '13px', margin: 0 }}>{error}</p>
        )}

        {/* ── Buttons ── */}
        <div style={btnRow}>
          <button style={btn('neutral')} onClick={onClose} disabled={running}>
            Close
          </button>
          <button
            style={btn(running ? 'neutral' : 'primary')}
            onClick={handleUpdate}
            disabled={running || loading}
          >
            {running ? 'Updating…' : 'Update now'}
          </button>
        </div>
      </div>
    </div>
  )
}

// App.jsx — root component
import React, { useState, useEffect, useCallback } from 'react'
import { InputBar }       from './components/InputBar.jsx'
import { PictogramGrid }  from './components/PictogramGrid.jsx'
import { SessionSidebar } from './components/SessionSidebar.jsx'
import { SettingsPanel }  from './components/SettingsPanel.jsx'
import { DatasetPanel }      from './components/DatasetPanel.jsx'
import { CategoryBrowser }  from './components/CategoryBrowser.jsx'
import { useAgent }       from './hooks/useAgent.js'
import { useSettings }    from './hooks/useSettings.js'

// ── Styles ────────────────────────────────────────────────────────────────────
const layout = {
  display: 'flex', flexDirection: 'column', height: '100vh',
}
const header = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  padding: '10px 16px',
  background: '#1a1a2e',
  color: '#fff',
  gap: '10px',
  flexShrink: 0,
}
const headerTitle = { fontSize: '16px', fontWeight: '700', letterSpacing: '0.5px' }
const headerRight = { display: 'flex', alignItems: 'center', gap: '10px' }
const ollamaDot = (ok) => ({
  width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0,
  background: ok ? '#68d391' : '#fc8181',
})
const modelSelect = {
  fontSize: '13px',
  background: 'rgba(255,255,255,0.1)',
  border: '1.5px solid rgba(255,255,255,0.25)',
  borderRadius: '8px',
  color: '#e2e8f0',
  padding: '4px 8px',
  cursor: 'pointer',
  outline: 'none',
  maxWidth: '180px',
}
const iconBtn = {
  background: 'none', border: '1.5px solid rgba(255,255,255,0.3)',
  borderRadius: '8px', color: '#fff', padding: '4px 10px',
  cursor: 'pointer', fontSize: '14px',
}
const main = { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }
const errorBanner = {
  background: '#fff5f5', color: '#c53030', fontSize: '13px',
  padding: '8px 16px', borderBottom: '1px solid #fed7d7',
}

export default function App() {
  const { runAgent, selectPictogram, resetSession, getSession, getHealth } = useAgent()
  const { getSettings, patchSettings } = useSettings()

  const [input,        setInput]        = useState('')
  const [loading,      setLoading]      = useState(false)
  const [pictograms,   setPictograms]   = useState([])
  const [selectedId,   setSelectedId]   = useState(null)
  const [toolsCalled,  setToolsCalled]  = useState(null)  // null = no turn yet
  const [sessionTurns, setSessionTurns] = useState([])
  const [health,       setHealth]       = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [showDatasets,  setShowDatasets]  = useState(false)
  const [showCategories, setShowCategories] = useState(false)
  const [error,        setError]        = useState(null)
  const [warmingUp,    setWarmingUp]    = useState(true)  // true until backend confirms model ready
  const [warmupError,  setWarmupError]  = useState(null)
  // Model selector
  const [models,        setModels]       = useState({})   // { "granite4:3b-h": {size_gb: 4.4}, … }
  const [currentModel,  setCurrentModel] = useState('')
  const [modelSwitching, setModelSwitching] = useState(false)

  // ── Bootstrap + warmup polling ───────────────────────────────────────────
  useEffect(() => {
    getSettings()
      .then(s => {
        setModels(s.models || {})
        setCurrentModel(m => m || s.agent_default_model || '')
      })
      .catch(() => {})

    // Poll /health until warming_up=false
    let cancelled = false
    async function pollHealth() {
      while (!cancelled) {
        try {
          const h = await getHealth()
          setHealth(h)
          setCurrentModel(m => m || h.model || '')
          if (!h.warming_up) {
            setWarmingUp(false)
            setWarmupError(h.warmup_error || null)
            break
          }
        } catch (_) {}
        await new Promise(r => setTimeout(r, 1500))
      }
    }
    pollHealth()
    return () => { cancelled = true }
  }, [])

  // ── Session refresh ───────────────────────────────────────────────────────
  const refreshSession = useCallback(async () => {
    try {
      const data = await getSession()
      setSessionTurns(data.turns || [])
    } catch (_) {}
  }, [getSession])

  // ── Core search ───────────────────────────────────────────────────────────
  async function _doSearch(text) {
    setLoading(true)
    setError(null)
    try {
      const data = await runAgent(text)
      setPictograms(data.pictograms || [])
      setToolsCalled(data.tools_called ?? null)
    } catch (e) {
      // Give a helpful hint when the model isn't pulled yet
      const msg = e.message || ''
      const hint = msg.toLowerCase().includes('not found') || msg.toLowerCase().includes('pull')
        ? ` — run: ollama pull ${currentModel}`
        : ''
      setError(`Agent error: ${msg}${hint}`)
      setPictograms([])
    } finally {
      setLoading(false)
    }
  }

  async function handleSubmit() {
    if (!input.trim() || loading) return
    setSelectedId(null)
    await _doSearch(input.trim())
  }

  // ── Pictogram selection → auto-continue ───────────────────────────────────
  async function handleSelect(pic) {
    setSelectedId(pic.id)
    try {
      await selectPictogram(pic.id)
      await refreshSession()
      setSelectedId(null)
      // Auto-continue with the same input so the agent suggests the next concept.
      // The updated session memory influences what the planner proposes next.
      if (input.trim()) {
        await _doSearch(input.trim())
      } else {
        setPictograms([])
      }
    } catch (e) {
      setError(`Select error: ${e.message}`)
    }
  }

  // ── Reset ─────────────────────────────────────────────────────────────────
  async function handleReset() {
    await resetSession().catch(() => {})
    setInput('')
    setPictograms([])
    setSelectedId(null)
    setSessionTurns([])
    setToolsCalled(null)
    setError(null)
  }

  // ── Model switch ──────────────────────────────────────────────────────────
  async function handleModelChange(e) {
    const model = e.target.value
    setCurrentModel(model)
    setModelSwitching(true)
    setWarmingUp(true)  // backend will load the new GGUF in background
    try {
      await patchSettings({ agent_default_model: model })
      // Re-enter warmup polling loop
      let tries = 0
      const pollSwitch = async () => {
        while (tries++ < 60) {
          try {
            const h = await getHealth()
            setHealth(h)
            if (!h.warming_up) {
              setWarmingUp(false)
              setWarmupError(h.warmup_error || null)
              break
            }
          } catch (_) {}
          await new Promise(r => setTimeout(r, 1500))
        }
      }
      pollSwitch()
    } catch (err) {
      setError(`Could not switch model: ${err.message}`)
      setWarmingUp(false)
    } finally {
      setModelSwitching(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  const modelKeys = Object.keys(models)

  return (
    <div style={layout}>
      {/* ── Header ── */}
      <div style={header}>
        <span style={headerTitle}>AAC Pictogram Agent</span>
        <div style={headerRight}>

          {/* Tools-called indicator */}
          {toolsCalled !== null && (
            <span
              title={toolsCalled
                ? 'Planner called tools (get_time / get_schedule)'
                : 'Planner did NOT call tools — input already explicit'}
              style={{
                fontSize: '11px',
                padding: '2px 7px',
                borderRadius: '10px',
                background: toolsCalled ? 'rgba(104,211,145,0.2)' : 'rgba(160,174,192,0.15)',
                color: toolsCalled ? '#68d391' : '#a0aec0',
                border: `1px solid ${toolsCalled ? '#68d391' : '#4a5568'}`,
                whiteSpace: 'nowrap',
                cursor: 'default',
              }}
            >
              {toolsCalled ? 'tool' : 'no tool'}
            </span>
          )}

          {/* Ollama status dot */}
          {health && (
            <span
              style={ollamaDot(health.ollama)}
              title={health.ollama ? 'Ollama reachable' : 'Ollama not reachable'}
            />
          )}

          {/* Model dropdown */}
          {modelKeys.length > 0 ? (
            <select
              style={modelSelect}
              value={currentModel}
              onChange={handleModelChange}
              disabled={modelSwitching || loading}
              title="Active Ollama model"
            >
              {modelKeys.map(m => (
                <option key={m} value={m}>
                  {m}{models[m]?.size_gb ? ` (${models[m].size_gb} GB)` : ''}
                </option>
              ))}
            </select>
          ) : (
            <span style={{ fontSize: '12px', color: '#a0aec0' }}>
              {currentModel || '…'}
            </span>
          )}

          <button style={iconBtn} onClick={() => setShowCategories(true)} title="Browse by category">
            Browse
          </button>
          <button style={iconBtn} onClick={() => setShowDatasets(true)} title="Local datasets">
            ⬇ Datasets
          </button>
          <button style={iconBtn} onClick={() => setShowSettings(true)} title="Settings">
            ⚙ Settings
          </button>
          <button style={iconBtn} onClick={handleReset} title="Reset session">
            ↺ Reset
          </button>
        </div>
      </div>

      {/* ── Warmup banner ── */}
      {warmingUp && (
        <div style={{
          background: '#fffbeb', color: '#92400e', fontSize: '13px',
          padding: '7px 16px', borderBottom: '1px solid #fde68a',
          display: 'flex', alignItems: 'center', gap: '8px',
        }}>
          <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⏳</span>
          Loading model into memory… first inference will start immediately after.
          {warmupError && <span style={{ color: '#c53030', marginLeft: 8 }}>⚠️ {warmupError}</span>}
        </div>
      )}

      {/* ── Error banner ── */}
      {error && <div style={errorBanner}>{error}</div>}

      {/* ── Input ── */}
      <InputBar
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        loading={loading}
        warmingUp={warmingUp}
      />

      {/* ── Pictogram grid ── */}
      <div style={main}>
        <PictogramGrid
          pictograms={pictograms}
          loading={loading}
          selectedId={selectedId}
          onSelect={handleSelect}
        />
      </div>

      {/* ── Session bar ── */}
      <SessionSidebar turns={sessionTurns} />

      {/* ── Category browser ── */}
      {showCategories && (
        <CategoryBrowser
          onClose={() => setShowCategories(false)}
          onSelect={async (pic) => {
            setShowCategories(false)
            await handleSelect(pic)
          }}
          lang="en"
        />
      )}

      {/* ── Dataset modal ── */}
      {showDatasets && (
        <DatasetPanel onClose={() => setShowDatasets(false)} />
      )}

      {/* ── Settings modal ── */}
      {showSettings && (
        <SettingsPanel
          onClose={() => {
            setShowSettings(false)
            // Refresh models list and health after settings change
            getSettings().then(s => setModels(s.models || {})).catch(() => {})
            getHealth().then(setHealth).catch(() => {})
          }}
        />
      )}
    </div>
  )
}

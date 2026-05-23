// CategoryBrowser.jsx — navigazione a 3 livelli per la ricerca manuale dei pittogrammi
//
// Livello 0 — macro-categorie (~15 card con emoji + immagine rappresentativa)
// Livello 1 — categorie ARASAAC figlie della macro selezionata
// Livello 2 — pittogrammi della categoria selezionata (riusa lo stile di PictogramCard)
//
// Chiamate API:
//   GET /api/categories?lang=en   → {macros: [...]}
//   GET /api/by_category?category=food&lang=en&max_results=50  → [...]
//
// Al tap su un pittogramma si chiama onSelect(pic) — stesso contratto di PictogramGrid.

import React, { useState, useEffect, useRef } from 'react'

// ── Stili ─────────────────────────────────────────────────────────────────────

const overlay = {
  position: 'fixed', inset: 0,
  background: 'rgba(0,0,0,0.55)',
  zIndex: 200,
  display: 'flex', alignItems: 'stretch', justifyContent: 'flex-end',
}

const panel = {
  background: '#fff',
  width: 'min(100vw, 520px)',
  display: 'flex', flexDirection: 'column',
  boxShadow: '-4px 0 24px rgba(0,0,0,0.25)',
}

const panelHeader = {
  display: 'flex', alignItems: 'center', gap: '8px',
  padding: '12px 16px',
  background: '#1a1a2e',
  color: '#fff',
  flexShrink: 0,
}

const closeBtn = {
  marginLeft: 'auto',
  background: 'none', border: 'none', color: '#fff',
  fontSize: '20px', cursor: 'pointer', lineHeight: 1,
  padding: '0 4px',
}

const backBtn = {
  background: 'rgba(255,255,255,0.15)',
  border: 'none', color: '#fff',
  borderRadius: '6px', padding: '3px 10px',
  fontSize: '13px', cursor: 'pointer',
}

const breadcrumb = {
  fontSize: '13px', color: 'rgba(255,255,255,0.7)',
  flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
}

const scrollArea = {
  flex: 1, overflowY: 'auto',
  padding: '12px',
}

const gridMacro = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
  gap: '10px',
}

const gridCat = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
  gap: '10px',
}

const gridPics = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
  gap: '8px',
}

const card = (selected) => ({
  border: selected ? '2.5px solid #4299e1' : '1.5px solid #e2e8f0',
  borderRadius: '10px',
  padding: '8px 6px',
  cursor: 'pointer',
  background: selected ? '#ebf8ff' : '#fafafa',
  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px',
  transition: 'box-shadow 0.15s, border-color 0.15s',
  userSelect: 'none',
})

const cardLabel = {
  fontSize: '11px', textAlign: 'center', color: '#2d3748',
  lineHeight: 1.2, maxWidth: '100%',
  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  width: '100%',
}

const cardCount = {
  fontSize: '10px', color: '#718096',
}

const thumbStyle = (size = 72) => ({
  width: `${size}px`, height: `${size}px`,
  objectFit: 'contain', borderRadius: '6px',
  background: '#f7fafc',
})

const emojiIcon = (size = 48) => ({
  fontSize: `${size * 0.55}px`,
  width: `${size}px`, height: `${size}px`,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  background: '#edf2f7', borderRadius: '8px',
})

const spinnerWrap = {
  display: 'flex', justifyContent: 'center', alignItems: 'center',
  padding: '40px',
}

const emptyMsg = {
  color: '#718096', fontSize: '14px', textAlign: 'center', padding: '32px',
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const API_BASE = '/api'

async function fetchCategories(lang = 'en') {
  const r = await fetch(`${API_BASE}/categories?lang=${lang}`)
  if (!r.ok) throw new Error(`/categories ${r.status}`)
  return r.json()   // {macros: [...]}
}

async function fetchByCategory(category, lang = 'en', maxResults = 60) {
  const params = new URLSearchParams({ category, lang, max_results: maxResults })
  const r = await fetch(`${API_BASE}/by_category?${params}`)
  if (!r.ok) throw new Error(`/by_category ${r.status}`)
  return r.json()   // [{id, label, image_url, categories, aac}]
}

// Thumbnail con fallback emoji se l'immagine non carica
function Thumb({ imageUrl, alt, size = 72, emoji = '🖼' }) {
  const [failed, setFailed] = useState(false)
  if (failed || !imageUrl) {
    return <div style={emojiIcon(size)}>{emoji}</div>
  }
  return (
    <img
      src={imageUrl}
      alt={alt}
      style={thumbStyle(size)}
      onError={() => setFailed(true)}
      loading="lazy"
    />
  )
}

function imageUrlForId(id) {
  return `${API_BASE}/images/${id}`
}

// ── Livello 0 — griglia macro-categorie ──────────────────────────────────────

function MacroGrid({ macros, onSelect }) {
  return (
    <div style={gridMacro}>
      {macros.map(mc => (
        <div
          key={mc.name}
          style={card(false)}
          onClick={() => onSelect(mc)}
          title={`${mc.count} pittogrammi`}
        >
          <Thumb
            imageUrl={mc.representative_id ? imageUrlForId(mc.representative_id) : null}
            alt={mc.name}
            size={72}
            emoji={mc.emoji}
          />
          <span style={cardLabel}>{mc.name}</span>
          <span style={cardCount}>{mc.count}</span>
        </div>
      ))}
    </div>
  )
}

// ── Livello 1 — griglia categorie ARASAAC ─────────────────────────────────────

function CategoryGrid({ categories, onSelect }) {
  if (!categories.length) return <p style={emptyMsg}>No categories found.</p>
  return (
    <div style={gridCat}>
      {categories.map(cat => (
        <div
          key={cat.name}
          style={card(false)}
          onClick={() => onSelect(cat)}
          title={`${cat.count} pittogrammi`}
        >
          <Thumb
            imageUrl={cat.representative_id ? imageUrlForId(cat.representative_id) : null}
            alt={cat.name}
            size={64}
            emoji="🏷"
          />
          <span style={cardLabel}>{cat.name}</span>
          <span style={cardCount}>{cat.count}</span>
        </div>
      ))}
    </div>
  )
}

// ── Livello 2 — griglia pittogrammi ──────────────────────────────────────────

function PicGrid({ pics, selectedId, onSelect }) {
  if (!pics.length) return <p style={emptyMsg}>Nessun pittogramma trovato.</p>
  return (
    <div style={gridPics}>
      {pics.map(p => (
        <div
          key={p.id}
          style={card(p.id === selectedId)}
          onClick={() => onSelect(p)}
          title={p.label}
        >
          <Thumb
            imageUrl={p.image_url}
            alt={p.label}
            size={80}
            emoji="🖼"
          />
          <span style={cardLabel}>{p.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Componente principale ─────────────────────────────────────────────────────

/**
 * CategoryBrowser
 *
 * Props:
 *   onClose()          — chiude il pannello (senza selezione)
 *   onSelect(pic)      — pittogramma scelto; stesso contratto di PictogramGrid
 *   lang               — lingua dataset (default "en")
 */
export function CategoryBrowser({ onClose, onSelect, lang = 'en' }) {
  // Livello di navigazione: 0 | 1 | 2
  const [level,      setLevel]      = useState(0)
  const [macros,     setMacros]     = useState([])
  const [activeMacro, setActiveMacro] = useState(null)   // oggetto macro
  const [activeCat,  setActiveCat]  = useState(null)     // oggetto categoria ARASAAC
  const [pics,       setPics]       = useState([])
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const scrollRef = useRef(null)

  // Load macro-categories on mount
  useEffect(() => {
    setLoading(true)
    fetchCategories(lang)
      .then(data => setMacros(data.macros || []))
      .catch(e => setError(`Failed to load categories: ${e.message}`))
      .finally(() => setLoading(false))
  }, [lang])

  // Scorre in cima ad ogni cambio livello
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [level])

  // ── Navigazione ──────────────────────────────────────────────────────────

  function selectMacro(mc) {
    setActiveMacro(mc)
    setLevel(1)
  }

  function selectCategory(cat) {
    setActiveCat(cat)
    setLevel(2)
    setLoading(true)
    setError(null)
    setPics([])
    fetchByCategory(cat.name, lang)
      .then(data => setPics(data))
      .catch(e => setError(`Failed to load pictograms: ${e.message}`))
      .finally(() => setLoading(false))
  }

  function goBack() {
    if (level === 2) { setLevel(1); setPics([]); setSelectedId(null) }
    else if (level === 1) { setLevel(0); setActiveMacro(null) }
  }

  // ── Selezione pittogramma ────────────────────────────────────────────────

  function handleSelectPic(pic) {
    setSelectedId(pic.id)
    // Normalizza il formato atteso da App.jsx (stesso di PictogramGrid)
    onSelect(pic)
  }

  // ── Breadcrumb ───────────────────────────────────────────────────────────

  function breadcrumbText() {
    if (level === 0) return 'All categories'
    if (level === 1) return activeMacro?.name || ''
    return `${activeMacro?.name} › ${activeCat?.name}`
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div style={overlay} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={panel}>
        {/* Header */}
        <div style={panelHeader}>
          {level > 0 && (
            <button style={backBtn} onClick={goBack} title="Go back">
              ← Back
            </button>
          )}
          <span style={breadcrumb}>{breadcrumbText()}</span>
          <button style={closeBtn} onClick={onClose} title="Close">✕</button>
        </div>

        {/* Titolo livello */}
        <div style={{ padding: '8px 16px 4px', borderBottom: '1px solid #e2e8f0', flexShrink: 0 }}>
          <span style={{ fontSize: '13px', color: '#4a5568', fontWeight: '600' }}>
            {level === 0 && 'Choose a category'}
            {level === 1 && `${activeMacro?.name} (choose a subcategory)`}
            {level === 2 && `${activeCat?.name} (${pics.length} pictograms)`}
          </span>
        </div>

        {/* Contenuto scrollabile */}
        <div style={scrollArea} ref={scrollRef}>
          {error && (
            <p style={{ color: '#c53030', fontSize: '13px', marginBottom: '12px' }}>{error}</p>
          )}

          {loading && (
            <div style={spinnerWrap}>
              <span style={{ fontSize: '22px' }}>⏳</span>
            </div>
          )}

          {!loading && !error && (
            <>
              {level === 0 && (
                <MacroGrid macros={macros} onSelect={selectMacro} />
              )}
              {level === 1 && activeMacro && (
                <CategoryGrid
                  categories={activeMacro.categories || []}
                  onSelect={selectCategory}
                />
              )}
              {level === 2 && (
                <PicGrid
                  pics={pics}
                  selectedId={selectedId}
                  onSelect={handleSelectPic}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

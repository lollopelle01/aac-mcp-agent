// components/PictogramGrid.jsx
import React, { useState, useEffect } from 'react'
import { PictogramCard } from './PictogramCard.jsx'

const styles = {
  wrapper: {
    display: 'grid',
    gridTemplateColumns: 'repeat(5, 1fr)',
    gap: '12px',
    padding: '16px',
  },
  spinner: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '200px',
    color: '#718096',
    fontSize: '15px',
    gap: '14px',
  },
  dotsRow: {
    display: 'flex',
    gap: '6px',
  },
  dot: (i, active) => ({
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: active === i ? '#4f7fe0' : '#cbd5e0',
    transition: 'background 0.3s',
  }),
  phase: {
    fontSize: '13px',
    color: '#718096',
    fontStyle: 'italic',
    textAlign: 'center',
    maxWidth: '220px',
    lineHeight: '1.4',
  },
  timer: {
    fontSize: '11px',
    color: '#a0aec0',
  },
  empty: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '200px',
    color: '#a0aec0',
    fontSize: '14px',
    fontStyle: 'italic',
  },
}

// Pipeline: planner LLM -> ARASAAC search -> deterministic ranking.
// Phases reflect the actual flow (no second LLM filter).
const PHASES = [
  { label: 'Planning concepts…',           seconds: 10 },
  { label: 'Searching ARASAAC catalogue…', seconds: Infinity },
]

function LoadingSpinner() {
  const [elapsed,   setElapsed]   = useState(0)
  const [activeDot, setActiveDot] = useState(0)

  useEffect(() => {
    const tick = setInterval(() => {
      setElapsed(s => s + 1)
      setActiveDot(d => (d + 1) % 3)
    }, 1000)
    return () => clearInterval(tick)
  }, [])

  // Which phase label to show
  let phaseLabel = PHASES[0].label
  let cumulative = 0
  for (const p of PHASES) {
    cumulative += p.seconds
    if (elapsed < cumulative) { phaseLabel = p.label; break }
    phaseLabel = p.label   // last phase stays
  }

  return (
    <div style={styles.spinner}>
      <div style={styles.dotsRow}>
        {[0, 1].map(i => <span key={i} style={styles.dot(i, activeDot % 2)} />)}
      </div>
      <span style={styles.phase}>{phaseLabel}</span>
      {elapsed >= 10 && (
        <span style={styles.timer}>
          {elapsed}s — small models on CPU can take 1–2 min
        </span>
      )}
    </div>
  )
}

// Inline keyframe injection (once)
if (typeof document !== 'undefined' && !document.getElementById('aac-pulse-style')) {
  const s = document.createElement('style')
  s.id = 'aac-pulse-style'
  s.textContent = `
    @keyframes pulse {
      0%, 100% { opacity: 0.3; transform: scale(0.8); }
      50%       { opacity: 1;   transform: scale(1.1); }
    }
  `
  document.head.appendChild(s)
}

export function PictogramGrid({ pictograms, loading, selectedId, onSelect }) {
  if (loading) return <LoadingSpinner />

  if (!pictograms || pictograms.length === 0) {
    return (
      <div style={styles.empty}>
        Type a description above and press Enter
      </div>
    )
  }

  return (
    <div style={styles.wrapper}>
      {pictograms.map((pic) => (
        <PictogramCard
          key={pic.id}
          pictogram={pic}
          selected={pic.id === selectedId}
          onClick={() => onSelect(pic)}
        />
      ))}
    </div>
  )
}

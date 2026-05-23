// components/SessionSidebar.jsx — Bottom bar showing pictograms selected so far
import React from 'react'

const styles = {
  bar: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 16px',
    background: '#fff',
    borderTop: '1px solid #e2e8f0',
    minHeight: '64px',
    overflowX: 'auto',
  },
  label: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#718096',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
    whiteSpace: 'nowrap',
    marginRight: '4px',
  },
  thumb: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '2px',
    flexShrink: 0,
  },
  img: {
    width: '40px',
    height: '40px',
    objectFit: 'contain',
    borderRadius: '6px',
    border: '1.5px solid #e2e8f0',
    background: '#f8f9fa',
  },
  thumbLabel: {
    fontSize: '10px',
    color: '#4a5568',
    maxWidth: '44px',
    textAlign: 'center',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  empty: {
    fontSize: '13px',
    color: '#a0aec0',
    fontStyle: 'italic',
  },
}

export function SessionSidebar({ turns }) {
  // Flatten all selected pictograms across turns (each turn has 1 after /select)
  const selected = turns.flatMap((t) =>
    t.pictograms.map((p) => ({ ...p, turnId: t.turn_id }))
  )

  return (
    <div style={styles.bar}>
      <span style={styles.label}>Session:</span>
      {selected.length === 0 ? (
        <span style={styles.empty}>No pictograms selected yet</span>
      ) : (
        selected.map((p, i) => (
          <div key={`${p.id}-${i}`} style={styles.thumb} title={p.label}>
            <img
              src={p.image_url}
              alt={p.label}
              style={styles.img}
              onError={(e) => { e.target.src = `/api/images/${p.id}` }}
            />
            <span style={styles.thumbLabel}>{p.label}</span>
          </div>
        ))
      )}
    </div>
  )
}

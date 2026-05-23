// components/PictogramCard.jsx
import React from 'react'

const styles = {
  card: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    background: '#fff',
    border: '2px solid #e2e8f0',
    borderRadius: '16px',
    padding: '12px 8px 8px',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    userSelect: 'none',
    minHeight: '130px',
    position: 'relative',
  },
  cardSelected: {
    border: '2px solid #4f7fe0',
    background: '#eef4ff',
    transform: 'scale(1.04)',
    boxShadow: '0 4px 16px rgba(79,127,224,0.25)',
  },
  img: {
    width: '80px',
    height: '80px',
    objectFit: 'contain',
    borderRadius: '8px',
  },
  label: {
    marginTop: '8px',
    fontSize: '13px',
    fontWeight: '600',
    textAlign: 'center',
    color: '#1a1a2e',
    lineHeight: 1.2,
  },
  aacBadge: {
    position: 'absolute',
    top: '6px',
    right: '6px',
    background: '#48bb78',
    color: '#fff',
    fontSize: '9px',
    fontWeight: '700',
    padding: '2px 5px',
    borderRadius: '6px',
    letterSpacing: '0.5px',
  },
}

export function PictogramCard({ pictogram, selected, onClick }) {
  return (
    <div
      style={{ ...styles.card, ...(selected ? styles.cardSelected : {}) }}
      onClick={onClick}
      role="button"
      aria-label={pictogram.label}
      aria-pressed={selected}
    >
      {pictogram.aac && <span style={styles.aacBadge}>AAC</span>}
      <img
        src={pictogram.image_url}
        alt={pictogram.label}
        style={styles.img}
        onError={(e) => {
          // Fallback to local backend endpoint if CDN unavailable
          e.target.src = `/api/images/${pictogram.id}`
        }}
      />
      <span style={styles.label}>{pictogram.label}</span>
    </div>
  )
}

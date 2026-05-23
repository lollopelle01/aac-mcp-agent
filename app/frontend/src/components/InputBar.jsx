// components/InputBar.jsx
import React, { useRef } from 'react'

const styles = {
  wrapper: {
    display: 'flex',
    gap: '10px',
    padding: '14px 16px',
    background: '#fff',
    borderBottom: '1px solid #e2e8f0',
  },
  textarea: {
    flex: 1,
    resize: 'none',
    border: '1.5px solid #cbd5e0',
    borderRadius: '10px',
    padding: '10px 14px',
    fontSize: '15px',
    fontFamily: 'inherit',
    outline: 'none',
    lineHeight: '1.5',
    transition: 'border-color 0.15s',
    background: '#f8f9fa',
  },
  textareaWarmingUp: {
    background: '#fefce8',
    color: '#92400e',
    border: '1.5px solid #fde68a',
  },
  button: {
    alignSelf: 'flex-end',
    padding: '10px 20px',
    background: '#4f7fe0',
    color: '#fff',
    border: 'none',
    borderRadius: '10px',
    fontSize: '15px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background 0.15s',
    whiteSpace: 'nowrap',
  },
  buttonDisabled: {
    background: '#a0aec0',
    cursor: 'not-allowed',
  },
}

export function InputBar({ value, onChange, onSubmit, loading, warmingUp = false }) {
  const ref = useRef(null)

  const isDisabled = loading || warmingUp

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!isDisabled && value.trim()) onSubmit()
    }
  }

  const placeholder = warmingUp
    ? 'Loading model, please wait\u2026'
    : "Describe what the person wants\u2026 (e.g. 'he wants something before going out')"

  return (
    <div style={styles.wrapper}>
      <textarea
        ref={ref}
        rows={2}
        style={{ ...styles.textarea, ...(warmingUp ? styles.textareaWarmingUp : {}) }}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKey}
        disabled={isDisabled}
      />
      <button
        style={{ ...styles.button, ...(isDisabled || !value.trim() ? styles.buttonDisabled : {}) }}
        onClick={onSubmit}
        disabled={isDisabled || !value.trim()}
      >
        {warmingUp ? '\u23f3' : loading ? '\u2026' : '\u2192 Search'}
      </button>
    </div>
  )
}

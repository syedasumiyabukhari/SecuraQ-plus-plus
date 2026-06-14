import React, { useState, useEffect } from 'react'
import { api } from '../services/api'

export default function Topbar({ title, subtitle }) {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    api.health().then(r => setHealth(r.data)).catch(() => setHealth(null))
  }, [])

  return (
    <header className="flex items-center justify-between px-6 py-3.5"
      style={{ borderBottom: '1px solid var(--border)', background: 'rgba(5,5,10,0.8)', backdropFilter: 'blur(12px)' }}>
      <div>
        {title && <h1 className="font-display font-semibold text-base" style={{ color: '#f0f0f8' }}>{title}</h1>}
        {subtitle && <p className="text-xs mt-0.5" style={{ color: 'var(--text-dim)' }}>{subtitle}</p>}
      </div>
      <div className="flex items-center gap-4 text-xs">
        <span className="font-mono px-2.5 py-1 rounded-full"
          style={{ background: 'rgba(200,169,110,.08)', border: '1px solid rgba(200,169,110,.2)', color: 'var(--gold-light)' }}>
          QENND-v2.0-q4
        </span>
        {health ? (
          <span className="dot-online" style={{ color: '#86efac' }}>
            ML {health.ml_pipeline ? 'Live' : 'Demo'}
          </span>
        ) : (
          <span className="dot-offline" style={{ color: '#fca5a5' }}>Offline</span>
        )}
      </div>
    </header>
  )
}

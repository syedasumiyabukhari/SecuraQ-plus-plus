import React from 'react'

const accents = {
  gold:  { border: 'rgba(200,169,110,.25)', glow: 'rgba(200,169,110,.08)', val: 'var(--gold-light)' },
  green: { border: 'rgba(34,197,94,.25)',   glow: 'rgba(34,197,94,.06)',   val: '#86efac' },
  red:   { border: 'rgba(239,68,68,.25)',   glow: 'rgba(239,68,68,.06)',   val: '#fca5a5' },
  blue:  { border: 'rgba(99,179,237,.25)',  glow: 'rgba(99,179,237,.06)',  val: '#90cdf4' },
  purple:{ border: 'rgba(167,139,250,.25)', glow: 'rgba(167,139,250,.06)', val: '#c4b5fd' },
}

export default function StatCard({ label, value, hint, accent = 'gold', icon }) {
  const a = accents[accent] || accents.gold
  return (
    <div className="rounded-xl p-4 flex flex-col gap-2"
      style={{ background: a.glow, border: `1px solid ${a.border}` }}>
      <div className="flex items-center justify-between">
        <span className="text-xs" style={{ color: 'var(--text-dim)' }}>{label}</span>
        {icon && <span className="text-base">{icon}</span>}
      </div>
      <div className="font-display font-bold text-2xl" style={{ color: a.val }}>{value}</div>
      {hint && <div className="text-[11px]" style={{ color: 'var(--text-dim)' }}>{hint}</div>}
    </div>
  )
}

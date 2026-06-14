import React, { useState, useEffect } from 'react'
import { api } from '../../services/api'

const ACTION_COLOR = {
  LOGIN:               '#86efac',
  LOGOUT:              'rgba(200,200,220,0.4)',
  REGISTER:            '#60a5fa',
  ADMIN_CREATE_USER:   '#a78bfa',
  ADMIN_UPDATE_USER:   '#fde047',
  ADMIN_DELETE_USER:   '#fca5a5',
  ADMIN_RESET_PASSWORD:'#fdba74',
}

export default function AdminAuditPage() {
  const [log, setLog]         = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter]   = useState('')
  const [limit, setLimit]     = useState(100)

  const load = () => {
    setLoading(true)
    api.adminAuditLog(limit)
      .then(r => setLog(r.data || []))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(load, [limit])

  const filtered = log.filter(a =>
    !filter || a.action?.includes(filter.toUpperCase()) ||
    a.email?.includes(filter) || a.detail?.includes(filter)
  )

  const actions = [...new Set(log.map(a => a.action))].sort()

  return (
    <div className="space-y-5 animate-fadeUp">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Audit Log</h2>
          <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
            All security-relevant platform events · {log.length} records
          </p>
        </div>
        <button onClick={load} className="text-xs px-4 py-2 rounded-lg transition-all flex items-center gap-2"
          style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.25)' }}>
          <span className={loading ? 'animate-spin' : ''}>⟳</span> Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap items-center">
        <input value={filter} onChange={e => setFilter(e.target.value)}
          placeholder="Filter by action, email, detail…"
          className="input-dark px-3 py-2 rounded-xl text-xs w-64" />
        <select value={limit} onChange={e => setLimit(+e.target.value)}
          className="input-dark px-3 py-2 rounded-xl text-xs">
          {[50, 100, 250, 500].map(n => <option key={n} value={n}>Last {n}</option>)}
        </select>
        <div className="flex gap-1.5 flex-wrap">
          {actions.map(a => (
            <button key={a} onClick={() => setFilter(f => f === a ? '' : a)}
              className="text-[10px] px-2 py-1 rounded transition-all"
              style={{
                background: filter === a ? 'rgba(200,169,110,0.12)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${filter === a ? 'rgba(200,169,110,0.4)' : 'rgba(200,200,220,0.08)'}`,
                color: filter === a ? 'var(--gold)' : 'rgba(200,200,220,0.4)',
              }}>
              {a}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="glass rounded-2xl overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>Loading…</div>
        ) : filtered.length === 0 ? (
          <div className="p-8 text-center text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>No matching records</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(255,255,255,0.02)' }}>
                  {['Time', 'Action', 'User', 'Detail', 'IP'].map(h => (
                    <th key={h} className="text-left px-4 py-3 font-medium"
                      style={{ color: 'rgba(200,200,220,0.45)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((a, i) => (
                  <tr key={a.id || i} className="border-b transition-all hover:bg-white/[0.015]"
                    style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                    <td className="px-4 py-2.5 font-mono text-[10px]" style={{ color: 'rgba(200,200,220,0.3)', whiteSpace: 'nowrap' }}>
                      {a.created_at?.replace('T', ' ').slice(0, 19)}
                    </td>
                    <td className="px-4 py-2.5">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono"
                        style={{
                          background: 'rgba(255,255,255,0.04)',
                          color: ACTION_COLOR[a.action] || 'rgba(200,200,220,0.5)',
                        }}>
                        {a.action}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono" style={{ color: 'rgba(200,200,220,0.55)' }}>{a.email || '—'}</td>
                    <td className="px-4 py-2.5" style={{ color: 'rgba(200,200,220,0.4)', maxWidth: 200 }}>
                      <span className="truncate block">{a.detail || '—'}</span>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-[10px]" style={{ color: 'rgba(200,200,220,0.25)' }}>
                      {a.ip || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

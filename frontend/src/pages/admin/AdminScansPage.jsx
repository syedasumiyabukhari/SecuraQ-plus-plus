import React, { useState, useEffect } from 'react'
import { api } from '../../services/api'

const SEV_STYLE = {
  CRITICAL: { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)',  text: '#fca5a5' },
  HIGH:     { bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.3)', text: '#fdba74' },
  MEDIUM:   { bg: 'rgba(234,179,8,0.12)',  border: 'rgba(234,179,8,0.3)',  text: '#fde047' },
  LOW:      { bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.3)',  text: '#86efac' },
}

function Modal({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className="relative z-10 w-full max-w-2xl glass rounded-2xl p-6 animate-fadeUp max-h-[80vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-5">
          <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>{title}</h3>
          <button onClick={onClose} className="text-xs" style={{ color: 'rgba(200,200,220,0.4)' }}>✕</button>
        </div>
        {children}
      </div>
    </div>
  )
}

export default function AdminScansPage() {
  const [scans, setScans]       = useState([])
  const [loading, setLoading]   = useState(true)
  const [filter, setFilter]     = useState('all')
  const [search, setSearch]     = useState('')
  const [detail, setDetail]     = useState(null)
  const [delTarget, setDelTarget] = useState(null)
  const [msg, setMsg]           = useState('')

  const load = () => {
    setLoading(true)
    api.listScans()
      .then(r => setScans(r.data || []))
      .catch(() => setMsg('Failed to load scans'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const openDetail = async (scan) => {
    try {
      const { data } = await api.getScanResults(scan.scan_id)
      setDetail(data)
    } catch {
      setMsg('Could not load scan details')
    }
  }

  const confirmDelete = async () => {
    try {
      await api.deleteScan(delTarget.scan_id)
      setMsg('Scan deleted')
      setDelTarget(null)
      load()
    } catch { setMsg('Delete failed') }
  }

  const filtered = scans
    .filter(s => {
      if (filter === 'completed') return s.status === 'completed'
      if (filter === 'failed')    return s.status === 'failed'
      if (filter === 'vulns')     return s.total_vulnerabilities > 0
      return true
    })
    .filter(s => s.filename?.toLowerCase().includes(search.toLowerCase()))

  const totalVulns = scans.reduce((a, s) => a + (s.total_vulnerabilities || 0), 0)

  return (
    <div className="space-y-5 animate-fadeUp">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Scan Management</h2>
            <span className="text-[10px] px-2 py-0.5 rounded font-mono"
              style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
              ADMIN
            </span>
          </div>
          <p className="text-xs" style={{ color: 'rgba(200,200,220,0.4)' }}>
            All platform scans · {scans.length} total · {totalVulns} vulnerabilities detected
          </p>
        </div>
        <button onClick={load}
          className="text-xs px-4 py-2 rounded-lg flex items-center gap-2 transition-all"
          style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.25)' }}>
          <span>↻</span> Refresh
        </button>
      </div>

      {msg && (
        <div className="text-xs rounded-lg px-3 py-2"
          style={{ background: 'rgba(200,169,110,0.08)', border: '1px solid rgba(200,169,110,0.2)', color: 'var(--gold)' }}>
          {msg}
        </div>
      )}

      {/* Stats bar */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { l: 'Total Scans',  v: scans.length,                                            c: 'var(--gold)' },
          { l: 'Completed',    v: scans.filter(s => s.status === 'completed').length,       c: '#86efac' },
          { l: 'Failed',       v: scans.filter(s => s.status === 'failed').length,          c: '#fca5a5' },
          { l: 'Vulns Found',  v: totalVulns,                                               c: '#fdba74' },
        ].map(s => (
          <div key={s.l} className="glass rounded-xl p-4 text-center">
            <div className="font-bold font-mono text-2xl" style={{ color: s.c }}>{s.v}</div>
            <div className="text-[11px] mt-1" style={{ color: 'rgba(200,200,220,0.35)' }}>{s.l}</div>
          </div>
        ))}
      </div>

      {/* Filters + search */}
      <div className="flex items-center gap-3 flex-wrap">
        {[['all','All'],['completed','Completed'],['failed','Failed'],['vulns','Has Vulns']].map(([k,l]) => (
          <button key={k} onClick={() => setFilter(k)}
            className="text-xs px-3 py-1.5 rounded-lg transition-all"
            style={{
              background: filter === k ? 'rgba(200,169,110,0.15)' : 'transparent',
              border: filter === k ? '1px solid rgba(200,169,110,0.4)' : '1px solid rgba(255,255,255,0.06)',
              color: filter === k ? 'var(--gold)' : 'rgba(200,200,220,0.5)',
            }}>
            {l}
          </button>
        ))}
        <input value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Filter by filename..."
          className="input-dark px-3 py-1.5 rounded-xl text-xs ml-auto w-48" />
      </div>

      {/* Table */}
      <div className="glass rounded-2xl overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>Loading scans...</div>
        ) : filtered.length === 0 ? (
          <div className="p-8 text-center text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>No scans found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(255,255,255,0.02)' }}>
                  {['File', 'Status', 'Vulnerabilities', 'Created', 'Actions'].map(h => (
                    <th key={h} className="text-left px-4 py-3 font-medium"
                      style={{ color: 'rgba(200,200,220,0.45)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((s, i) => (
                  <tr key={s.scan_id} className="border-b transition-all"
                    style={{ borderColor: 'rgba(255,255,255,0.04)', background: i%2===0?'transparent':'rgba(255,255,255,0.01)' }}>
                    <td className="px-4 py-3">
                      <div className="font-mono text-xs" style={{ color: '#e8e8f0' }}>{s.filename}</div>
                      <div className="text-[10px] font-mono mt-0.5 truncate max-w-[160px]"
                        style={{ color: 'rgba(200,200,220,0.25)' }}>{s.scan_id}</div>
                    </td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded text-[10px]"
                        style={{
                          background: s.status==='completed' ? 'rgba(34,197,94,0.1)' : s.status==='failed' ? 'rgba(239,68,68,0.1)' : 'rgba(234,179,8,0.1)',
                          color: s.status==='completed' ? '#86efac' : s.status==='failed' ? '#fca5a5' : '#fde047',
                        }}>
                        {s.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {s.total_vulnerabilities > 0 ? (
                        <span className="font-mono font-bold" style={{ color: '#fca5a5' }}>
                          {s.total_vulnerabilities} found
                        </span>
                      ) : (
                        <span style={{ color: '#86efac' }}>Clean</span>
                      )}
                    </td>
                    <td className="px-4 py-3 font-mono text-[10px]" style={{ color: 'rgba(200,200,220,0.3)' }}>
                      {s.created_at?.slice(0,16).replace('T',' ')}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <button onClick={() => openDetail(s)}
                          className="text-[10px] px-2 py-1 rounded transition-all"
                          style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.2)' }}>
                          View
                        </button>
                        <button
                          onClick={() => window.open(`/api/download-report/${s.scan_id}`, '_blank')}
                          className="text-[10px] px-2 py-1 rounded transition-all"
                          style={{ color: '#60a5fa', border: '1px solid rgba(96,165,250,0.2)' }}>
                          Report
                        </button>
                        <button onClick={() => setDelTarget(s)}
                          className="text-[10px] px-2 py-1 rounded transition-all"
                          style={{ color: '#fca5a5', border: '1px solid rgba(239,68,68,0.2)' }}>
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {detail && (
        <Modal title={`Scan: ${detail.filename}`} onClose={() => setDetail(null)}>
          <div className="space-y-4">
            {/* Meta */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              {[
                ['Scan ID',   detail.scan_id],
                ['File',      detail.filename],
                ['Status',    detail.status],
                ['Findings',  detail.total_vulnerabilities],
                ['Created',   detail.created_at?.slice(0,16).replace('T',' ')],
                ['Completed', detail.completed_at?.slice(0,16).replace('T',' ') || '—'],
              ].map(([k,v]) => (
                <div key={k} className="rounded-lg p-3"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div className="text-[10px] mb-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>{k}</div>
                  <div className="font-mono" style={{ color: '#e8e8f0' }}>{v}</div>
                </div>
              ))}
            </div>

            {/* Vulnerabilities */}
            {detail.vulnerabilities?.length > 0 ? (
              <div className="space-y-2">
                <p className="text-xs font-medium" style={{ color: 'rgba(200,200,220,0.5)' }}>
                  Vulnerabilities ({detail.vulnerabilities.length})
                </p>
                {detail.vulnerabilities.map((v, i) => {
                  const st = SEV_STYLE[v.severity] || SEV_STYLE.MEDIUM
                  return (
                    <div key={i} className="rounded-xl p-4"
                      style={{ background: st.bg, border: `1px solid ${st.border}` }}>
                      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-sm" style={{ color: st.text }}>{v.type}</span>
                          <span className="text-[10px] px-2 py-0.5 rounded font-mono"
                            style={{ background: 'rgba(0,0,0,0.3)', color: st.text }}>{v.severity}</span>
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                            style={{ background: 'rgba(0,0,0,0.2)', color: 'rgba(200,200,220,0.5)' }}>{v.cwe}</span>
                        </div>
                        <span className="text-xs" style={{ color: 'rgba(200,200,220,0.45)' }}>
                          Line {v.line_number} · {(v.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-xs mb-2" style={{ color: 'rgba(200,200,220,0.7)' }}>{v.description}</p>
                      <div className="rounded-lg px-3 py-2"
                        style={{ background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(0,0,0,0.4)' }}>
                        <code className="text-xs font-mono" style={{ color: '#fde68a' }}>{v.code_snippet}</code>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="rounded-xl p-4 text-center"
                style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.25)' }}>
                <p className="text-sm" style={{ color: '#86efac' }}>No vulnerabilities detected</p>
              </div>
            )}

            <button onClick={() => window.open(`/api/download-report/${detail.scan_id}`, '_blank')}
              className="btn-gold w-full py-2.5 rounded-xl text-sm font-semibold">
              Download Full Report
            </button>
          </div>
        </Modal>
      )}

      {/* Delete confirm */}
      {delTarget && (
        <Modal title="Delete Scan" onClose={() => setDelTarget(null)}>
          <p className="text-sm mb-5" style={{ color: 'rgba(200,200,220,0.6)' }}>
            Permanently delete scan for <strong style={{ color: '#e8e8f0' }}>{delTarget.filename}</strong>?
          </p>
          <div className="flex gap-3">
            <button onClick={() => setDelTarget(null)}
              className="flex-1 py-2.5 rounded-xl text-sm"
              style={{ border: '1px solid var(--border)', color: 'rgba(200,200,220,0.5)' }}>
              Cancel
            </button>
            <button onClick={confirmDelete}
              className="flex-1 py-2.5 rounded-xl text-sm font-semibold"
              style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
              Delete
            </button>
          </div>
        </Modal>
      )}
    </div>
  )
}

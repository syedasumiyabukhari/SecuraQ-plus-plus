import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../services/api'

const SEV = {
  CRITICAL: { color: '#fca5a5', bg: 'rgba(239,68,68,0.15)' },
  HIGH:     { color: '#fdba74', bg: 'rgba(249,115,22,0.15)' },
  MEDIUM:   { color: '#fde047', bg: 'rgba(234,179,8,0.15)'  },
  LOW:      { color: '#86efac', bg: 'rgba(34,197,94,0.15)'  },
}

export default function ReportsPage() {
  const [scans, setScans]     = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter]   = useState('all')
  const [selected, setSelected] = useState(null)
  const [detail, setDetail]   = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    api.listScans()
      .then(r => setScans(r.data || []))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const filtered = scans.filter(s =>
    filter === 'all' ? true :
    filter === 'vuln' ? s.total_vulnerabilities > 0 :
    s.status === filter
  )

  const openDetail = async (scan) => {
    setSelected(scan.scan_id)
    setDetailLoading(true)
    try {
      const { data } = await api.getScanResults(scan.scan_id)
      setDetail(data)
    } catch { setDetail(null) }
    finally { setDetailLoading(false) }
  }

  const deleteScan = async (id, e) => {
    e.stopPropagation()
    if (!confirm('Delete this scan record?')) return
    await api.deleteScan(id).catch(() => {})
    setScans(s => s.filter(x => x.scan_id !== id))
    if (selected === id) { setSelected(null); setDetail(null) }
  }

  return (
    <div className="space-y-5 animate-fadeUp">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Reports</h2>
          <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
            Scan history · {scans.length} total records
          </p>
        </div>
        <button onClick={() => navigate('/scan')} className="btn-gold px-4 py-2 rounded-lg text-xs font-semibold">
          + New Scan
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        {[['all','All'],['completed','Completed'],['vuln','Has Vulns'],['failed','Failed']].map(([v, l]) => (
          <button key={v} onClick={() => setFilter(v)}
            className="text-xs px-3 py-1.5 rounded-lg transition-all"
            style={{
              background: filter === v ? 'rgba(200,169,110,0.15)' : 'rgba(255,255,255,0.03)',
              border: `1px solid ${filter === v ? 'rgba(200,169,110,0.4)' : 'rgba(200,200,220,0.08)'}`,
              color: filter === v ? 'var(--gold)' : 'rgba(200,200,220,0.5)',
            }}>
            {l}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* List */}
        <div className="lg:col-span-2 glass rounded-2xl overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>Loading…</div>
          ) : filtered.length === 0 ? (
            <div className="p-8 text-center">
              <div className="text-3xl mb-3">◻</div>
              <p className="text-xs" style={{ color: 'rgba(200,200,220,0.35)' }}>No matching scans</p>
            </div>
          ) : (
            <div className="divide-y" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
              {filtered.map(s => (
                <div key={s.scan_id}
                  onClick={() => openDetail(s)}
                  className="px-4 py-3 cursor-pointer transition-all flex items-center justify-between gap-3"
                  style={{
                    background: selected === s.scan_id ? 'rgba(200,169,110,0.08)' : 'transparent',
                    borderLeft: selected === s.scan_id ? '2px solid var(--gold)' : '2px solid transparent',
                  }}>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                        s.status === 'completed' ? 'bg-green-400' :
                        s.status === 'failed' ? 'bg-red-400' : 'bg-yellow-400'}`} />
                      <span className="text-xs font-mono truncate" style={{ color: '#e8e8f0' }}>{s.filename}</span>
                    </div>
                    <div className="text-[10px] mt-0.5 ml-4" style={{ color: 'rgba(200,200,220,0.3)' }}>
                      {s.created_at ? new Date(s.created_at).toLocaleString() : ''}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {s.status === 'completed' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                        style={{
                          background: s.total_vulnerabilities > 0 ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.1)',
                          color: s.total_vulnerabilities > 0 ? '#fca5a5' : '#86efac',
                        }}>
                        {s.total_vulnerabilities}
                      </span>
                    )}
                    <button onClick={e => deleteScan(s.scan_id, e)}
                      className="text-xs transition-all opacity-0 group-hover:opacity-100"
                      style={{ color: 'rgba(239,68,68,0.5)' }}
                      onMouseEnter={e => e.target.style.color = '#ef4444'}
                      onMouseLeave={e => e.target.style.color = 'rgba(239,68,68,0.5)'}>
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Detail panel */}
        <div className="lg:col-span-3 glass rounded-2xl p-5">
          {!selected ? (
            <div className="h-full flex items-center justify-center text-center py-16">
              <div>
                <div className="text-4xl mb-3" style={{ color: 'rgba(200,200,220,0.1)' }}>◷</div>
                <p className="text-sm" style={{ color: 'rgba(200,200,220,0.3)' }}>Select a scan to view details</p>
              </div>
            </div>
          ) : detailLoading ? (
            <div className="h-full flex items-center justify-center">
              <span className="text-sm animate-spin" style={{ color: 'var(--gold)' }}>⟳</span>
            </div>
          ) : detail ? (
            <div className="space-y-4">
              <div className="flex items-start justify-between flex-wrap gap-3">
                <div>
                  <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>{detail.filename}</h3>
                  <p className="text-[10px] font-mono mt-0.5" style={{ color: 'rgba(200,200,220,0.3)' }}>{detail.scan_id}</p>
                </div>
                <button onClick={() => window.open(`/api/download-report/${detail.scan_id}`, '_blank')}
                  className="text-xs px-3 py-1.5 rounded-lg"
                  style={{ background: 'rgba(200,169,110,0.1)', border: '1px solid rgba(200,169,110,0.25)', color: 'var(--gold)' }}>
                  📄 Download PDF
                </button>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { l: 'Findings', v: detail.total_vulnerabilities, c: detail.total_vulnerabilities > 0 ? '#fca5a5' : '#86efac' },
                  { l: 'Status', v: detail.status, c: detail.status === 'completed' ? '#86efac' : '#fca5a5' },
                  { l: 'Graphs', v: Object.keys(detail.graph_summary || {}).length, c: 'var(--gold)' },
                ].map(s => (
                  <div key={s.l} className="rounded-lg p-3 text-center"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div className="text-lg font-bold font-mono" style={{ color: s.c }}>{s.v}</div>
                    <div className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>{s.l}</div>
                  </div>
                ))}
              </div>

              {/* Vulns */}
              {detail.vulnerabilities?.length === 0 ? (
                <div className="rounded-xl p-4 text-center"
                  style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
                  <p className="text-xs" style={{ color: '#86efac' }}>✅ No vulnerabilities detected</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {detail.vulnerabilities.map((v, i) => {
                    const s = SEV[v.severity] || SEV.MEDIUM
                    return (
                      <div key={i} className="rounded-lg p-3"
                        style={{ background: s.bg, border: `1px solid ${s.color}30` }}>
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold" style={{ color: s.color }}>{v.type}</span>
                            {v.cwe && <span className="text-[9px] font-mono" style={{ color: 'rgba(200,200,220,0.4)' }}>{v.cwe}</span>}
                          </div>
                          <span className="text-[10px]" style={{ color: 'rgba(200,200,220,0.4)' }}>
                            {(v.confidence * 100).toFixed(0)}% · L{v.line_number}
                          </span>
                        </div>
                        <p className="text-[11px] leading-relaxed" style={{ color: 'rgba(200,200,220,0.65)' }}>{v.description}</p>
                        <code className="mt-1.5 block text-[10px] font-mono px-2 py-1 rounded"
                          style={{ background: 'rgba(0,0,0,0.3)', color: '#fde68a' }}>
                          {v.code_snippet}
                        </code>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>Failed to load details</div>
          )}
        </div>
      </div>
    </div>
  )
}

import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../services/api'

const Card = ({ children, className = '' }) => (
  <div className={`glass rounded-2xl p-5 ${className}`}>{children}</div>
)

export default function AdminDashboard() {
  const [stats, setStats]     = useState(null)
  const [audit, setAudit]     = useState([])
  const [scans, setScans]     = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([
      api.adminStats(),
      api.adminAuditLog(),
      api.listScans(),
    ]).then(([s, a, sc]) => {
      setStats(s.data)
      setAudit(a.data || [])
      setScans(sc.data || [])
    }).catch(() => {})
    .finally(() => setLoading(false))
  }, [])

  const StatBox = ({ label, value, sub, color = 'var(--gold)' }) => (
    <Card>
      <div className="text-xs mb-1" style={{ color: 'rgba(200,200,220,0.4)' }}>{label}</div>
      <div className="font-display font-bold text-3xl" style={{ color }}>{loading ? '…' : value}</div>
      {sub && <div className="text-[11px] mt-1" style={{ color: 'rgba(200,200,220,0.3)' }}>{sub}</div>}
    </Card>
  )

  return (
    <div className="space-y-6 animate-fadeUp">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Admin Portal</h2>
            <span className="text-[10px] px-2 py-0.5 rounded font-mono"
              style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
              ADMIN
            </span>
          </div>
          <p className="text-xs" style={{ color: 'rgba(200,200,220,0.4)' }}>
            Platform overview · User management · Audit logs
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => navigate('/admin/users')}
            className="btn-gold px-4 py-2 rounded-lg text-xs font-semibold">
            Manage Users
          </button>
          <button onClick={() => navigate('/admin/audit')}
            className="text-xs px-4 py-2 rounded-lg transition-all"
            style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.25)' }}>
            Audit Log
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatBox label="Total Users"    value={stats?.totalUsers}   sub="All accounts"        color="var(--gold)" />
        <StatBox label="Active Users"   value={stats?.activeUsers}  sub="Enabled accounts"    color="#86efac" />
        <StatBox label="Admin Accounts" value={stats?.adminCount}   sub="Privileged roles"    color="#fca5a5" />
        <StatBox label="Logins (24h)"   value={stats?.recentLogins} sub="Last 24 hours"       color="#60a5fa" />
      </div>

      {/* Scan overview + recent audit */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Scan summary */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Platform Scans</h3>
            <span className="text-xs font-mono" style={{ color: 'rgba(200,200,220,0.3)' }}>{scans.length} total</span>
          </div>
          <div className="grid grid-cols-3 gap-2 mb-4">
            {[
              { l: 'Completed', v: scans.filter(s => s.status === 'completed').length, c: '#86efac' },
              { l: 'Failed',    v: scans.filter(s => s.status === 'failed').length,    c: '#fca5a5' },
              { l: 'Vulns',     v: scans.reduce((a, s) => a + (s.total_vulnerabilities || 0), 0), c: 'var(--gold)' },
            ].map(s => (
              <div key={s.l} className="rounded-lg p-3 text-center"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div className="font-bold font-mono text-xl" style={{ color: s.c }}>{s.v}</div>
                <div className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>{s.l}</div>
              </div>
            ))}
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {scans.slice(0, 6).map(s => (
              <div key={s.scan_id} className="flex items-center justify-between px-2 py-1.5 rounded"
                style={{ background: 'rgba(255,255,255,0.02)' }}>
                <span className="text-xs font-mono truncate" style={{ color: 'rgba(200,200,220,0.7)' }}>{s.filename}</span>
                <span className={`text-[10px] ml-2 flex-shrink-0 ${
                  s.status === 'completed' ? 'text-green-400' : 'text-red-400'}`}>
                  {s.total_vulnerabilities > 0 ? `${s.total_vulnerabilities} vulns` : s.status}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Recent audit */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Recent Activity</h3>
            <button onClick={() => navigate('/admin/audit')}
              className="text-[10px]" style={{ color: 'var(--gold)' }}>View all →</button>
          </div>
          {audit.length === 0 ? (
            <div className="text-center py-6 text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>No activity yet</div>
          ) : (
            <div className="space-y-1 max-h-52 overflow-y-auto">
              {audit.slice(0, 10).map((a, i) => (
                <div key={i} className="flex items-start gap-2 py-1.5 border-b last:border-0"
                  style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                  <span className="text-[10px] font-mono mt-0.5 flex-shrink-0 w-20 truncate"
                    style={{ color: 'rgba(200,200,220,0.25)' }}>
                    {a.created_at?.slice(11, 19)}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded flex-shrink-0"
                    style={{ background: 'rgba(200,169,110,0.08)', color: 'var(--gold-dim)' }}>
                    {a.action}
                  </span>
                  <span className="text-[10px] truncate" style={{ color: 'rgba(200,200,220,0.45)' }}>
                    {a.email || a.detail}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

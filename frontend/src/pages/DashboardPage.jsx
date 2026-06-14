import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, BarChart, Bar
} from 'recharts'
import { api } from '../services/api'
import { useAuth } from '../context/AuthContext'

const COLORS = { CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e' }
const PIE_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e']
const ttStyle = { backgroundColor: '#07070f', border: '1px solid rgba(200,169,110,0.2)', fontSize: 11, color: '#e8e8f0' }

const Card = ({ children, className = '' }) => (
  <div className={`glass rounded-2xl p-5 ${className}`}>{children}</div>
)

const StatCard = ({ label, value, sub, accent = 'gold' }) => {
  const colors = { gold: 'var(--gold)', green: '#22c55e', red: '#ef4444', blue: '#60a5fa', purple: '#a78bfa' }
  return (
    <Card>
      <div className="text-xs mb-1" style={{ color: 'rgba(200,200,220,0.4)' }}>{label}</div>
      <div className="font-display font-bold text-3xl" style={{ color: colors[accent] }}>{value}</div>
      {sub && <div className="text-[11px] mt-1" style={{ color: 'rgba(200,200,220,0.35)' }}>{sub}</div>}
    </Card>
  )
}

// Static performance data (model training metrics)
const perfData = [
  { epoch: 1, BO: 0.81, FS: 0.72, UAF: 0.85 },
  { epoch: 2, BO: 0.85, FS: 0.75, UAF: 0.88 },
  { epoch: 3, BO: 0.88, FS: 0.78, UAF: 0.91 },
  { epoch: 4, BO: 0.90, FS: 0.80, UAF: 0.93 },
  { epoch: 5, BO: 0.91, FS: 0.81, UAF: 0.94 },
]


export default function DashboardPage() {
  const [scans, setScans]         = useState([])
  const [mlOnline, setMlOnline]   = useState(null)
  const { user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    api.listScans().then(r => setScans(r.data || [])).catch(() => {})
    api.health().then(r => setMlOnline(r.data?.ml_pipeline)).catch(() => setMlOnline(false))
  }, [])

  const completed  = scans.filter(s => s.status === 'completed').length
  const totalVulns = scans.reduce((a, s) => a + (s.total_vulnerabilities || 0), 0)

  const severityData = (() => {
    const counts = {}
    scans.forEach(s => {
      Object.entries(s.severity_counts || {}).forEach(([sev, n]) => {
        const k = sev.charAt(0).toUpperCase() + sev.slice(1).toLowerCase()
        counts[k] = (counts[k] || 0) + n
      })
    })
    return Object.entries(counts).filter(([, n]) => n > 0).map(([name, value]) => ({ name, value }))
  })()

  const vulnTypeData = (() => {
    const counts = {}
    scans.forEach(s => {
      Object.entries(s.type_counts || {}).forEach(([type, n]) => {
        counts[type] = (counts[type] || 0) + n
      })
    })
    return Object.entries(counts).map(([type, count]) => ({ type, count }))
  })()

  return (
    <div className="space-y-6 animate-fadeUp">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>
            Welcome back, {user?.full_name?.split(' ')[0]}
          </h2>
          <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
            Quantum-Enhanced Vulnerability Detection · QEGVD v2.0
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <span className="px-3 py-1.5 rounded-full text-xs font-mono"
            style={{ background: 'rgba(200,169,110,0.08)', border: '1px solid rgba(200,169,110,0.2)', color: 'var(--gold-light)' }}>
            QENND-v2.0-q4
          </span>
          <span className="px-3 py-1.5 rounded-full text-xs"
            style={{
              background: mlOnline ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
              border: `1px solid ${mlOnline ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
              color: mlOnline ? '#86efac' : '#fca5a5',
            }}>
            {mlOnline === null ? '…' : mlOnline ? '🟢 ML Online' : '🔴 Demo Mode'}
          </span>
        </div>
      </div>

      {/* Stat row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Scans"      value={scans.length}  sub="All time"           accent="gold"   />
        <StatCard label="Completed"        value={completed}     sub="Successfully analyzed" accent="green"  />
        <StatCard label="Vulnerabilities"  value={totalVulns}    sub="Detected across scans" accent="red"    />
        <StatCard label="Model F1 (UAF)"   value="0.915"         sub="Best classifier"    accent="blue"   />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* F1 trend */}
        <Card className="lg:col-span-2">
          <div className="mb-4">
            <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Model F1 Score by Classifier</h3>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>Training epochs · BO · FS · UAF</p>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={perfData}>
                <XAxis dataKey="epoch" stroke="rgba(200,200,220,0.15)" tick={{ fill: 'rgba(200,200,220,0.35)', fontSize: 11 }} />
                <YAxis domain={[0.6, 1]} stroke="rgba(200,200,220,0.15)" tick={{ fill: 'rgba(200,200,220,0.35)', fontSize: 11 }} />
                <Tooltip contentStyle={ttStyle} formatter={v => v.toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 11, color: 'rgba(200,200,220,0.5)' }} />
                <Line type="monotone" dataKey="BO"  stroke="var(--gold-light)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="FS"  stroke="#a78bfa"           strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="UAF" stroke="#34d399"           strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Severity pie */}
        <Card>
          <div className="mb-4">
            <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Severity Distribution</h3>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>All detected vulns</p>
          </div>
          <div className="h-48">
            {severityData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs" style={{ color: 'rgba(200,200,220,0.25)' }}>
                No vulnerability data yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={severityData} cx="50%" cy="50%" innerRadius={40} outerRadius={70}
                    dataKey="value" nameKey="name" paddingAngle={3}>
                    {severityData.map((entry, i) => (
                      <Cell key={i} fill={COLORS[entry.name.toUpperCase()] || PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={ttStyle} />
                  <Legend wrapperStyle={{ fontSize: 10, color: 'rgba(200,200,220,0.5)' }} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      </div>

      {/* Vuln type bar + Recent scans */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card>
          <div className="mb-4">
            <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Vulnerability Types</h3>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>Detection breakdown</p>
          </div>
          <div className="h-40">
            {vulnTypeData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs" style={{ color: 'rgba(200,200,220,0.25)' }}>
                No vulnerability data yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={vulnTypeData} layout="vertical">
                  <XAxis type="number" stroke="rgba(200,200,220,0.15)" tick={{ fill: 'rgba(200,200,220,0.35)', fontSize: 10 }} />
                  <YAxis type="category" dataKey="type" width={90} tick={{ fill: 'rgba(200,200,220,0.5)', fontSize: 10 }} />
                  <Tooltip contentStyle={ttStyle} />
                  <Bar dataKey="count" fill="var(--gold)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        {/* Recent scans */}
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Recent Scans</h3>
            <button onClick={() => navigate('/scan')}
              className="text-xs px-3 py-1 rounded-lg transition-all"
              style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.2)' }}>
              + New Scan
            </button>
          </div>
          {scans.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-3xl mb-3">⬡</div>
              <p className="text-sm" style={{ color: 'rgba(200,200,220,0.4)' }}>No scans yet.</p>
              <button onClick={() => navigate('/scan')} className="btn-gold mt-3 px-5 py-2 rounded-lg text-xs">
                Run First Scan
              </button>
            </div>
          ) : (
            <div className="space-y-2 max-h-52 overflow-y-auto">
              {scans.slice(0, 8).map(s => (
                <div key={s.scan_id} className="flex items-center justify-between px-3 py-2 rounded-lg transition-all"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <div className="flex items-center gap-3 min-w-0">
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                      s.status === 'completed' ? 'bg-green-400' :
                      s.status === 'failed' ? 'bg-red-400' : 'bg-yellow-400'}`} />
                    <span className="text-xs font-mono truncate" style={{ color: '#e8e8f0' }}>{s.filename}</span>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    {s.status === 'completed' && (
                      <span className="text-[10px] px-2 py-0.5 rounded"
                        style={{ background: s.total_vulnerabilities > 0 ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)',
                          color: s.total_vulnerabilities > 0 ? '#fca5a5' : '#86efac' }}>
                        {s.total_vulnerabilities} vuln{s.total_vulnerabilities !== 1 ? 's' : ''}
                      </span>
                    )}
                    <span className="text-[10px] capitalize" style={{ color: 'rgba(200,200,220,0.3)' }}>{s.status}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

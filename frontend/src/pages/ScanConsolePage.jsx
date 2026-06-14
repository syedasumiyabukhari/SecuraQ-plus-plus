import React, { useState, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../services/api'
import GraphViewerModal from '../components/GraphViewerModal'

const SEVERITY_STYLE = {
  CRITICAL: { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)',  text: '#fca5a5' },
  HIGH:     { bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.35)', text: '#fdba74' },
  MEDIUM:   { bg: 'rgba(234,179,8,0.12)',  border: 'rgba(234,179,8,0.35)',  text: '#fde047' },
  LOW:      { bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.35)',  text: '#86efac' },
}

const C = {
  gold:   '#c8a96e',
  red:    '#ef4444',
  orange: '#f97316',
  yellow: '#eab308',
  green:  '#4ade80',
  blue:   '#60a5fa',
  purple: '#a78bfa',
  cyan:   '#22d3ee',
  pink:   '#f472b6',
}

const TT = {
  contentStyle: {
    background: 'rgba(12,12,26,0.97)',
    border: '1px solid rgba(200,169,110,0.2)',
    borderRadius: 8,
    fontSize: 11,
  },
  labelStyle: { color: 'rgba(200,200,220,0.7)' },
  itemStyle:  { color: C.gold },
}

const TICK = { fontSize: 9, fill: 'rgba(200,200,220,0.4)' }

// The 8 QEGVD graph types — order matches the Graph Bundle tags
const GRAPH_TYPES = ['AST', 'CFG', 'DFG', 'PDG', 'TPG', 'MAG', 'CG', 'FSG']

const GRAPH_INFO = {
  AST: { full: 'Abstract Syntax Tree',     color: C.cyan   },
  CFG: { full: 'Control Flow Graph',       color: C.blue   },
  DFG: { full: 'Data Flow Graph',          color: C.purple },
  PDG: { full: 'Program Dependence Graph', color: C.gold   },
  TPG: { full: 'Token Path Graph',         color: C.green  },
  MAG: { full: 'Multi-Aspect Graph',       color: C.orange },
  CG:  { full: 'Call Graph',              color: C.red    },
  FSG: { full: 'Function Sequence Graph',  color: C.pink   },
}

const downloadSVG = (el, name) => {
  if (!el) return
  const svg = el.querySelector('svg')
  if (!svg) return
  const clone = svg.cloneNode(true)
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  bg.setAttribute('width', '100%')
  bg.setAttribute('height', '100%')
  bg.setAttribute('fill', '#0d0d1f')
  clone.insertBefore(bg, clone.firstChild)
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' })
  const url  = URL.createObjectURL(blob)
  Object.assign(document.createElement('a'), { href: url, download: name + '.svg' }).click()
  URL.revokeObjectURL(url)
}

const Skeleton = ({ h = 120 }) => (
  <div className="animate-pulse rounded-lg" style={{ height: h, background: 'rgba(200,200,220,0.04)' }} />
)

const DlBtn = ({ onClick }) => (
  <button
    onClick={onClick}
    title="Download chart as SVG"
    className="text-[10px] px-1.5 py-0.5 rounded opacity-30 hover:opacity-90 transition-opacity"
    style={{ color: C.gold, border: '1px solid rgba(200,169,110,0.2)' }}>
    ⬇
  </button>
)

export default function ScanConsolePage() {
  const [file, setFile]               = useState(null)
  const [scanId, setScanId]           = useState(null)
  const [status, setStatus]           = useState('idle')
  const [progress, setProgress]       = useState(0)
  const [stage, setStage]             = useState('Ready')
  const [logs, setLogs]               = useState([])
  const [results, setResults]         = useState(null)
  const [graphsBuilt, setGraphsBuilt] = useState({})   // live graph data during scan
  const [chartsSaved, setChartsSaved]   = useState(false)
  const [chartsSaving, setChartsSaving] = useState(false)
  const [graphModal, setGraphModal]     = useState(null)  // { type, info, meta }

  const logsRef = useRef(null)
  const wsRef   = useRef(null)

  // One ref per graph box
  const cr1 = useRef(null); const cr2 = useRef(null)
  const cr3 = useRef(null); const cr4 = useRef(null)
  const cr5 = useRef(null); const cr6 = useRef(null)
  const cr7 = useRef(null); const cr8 = useRef(null)
  const chartRefs = [cr1, cr2, cr3, cr4, cr5, cr6, cr7, cr8]

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight
  }, [logs])

  useEffect(() => () => wsRef.current?.close(), [])

  const addLog = msg => {
    const ts = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, { ts, msg }])
  }

  const reset = () => {
    wsRef.current?.close()
    setFile(null); setScanId(null); setStatus('idle')
    setProgress(0); setStage('Ready'); setLogs([]); setResults(null)
    setGraphsBuilt({})
    setChartsSaved(false); setChartsSaving(false)
  }

  const handleFile = e => {
    const f = e.target.files?.[0]
    if (!f) return
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['c', 'cpp', 'cc', 'cxx', 'h', 'hpp'].includes(ext)) {
      alert('Only C/C++ files allowed (.c, .cpp, .h, .hpp)')
      e.target.value = ''; return
    }
    setFile(f); setStatus('idle'); setProgress(0)
    setStage('Ready'); setLogs([]); setResults(null); setScanId(null)
    setChartsSaved(false)
  }

  const fetchResults = async id => {
    try {
      const { data } = await api.getScanResults(id)
      setResults(data); setStatus('completed'); setStage('Analysis Complete')
      addLog(`🎉 Complete — ${data.total_vulnerabilities} finding(s) detected`)
    } catch (err) {
      addLog(`❌ Failed to fetch results: ${err.message}`)
    }
  }

  const startScan = async () => {
    if (!file) return
    try {
      setStatus('uploading'); setStage('Uploading…')
      addLog(`📤 Uploading: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`)

      const form = new FormData(); form.append('file', file)
      const { data: up } = await api.uploadFile(form)
      const id = up.scan_id
      setScanId(id)
      addLog(`✅ Upload OK — Scan ID: ${id}`)

      setStatus('scanning')
      addLog('🔌 Connecting to live stream…')
      const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/scan/${id}`)
      wsRef.current = ws

      ws.onopen = async () => {
        addLog('📡 Stream connected')
        try {
          await api.startScan(id)
          addLog('🚀 Scan engine started')
        } catch (err) {
          addLog(`❌ Could not start scan: ${err.message}`)
          setStatus('error')
        }
      }

      ws.onmessage = e => {
        const d = JSON.parse(e.data)
        if (d.type === 'log' && d.log) { addLog(d.log); return }
        if (typeof d.progress === 'number') setProgress(d.progress)
        if (d.stage) setStage(d.stage)
        // Step-by-step graph reveal
        if (d.graph) {
          setGraphsBuilt(prev => ({
            ...prev,
            [d.graph.type]: { nodes: d.graph.nodes, edges: d.graph.edges }
          }))
        }
        if (d.status === 'completed') { ws.close(); fetchResults(id) }
      }

      ws.onerror = () => { addLog('❌ WebSocket error'); setStatus('error') }
      ws.onclose = () => {}
    } catch (err) {
      addLog(`❌ ${err.response?.data?.detail || err.message}`)
      setStatus('error'); setStage('Failed')
    }
  }

  const handleSaveAllCharts = async () => {
    if (!scanId || chartsSaving || chartsSaved) return
    setChartsSaving(true)
    const charts = {}
    chartRefs.forEach((ref, i) => {
      const svg = ref.current?.querySelector('svg')
      if (!svg) return
      const clone = svg.cloneNode(true)
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      bg.setAttribute('width', '100%'); bg.setAttribute('height', '100%')
      bg.setAttribute('fill', '#0d0d1f')
      clone.insertBefore(bg, clone.firstChild)
      charts[GRAPH_TYPES[i].toLowerCase()] = new XMLSerializer().serializeToString(clone)
    })
    try {
      await api.saveCharts(scanId, { charts })
      setChartsSaved(true)
    } catch (err) {
      console.error('Failed to save charts:', err)
    } finally {
      setChartsSaving(false)
    }
  }

  const stopScan = async () => {
    wsRef.current?.close()
    if (scanId) {
      try { await api.stopScan(scanId) } catch { /* ignore */ }
    }
    setStatus('error'); setStage('Stopped by user')
    addLog('⏹ Scan stopped by user')
  }

  const busy = status === 'uploading' || status === 'scanning'
  const sev  = results?.vulnerabilities?.reduce((a, v) => {
    a[v.severity] = (a[v.severity] || 0) + 1; return a
  }, {}) || {}

  // Graph data: live during scan, from results after completion
  const gs = results?.graph_summary || graphsBuilt
  const maxNodes = Math.max(1, ...Object.values(gs).map(g => g.nodes))

  return (
    <div className="space-y-5 animate-fadeUp max-w-6xl">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Scan Console</h2>
          <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
            Upload C/C++ source · Quantum-Hybrid QEGVD analysis · BO · FS · UAF
          </p>
        </div>
        {results && (
          <button onClick={reset}
            className="text-xs px-4 py-2 rounded-lg transition-all"
            style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.25)' }}>
            ↺ New Scan
          </button>
        )}
      </div>

      {/* ── Controls + Log ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Controls */}
        <div className="glass rounded-2xl p-5 space-y-5">
          <div>
            <label className="block text-xs font-medium mb-2" style={{ color: 'rgba(200,200,220,0.6)' }}>
              Source File
            </label>
            <label
              className={`flex flex-col items-center justify-center w-full h-28 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
                file
                  ? 'border-[rgba(200,169,110,0.4)]'
                  : 'border-[rgba(200,200,220,0.1)] hover:border-[rgba(200,169,110,0.25)]'
              }`}
              style={{ background: file ? 'rgba(200,169,110,0.04)' : 'rgba(255,255,255,0.01)' }}>
              <input type="file" className="hidden" accept=".c,.cpp,.cc,.cxx,.h,.hpp"
                onChange={handleFile} disabled={busy} />
              {file ? (
                <>
                  <span className="text-2xl mb-1">📄</span>
                  <span className="text-xs font-medium" style={{ color: 'var(--gold)' }}>{file.name}</span>
                  <span className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </>
              ) : (
                <>
                  <span className="text-2xl mb-1" style={{ color: 'rgba(200,200,220,0.2)' }}>⬡</span>
                  <span className="text-xs" style={{ color: 'rgba(200,200,220,0.35)' }}>Click to select .c / .cpp</span>
                </>
              )}
            </label>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs" style={{ color: 'rgba(200,200,220,0.45)' }}>
              <span>{stage}</span>
              <span className="font-mono" style={{ color: 'var(--gold)' }}>{Math.round(progress)}%</span>
            </div>
            <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(200,200,220,0.08)' }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${progress}%`,
                  background: status === 'error'
                    ? '#ef4444'
                    : 'linear-gradient(90deg, var(--gold-dim), var(--gold-light))',
                }} />
            </div>
          </div>

          <button
            onClick={startScan}
            disabled={!file || busy}
            className="btn-gold w-full py-3 rounded-xl text-sm font-semibold flex items-center justify-center gap-2">
            {busy ? (
              <><span className="animate-spin">⟳</span> {status === 'uploading' ? 'Uploading…' : 'Scanning…'}</>
            ) : '⚛ Start Quantum Scan'}
          </button>

          {/* FR-M5.3: Stop scan button */}
          {status === 'scanning' && (
            <button
              onClick={stopScan}
              className="w-full py-2.5 rounded-xl text-sm font-medium flex items-center justify-center gap-2 transition-all"
              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
              ⏹ Stop Scan
            </button>
          )}

          {scanId && (
            <div className="text-[10px] font-mono text-center" style={{ color: 'rgba(200,200,220,0.25)' }}>
              {scanId}
            </div>
          )}
        </div>

        {/* Log stream */}
        <div className="lg:col-span-2 glass rounded-2xl p-4 flex flex-col" style={{ minHeight: 280 }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium" style={{ color: 'rgba(200,200,220,0.5)' }}>System Log</span>
            <span
              className={`w-2 h-2 rounded-full ${busy ? 'bg-green-400' : 'bg-gray-600'}`}
              style={busy ? { animation: 'pulse-green 2s ease infinite' } : {}} />
          </div>
          <div
            ref={logsRef}
            className="flex-1 overflow-y-auto rounded-xl p-3 font-mono text-[11px] space-y-0.5"
            style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.04)', maxHeight: 260 }}>
            {logs.length === 0
              ? <span style={{ color: 'rgba(200,200,220,0.2)' }}>Awaiting scan…</span>
              : logs.map((l, i) => (
                <div key={i} className="flex gap-2 border-b pb-0.5"
                  style={{ borderColor: 'rgba(255,255,255,0.03)', color: '#4ade80' }}>
                  <span style={{ color: 'rgba(200,200,220,0.2)', flexShrink: 0 }}>{l.ts}</span>
                  <span>{l.msg}</span>
                </div>
              ))
            }
          </div>
        </div>
      </div>

      {/* FR-M11.7: Fallback simulation notice — shown when ML pipeline not available */}
      {results && !results.graph_summary?.AST && (
        <div className="rounded-xl px-4 py-3 flex items-start gap-3 animate-fadeUp"
          style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.22)' }}>
          <span style={{ color: '#fde047', flexShrink: 0 }}>⚠</span>
          <p className="text-xs leading-relaxed" style={{ color: '#fde047' }}>
            <strong>Demo / Static Mode:</strong> The QEGVD ML pipeline (PyTorch + PennyLane) is not loaded on this server.
            Results are produced by the FS-Direct stacking classifier and rule-based detectors.
            Graph visualisations are unavailable in this mode.
          </p>
        </div>
      )}

      {/* ── Graph Bundle — step-by-step reveal ────────────────────────────── */}
      {status !== 'idle' && (
        <div className="space-y-3 animate-fadeUp">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold" style={{ color: 'rgba(200,200,220,0.55)' }}>
                Graph Bundle
              </span>
              <span className="text-[10px] ml-2" style={{ color: 'rgba(200,200,220,0.3)' }}>
                {Object.keys(gs).length}/8 constructed
              </span>
              {status === 'scanning' && Object.keys(gs).length < 8 && (
                <span className="text-[10px] ml-2 animate-pulse" style={{ color: 'var(--gold)' }}>
                  ⟳ Building…
                </span>
              )}
            </div>
            {results && Object.keys(gs).length > 0 && (
              <button
                onClick={handleSaveAllCharts}
                disabled={chartsSaving || chartsSaved}
                className="text-xs px-3 py-1.5 rounded-lg transition-all"
                style={{
                  background: chartsSaved ? 'rgba(34,197,94,0.1)' : 'rgba(200,169,110,0.08)',
                  border: chartsSaved ? '1px solid rgba(34,197,94,0.3)' : '1px solid rgba(200,169,110,0.22)',
                  color: chartsSaved ? '#86efac' : 'var(--gold)',
                  opacity: chartsSaving ? 0.6 : 1,
                  cursor: (chartsSaving || chartsSaved) ? 'default' : 'pointer',
                }}>
                {chartsSaving ? '⟳ Saving…' : chartsSaved ? '✓ Charts Saved' : '⬇ Save All Charts'}
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {GRAPH_TYPES.map((gt, i) => {
              const info  = GRAPH_INFO[gt]
              const gData = gs[gt] || null
              const ref   = chartRefs[i]
              const isBuilt    = !!gData
              const isBuilding = !isBuilt && status === 'scanning'
              const barData = gData
                ? [
                    { label: 'Nodes', value: gData.nodes, fill: info.color },
                    { label: 'Edges', value: gData.edges, fill: 'rgba(200,200,220,0.18)' },
                  ]
                : []

              return (
                <div
                  key={gt}
                  ref={ref}
                  className="glass rounded-2xl p-4 flex flex-col transition-all duration-500"
                  style={{
                    minHeight: 210,
                    opacity: isBuilt ? 1 : 0.45,
                    border: isBuilding
                      ? `1px solid ${info.color}55`
                      : isBuilt
                        ? `1px solid ${info.color}33`
                        : '1px solid rgba(255,255,255,0.04)',
                    boxShadow: isBuilt ? `0 0 12px ${info.color}18` : 'none',
                  }}>

                  {/* Header */}
                  <div className="flex items-start justify-between mb-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-bold font-mono leading-none"
                        style={{ color: info.color }}>{gt}</span>
                      {isBuilding && (
                        <span className="text-[9px] animate-pulse px-1.5 py-0.5 rounded font-mono"
                          style={{ background: `${info.color}18`, color: info.color }}>
                          building…
                        </span>
                      )}
                      {isBuilt && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded font-mono"
                          style={{ background: 'rgba(34,197,94,0.1)', color: '#86efac' }}>
                          ✓
                        </span>
                      )}
                    </div>
                    {isBuilt && results && (
                      <DlBtn onClick={() => downloadSVG(ref.current, gt.toLowerCase() + '_graph')} />
                    )}
                  </div>
                  <p className="text-[9px] mb-3 leading-tight"
                    style={{ color: 'rgba(200,200,220,0.35)' }}>
                    {info.full}
                  </p>

                  {/* Chart / state area */}
                  {isBuilding ? (
                    <div className="flex flex-col items-center justify-center gap-2 flex-1">
                      <div className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin"
                        style={{ borderColor: `${info.color}44`, borderTopColor: info.color }} />
                      <span className="text-[10px]" style={{ color: info.color }}>
                        Constructing…
                      </span>
                    </div>
                  ) : !isBuilt ? (
                    <div className="flex flex-col items-center justify-center gap-2 flex-1">
                      <span className="text-[10px]" style={{ color: 'rgba(200,200,220,0.2)' }}>
                        Queued
                      </span>
                    </div>
                  ) : (
                    <>
                      <ResponsiveContainer width="100%" height={100}>
                        <BarChart data={barData}
                          margin={{ top: 2, right: 4, bottom: 0, left: -28 }}
                          barCategoryGap="30%">
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey="label" tick={TICK} />
                          <YAxis tick={TICK} allowDecimals={false} />
                          <RTooltip {...TT} formatter={(v, n) => [v, n]} labelFormatter={l => l} />
                          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                            {barData.map((d, idx) => <Cell key={idx} fill={d.fill} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>

                      <div className="flex items-center justify-between mt-2 px-0.5">
                        <div className="text-center">
                          <div className="text-base font-bold font-mono leading-none"
                            style={{ color: info.color }}>{gData.nodes}</div>
                          <div className="text-[9px] mt-0.5 tracking-wide"
                            style={{ color: 'rgba(200,200,220,0.35)' }}>NODES</div>
                        </div>
                        <div style={{ width: 1, height: 28, background: 'rgba(200,200,220,0.08)' }} />
                        <div className="text-center">
                          <div className="text-base font-bold font-mono leading-none"
                            style={{ color: 'rgba(200,200,220,0.5)' }}>{gData.edges}</div>
                          <div className="text-[9px] mt-0.5 tracking-wide"
                            style={{ color: 'rgba(200,200,220,0.25)' }}>EDGES</div>
                        </div>
                        <div style={{ width: 1, height: 28, background: 'rgba(200,200,220,0.08)' }} />
                        <div className="text-center">
                          <div className="text-[10px] font-mono font-semibold leading-none"
                            style={{ color: 'rgba(200,200,220,0.4)' }}>
                            {gData.nodes > 0 ? (gData.edges / gData.nodes).toFixed(1) : '0.0'}
                          </div>
                          <div className="text-[9px] mt-0.5 tracking-wide"
                            style={{ color: 'rgba(200,200,220,0.2)' }}>E/N</div>
                        </div>
                      </div>

                      {/* View Graph button */}
                      <button
                        onClick={() => setGraphModal({ type: gt, info, meta: gData })}
                        className="mt-3 w-full py-1.5 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5 transition-all"
                        style={{
                          background: `${info.color}14`,
                          border: `1px solid ${info.color}40`,
                          color: info.color,
                        }}>
                        ◎ View Graph
                      </button>
                    </>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Vulnerability findings ─────────────────────────────────────────── */}
      {results && (
        <div className="glass rounded-2xl p-6 animate-fadeUp space-y-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h3 className="font-display font-semibold text-lg" style={{ color: '#f0f0f8' }}>
                Vulnerability Findings
              </h3>
              <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.4)' }}>
                {results.filename} · {results.total_vulnerabilities} finding(s)
              </p>
            </div>
            <div className="flex gap-2 flex-wrap">
              {Object.entries(sev).map(([s, n]) => {
                const st = SEVERITY_STYLE[s] || {}
                return (
                  <span key={s} className="text-xs px-2 py-1 rounded"
                    style={{ background: st.bg, border: `1px solid ${st.border}`, color: st.text }}>
                    {n} {s}
                  </span>
                )
              })}
              <button
                onClick={() => window.open(`/api/download-report/${scanId}`, '_blank')}
                className="text-xs px-3 py-1.5 rounded-lg transition-all"
                style={{
                  background: 'rgba(200,169,110,0.1)',
                  border: '1px solid rgba(200,169,110,0.25)',
                  color: 'var(--gold)',
                }}>
                📄 Download PDF Report
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {results.total_vulnerabilities === 0 ? (
              <div className="rounded-xl p-5 text-center"
                style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.25)' }}>
                <div className="text-2xl mb-2">✅</div>
                <p className="text-sm font-medium" style={{ color: '#86efac' }}>No vulnerabilities detected</p>
                <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
                  All three classifiers (BO, FS, UAF) returned safe predictions.
                </p>
              </div>
            ) : (
              results.vulnerabilities.map((v, i) => {
                const st = SEVERITY_STYLE[v.severity] || SEVERITY_STYLE.MEDIUM
                return (
                  <div key={i} className="rounded-xl p-4 transition-all"
                    style={{ background: st.bg, border: `1px solid ${st.border}` }}>
                    <div className="flex items-start justify-between gap-4 mb-2 flex-wrap">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-display font-semibold text-sm" style={{ color: st.text }}>
                          {v.type}
                        </span>
                        <span className="text-[10px] px-2 py-0.5 rounded font-mono"
                          style={{ background: 'rgba(0,0,0,0.3)', color: st.text }}>
                          {v.severity}
                        </span>
                        {v.cwe && (
                          <span className="text-[10px] px-2 py-0.5 rounded font-mono"
                            style={{ background: 'rgba(0,0,0,0.2)', color: 'rgba(200,200,220,0.5)' }}>
                            {v.cwe}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-xs" style={{ color: 'rgba(200,200,220,0.45)' }}>
                        <span>Line {v.line_number}</span>
                        <span>
                          Confidence:{' '}
                          <span className="font-mono" style={{ color: st.text }}>
                            {(v.confidence * 100).toFixed(1)}%
                          </span>
                        </span>
                        {v.detector && (
                          <span className="font-mono text-[10px]">{v.detector}</span>
                        )}
                      </div>
                    </div>
                    <p className="text-xs leading-relaxed mb-3" style={{ color: 'rgba(200,200,220,0.7)' }}>
                      {v.description}
                    </p>
                    <div className="rounded-lg p-3"
                      style={{ background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(0,0,0,0.4)' }}>
                      <code className="text-xs font-mono" style={{ color: '#fde68a' }}>
                        {v.code_snippet}
                      </code>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      )}

      {/* Graph viewer modal */}
      {graphModal && (
        <GraphViewerModal
          graphType={graphModal.type}
          graphInfo={graphModal.info}
          graphMeta={graphModal.meta}
          results={results}
          onClose={() => setGraphModal(null)}
        />
      )}
    </div>
  )
}

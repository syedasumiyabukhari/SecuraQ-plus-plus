import React, { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../services/api'

const PATCHES = {
  'Buffer Overflow': {
    cwe: 'CWE-121/122', icon: '🔴',
    title: 'Buffer Overflow Mitigation',
    desc: 'Unbounded writes to stack/heap buffers allow attackers to corrupt adjacent memory.',
    fixes: [
      { label: 'Replace strcpy → strncpy', before: 'strcpy(dest, src);', after: 'strncpy(dest, src, sizeof(dest) - 1);\ndest[sizeof(dest) - 1] = \'\\0\';' },
      { label: 'Replace gets → fgets', before: 'gets(buf);', after: 'fgets(buf, sizeof(buf), stdin);' },
      { label: 'Replace sprintf → snprintf', before: 'sprintf(buf, fmt, arg);', after: 'snprintf(buf, sizeof(buf), fmt, arg);' },
    ],
    refs: ['CERT C MEM35-C', 'OWASP Buffer Overflow', 'CWE-121'],
  },
  'Format String': {
    cwe: 'CWE-134', icon: '🟠',
    title: 'Format String Vulnerability',
    desc: 'User-controlled format strings allow arbitrary memory read/write via %n or %x directives.',
    fixes: [
      { label: 'Harden printf', before: 'printf(user_input);', after: 'printf("%s", user_input);' },
      { label: 'Harden fprintf', before: 'fprintf(stderr, user_msg);', after: 'fprintf(stderr, "%s", user_msg);' },
      { label: 'Harden syslog', before: 'syslog(LOG_ERR, user_input);', after: 'syslog(LOG_ERR, "%s", user_input);' },
    ],
    refs: ['CERT C FIO30-C', 'OWASP Format String', 'CWE-134'],
  },
  'Use-After-Free': {
    cwe: 'CWE-416', icon: '🟡',
    title: 'Use-After-Free Prevention',
    desc: 'Accessing heap memory after free() allows attackers to control the reallocated chunk.',
    fixes: [
      { label: 'NULL after free', before: 'free(ptr);\nuse(ptr);', after: 'free(ptr);\nptr = NULL;\n// Now safe — ptr != NULL check will catch misuse' },
      { label: 'RAII pattern (C++)', before: 'int* p = new int(5);\ndelete p;\nuse(*p);', after: 'std::unique_ptr<int> p = std::make_unique<int>(5);\n// Automatically freed, no dangling pointer' },
      { label: 'Double-free guard', before: 'free(ptr);\n/* ... */\nfree(ptr); // BUG', after: 'free(ptr); ptr = NULL;\n/* ... */\nif (ptr) { free(ptr); ptr = NULL; }' },
    ],
    refs: ['CERT C MEM30-C', 'CWE-416', 'ISO/IEC TS 17961:2013'],
  },
}

// ── helpers ───────────────────────────────────────────────────────────────────

function SevBadge({ sev }) {
  const C = {
    CRITICAL: { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)',  text: '#fca5a5' },
    HIGH:     { bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.35)', text: '#fdba74' },
    MEDIUM:   { bg: 'rgba(234,179,8,0.12)',  border: 'rgba(234,179,8,0.35)',  text: '#fde047' },
    LOW:      { bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.35)',  text: '#86efac' },
  }
  const c = C[sev] || C.LOW
  return (
    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded"
      style={{ background: c.bg, border: `1px solid ${c.border}`, color: c.text }}>
      {sev}
    </span>
  )
}

function VulnRow({ v, accent }) {
  return (
    <div className="flex items-start justify-between rounded-lg px-3 py-2 gap-3"
      style={{ background: 'rgba(255,255,255,0.02)', border: `1px solid ${accent}22` }}>
      <div className="min-w-0">
        <code className="text-[11px] font-mono block truncate" style={{ color: '#fde68a' }}>
          {v.code_snippet}
        </code>
        <span className="text-[10px] mt-0.5 block" style={{ color: 'rgba(200,200,220,0.35)' }}>
          Line {v.line_number} · {v.detector}
        </span>
      </div>
      <SevBadge sev={v.severity} />
    </div>
  )
}

// ── Auto-Fix panel ────────────────────────────────────────────────────────────

function AutoFixPanel({ scanId, vulnCount }) {
  const [state, setState]       = useState('idle') // idle | loading | done | error
  const [preview, setPreview]   = useState(null)
  const [showCode, setShowCode] = useState(false)
  const [errMsg, setErrMsg]     = useState('')

  const loadPreview = async () => {
    setState('loading')
    setErrMsg('')
    try {
      const { data } = await api.autoFixPreview(scanId)
      setPreview(data)
      setState('done')
    } catch (e) {
      setErrMsg(e?.response?.data?.detail || 'Failed to generate fix preview.')
      setState('error')
    }
  }

  const downloadFixed = async () => {
    try {
      const { data } = await api.autoFixDownload(scanId)
      const url  = URL.createObjectURL(new Blob([data], { type: 'text/plain' }))
      const a    = document.createElement('a')
      a.href     = url
      a.download = preview?.patched_filename || 'fixed.c'
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      // silently ignore — browser will show an error if needed
    }
  }

  // Diff: highlight lines that changed
  const diffLines = preview ? buildDiff(preview.original_code, preview.patched_code) : []

  return (
    <div className="glass rounded-2xl p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>
            Auto-Fix File
          </h3>
          <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.4)' }}>
            Automatically patch {vulnCount} detected vulnerabilit{vulnCount !== 1 ? 'ies' : 'y'} in the source file
          </p>
        </div>
        {state === 'done' && preview && (
          <span className="text-[10px] px-2 py-1 rounded font-mono"
            style={{ background: 'rgba(74,222,128,0.1)', border: '1px solid rgba(74,222,128,0.25)', color: '#4ade80', flexShrink: 0 }}>
            {preview.fix_count} fix{preview.fix_count !== 1 ? 'es' : ''} ready
          </span>
        )}
      </div>

      {/* Idle — show generate button */}
      {state === 'idle' && (
        <button onClick={loadPreview}
          className="w-full py-2.5 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2"
          style={{
            background: 'rgba(200,169,110,0.1)',
            border: '1px solid rgba(200,169,110,0.3)',
            color: 'var(--gold)',
          }}>
          ◆ Generate Auto-Fix Preview
        </button>
      )}

      {/* Loading */}
      {state === 'loading' && (
        <div className="flex items-center gap-3 py-3">
          <span className="animate-spin text-lg" style={{ color: 'var(--gold)' }}>⟳</span>
          <span className="text-xs" style={{ color: 'rgba(200,200,220,0.5)' }}>Generating patched file…</span>
        </div>
      )}

      {/* Error */}
      {state === 'error' && (
        <div className="rounded-xl px-4 py-3 text-sm flex items-center justify-between gap-3"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
          <span>{errMsg}</span>
          <button onClick={() => setState('idle')} className="text-[10px] underline opacity-60 hover:opacity-100">Retry</button>
        </div>
      )}

      {/* Done — fixes list + code preview + download */}
      {state === 'done' && preview && (
        <div className="space-y-3">
          {/* Applied fixes list */}
          {preview.applied_fixes.length > 0 ? (
            <div className="rounded-xl overflow-hidden"
              style={{ border: '1px solid rgba(74,222,128,0.15)' }}>
              <div className="px-3 py-2 text-[10px] font-medium"
                style={{ background: 'rgba(74,222,128,0.06)', color: '#4ade80' }}>
                Applied Fixes ({preview.fix_count})
              </div>
              <div className="divide-y" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
                {preview.applied_fixes.map((fix, i) => (
                  <div key={i} className="px-3 py-2 text-[11px] font-mono flex items-center gap-2"
                    style={{ color: '#86efac', background: 'rgba(255,255,255,0.01)' }}>
                    <span style={{ color: '#4ade80' }}>✓</span> {fix}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-xl px-4 py-3 text-xs"
              style={{ background: 'rgba(234,179,8,0.06)', border: '1px solid rgba(234,179,8,0.2)', color: '#fde047' }}>
              No automatic fixes could be applied — the patterns may use complex expressions.
              Review the remediation patterns above and apply changes manually.
            </div>
          )}

          {/* Code diff toggle */}
          {preview.fix_count > 0 && (
            <button onClick={() => setShowCode(v => !v)}
              className="text-[11px] px-3 py-1.5 rounded-lg transition-all"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.07)',
                color: 'rgba(200,200,220,0.5)',
              }}>
              {showCode ? '▲ Hide' : '▼ Show'} patched code
            </button>
          )}

          {showCode && (
            <div className="rounded-xl overflow-hidden"
              style={{ border: '1px solid rgba(255,255,255,0.06)', maxHeight: 360, overflowY: 'auto' }}>
              <div className="px-3 py-2 text-[10px] font-medium sticky top-0"
                style={{ background: '#161b22', color: 'rgba(200,200,220,0.4)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                {preview.patched_filename} — changed lines highlighted
              </div>
              <div style={{ background: '#0d1117' }}>
                {diffLines.map((dl, i) => (
                  <div key={i} className="flex text-[11px] font-mono"
                    style={{
                      background: dl.type === 'added'   ? 'rgba(74,222,128,0.08)'
                                : dl.type === 'removed' ? 'rgba(239,68,68,0.08)'
                                : 'transparent',
                      borderLeft: dl.type === 'added'   ? '2px solid #4ade80'
                                : dl.type === 'removed' ? '2px solid #f87171'
                                : '2px solid transparent',
                    }}>
                    <span className="select-none w-10 flex-shrink-0 text-right pr-3 pt-0.5 text-[10px]"
                      style={{ color: 'rgba(200,200,220,0.2)', borderRight: '1px solid rgba(255,255,255,0.04)' }}>
                      {dl.lineNo || ''}
                    </span>
                    <span className="px-3 py-0.5 whitespace-pre"
                      style={{
                        color: dl.type === 'added'   ? '#86efac'
                             : dl.type === 'removed' ? '#fca5a5'
                             : '#a8b8c8',
                      }}>
                      {dl.type === 'added' ? '+ ' : dl.type === 'removed' ? '- ' : '  '}{dl.text}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Download button */}
          {preview.fix_count > 0 && (
            <button onClick={downloadFixed}
              className="w-full py-2.5 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2"
              style={{
                background: 'rgba(74,222,128,0.1)',
                border: '1px solid rgba(74,222,128,0.3)',
                color: '#4ade80',
              }}>
              ↓ Download Fixed File — {preview.patched_filename}
            </button>
          )}

          <button onClick={() => { setState('idle'); setPreview(null); setShowCode(false) }}
            className="text-[10px] w-full text-center transition-all"
            style={{ color: 'rgba(200,200,220,0.25)' }}>
            Reset
          </button>
        </div>
      )}
    </div>
  )
}

// Simple line-diff: returns array of { type: 'same'|'added'|'removed', text, lineNo }
function buildDiff(original, patched) {
  const origLines   = original.split('\n')
  const patchLines  = patched.split('\n')
  const result      = []
  const maxLen      = Math.max(origLines.length, patchLines.length)

  let patchIdx = 0
  for (let i = 0; i < origLines.length; i++) {
    const o = origLines[i]
    if (patchIdx >= patchLines.length) {
      result.push({ type: 'removed', text: o, lineNo: '' })
      continue
    }
    const p = patchLines[patchIdx]
    if (o === p) {
      result.push({ type: 'same', text: o, lineNo: patchIdx + 1 })
      patchIdx++
    } else {
      // emit removed original line, then all new patched lines until they catch up
      result.push({ type: 'removed', text: o, lineNo: '' })
      // consume added lines that replaced this one
      while (patchIdx < patchLines.length) {
        const next = patchLines[patchIdx]
        // heuristic: if the next patched line also doesn't match the next original line, it's added
        const nextOrig = origLines[i + 1]
        if (next === nextOrig || (patchIdx > 0 && next === origLines[i])) break
        result.push({ type: 'added', text: next, lineNo: patchIdx + 1 })
        patchIdx++
        if (patchIdx < patchLines.length && patchLines[patchIdx] === (origLines[i + 1] ?? null)) break
      }
    }
  }
  // remaining added lines at end
  while (patchIdx < patchLines.length) {
    result.push({ type: 'added', text: patchLines[patchIdx], lineNo: patchIdx + 1 })
    patchIdx++
  }
  return result
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function PatchEnginePage() {
  const [scans, setScans]   = useState([])
  const [sel, setSel]       = useState(null)
  const [detail, setDetail] = useState(null)
  const [tab, setTab]       = useState(null)
  const [rejected, setRejected] = useState({})
  const [aiResults, setAiResults] = useState({}) // { key: { loading, fix, error } }

  // patch-validation state
  const [patchFile, setPatchFile]         = useState(null)
  const [patchProgress, setPatchProgress] = useState(0)
  const [patchStage, setPatchStage]       = useState('')
  const [patchStatus, setPatchStatus]     = useState('idle') // idle | scanning | done | error
  const [comparison, setComparison]       = useState(null)
  const [patchError, setPatchError]       = useState('')

  const pollRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    api.listScans()
      .then(r => {
        const completed = (r.data || []).filter(s => s.status === 'completed' && s.total_vulnerabilities > 0)
        setScans(completed)
      })
      .catch(() => {})
  }, [])

  const clearPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const openScan = async (scan) => {
    clearPoll()
    setSel(scan.scan_id)
    setDetail(null)
    setTab(null)
    setPatchFile(null)
    setPatchStatus('idle')
    setComparison(null)
    setPatchError('')
    try {
      const { data } = await api.getScanResults(scan.scan_id)
      setDetail(data)
      const types = [...new Set(data.vulnerabilities.map(v => v.type))]
      setTab(types[0] || null)
    } catch { setDetail(null) }
  }

  const pollComparison = useCallback((originalId, pId) => {
    clearPoll()
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await api.getComparison(originalId, pId)
        if (data.status === 'completed') {
          clearPoll()
          setComparison(data)
          setPatchStatus('done')
        } else {
          setPatchProgress(data.progress || 0)
          setPatchStage(data.stage || '')
        }
      } catch {
        clearPoll()
        setPatchStatus('error')
        setPatchError('Failed to retrieve comparison results.')
      }
    }, 1500)
  }, [clearPoll])

  useEffect(() => () => clearPoll(), [clearPoll])

  const submitPatch = async () => {
    if (!patchFile || !sel) return
    setPatchStatus('scanning')
    setPatchProgress(0)
    setPatchStage('Uploading…')
    setPatchError('')
    setComparison(null)
    try {
      const fd = new FormData()
      fd.append('file', patchFile)
      const { data } = await api.validatePatch(sel, fd)
      setPatchStage('Scan started…')
      pollComparison(sel, data.scan_id)
    } catch (e) {
      setPatchStatus('error')
      setPatchError(e?.response?.data?.detail || 'Upload failed.')
    }
  }

  const resetValidation = () => {
    clearPoll()
    setPatchFile(null)
    setPatchProgress(0)
    setPatchStage('')
    setPatchStatus('idle')
    setComparison(null)
    setPatchError('')
    if (fileRef.current) fileRef.current.value = ''
  }

  const vulnTypes   = detail ? [...new Set(detail.vulnerabilities.map(v => v.type))] : []
  const patch       = tab ? PATCHES[tab] : null
  const vulnsOfType = detail?.vulnerabilities.filter(v => v.type === tab) || []

  return (
    <div className="space-y-5 animate-fadeUp">
      <div>
        <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Patch Engine</h2>
        <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
          CWE-mapped remediation, auto-fix generation, and patch validation
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Scan picker */}
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b text-xs font-medium"
            style={{ borderColor: 'rgba(255,255,255,0.05)', color: 'rgba(200,200,220,0.5)' }}>
            Scans with findings
          </div>
          {scans.length === 0 ? (
            <div className="p-6 text-center">
              <div className="text-3xl mb-2" style={{ color: 'rgba(200,200,220,0.1)' }}>◆</div>
              <p className="text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>No completed scans with vulnerabilities</p>
            </div>
          ) : (
            <div className="divide-y" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
              {scans.map(s => (
                <button key={s.scan_id} onClick={() => openScan(s)}
                  className="w-full px-4 py-3 text-left transition-all"
                  style={{
                    background: sel === s.scan_id ? 'rgba(200,169,110,0.08)' : 'transparent',
                    borderLeft: sel === s.scan_id ? '2px solid var(--gold)' : '2px solid transparent',
                  }}>
                  <div className="text-xs font-mono truncate" style={{ color: '#e8e8f0' }}>{s.filename}</div>
                  <div className="text-[10px] mt-0.5" style={{ color: '#fca5a5' }}>
                    {s.total_vulnerabilities} vulnerability{s.total_vulnerabilities !== 1 ? 's' : ''}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="lg:col-span-3 space-y-4">
          {/* FR-M7.10: Warning banner */}
          <div className="rounded-xl px-4 py-3 flex items-start gap-3"
            style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.22)' }}>
            <span style={{ color: '#fde047', flexShrink: 0 }}>⚠</span>
            <p className="text-xs leading-relaxed" style={{ color: '#fde047' }}>
              Patches are <strong>automatically generated suggestions</strong>. Review all changes carefully before applying to production code.
              Auto-fix applies pattern-based transformations — context-specific logic may require manual adjustment.
            </p>
          </div>

          {!sel ? (
            <div className="glass rounded-2xl p-12 text-center">
              <div className="text-5xl mb-4" style={{ color: 'rgba(200,200,220,0.08)' }}>◆</div>
              <p className="text-sm" style={{ color: 'rgba(200,200,220,0.3)' }}>Select a scan to view patch recommendations</p>
            </div>
          ) : !detail ? (
            <div className="glass rounded-2xl p-12 text-center">
              <span className="animate-spin text-2xl" style={{ color: 'var(--gold)' }}>⟳</span>
            </div>
          ) : (
            <>
              {/* FR-M7.9: Scan metadata + timestamp */}
              <div className="glass rounded-2xl px-5 py-3 flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-xs font-mono" style={{ color: 'rgba(200,200,220,0.5)' }}>
                    {detail.filename}
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded font-mono"
                    style={{ background: 'rgba(239,68,68,0.1)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.2)' }}>
                    {detail.total_vulnerabilities} finding{detail.total_vulnerabilities !== 1 ? 's' : ''}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[10px]" style={{ color: 'rgba(200,200,220,0.35)' }}>
                  {detail.completed_at && (
                    <span>Scanned: {new Date(detail.completed_at).toLocaleString()}</span>
                  )}
                  <span className="font-mono">{sel?.slice(0, 8)}…</span>
                </div>
              </div>

              {/* ── AUTO-FIX PANEL ─────────────────────────────────────────── */}
              <AutoFixPanel scanId={sel} vulnCount={detail.total_vulnerabilities} />

              {/* Vuln type tabs */}
              <div className="flex gap-2 flex-wrap">
                {vulnTypes.map(t => {
                  const p = PATCHES[t]
                  return (
                    <button key={t} onClick={() => setTab(t)}
                      className="text-xs px-3 py-2 rounded-lg transition-all flex items-center gap-1.5"
                      style={{
                        background: tab === t ? 'rgba(200,169,110,0.12)' : 'rgba(255,255,255,0.03)',
                        border: `1px solid ${tab === t ? 'rgba(200,169,110,0.4)' : 'rgba(200,200,220,0.08)'}`,
                        color: tab === t ? 'var(--gold)' : 'rgba(200,200,220,0.5)',
                      }}>
                      {p?.icon || '⚠'} {t}
                      <span className="text-[9px] ml-1 opacity-60">
                        ({detail.vulnerabilities.filter(v => v.type === t).length})
                      </span>
                    </button>
                  )
                })}
              </div>

              {/* Patch recommendations */}
              {patch && (
                <div className="glass rounded-2xl p-5 space-y-5">
                  <div className="flex items-start gap-3">
                    <span className="text-3xl">{patch.icon}</span>
                    <div>
                      <h3 className="font-display font-semibold text-base" style={{ color: '#f0f0f8' }}>{patch.title}</h3>
                      <p className="text-xs mt-1 leading-relaxed" style={{ color: 'rgba(200,200,220,0.5)' }}>{patch.desc}</p>
                      <div className="flex gap-2 mt-2 flex-wrap">
                        {patch.refs.map(r => (
                          <span key={r} className="text-[10px] px-2 py-0.5 rounded font-mono"
                            style={{ background: 'rgba(200,169,110,0.08)', border: '1px solid rgba(200,169,110,0.15)', color: 'var(--gold-dim)' }}>
                            {r}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {vulnsOfType.length > 0 && (
                    <div>
                      <p className="text-xs font-medium mb-2" style={{ color: 'rgba(200,200,220,0.5)' }}>
                        Detected Instances ({vulnsOfType.length})
                      </p>
                      <div className="space-y-1.5">
                        {vulnsOfType.map((v, i) => (
                          <div key={i} className="flex items-center justify-between rounded-lg px-3 py-2"
                            style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <code className="text-[11px] font-mono" style={{ color: '#fde68a' }}>{v.code_snippet}</code>
                            <span className="text-[10px] ml-3 flex-shrink-0" style={{ color: 'rgba(200,200,220,0.35)' }}>
                              Line {v.line_number} · {(v.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div>
                    <p className="text-xs font-medium mb-3" style={{ color: 'rgba(200,200,220,0.5)' }}>Remediation Patterns</p>
                    <div className="space-y-4">
                      {patch.fixes.map((fix, i) => {
                        const key = `${tab}_${i}`
                        const isRejected = rejected[key]
                        const ai = aiResults[key] || {}
                        const hasAiFix = !!ai.fix

                        const handleAiImprove = async () => {
                          if (ai.loading) return
                          setAiResults(p => ({ ...p, [key]: { loading: true } }))
                          try {
                            const snippet = vulnsOfType[0]?.code_snippet || fix.before
                            const { data } = await api.aiImprove({
                              scan_id: sel,
                              vuln_type: tab,
                              code_snippet: snippet,
                              fix_label: fix.label,
                            })
                            setAiResults(p => ({ ...p, [key]: { loading: false, fix: data.improved_fix } }))
                          } catch (e) {
                            const msg = e?.response?.data?.detail || 'AI unavailable — set ANTHROPIC_API_KEY'
                            setAiResults(p => ({ ...p, [key]: { loading: false, error: msg } }))
                          }
                        }

                        return (
                        <div key={i} style={{ opacity: isRejected ? 0.4 : 1, transition: 'opacity 0.3s' }}>
                          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                            <p className="text-xs font-medium" style={{ color: isRejected ? 'rgba(200,200,220,0.35)' : 'var(--gold)' }}>
                              {fix.label}
                              {isRejected && <span className="ml-2 text-[9px] font-mono" style={{ color: '#f87171' }}>REJECTED</span>}
                              {hasAiFix && <span className="ml-2 text-[9px] font-mono" style={{ color: '#86efac' }}>AI IMPROVED</span>}
                            </p>
                            <div className="flex gap-1.5">
                              {!isRejected && (
                                <button
                                  onClick={handleAiImprove}
                                  disabled={ai.loading}
                                  className="text-[10px] px-2 py-1 rounded transition-all"
                                  style={{
                                    background: hasAiFix ? 'rgba(34,197,94,0.12)' : 'rgba(200,169,110,0.08)',
                                    border: `1px solid ${hasAiFix ? 'rgba(34,197,94,0.3)' : 'rgba(200,169,110,0.2)'}`,
                                    color: hasAiFix ? '#86efac' : 'var(--gold)',
                                    opacity: ai.loading ? 0.6 : 1,
                                    cursor: ai.loading ? 'wait' : 'pointer',
                                  }}>
                                  {ai.loading ? '⟳ Thinking…' : hasAiFix ? '✓ AI Fix Applied' : '⚡ AI Improve'}
                                </button>
                              )}
                              <button
                                onClick={() => setRejected(p => ({ ...p, [key]: !p[key] }))}
                                className="text-[10px] px-2 py-1 rounded transition-all"
                                style={{
                                  background: isRejected ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.03)',
                                  border: `1px solid ${isRejected ? 'rgba(239,68,68,0.4)' : 'rgba(255,255,255,0.08)'}`,
                                  color: isRejected ? '#fca5a5' : 'rgba(200,200,220,0.4)',
                                }}>
                                {isRejected ? '↩ Restore' : '✗ Reject'}
                              </button>
                            </div>
                          </div>
                          {ai.error && (
                            <div className="mb-2 px-3 py-1.5 rounded text-[10px]"
                              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#fca5a5' }}>
                              {ai.error}
                            </div>
                          )}
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            <div className="rounded-lg overflow-hidden" style={{ border: '1px solid rgba(239,68,68,0.25)' }}>
                              <div className="px-3 py-1.5 text-[10px] font-medium flex items-center gap-1.5"
                                style={{ background: 'rgba(239,68,68,0.12)', color: '#fca5a5' }}>
                                ✗ Vulnerable
                              </div>
                              <pre className="px-3 py-2.5 text-[11px] font-mono overflow-x-auto"
                                style={{ background: 'rgba(239,68,68,0.05)', color: '#fca5a5', margin: 0 }}>
                                {fix.before}
                              </pre>
                            </div>
                            <div className="rounded-lg overflow-hidden" style={{ border: `1px solid ${hasAiFix ? 'rgba(34,197,94,0.45)' : 'rgba(34,197,94,0.25)'}` }}>
                              <div className="px-3 py-1.5 text-[10px] font-medium flex items-center gap-1.5"
                                style={{ background: 'rgba(34,197,94,0.12)', color: '#86efac' }}>
                                ✓ Patched {hasAiFix && <span className="ml-1 text-[9px] font-mono opacity-70">· AI-generated</span>}
                              </div>
                              <pre className="px-3 py-2.5 text-[11px] font-mono overflow-x-auto"
                                style={{ background: 'rgba(34,197,94,0.05)', color: '#86efac', margin: 0 }}>
                                {hasAiFix ? ai.fix : fix.after}
                              </pre>
                            </div>
                          </div>
                        </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* ── VALIDATE PATCH ─────────────────────────────────────────── */}
              <div className="glass rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>
                      Validate Your Patch
                    </h3>
                    <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.4)' }}>
                      Upload your manually-fixed file to verify which vulnerabilities were resolved
                    </p>
                  </div>
                  {patchStatus !== 'idle' && (
                    <button onClick={resetValidation}
                      className="text-[11px] px-3 py-1.5 rounded-lg transition-all"
                      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: 'rgba(200,200,220,0.5)' }}>
                      Reset
                    </button>
                  )}
                </div>

                {patchStatus === 'idle' && (
                  <div className="space-y-3">
                    <div
                      className="rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition-all"
                      style={{ borderColor: patchFile ? 'rgba(200,169,110,0.4)' : 'rgba(200,200,220,0.12)' }}
                      onClick={() => fileRef.current?.click()}>
                      <input ref={fileRef} type="file" accept=".c,.cpp,.cc,.cxx,.h,.hpp"
                        className="hidden"
                        onChange={e => setPatchFile(e.target.files[0] || null)} />
                      {patchFile ? (
                        <>
                          <p className="text-sm font-mono" style={{ color: 'var(--gold)' }}>{patchFile.name}</p>
                          <p className="text-[10px] mt-1" style={{ color: 'rgba(200,200,220,0.35)' }}>
                            {(patchFile.size / 1024).toFixed(1)} KB · click to change
                          </p>
                        </>
                      ) : (
                        <>
                          <p className="text-sm" style={{ color: 'rgba(200,200,220,0.35)' }}>Click to select patched C/C++ file</p>
                          <p className="text-[10px] mt-1" style={{ color: 'rgba(200,200,220,0.2)' }}>.c .cpp .cc .cxx .h .hpp</p>
                        </>
                      )}
                    </div>
                    <button disabled={!patchFile} onClick={submitPatch}
                      className="w-full py-2.5 rounded-xl text-sm font-medium transition-all"
                      style={{
                        background: patchFile ? 'rgba(200,169,110,0.15)' : 'rgba(255,255,255,0.03)',
                        border: `1px solid ${patchFile ? 'rgba(200,169,110,0.4)' : 'rgba(255,255,255,0.06)'}`,
                        color: patchFile ? 'var(--gold)' : 'rgba(200,200,220,0.2)',
                        cursor: patchFile ? 'pointer' : 'not-allowed',
                      }}>
                      Run Patch Validation
                    </button>
                  </div>
                )}

                {patchStatus === 'scanning' && (
                  <div className="flex items-center gap-3">
                    <span className="animate-spin text-lg" style={{ color: 'var(--gold)' }}>⟳</span>
                    <div className="flex-1">
                      <p className="text-xs" style={{ color: '#e8e8f0' }}>{patchStage || 'Scanning…'}</p>
                      <div className="mt-1.5 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                        <div className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${patchProgress}%`, background: 'var(--gold)' }} />
                      </div>
                    </div>
                    <span className="text-xs font-mono" style={{ color: 'var(--gold)' }}>{patchProgress}%</span>
                  </div>
                )}

                {patchStatus === 'error' && (
                  <div className="rounded-xl px-4 py-3 text-sm"
                    style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
                    {patchError}
                  </div>
                )}

                {patchStatus === 'done' && comparison && (
                  <ComparisonResults comparison={comparison} />
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Comparison results ────────────────────────────────────────────────────────

function ComparisonResults({ comparison }) {
  const { summary, fixed, still_present, new: newVulns } = comparison
  const [activeSection, setActiveSection] = useState(
    fixed.length > 0 ? 'fixed' : still_present.length > 0 ? 'still' : 'new'
  )
  const pct        = summary.improvement_pct
  const scoreColor = pct === 100 ? '#4ade80' : pct >= 50 ? '#facc15' : '#f87171'

  return (
    <div className="space-y-4">
      <div className="rounded-xl p-4"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-medium" style={{ color: 'rgba(200,200,220,0.5)' }}>
            Patch Validation · <span style={{ color: '#e8e8f0' }}>{comparison.patch_filename}</span>
          </p>
          <span className="text-lg font-bold font-mono" style={{ color: scoreColor }}>{pct}% fixed</span>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Fixed',         count: summary.fixed_count,         color: '#4ade80', icon: '✓' },
            { label: 'Still Present', count: summary.still_present_count, color: '#facc15', icon: '⚠' },
            { label: 'New',           count: summary.new_count,           color: '#f87171', icon: '!' },
          ].map(({ label, count, color, icon }) => (
            <div key={label} className="rounded-lg p-3 text-center"
              style={{ background: `${color}10`, border: `1px solid ${color}30` }}>
              <div className="text-xl font-bold" style={{ color }}>{count}</div>
              <div className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.45)' }}>{icon} {label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-2">
        {[
          { key: 'fixed', label: `✓ Fixed (${summary.fixed_count})`,                  color: '#4ade80' },
          { key: 'still', label: `⚠ Still Present (${summary.still_present_count})`,  color: '#facc15' },
          { key: 'new',   label: `! New (${summary.new_count})`,                       color: '#f87171' },
        ].map(({ key, label, color }) => (
          <button key={key} onClick={() => setActiveSection(key)}
            className="text-[11px] px-3 py-1.5 rounded-lg transition-all"
            style={{
              background: activeSection === key ? `${color}15` : 'rgba(255,255,255,0.03)',
              border: `1px solid ${activeSection === key ? `${color}50` : 'rgba(255,255,255,0.06)'}`,
              color: activeSection === key ? color : 'rgba(200,200,220,0.4)',
            }}>
            {label}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        {activeSection === 'fixed' && (
          fixed.length === 0
            ? <EmptySection text="No vulnerabilities were fixed." />
            : fixed.map((v, i) => <VulnRow key={i} v={v} accent="#4ade80" />)
        )}
        {activeSection === 'still' && (
          still_present.length === 0
            ? <EmptySection text="All original vulnerabilities were resolved." />
            : still_present.map((v, i) => <VulnRow key={i} v={v} accent="#facc15" />)
        )}
        {activeSection === 'new' && (
          newVulns.length === 0
            ? <EmptySection text="No new vulnerabilities were introduced." />
            : newVulns.map((v, i) => <VulnRow key={i} v={v} accent="#f87171" />)
        )}
      </div>
    </div>
  )
}

function EmptySection({ text }) {
  return (
    <div className="rounded-lg px-4 py-6 text-center text-xs"
      style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid rgba(255,255,255,0.05)', color: 'rgba(200,200,220,0.3)' }}>
      {text}
    </div>
  )
}

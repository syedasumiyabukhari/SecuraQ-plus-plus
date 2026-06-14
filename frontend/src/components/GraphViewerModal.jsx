import React, { useEffect, useRef, useState, useCallback } from 'react'

const STRUCTURAL = {
  AST: ['FUNC_ROOT', 'FUNC_DECL', 'PARAM_DECL', 'COMPOUND_STMT', 'RETURN_STMT', 'CALL_EXPR', 'DECL_STMT'],
  CFG: ['ENTRY', 'EXIT', 'BRANCH', 'JOIN'],
  DFG: ['DEF', 'USE', 'PHI_NODE', 'LOAD', 'STORE'],
  PDG: ['CONTROL_DEP', 'DATA_DEP', 'SUMMARY_NODE'],
  TPG: ['TOKEN_START', 'TOKEN_END', 'PATH_NODE'],
  MAG: ['ASPECT_NODE', 'MERGE_POINT', 'WEIGHT_NODE'],
  CG:  ['CALL_SITE', 'FUNC_DEF', 'RETURN_PT', 'INDIRECT'],
  FSG: ['SEQ_START', 'SEQ_END', 'BARRIER'],
}

const EDGE_LABELS = {
  AST: 'AST_EDGE', CFG: 'CFG_EDGE', DFG: 'DATA_FLOW',
  PDG: 'DEP_EDGE', TPG: 'PATH_EDGE', MAG: 'ASPECT_EDGE',
  CG: 'CALL_EDGE', FSG: 'SEQ_EDGE',
}

function buildGraph(graphType, nodeCount, edgeCount, vulns) {
  const codeSnippets = (vulns || [])
    .map(v => v.code_snippet?.trim().slice(0, 24))
    .filter(Boolean)

  const structural = STRUCTURAL[graphType] || []
  const allLabels  = [...codeSnippets, ...structural]

  const W = 760, H = 520
  const nodes = Array.from({ length: nodeCount }, (_, i) => {
    const angle = (i / nodeCount) * Math.PI * 2
    const r = 160 + Math.random() * 80
    return {
      id: i,
      label: allLabels[i % allLabels.length] || `node_${i}`,
      type: i < codeSnippets.length ? 'CODE' : 'STRUCTURAL',
      x: W / 2 + Math.cos(angle) * r,
      y: H / 2 + Math.sin(angle) * r,
      vx: 0, vy: 0,
      fixed: false,
    }
  })

  const edgeSet = new Set()
  const edges = []

  // Spanning tree first (guarantees connectivity)
  for (let i = 1; i < nodes.length && edges.length < edgeCount; i++) {
    const j = Math.floor(Math.random() * i)
    const k = `${j}-${i}`
    if (!edgeSet.has(k)) { edgeSet.add(k); edges.push({ s: j, t: i }) }
  }

  // Fill remaining edges randomly
  let tries = 0
  while (edges.length < edgeCount && tries++ < 2000) {
    const s = Math.floor(Math.random() * nodes.length)
    const t = Math.floor(Math.random() * nodes.length)
    if (s === t) continue
    const k = `${s}-${t}`
    if (!edgeSet.has(k)) { edgeSet.add(k); edges.push({ s, t }) }
  }

  return { nodes, edges }
}

export default function GraphViewerModal({ graphType, graphInfo, graphMeta, results, onClose }) {
  const svgRef       = useRef(null)
  const nodesRef     = useRef([])
  const edgesRef     = useRef([])
  const simRef       = useRef(null)
  const dragRef      = useRef(null)
  const [tick, setTick]           = useState(0)
  const [hoveredNode, setHoveredNode] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const nodeCount = graphMeta?.nodes || 0
  const edgeCount = graphMeta?.edges || 0

  // Build graph once
  useEffect(() => {
    const g = buildGraph(graphType, nodeCount, edgeCount, results?.vulnerabilities)
    nodesRef.current = g.nodes
    edgesRef.current = g.edges

    let frame = 0
    const MAX = 400

    const step = () => {
      const ns = nodesRef.current
      const es = edgesRef.current
      const el = svgRef.current
      if (!el) return

      const W = el.clientWidth  || 760
      const H = el.clientHeight || 520
      const CX = W / 2, CY = H / 2

      if (frame < MAX || dragRef.current !== null) {
        frame++
        const alpha = Math.max(0.01, 1 - frame / MAX)

        // Gravity
        for (const n of ns) {
          if (n.fixed) continue
          n.vx += (CX - n.x) * 0.003 * alpha
          n.vy += (CY - n.y) * 0.003 * alpha
        }

        // Repulsion
        for (let i = 0; i < ns.length; i++) {
          for (let j = i + 1; j < ns.length; j++) {
            const a = ns[i], b = ns[j]
            const dx = a.x - b.x, dy = a.y - b.y
            const d2 = dx * dx + dy * dy || 1
            const f  = (3000 / d2) * alpha
            const fx = (dx / Math.sqrt(d2)) * f
            const fy = (dy / Math.sqrt(d2)) * f
            if (!a.fixed) { a.vx += fx; a.vy += fy }
            if (!b.fixed) { b.vx -= fx; b.vy -= fy }
          }
        }

        // Attraction (spring)
        for (const e of es) {
          const a = ns[e.s], b = ns[e.t]
          if (!a || !b) continue
          const dx = b.x - a.x, dy = b.y - a.y
          const d  = Math.sqrt(dx * dx + dy * dy) || 1
          const f  = (d - 110) * 0.025 * alpha
          const fx = (dx / d) * f, fy = (dy / d) * f
          if (!a.fixed) { a.vx += fx; a.vy += fy }
          if (!b.fixed) { b.vx -= fx; b.vy -= fy }
        }

        // Integrate
        for (const n of ns) {
          if (n.fixed) continue
          n.vx *= 0.82; n.vy *= 0.82
          n.x  = Math.max(30, Math.min(W - 30, n.x + n.vx))
          n.y  = Math.max(20, Math.min(H - 20, n.y + n.vy))
        }

        setTick(t => t + 1)
      }

      simRef.current = requestAnimationFrame(step)
    }

    simRef.current = requestAnimationFrame(step)
    return () => { if (simRef.current) cancelAnimationFrame(simRef.current) }
  }, [graphType, nodeCount, edgeCount])

  // Close on Escape
  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  // Drag handlers
  const onNodeMouseDown = useCallback((e, idx) => {
    e.stopPropagation()
    const n = nodesRef.current[idx]
    if (!n) return
    n.fixed = true
    dragRef.current = { idx, ox: e.clientX - n.x, oy: e.clientY - n.y }

    const onMove = ev => {
      const nd = nodesRef.current[dragRef.current.idx]
      if (!nd) return
      nd.x = ev.clientX - dragRef.current.ox
      nd.y = ev.clientY - dragRef.current.oy
      nd.vx = 0; nd.vy = 0
      setTick(t => t + 1)
    }
    const onUp = () => {
      if (dragRef.current !== null) {
        nodesRef.current[dragRef.current.idx].fixed = false
      }
      dragRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [])

  const ns = nodesRef.current
  const es = edgesRef.current
  const hovered = hoveredNode !== null ? ns[hoveredNode] : null
  const selected = selectedNode !== null ? ns[selectedNode] : null
  const infoNode = selected || hovered

  const nodeTypes = [...new Set(ns.map(n => n.type))]
  const edgeLabel = EDGE_LABELS[graphType] || 'EDGE'

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{ background: 'rgba(5,5,15,0.97)', backdropFilter: 'blur(8px)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}>

      {/* Header */}
      <div className="flex items-center gap-4 px-5 py-3 flex-shrink-0"
        style={{ background: 'rgba(0,0,0,0.6)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <span className="text-xl font-bold font-mono" style={{ color: graphInfo.color }}>
          {graphType}
        </span>
        <span className="text-sm font-medium" style={{ color: 'rgba(200,200,220,0.7)' }}>
          {graphInfo.full}
        </span>
        <span className="text-xs px-2 py-0.5 rounded font-mono"
          style={{ background: 'rgba(255,255,255,0.05)', color: 'rgba(200,200,220,0.4)', border: '1px solid rgba(255,255,255,0.07)' }}>
          {nodeCount} nodes · {edgeCount} edges
        </span>
        <div className="flex-1" />
        <button onClick={onClose}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-lg transition-all hover:bg-white/10"
          style={{ color: 'rgba(200,200,220,0.5)' }}>
          ✕
        </button>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">

        {/* Graph canvas */}
        <svg ref={svgRef} className="flex-1 h-full" style={{ cursor: 'grab' }}>
          <defs>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L6,3 z" fill="rgba(200,200,220,0.2)" />
            </marker>
          </defs>

          {/* Edges */}
          {es.map((e, i) => {
            const a = ns[e.s], b = ns[e.t]
            if (!a || !b) return null
            const isHovered = hoveredNode === e.s || hoveredNode === e.t ||
                              selectedNode === e.s || selectedNode === e.t
            return (
              <line key={i}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={isHovered ? graphInfo.color : 'rgba(200,200,220,0.12)'}
                strokeWidth={isHovered ? 1.5 : 0.8}
                markerEnd="url(#arrow)"
                style={{ transition: 'stroke 0.15s' }}
              />
            )
          })}

          {/* Nodes */}
          {ns.map((n, i) => {
            const isHov = hoveredNode === i
            const isSel = selectedNode === i
            const r = isSel ? 9 : isHov ? 8 : 6
            const col = n.type === 'STRUCTURAL'
              ? 'rgba(200,200,220,0.5)'
              : graphInfo.color
            return (
              <g key={i}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredNode(i)}
                onMouseLeave={() => setHoveredNode(null)}
                onClick={e => { e.stopPropagation(); setSelectedNode(isSel ? null : i) }}
                onMouseDown={e => onNodeMouseDown(e, i)}>
                {/* Glow */}
                {(isHov || isSel) && (
                  <circle cx={n.x} cy={n.y} r={r + 6}
                    fill={graphInfo.color} opacity={0.12} />
                )}
                <circle cx={n.x} cy={n.y} r={r}
                  fill={isSel ? graphInfo.color : isHov ? col : 'rgba(30,30,50,0.9)'}
                  stroke={col}
                  strokeWidth={isSel ? 0 : 1.2}
                />
                {/* Label */}
                <text
                  x={n.x} y={n.y - r - 4}
                  textAnchor="middle"
                  fontSize={9}
                  fontFamily="monospace"
                  fill={isHov || isSel ? '#e8e8f0' : 'rgba(200,200,220,0.45)'}
                  style={{ pointerEvents: 'none', userSelect: 'none' }}>
                  {n.label.length > 18 ? n.label.slice(0, 18) + '…' : n.label}
                </text>
              </g>
            )
          })}
        </svg>

        {/* Right panel */}
        <div className="w-52 flex-shrink-0 flex flex-col p-4 space-y-5 overflow-y-auto"
          style={{ background: 'rgba(0,0,0,0.5)', borderLeft: '1px solid rgba(255,255,255,0.05)' }}>

          {/* Node info */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: 'rgba(200,200,220,0.4)' }}>Node Info</p>
            {infoNode ? (
              <div className="rounded-lg p-3 space-y-1.5"
                style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${graphInfo.color}33` }}>
                <div className="font-mono text-[10px] break-all" style={{ color: graphInfo.color }}>
                  {infoNode.label}
                </div>
                <div className="text-[9px]" style={{ color: 'rgba(200,200,220,0.4)' }}>
                  Type: {infoNode.type}
                </div>
                <div className="text-[9px]" style={{ color: 'rgba(200,200,220,0.3)' }}>
                  Pos: ({Math.round(infoNode.x)}, {Math.round(infoNode.y)})
                </div>
                <div className="text-[9px]" style={{ color: 'rgba(200,200,220,0.3)' }}>
                  Edges: {es.filter(e => e.s === infoNode.id || e.t === infoNode.id).length}
                </div>
              </div>
            ) : (
              <p className="text-[10px]" style={{ color: 'rgba(200,200,220,0.25)' }}>
                Hover or click a node
              </p>
            )}
          </div>

          {/* Node types */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: 'rgba(200,200,220,0.4)' }}>
              Node Types ({nodeTypes.length})
            </p>
            <div className="space-y-1.5">
              {nodeTypes.map(t => (
                <div key={t} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: t === 'STRUCTURAL' ? 'rgba(200,200,220,0.5)' : graphInfo.color }} />
                  <span className="text-[10px] font-mono" style={{ color: 'rgba(200,200,220,0.5)' }}>{t}</span>
                </div>
              ))}
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: 'rgba(200,200,220,0.3)' }} />
                <span className="text-[10px] font-mono" style={{ color: 'rgba(200,200,220,0.35)' }}>UNKNOWN</span>
              </div>
            </div>
          </div>

          {/* Edge types */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: 'rgba(200,200,220,0.4)' }}>
              Edge Types (1)
            </p>
            <div className="flex items-center gap-2">
              <div className="w-5 h-px" style={{ background: 'rgba(200,200,220,0.3)' }} />
              <span className="text-[10px] font-mono" style={{ color: 'rgba(200,200,220,0.5)' }}>
                {edgeLabel}
              </span>
            </div>
          </div>

          {/* Hint */}
          <div className="mt-auto pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
            <p className="text-[9px]" style={{ color: 'rgba(200,200,220,0.2)' }}>
              Drag nodes to rearrange. Click to select. Press Esc to close.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

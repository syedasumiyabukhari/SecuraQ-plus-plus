import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function LandingPage() {
  const navigate = useNavigate()
  const { user } = useAuth()

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--surface-0)' }}>
      {/* Nav */}
      <header className="flex items-center justify-between px-8 py-5" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center text-lg"
            style={{ background: 'linear-gradient(135deg, var(--gold), var(--gold-dim))' }}>⚛</div>
          <span className="font-display font-bold text-lg" style={{ color: 'var(--gold-light)' }}>SecuraQ++</span>
        </div>
        <div className="flex gap-3">
          {user ? (
            <button onClick={() => navigate('/dashboard')} className="btn-gold px-5 py-2 rounded-lg text-sm">
              Dashboard →
            </button>
          ) : (
            <>
              <button onClick={() => navigate('/login')}
                className="px-5 py-2 rounded-lg text-sm transition-all"
                style={{ color: 'var(--gold)', border: '1px solid var(--border)' }}
                onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--gold)'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}>
                Sign In
              </button>
              <button onClick={() => navigate('/register')} className="btn-gold px-5 py-2 rounded-lg text-sm">
                Get Started
              </button>
            </>
          )}
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-8 py-24 text-center animate-fadeUp">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs mb-8"
          style={{ background: 'rgba(200,169,110,0.08)', border: '1px solid rgba(200,169,110,0.2)', color: 'var(--gold)' }}>
          <span className="status-online">QENND-v2.0 — Now with 3-classifier hybrid fusion</span>
        </div>

        <h1 className="font-display font-bold text-5xl md:text-7xl mb-6 leading-tight">
          <span className="text-gold-gradient">Quantum-Enhanced</span>
          <br />
          <span style={{ color: '#e8e8f0' }}>Vulnerability Detection</span>
        </h1>

        <p className="text-lg max-w-2xl mb-10" style={{ color: 'rgba(200,200,220,0.5)', lineHeight: 1.7 }}>
          SecuraQ++ combines GAT graph neural networks with a 4-qubit variational
          quantum circuit to detect Buffer Overflow, Format String, and Use-After-Free
          vulnerabilities in C/C++ code — with sub-second inference.
        </p>

        <div className="flex flex-wrap gap-4 justify-center">
          <button onClick={() => navigate('/register')} className="btn-gold px-8 py-3 rounded-xl text-base font-semibold">
            Start Scanning Free
          </button>
          <button onClick={() => navigate('/login')}
            className="px-8 py-3 rounded-xl text-base transition-all"
            style={{ color: 'rgba(200,200,220,0.7)', border: '1px solid var(--border)' }}>
            Sign In
          </button>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-8 mt-20 max-w-2xl w-full">
          {[
            { val: '3', label: 'Vulnerability Classes', sub: 'BO · FS · UAF' },
            { val: '4', label: 'Qubit VQC Circuit',     sub: 'PennyLane backend' },
            { val: '9', label: 'Pipeline Stages',        sub: 'End-to-end QEGVD' },
          ].map(s => (
            <div key={s.val} className="text-center">
              <div className="font-display font-bold text-4xl text-gold-gradient">{s.val}</div>
              <div className="text-sm font-medium mt-1" style={{ color: 'rgba(200,200,220,0.7)' }}>{s.label}</div>
              <div className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.3)' }}>{s.sub}</div>
            </div>
          ))}
        </div>

        {/* Feature cards */}
        <div className="grid md:grid-cols-3 gap-4 mt-16 max-w-4xl w-full text-left">
          {[
            { icon: '🧠', title: 'Multi-View GAT', desc: '8 graph views (AST, CFG, DFG, PDG, TPG, MAG, CG, FSG) encoded via 4-layer GATConv.' },
            { icon: '⚛️', title: 'Quantum VQC', desc: '4-qubit variational circuit with RY/RZ rotations and CNOT entanglement for feature compression.' },
            { icon: '🛡️', title: 'Patch Recommendations', desc: 'Automated CWE-mapped patch suggestions with code-level fixes for every detected vulnerability.' },
          ].map(f => (
            <div key={f.title} className="glass rounded-xl p-5">
              <div className="text-2xl mb-3">{f.icon}</div>
              <div className="font-display font-semibold text-sm mb-2" style={{ color: '#e8e8f0' }}>{f.title}</div>
              <div className="text-xs leading-relaxed" style={{ color: 'rgba(200,200,220,0.45)' }}>{f.desc}</div>
            </div>
          ))}
        </div>
      </main>

      <footer className="py-6 text-center text-xs" style={{ color: 'rgba(200,200,220,0.2)', borderTop: '1px solid var(--border)' }}>
        SecuraQ++ · QEGVD Platform v2.0 · Quantum-Hybrid Vulnerability Detection
      </footer>
    </div>
  )
}

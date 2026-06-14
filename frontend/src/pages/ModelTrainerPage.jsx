import React, { useState } from 'react'

const MODELS = [
  { name: 'BO Classifier (MLP)',    acc: 90.5, f1: 0.912, precision: 0.851, recall: 0.981, status: 'trained', trained: '2025-12-01' },
  { name: 'FS Classifier (Hybrid)', acc: 82.0, f1: 0.819, precision: 0.831, recall: 0.808, status: 'trained', trained: '2026-05-05' },
  { name: 'UAF Classifier (MLP)',   acc: 91.8, f1: 0.915, precision: 0.908, recall: 0.922, status: 'trained', trained: '2025-12-01' },
]

const STAGES = [
  { id: 1, name: 'Graph Construction',     detail: 'AST / CFG / DFG / PDG / TPG / MAG / CG / FSG' },
  { id: 2, name: 'GAT Encoder Training',   detail: '4-layer GATConv · heads=8 · 128-dim output' },
  { id: 3, name: 'Classical Encoder',      detail: 'MLP 128 → 32-dim compression' },
  { id: 4, name: 'QAFA Selection',         detail: 'Quantum-Assisted Feature Analysis · top-16' },
  { id: 5, name: 'VQC Training',           detail: '4-qubit PennyLane circuit · Adam optimiser' },
  { id: 6, name: 'Hybrid Fusion',          detail: 'Residual concat 32+4 → 36-dim MLP head' },
  { id: 7, name: 'FS Direct Classifier',   detail: 'Stacking GBM+RF+ET+LR · TF-IDF n-grams' },
]

export default function ModelTrainerPage() {
  const [selected, setSelected] = useState(null)

  return (
    <div className="space-y-5 animate-fadeUp max-w-4xl">
      <div>
        <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Model Trainer</h2>
        <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
          QEGVD training pipeline · BO · FS · UAF classifiers · Quantum-Classical hybrid
        </p>
      </div>

      {/* Notice */}
      <div className="rounded-xl px-4 py-3 flex items-start gap-3"
        style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.2)' }}>
        <span style={{ color: '#fde047', flexShrink: 0 }}>⚠</span>
        <p className="text-xs leading-relaxed" style={{ color: '#fde047' }}>
          Model retraining requires the full QEGVD training pipeline and GPU resources.
          Training is executed offline via <code className="font-mono">train_qegvd.py</code> and <code className="font-mono">train_fs_direct.py</code>.
          This panel shows current model performance metrics.
        </p>
      </div>

      {/* Model cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {MODELS.map((m, i) => (
          <div key={i}
            onClick={() => setSelected(selected === i ? null : i)}
            className="glass rounded-2xl p-5 cursor-pointer transition-all"
            style={{
              border: selected === i ? '1px solid rgba(200,169,110,0.35)' : '1px solid rgba(255,255,255,0.06)',
              background: selected === i ? 'rgba(200,169,110,0.04)' : undefined,
            }}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs font-medium" style={{ color: '#e8e8f0' }}>{m.name}</p>
                <p className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>Trained: {m.trained}</p>
              </div>
              <span className="text-[9px] px-2 py-0.5 rounded font-mono"
                style={{ background: 'rgba(34,197,94,0.1)', color: '#86efac', border: '1px solid rgba(34,197,94,0.2)' }}>
                ACTIVE
              </span>
            </div>
            <div className="text-3xl font-bold font-mono" style={{ color: 'var(--gold)' }}>
              {m.acc}%
            </div>
            <div className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>Accuracy</div>

            {selected === i && (
              <div className="mt-3 pt-3 border-t space-y-1.5 animate-fadeUp"
                style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
                {[
                  { label: 'F1 Score',  value: m.f1.toFixed(3) },
                  { label: 'Precision', value: m.precision.toFixed(3) },
                  { label: 'Recall',    value: m.recall.toFixed(3) },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between text-[11px]">
                    <span style={{ color: 'rgba(200,200,220,0.45)' }}>{label}</span>
                    <span className="font-mono" style={{ color: '#e8e8f0' }}>{value}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Training pipeline steps */}
      <div className="glass rounded-2xl p-5">
        <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#f0f0f8' }}>QEGVD Training Pipeline</h3>
        <div className="space-y-0">
          {STAGES.map((s, i) => (
            <div key={i} className="flex items-start gap-3 py-3 border-b last:border-0 relative"
              style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
              {/* Connector line */}
              {i < STAGES.length - 1 && (
                <div className="absolute left-[18px] top-8 w-px h-full"
                  style={{ background: 'rgba(200,169,110,0.15)' }} />
              )}
              <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 z-10"
                style={{ background: 'rgba(200,169,110,0.12)', border: '1px solid rgba(200,169,110,0.25)' }}>
                <span className="text-[10px] font-mono font-bold" style={{ color: 'var(--gold)' }}>{s.id}</span>
              </div>
              <div className="min-w-0">
                <p className="text-xs font-medium" style={{ color: '#e8e8f0' }}>{s.name}</p>
                <p className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>{s.detail}</p>
              </div>
              <span className="ml-auto flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded"
                style={{ background: 'rgba(34,197,94,0.1)', color: '#86efac', border: '1px solid rgba(34,197,94,0.2)' }}>
                OK
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

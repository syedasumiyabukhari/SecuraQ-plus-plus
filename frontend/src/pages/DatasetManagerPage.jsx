import React from 'react'

const DATASETS = [
  { name: 'Juliet Test Suite — CWE-121 (Stack BO)',  samples: 3200, split: '80/10/10', status: 'loaded' },
  { name: 'Juliet Test Suite — CWE-134 (Format Str)', samples: 2800, split: '80/10/10', status: 'loaded' },
  { name: 'Juliet Test Suite — CWE-416 (UAF)',        samples: 2600, split: '80/10/10', status: 'loaded' },
  { name: 'NIST SARD C/C++ corpus',                   samples: 0,    split: '—',        status: 'pending' },
]

const PIPELINE_STEPS = [
  'Identifier masking (variable / function names → __VAR__, __FUNC__)',
  'Tokenisation & TF-IDF feature extraction',
  'Graph construction (AST / CFG / DFG / PDG / TPG / MAG / CG / FSG)',
  'GAT embedding (128-dim) → classical encoder (32-dim)',
  'QAFA feature selection (top-16 dimensions)',
  'Train / validation / test split stratified by vulnerability type',
]

export default function DatasetManagerPage() {
  return (
    <div className="space-y-5 animate-fadeUp max-w-4xl">
      <div>
        <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Dataset Manager</h2>
        <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
          Manage training datasets for the QEGVD pipeline · Juliet Test Suite · CWE-121 · CWE-134 · CWE-416
        </p>
      </div>

      {/* Notice */}
      <div className="rounded-xl px-4 py-3 flex items-start gap-3"
        style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.2)' }}>
        <span style={{ color: '#fde047', flexShrink: 0 }}>⚠</span>
        <p className="text-xs leading-relaxed" style={{ color: '#fde047' }}>
          Dataset upload and retraining is managed offline through the QEGVD training scripts.
          This panel provides visibility into loaded datasets. Contact the system administrator to add new datasets.
        </p>
      </div>

      {/* Dataset table */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-5 py-4 border-b flex items-center justify-between"
          style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>Loaded Datasets</h3>
          <span className="text-[10px] px-2 py-1 rounded font-mono"
            style={{ background: 'rgba(200,169,110,0.08)', border: '1px solid rgba(200,169,110,0.2)', color: 'var(--gold)' }}>
            {DATASETS.filter(d => d.status === 'loaded').length} / {DATASETS.length} active
          </span>
        </div>
        <div className="divide-y" style={{ divideColor: 'rgba(255,255,255,0.04)' }}>
          {DATASETS.map((d, i) => (
            <div key={i} className="px-5 py-3 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0">
                <span className={`w-2 h-2 rounded-full flex-shrink-0 ${d.status === 'loaded' ? 'bg-green-400' : 'bg-yellow-500'}`} />
                <div className="min-w-0">
                  <p className="text-xs font-medium truncate" style={{ color: '#e8e8f0' }}>{d.name}</p>
                  <p className="text-[10px] mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>
                    Split: {d.split}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4 flex-shrink-0 text-right">
                <div>
                  <div className="text-sm font-bold font-mono" style={{ color: d.status === 'loaded' ? 'var(--gold)' : 'rgba(200,200,220,0.3)' }}>
                    {d.samples > 0 ? d.samples.toLocaleString() : '—'}
                  </div>
                  <div className="text-[9px]" style={{ color: 'rgba(200,200,220,0.3)' }}>samples</div>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded font-mono"
                  style={{
                    background: d.status === 'loaded' ? 'rgba(34,197,94,0.1)' : 'rgba(234,179,8,0.1)',
                    color: d.status === 'loaded' ? '#86efac' : '#fde047',
                    border: `1px solid ${d.status === 'loaded' ? 'rgba(34,197,94,0.25)' : 'rgba(234,179,8,0.25)'}`,
                  }}>
                  {d.status.toUpperCase()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Preprocessing pipeline */}
      <div className="glass rounded-2xl p-5">
        <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#f0f0f8' }}>Preprocessing Pipeline</h3>
        <div className="space-y-2">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={i} className="flex items-start gap-3 py-2 border-b last:border-0"
              style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
              <span className="text-[10px] font-mono w-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--gold-dim)' }}>
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="text-xs" style={{ color: 'rgba(200,200,220,0.65)' }}>{step}</span>
              <span className="ml-auto flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded"
                style={{ background: 'rgba(34,197,94,0.1)', color: '#86efac', border: '1px solid rgba(34,197,94,0.2)' }}>
                OK
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Stats summary */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Total Samples', value: '8,600', color: 'var(--gold)' },
          { label: 'Vulnerability Types', value: '3', color: '#60a5fa' },
          { label: 'Balanced Classes', value: 'Yes', color: '#86efac' },
        ].map(s => (
          <div key={s.label} className="glass rounded-2xl p-4 text-center">
            <div className="font-display font-bold text-2xl" style={{ color: s.color }}>{s.value}</div>
            <div className="text-[10px] mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

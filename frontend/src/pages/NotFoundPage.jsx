import React from 'react'
import { useNavigate } from 'react-router-dom'

export default function NotFoundPage() {
  const navigate = useNavigate()
  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--surface-0)' }}>
      <div className="text-center animate-fadeUp">
        <div className="font-display font-bold text-8xl text-gold-gradient mb-4">404</div>
        <p className="text-lg mb-2" style={{ color: '#e8e8f0' }}>Page not found</p>
        <p className="text-sm mb-8" style={{ color: 'rgba(200,200,220,0.4)' }}>This route doesn't exist in the QEGVD platform.</p>
        <button onClick={() => navigate('/dashboard')} className="btn-gold px-6 py-2.5 rounded-xl text-sm">
          Back to Dashboard
        </button>
      </div>
    </div>
  )
}

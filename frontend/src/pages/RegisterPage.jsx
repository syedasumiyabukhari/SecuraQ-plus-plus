import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'

function EyeIcon({ open }) {
  return open ? (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
    </svg>
  ) : (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19M1 1l22 22"/>
    </svg>
  )
}

function CheckIcon({ ok }) {
  return ok ? (
    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="#22c55e" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/></svg>
  ) : (
    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="#6b7280" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
  )
}

function getChecks(p) {
  return {
    length:    p.length >= 8,
    uppercase: /[A-Z]/.test(p),
    number:    /[0-9]/.test(p),
    special:   /[^A-Za-z0-9]/.test(p),
  }
}

// FR-M1.2: institutional / academic email domains
function isInstitutionalEmail(email) {
  const domain = email.split('@')[1]?.toLowerCase() || ''
  return (
    /\.edu$/.test(domain) ||          // .edu TLD
    /\.edu\.[a-z]{2}$/.test(domain) || // .edu.au, .edu.pk, .edu.sg …
    /\.ac\.[a-z]{2,}$/.test(domain) || // .ac.uk, .ac.nz, .ac.za …
    /\.university$/.test(domain) ||
    /\.institute$/.test(domain) ||
    /university\./.test(domain) ||     // university.example.com
    domain === 'securaqpp.local'       // dev / admin bypass
  )
}

export default function RegisterPage() {
  const [form, setForm]           = useState({ full_name: '', email: '', password: '', confirm: '', admin_code: '' })
  const [showPass, setShowPass]   = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [termsAccepted, setTerms] = useState(false)
  const [error, setError]         = useState('')
  const [loading, setLoading]     = useState(false)
  const navigate = useNavigate()

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }))

  const checks  = getChecks(form.password)
  const isStrong = Object.values(checks).every(Boolean)

  const strengthScore = Object.values(checks).filter(Boolean).length
  const strengthColor = ['', '#ef4444', '#f97316', '#eab308', '#22c55e'][strengthScore]
  const strengthLabel = ['', 'Weak', 'Fair', 'Good', 'Strong'][strengthScore]

  const emailOk = form.email ? isInstitutionalEmail(form.email) : true

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!isInstitutionalEmail(form.email))
      return setError('Please use an institutional or academic email address (e.g. .edu, .ac.uk)')
    if (!isStrong) return setError('Please meet all password requirements before continuing')
    if (form.password !== form.confirm) return setError('Passwords do not match')
    if (!termsAccepted) return setError('You must accept the Terms of Use and Privacy Policy')
    setLoading(true)
    try {
      await api.register({ full_name: form.full_name, email: form.email, password: form.password, admin_code: form.admin_code || undefined })
      navigate('/login', { state: { registered: true } })
    } catch (err) {
      setError(err.response?.data?.error || 'Registration failed')
    } finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12" style={{ background: 'var(--surface-0)' }}>
      <div className="w-full max-w-md animate-fadeUp">
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-4"
            style={{ background: 'linear-gradient(135deg, var(--gold), var(--gold-dim))' }}>⚛</div>
          <h1 className="font-display font-bold text-2xl" style={{ color: 'var(--gold-light)' }}>Create Account</h1>
          <p className="text-sm mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>Join SecuraQ++ — Quantum Vulnerability Detection</p>
        </div>

        <div className="glass rounded-2xl p-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(200,200,220,0.6)' }}>Full Name</label>
              <input value={form.full_name} onChange={set('full_name')} required
                className="input-dark w-full px-4 py-3 rounded-xl text-sm" placeholder="Jane Smith" />
            </div>

            {/* Email — FR-M1.2 */}
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(200,200,220,0.6)' }}>
                Institutional Email
              </label>
              <input type="email" value={form.email} onChange={set('email')} required
                className="input-dark w-full px-4 py-3 rounded-xl text-sm"
                placeholder="jane@university.edu" />
              {form.email && !emailOk && (
                <p className="text-[11px] mt-1.5 flex items-center gap-1" style={{ color: '#f97316' }}>
                  <span>⚠</span> Use an institutional email (.edu, .ac.uk, university domain)
                </p>
              )}
              {form.email && emailOk && (
                <p className="text-[11px] mt-1.5 flex items-center gap-1" style={{ color: '#22c55e' }}>
                  <span>✓</span> Institutional email accepted
                </p>
              )}
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(200,200,220,0.6)' }}>Password</label>
              <div className="relative">
                <input type={showPass ? 'text' : 'password'} value={form.password} onChange={set('password')} required
                  className="input-dark w-full px-4 py-3 pr-11 rounded-xl text-sm" placeholder="Create a strong password" />
                <button type="button" onClick={() => setShowPass(v => !v)} tabIndex={-1}
                  className="absolute right-3 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-90 transition-opacity"
                  style={{ color: 'var(--text-dim)' }}>
                  <EyeIcon open={showPass} />
                </button>
              </div>

              {form.password && (
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex gap-1 flex-1">
                    {[1,2,3,4].map(i => (
                      <div key={i} className="h-1 flex-1 rounded-full transition-all duration-300"
                        style={{ background: i <= strengthScore ? strengthColor : 'rgba(200,200,220,0.1)' }} />
                    ))}
                  </div>
                  <span className="text-[10px] w-10 text-right" style={{ color: strengthColor }}>{strengthLabel}</span>
                </div>
              )}

              {form.password && (
                <ul className="mt-2 space-y-1">
                  {[
                    [checks.length,    'At least 8 characters'],
                    [checks.uppercase, 'One uppercase letter (A–Z)'],
                    [checks.number,    'One number (0–9)'],
                    [checks.special,   'One special character (!@#$…)'],
                  ].map(([ok, label]) => (
                    <li key={label} className="flex items-center gap-1.5 text-[11px]"
                      style={{ color: ok ? '#22c55e' : '#9ca3af' }}>
                      <CheckIcon ok={ok} /> {label}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(200,200,220,0.6)' }}>Confirm Password</label>
              <div className="relative">
                <input type={showConfirm ? 'text' : 'password'} value={form.confirm} onChange={set('confirm')} required
                  className="input-dark w-full px-4 py-3 pr-11 rounded-xl text-sm" placeholder="••••••••" />
                <button type="button" onClick={() => setShowConfirm(v => !v)} tabIndex={-1}
                  className="absolute right-3 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-90 transition-opacity"
                  style={{ color: 'var(--text-dim)' }}>
                  <EyeIcon open={showConfirm} />
                </button>
              </div>
              {form.confirm && form.confirm !== form.password && (
                <p className="text-[11px] mt-1" style={{ color: '#f87171' }}>Passwords do not match</p>
              )}
            </div>

            {/* Optional admin invite code */}
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(200,200,220,0.4)' }}>
                Admin Invite Code <span style={{ color: 'rgba(200,200,220,0.25)' }}>(optional)</span>
              </label>
              <input value={form.admin_code} onChange={set('admin_code')}
                className="input-dark w-full px-4 py-3 rounded-xl text-sm" placeholder="Leave blank for standard account" />
            </div>

            {/* FR-M1.4: Terms & Privacy checkbox */}
            <label className="flex items-start gap-3 cursor-pointer group">
              <div className="relative flex-shrink-0 mt-0.5">
                <input type="checkbox" className="sr-only" checked={termsAccepted} onChange={e => setTerms(e.target.checked)} />
                <div className="w-4 h-4 rounded flex items-center justify-center transition-all"
                  style={{
                    background: termsAccepted ? 'var(--gold)' : 'transparent',
                    border: `1.5px solid ${termsAccepted ? 'var(--gold)' : 'rgba(200,200,220,0.25)'}`,
                  }}>
                  {termsAccepted && (
                    <svg width="10" height="10" fill="none" viewBox="0 0 24 24" stroke="#0a0a0f" strokeWidth="3.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
                    </svg>
                  )}
                </div>
              </div>
              <span className="text-[11px] leading-relaxed" style={{ color: 'rgba(200,200,220,0.5)' }}>
                I agree to the{' '}
                <span style={{ color: 'var(--gold)' }} className="hover:underline cursor-pointer">Terms of Use</span>
                {' '}and{' '}
                <span style={{ color: 'var(--gold)' }} className="hover:underline cursor-pointer">Privacy Policy</span>.
                I understand that SecuraQ++ is for authorised security research only.
              </span>
            </label>

            {error && <div className="text-xs text-red-400 bg-red-900/20 border border-red-500/20 rounded-lg px-3 py-2">{error}</div>}

            <button type="submit" disabled={loading || !isStrong || !termsAccepted}
              className="btn-gold w-full py-3 rounded-xl text-sm font-semibold mt-2 disabled:opacity-40 disabled:cursor-not-allowed">
              {loading ? 'Creating account…' : 'Create Account'}
            </button>
            <p className="text-center text-xs" style={{ color: 'rgba(200,200,220,0.35)' }}>
              Already have an account?{' '}
              <Link to="/login" style={{ color: 'var(--gold)' }} className="hover:underline">Sign in</Link>
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}

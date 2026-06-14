import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { api } from '../services/api'
import { jwtDecode } from 'jwt-decode'

const PURPLE = '#8b5cf6'
const PURPLE_DIM = '#6d28d9'
const PURPLE_BG  = 'rgba(139,92,246,0.10)'
const PURPLE_BORDER = 'rgba(139,92,246,0.30)'

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

export default function AdminLoginPage() {
  const [step, setStep]         = useState('login')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [showPass, setShowPass] = useState(false)
  const [code, setCode]         = useState('')
  const [tempToken, setTempToken] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [fpEmail, setFpEmail]   = useState('')
  const [fpToken, setFpToken]   = useState('')
  const [fpCode, setFpCode]     = useState('')
  const [fpNew, setFpNew]       = useState('')
  const [fpSent, setFpSent]     = useState(false)

  const { login, user, isAdmin } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (user) navigate(isAdmin ? '/admin' : '/dashboard', { replace: true })
  }, [user, isAdmin, navigate])

  async function handleLogin(e) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const { data } = await api.login({ email, password, remember_me: false })
      if (data.requires2FA) {
        setTempToken(data.tempToken)
        setStep('2fa')
      } else {
        const decoded = jwtDecode(data.token)
        if (decoded.role !== 'admin') {
          setError('This portal is for administrators only. Please use the user login page.')
          return
        }
        login(data.token)
        navigate('/admin')
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Login failed')
    } finally { setLoading(false) }
  }

  async function handle2FA(e) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const { data } = await api.login2fa({ tempToken, code })
      const decoded = jwtDecode(data.token)
      if (decoded.role !== 'admin') {
        setError('This portal is for administrators only.')
        return
      }
      login(data.token)
      navigate('/admin')
    } catch (err) {
      setError(err.response?.data?.error || 'Invalid code')
    } finally { setLoading(false) }
  }

  async function handleForgotSend(e) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const { data } = await api.forgotPassword(fpEmail)
      setFpToken(data.tempToken)
      setFpSent(true)
    } catch (err) {
      setError(err.response?.data?.error || 'Could not send reset code')
    } finally { setLoading(false) }
  }

  async function handleForgotReset(e) {
    e.preventDefault()
    if (fpNew.length < 8) return setError('Password must be at least 8 characters')
    setError(''); setLoading(true)
    try {
      await api.resetPassword({ tempToken: fpToken, code: fpCode, new_password: fpNew })
      setStep('login')
      setFpEmail(''); setFpCode(''); setFpNew(''); setFpToken(''); setFpSent(false)
    } catch (err) {
      setError(err.response?.data?.error || 'Reset failed')
    } finally { setLoading(false) }
  }

  const inputStyle = {
    background: 'rgba(15,10,30,0.7)',
    border: `1px solid ${PURPLE_BORDER}`,
    borderRadius: 10,
    color: '#f0f0f8',
    padding: '10px 14px',
    width: '100%',
    fontSize: 13,
    outline: 'none',
  }

  const btnStyle = {
    background: `linear-gradient(135deg, ${PURPLE}, ${PURPLE_DIM})`,
    border: 'none',
    borderRadius: 10,
    color: '#fff',
    width: '100%',
    padding: '11px 0',
    fontWeight: 600,
    fontSize: 14,
    cursor: 'pointer',
    opacity: loading ? 0.7 : 1,
  }

  const ghostBtnStyle = {
    background: 'rgba(255,255,255,0.03)',
    border: `1px solid ${PURPLE_BORDER}`,
    borderRadius: 10,
    color: PURPLE,
    width: '100%',
    padding: '10px 0',
    fontWeight: 500,
    fontSize: 13,
    cursor: 'pointer',
    marginTop: 8,
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4"
      style={{ background: `radial-gradient(ellipse at 30% 40%, rgba(139,92,246,.06) 0%, transparent 60%), var(--surface-0)` }}>

      <div className="fixed inset-0 pointer-events-none opacity-[0.025]"
        style={{ backgroundImage: `linear-gradient(${PURPLE} 1px, transparent 1px), linear-gradient(90deg, ${PURPLE} 1px, transparent 1px)`,
          backgroundSize: '40px 40px' }} />

      <div className="w-full max-w-sm animate-fadeUp">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-4"
            style={{ background: `linear-gradient(135deg, ${PURPLE}, ${PURPLE_DIM})`, boxShadow: `0 8px 32px rgba(139,92,246,.30)` }}>
            🛡️
          </div>
          <h1 className="font-display font-bold text-2xl" style={{ color: PURPLE }}>Administrator Portal</h1>
          <p className="text-xs mt-1" style={{ color: 'var(--text-dim)' }}>SecuraQ++ Platform Administration</p>
        </div>

        <div className="glass rounded-2xl p-8" style={{ border: `1px solid ${PURPLE_BORDER}` }}>

          {/* ── Sign In ── */}
          {step === 'login' && (
            <>
              <h2 className="font-display font-semibold text-lg mb-6" style={{ color: '#f0f0f8' }}>Admin Sign In</h2>
              <form onSubmit={handleLogin} className="space-y-4">
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>Admin Email</label>
                  <input type="email" value={email}
                    onChange={e => { setEmail(e.target.value); setError('') }}
                    style={inputStyle} placeholder="admin@securaqpp.local" required autoFocus />
                </div>
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>Password</label>
                  <div className="relative">
                    <input type={showPass ? 'text' : 'password'} value={password}
                      onChange={e => { setPassword(e.target.value); setError('') }}
                      style={{ ...inputStyle, paddingRight: 40 }}
                      placeholder="••••••••" required />
                    <button type="button" onClick={() => setShowPass(v => !v)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-90 transition-opacity"
                      style={{ color: 'var(--text-dim)' }} tabIndex={-1}>
                      <EyeIcon open={showPass} />
                    </button>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button type="button" onClick={() => { setStep('forgot'); setError('') }}
                    className="text-[11px] hover:underline transition-all" style={{ color: PURPLE }}>
                    Forgot password?
                  </button>
                </div>

                {error && (
                  <div className="text-xs rounded-lg px-3 py-2" style={{ color: '#fca5a5', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                    {error}
                  </div>
                )}

                <button type="submit" disabled={loading} style={btnStyle}>
                  {loading ? 'Signing in…' : 'Sign In as Administrator'}
                </button>
              </form>

              <div className="mt-5 pt-4" style={{ borderTop: `1px solid ${PURPLE_BORDER}` }}>
                <p className="text-center text-xs" style={{ color: 'var(--text-dim)' }}>
                  Not an admin?{' '}
                  <a href="/login" style={{ color: PURPLE }} className="hover:underline">User login →</a>
                </p>
              </div>
            </>
          )}

          {/* ── 2FA OTP ── */}
          {step === '2fa' && (
            <>
              <div className="flex items-center justify-center w-12 h-12 rounded-xl mb-4 mx-auto"
                style={{ background: PURPLE_BG, border: `1px solid ${PURPLE_BORDER}` }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5" style={{ color: PURPLE }}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25H4.5a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75"/>
                </svg>
              </div>
              <h2 className="font-display font-semibold text-lg mb-1 text-center" style={{ color: '#f0f0f8' }}>Verify Identity</h2>
              <p className="text-xs mb-1 text-center" style={{ color: 'var(--text-dim)' }}>Admin OTP sent to</p>
              <p className="text-xs mb-6 text-center font-mono" style={{ color: PURPLE }}>{email}</p>
              <form onSubmit={handle2FA} className="space-y-4">
                <input type="tel" inputMode="numeric" pattern="[0-9]{6}" maxLength={6}
                  value={code} onChange={e => setCode(e.target.value.replace(/\D/g, ''))}
                  style={{ ...inputStyle, textAlign: 'center', fontSize: 24, fontFamily: 'monospace', letterSpacing: '0.5em' }}
                  placeholder="000000" autoComplete="one-time-code" autoFocus required />
                {error && (
                  <div className="text-xs rounded-lg px-3 py-2" style={{ color: '#fca5a5', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                    {error}
                  </div>
                )}
                <button type="submit" disabled={loading || code.length !== 6} style={btnStyle}>
                  {loading ? 'Verifying…' : 'Verify OTP'}
                </button>
                <button type="button" onClick={() => { setStep('login'); setCode(''); setError('') }} style={ghostBtnStyle}>
                  ← Back to sign in
                </button>
              </form>
            </>
          )}

          {/* ── Forgot — send code ── */}
          {step === 'forgot' && !fpSent && (
            <>
              <h2 className="font-display font-semibold text-lg mb-2" style={{ color: '#f0f0f8' }}>Reset Password</h2>
              <p className="text-xs mb-5" style={{ color: 'var(--text-dim)' }}>
                Enter your admin email and we'll send a one-time reset code.
              </p>
              <form onSubmit={handleForgotSend} className="space-y-4">
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>Admin Email</label>
                  <input type="email" value={fpEmail} onChange={e => setFpEmail(e.target.value)}
                    style={inputStyle} placeholder="admin@securaqpp.local" required autoFocus />
                </div>
                {error && (
                  <div className="text-xs rounded-lg px-3 py-2" style={{ color: '#fca5a5', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                    {error}
                  </div>
                )}
                <button type="submit" disabled={loading} style={btnStyle}>
                  {loading ? 'Sending…' : 'Send Reset Code'}
                </button>
                <button type="button" onClick={() => { setStep('login'); setError('') }} style={ghostBtnStyle}>
                  ← Back to sign in
                </button>
              </form>
            </>
          )}

          {/* ── Forgot — verify + new password ── */}
          {step === 'forgot' && fpSent && (
            <>
              <div className="flex items-center justify-center w-12 h-12 rounded-xl mb-4 mx-auto"
                style={{ background: PURPLE_BG, border: `1px solid ${PURPLE_BORDER}` }}>
                <span style={{ color: PURPLE, fontSize: 22 }}>🔑</span>
              </div>
              <h2 className="font-display font-semibold text-lg mb-1 text-center" style={{ color: '#f0f0f8' }}>Set New Password</h2>
              <p className="text-xs mb-6 text-center" style={{ color: 'var(--text-dim)' }}>
                Code sent to <span className="font-mono" style={{ color: PURPLE }}>{fpEmail}</span>
              </p>
              <form onSubmit={handleForgotReset} className="space-y-4">
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>6-digit code</label>
                  <input type="tel" inputMode="numeric" maxLength={6}
                    value={fpCode} onChange={e => setFpCode(e.target.value.replace(/\D/g, ''))}
                    style={{ ...inputStyle, textAlign: 'center', fontSize: 24, fontFamily: 'monospace', letterSpacing: '0.5em' }}
                    placeholder="000000" autoFocus required />
                </div>
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>New Password</label>
                  <input type="password" value={fpNew} onChange={e => setFpNew(e.target.value)}
                    style={inputStyle} placeholder="Min. 8 characters" required />
                </div>
                {error && (
                  <div className="text-xs rounded-lg px-3 py-2" style={{ color: '#fca5a5', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
                    {error}
                  </div>
                )}
                <button type="submit" disabled={loading || fpCode.length !== 6} style={btnStyle}>
                  {loading ? 'Resetting…' : 'Reset Password'}
                </button>
                <button type="button" onClick={() => { setFpSent(false); setError('') }} style={ghostBtnStyle}>
                  ← Resend code
                </button>
              </form>
            </>
          )}
        </div>

        <p className="text-center text-[10px] mt-4" style={{ color: 'rgba(200,200,220,0.2)' }}>
          SecuraQ++ · Administrator Access Only
        </p>
      </div>
    </div>
  )
}

import React, { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { api } from '../services/api'
import { jwtDecode } from 'jwt-decode'

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

export default function LoginPage() {
  const [step, setStep]             = useState('login')   // login | 2fa | forgot | forgot_verify
  const [email, setEmail]           = useState('')
  const [password, setPassword]     = useState('')
  const [showPass, setShowPass]     = useState(false)
  const [rememberMe, setRememberMe] = useState(false)
  const [code, setCode]             = useState('')
  const [tempToken, setTempToken]   = useState('')
  const [error, setError]           = useState('')
  const [fieldError, setFieldError] = useState('')
  const [loading, setLoading]       = useState(false)
  // forgot-password state
  const [fpEmail, setFpEmail]       = useState('')
  const [fpToken, setFpToken]       = useState('')
  const [fpCode, setFpCode]         = useState('')
  const [fpNew, setFpNew]           = useState('')
  const [fpSent, setFpSent]         = useState(false)
  const { login, user, isAdmin } = useAuth()
  const navigate  = useNavigate()

  useEffect(() => {
    if (user) navigate(isAdmin ? '/admin' : '/dashboard', { replace: true })
  }, [user, isAdmin, navigate])

  async function handleLogin(e) {
    e.preventDefault()
    setError(''); setFieldError(''); setLoading(true)
    try {
      const { data } = await api.login({ email, password, remember_me: rememberMe })
      if (data.requires2FA) {
        setTempToken(data.tempToken)
        setStep('2fa')
      } else {
        const decoded = jwtDecode(data.token)
        login(data.token)
        navigate(decoded.role === 'admin' ? '/admin' : '/dashboard')
      }
    } catch (err) {
      const msg   = err.response?.data?.error || 'Login failed'
      const field = err.response?.data?.field || ''
      setError(msg); setFieldError(field)
    } finally { setLoading(false) }
  }

  async function handle2FA(e) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const { data } = await api.login2fa({ tempToken, code })
      const decoded = jwtDecode(data.token)
      login(data.token)
      navigate(decoded.role === 'admin' ? '/admin' : '/dashboard')
    } catch (err) {
      setError(err.response?.data?.error || 'Invalid code')
    } finally { setLoading(false) }
  }

  // FR-M2.5: Forgot password — send OTP
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

  // FR-M2.5: Forgot password — verify OTP + set new password
  async function handleForgotReset(e) {
    e.preventDefault()
    if (fpNew.length < 8) return setError('New password must be at least 8 characters')
    setError(''); setLoading(true)
    try {
      await api.resetPassword({ tempToken: fpToken, code: fpCode, new_password: fpNew })
      setStep('login')
      setError('')
      setFpEmail(''); setFpCode(''); setFpNew(''); setFpToken(''); setFpSent(false)
    } catch (err) {
      setError(err.response?.data?.error || 'Reset failed')
    } finally { setLoading(false) }
  }

  const inputBorder = (field) =>
    fieldError === field ? 'border border-red-500/60 focus:border-red-400' : ''

  return (
    <div className="min-h-screen flex items-center justify-center p-4"
      style={{ background: 'radial-gradient(ellipse at 30% 40%, rgba(200,169,110,.04) 0%, transparent 60%), var(--surface-0)' }}>

      <div className="fixed inset-0 pointer-events-none opacity-[0.03]"
        style={{ backgroundImage: 'linear-gradient(var(--gold) 1px, transparent 1px), linear-gradient(90deg, var(--gold) 1px, transparent 1px)',
          backgroundSize: '40px 40px' }} />

      <div className="w-full max-w-sm animate-fadeUp">
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-4"
            style={{ background: 'linear-gradient(135deg,var(--gold),var(--gold-dim))', boxShadow: '0 8px 32px rgba(200,169,110,.25)' }}>
            ⚛️
          </div>
          <h1 className="font-display font-bold text-2xl text-gold-gradient">SecuraQ++</h1>
          <p className="text-xs mt-1" style={{ color: 'var(--text-dim)' }}>Quantum-Enhanced Vulnerability Detection</p>
        </div>

        <div className="glass rounded-2xl p-8">

          {/* ── Sign In ───────────────────────────────────────────────── */}
          {step === 'login' && (
            <>
              <h2 className="font-display font-semibold text-lg mb-6" style={{ color: '#f0f0f8' }}>Sign In</h2>
              <form onSubmit={handleLogin} className="space-y-4">
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>Email</label>
                  <input
                    type="email" value={email}
                    onChange={e => { setEmail(e.target.value); setFieldError(''); setError('') }}
                    className={`input-dark ${inputBorder('email')}`}
                    placeholder="you@university.edu" required autoFocus />
                  {fieldError === 'email' && <p className="text-xs mt-1" style={{ color: '#f87171' }}>{error}</p>}
                </div>
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>Password</label>
                  <div className="relative">
                    <input
                      type={showPass ? 'text' : 'password'} value={password}
                      onChange={e => { setPassword(e.target.value); setFieldError(''); setError('') }}
                      className={`input-dark pr-10 ${inputBorder('password')}`}
                      placeholder="••••••••" required />
                    <button type="button" onClick={() => setShowPass(v => !v)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-90 transition-opacity"
                      style={{ color: 'var(--text-dim)' }} tabIndex={-1}>
                      <EyeIcon open={showPass} />
                    </button>
                  </div>
                  {fieldError === 'password' && <p className="text-xs mt-1" style={{ color: '#f87171' }}>{error}</p>}
                </div>

                {/* FR-M2.3: Remember Me + Forgot Password row */}
                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <div className="relative">
                      <input type="checkbox" className="sr-only" checked={rememberMe} onChange={e => setRememberMe(e.target.checked)} />
                      <div className="w-4 h-4 rounded flex items-center justify-center transition-all"
                        style={{
                          background: rememberMe ? 'var(--gold)' : 'transparent',
                          border: `1.5px solid ${rememberMe ? 'var(--gold)' : 'rgba(200,200,220,0.25)'}`,
                        }}>
                        {rememberMe && (
                          <svg width="10" height="10" fill="none" viewBox="0 0 24 24" stroke="#0a0a0f" strokeWidth="3.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
                          </svg>
                        )}
                      </div>
                    </div>
                    <span className="text-[11px]" style={{ color: 'rgba(200,200,220,0.5)' }}>Remember me (7 days)</span>
                  </label>
                  <button type="button" onClick={() => { setStep('forgot'); setError('') }}
                    className="text-[11px] hover:underline transition-all" style={{ color: 'var(--gold)' }}>
                    Forgot password?
                  </button>
                </div>

                {error && !fieldError && (
                  <div className="text-xs text-red-400 bg-red-900/20 border border-red-500/20 rounded-lg px-3 py-2">{error}</div>
                )}
                <button type="submit" disabled={loading} className="btn-gold w-full py-2.5">
                  {loading ? 'Signing in…' : 'Sign In'}
                </button>
              </form>
              <p className="text-center text-xs mt-5" style={{ color: 'var(--text-dim)' }}>
                No account?{' '}
                <Link to="/register" style={{ color: 'var(--gold)' }} className="hover:underline">Create one</Link>
              </p>
              <div className="mt-4 pt-4" style={{ borderTop: '1px solid rgba(200,169,110,0.12)' }}>
                <p className="text-center text-[11px]" style={{ color: 'rgba(200,200,220,0.3)' }}>
                  Administrator?{' '}
                  <Link to="/admin/login" style={{ color: 'rgba(139,92,246,0.8)' }} className="hover:underline">
                    Sign in to Admin Portal →
                  </Link>
                </p>
              </div>
            </>
          )}

          {/* ── Email OTP (2FA) ───────────────────────────────────────── */}
          {step === '2fa' && (
            <>
              <div className="flex items-center justify-center w-12 h-12 rounded-xl mb-4 mx-auto"
                style={{ background: 'rgba(200,169,110,0.12)', border: '1px solid rgba(200,169,110,0.25)' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--gold)' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25H4.5a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75"/>
                </svg>
              </div>
              <h2 className="font-display font-semibold text-lg mb-1 text-center" style={{ color: '#f0f0f8' }}>Check your email</h2>
              <p className="text-xs mb-1 text-center" style={{ color: 'var(--text-dim)' }}>We sent a 6-digit verification code to</p>
              <p className="text-xs mb-6 text-center font-mono" style={{ color: 'var(--gold)' }}>{email}</p>
              <form onSubmit={handle2FA} className="space-y-4">
                <input type="tel" inputMode="numeric" pattern="[0-9]{6}" maxLength={6}
                  value={code} onChange={e => setCode(e.target.value.replace(/\D/g,''))}
                  className="input-dark text-center font-mono text-2xl tracking-[.5em]"
                  placeholder="000000" autoComplete="one-time-code" autoFocus required />
                {error && <div className="text-xs text-red-400 bg-red-900/20 border border-red-500/20 rounded-lg px-3 py-2">{error}</div>}
                <button type="submit" disabled={loading || code.length !== 6} className="btn-gold w-full py-2.5">
                  {loading ? 'Verifying…' : 'Verify'}
                </button>
                <button type="button" onClick={() => { setStep('login'); setCode(''); setError('') }} className="btn-ghost w-full">
                  ← Back to sign in
                </button>
              </form>
            </>
          )}

          {/* ── Forgot Password — send code ───────────────────────────── */}
          {step === 'forgot' && !fpSent && (
            <>
              <h2 className="font-display font-semibold text-lg mb-2" style={{ color: '#f0f0f8' }}>Reset Password</h2>
              <p className="text-xs mb-5" style={{ color: 'var(--text-dim)' }}>
                Enter your account email and we'll send a one-time code to reset your password.
              </p>
              <form onSubmit={handleForgotSend} className="space-y-4">
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>Email</label>
                  <input type="email" value={fpEmail} onChange={e => setFpEmail(e.target.value)}
                    className="input-dark" placeholder="you@university.edu" required autoFocus />
                </div>
                {error && <div className="text-xs text-red-400 bg-red-900/20 border border-red-500/20 rounded-lg px-3 py-2">{error}</div>}
                <button type="submit" disabled={loading} className="btn-gold w-full py-2.5">
                  {loading ? 'Sending…' : 'Send Reset Code'}
                </button>
                <button type="button" onClick={() => { setStep('login'); setError('') }} className="btn-ghost w-full">
                  ← Back to sign in
                </button>
              </form>
            </>
          )}

          {/* ── Forgot Password — enter code + new password ───────────── */}
          {step === 'forgot' && fpSent && (
            <>
              <div className="flex items-center justify-center w-12 h-12 rounded-xl mb-4 mx-auto"
                style={{ background: 'rgba(200,169,110,0.12)', border: '1px solid rgba(200,169,110,0.25)' }}>
                <span style={{ color: 'var(--gold)', fontSize: 22 }}>🔑</span>
              </div>
              <h2 className="font-display font-semibold text-lg mb-1 text-center" style={{ color: '#f0f0f8' }}>New Password</h2>
              <p className="text-xs mb-6 text-center" style={{ color: 'var(--text-dim)' }}>
                Code sent to <span className="font-mono" style={{ color: 'var(--gold)' }}>{fpEmail}</span>
              </p>
              <form onSubmit={handleForgotReset} className="space-y-4">
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>6-digit code</label>
                  <input type="tel" inputMode="numeric" maxLength={6}
                    value={fpCode} onChange={e => setFpCode(e.target.value.replace(/\D/g,''))}
                    className="input-dark text-center font-mono text-2xl tracking-[.5em]"
                    placeholder="000000" autoFocus required />
                </div>
                <div>
                  <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-dim)' }}>New Password</label>
                  <input type="password" value={fpNew} onChange={e => setFpNew(e.target.value)}
                    className="input-dark" placeholder="Min. 8 characters" required />
                </div>
                {error && <div className="text-xs text-red-400 bg-red-900/20 border border-red-500/20 rounded-lg px-3 py-2">{error}</div>}
                <button type="submit" disabled={loading || fpCode.length !== 6} className="btn-gold w-full py-2.5">
                  {loading ? 'Resetting…' : 'Reset Password'}
                </button>
                <button type="button" onClick={() => { setFpSent(false); setError('') }} className="btn-ghost w-full">
                  ← Resend code
                </button>
              </form>
            </>
          )}

        </div>
      </div>
    </div>
  )
}

import React, { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { api } from '../services/api'

const Section = ({ title, children }) => (
  <div className="glass rounded-2xl p-6">
    <h3 className="font-display font-semibold text-sm mb-5" style={{ color: '#f0f0f8', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '1rem' }}>
      {title}
    </h3>
    {children}
  </div>
)

const Field = ({ label, children }) => (
  <div>
    <label className="block text-xs font-medium mb-1.5" style={{ color: 'rgba(200,200,220,0.6)' }}>{label}</label>
    {children}
  </div>
)

// FR-M10.3: Dark mode toggle — persisted in localStorage and applied via CSS class
function useDarkMode() {
  const [dark, setDark] = useState(() => localStorage.getItem('sqpp_theme') !== 'light')
  const toggle = () => setDark(d => {
    const next = !d
    localStorage.setItem('sqpp_theme', next ? 'dark' : 'light')
    document.documentElement.classList.toggle('theme-light', !next)
    return next
  })
  return [dark, toggle]
}

// FR-M10.4: Email notifications switch — persisted in localStorage
function useEmailNotifs() {
  const [on, setOn] = useState(() => localStorage.getItem('sqpp_email_notifs') !== 'false')
  const toggle = () => setOn(v => {
    const next = !v
    localStorage.setItem('sqpp_email_notifs', String(next))
    return next
  })
  return [on, toggle]
}

export default function ProfilePage() {
  const { user, login } = useAuth()
  const [profile, setProfile]     = useState({ full_name: '' })
  const [pwForm, setPwForm]       = useState({ current: '', next: '', confirm: '' })
  const [twoFA, setTwoFA]         = useState({ enabled: false, qr: '', secret: '', code: '', disablePw: '' })
  const [msgs, setMsgs]           = useState({})
  const [loading, setLoading]     = useState({})
  const [darkMode, toggleDark]    = useDarkMode()
  const [emailNotifs, toggleNotifs] = useEmailNotifs()

  const setMsg = (k, v) => setMsgs(m => ({ ...m, [k]: v }))
  const setBusy = (k, v) => setLoading(l => ({ ...l, [k]: v }))

  useEffect(() => {
    api.me().then(r => {
      const u = r.data
      setProfile({ full_name: u.full_name || '' })
      setTwoFA(t => ({ ...t, enabled: !!u.totp_enabled }))
    }).catch(() => {})
  }, [])

  const saveProfile = async () => {
    setBusy('profile', true); setMsg('profile', '')
    try {
      await api.updateProfile({ full_name: profile.full_name })
      setMsg('profile', { ok: 'Profile updated' })
    } catch (e) {
      setMsg('profile', { err: e.response?.data?.error || 'Failed' })
    } finally { setBusy('profile', false) }
  }

  const savePassword = async () => {
    setMsg('pw', '')
    if (pwForm.next !== pwForm.confirm) return setMsg('pw', { err: 'Passwords do not match' })
    if (pwForm.next.length < 8) return setMsg('pw', { err: 'Min. 8 characters' })
    setBusy('pw', true)
    try {
      await api.updatePassword({ current_password: pwForm.current, new_password: pwForm.next })
      setMsg('pw', { ok: 'Password updated' })
      setPwForm({ current: '', next: '', confirm: '' })
    } catch (e) {
      setMsg('pw', { err: e.response?.data?.error || 'Failed' })
    } finally { setBusy('pw', false) }
  }

  const setup2fa = async () => {
    setBusy('2fa', true); setMsg('2fa', '')
    try {
      const { data } = await api.setup2fa()
      setTwoFA(t => ({ ...t, qr: data.qrCode, secret: data.secret }))
    } catch { setMsg('2fa', { err: 'Setup failed' }) }
    finally { setBusy('2fa', false) }
  }

  const verify2fa = async () => {
    setBusy('2fa_verify', true); setMsg('2fa', '')
    try {
      await api.verify2fa(twoFA.code)
      setTwoFA(t => ({ ...t, enabled: true, qr: '', code: '' }))
      setMsg('2fa', { ok: '2FA enabled!' })
    } catch (e) {
      setMsg('2fa', { err: e.response?.data?.error || 'Invalid code' })
    } finally { setBusy('2fa_verify', false) }
  }

  const disable2fa = async () => {
    setBusy('2fa_dis', true); setMsg('2fa', '')
    try {
      await api.disable2fa(twoFA.disablePw)
      setTwoFA(t => ({ ...t, enabled: false, disablePw: '' }))
      setMsg('2fa', { ok: '2FA disabled' })
    } catch (e) {
      setMsg('2fa', { err: e.response?.data?.error || 'Failed' })
    } finally { setBusy('2fa_dis', false) }
  }

  const Msg = ({ k }) => {
    const m = msgs[k]
    if (!m) return null
    return (
      <div className={`text-xs rounded-lg px-3 py-2 mt-2 ${m.ok ? 'text-green-400 bg-green-900/20 border border-green-500/20' : 'text-red-400 bg-red-900/20 border border-red-500/20'}`}>
        {m.ok || m.err}
      </div>
    )
  }

  return (
    <div className="space-y-5 animate-fadeUp max-w-2xl">
      <div>
        <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>Profile & Security</h2>
        <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
          Manage your account, password, and two-factor authentication
        </p>
      </div>

      {/* Account info */}
      <Section title="Account Information">
        <div className="flex items-center gap-4 mb-5">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-xl font-bold flex-shrink-0"
            style={{ background: 'linear-gradient(135deg, var(--gold), var(--gold-dim))', color: '#0a0a0f' }}>
            {user?.full_name?.[0]?.toUpperCase()}
          </div>
          <div>
            <div className="font-medium" style={{ color: '#e8e8f0' }}>{user?.full_name}</div>
            <div className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.4)' }}>{user?.email}</div>
            <span className="text-[10px] px-2 py-0.5 rounded mt-1 inline-block font-mono"
              style={{ background: user?.role === 'admin' ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)',
                color: user?.role === 'admin' ? '#fca5a5' : '#86efac' }}>
              {user?.role}
            </span>
          </div>
        </div>
        <div className="space-y-3">
          <Field label="Full Name">
            <input value={profile.full_name} onChange={e => setProfile(p => ({ ...p, full_name: e.target.value }))}
              className="input-dark w-full px-4 py-2.5 rounded-xl text-sm" />
          </Field>
          <Field label="Email">
            <input value={user?.email || ''} disabled
              className="input-dark w-full px-4 py-2.5 rounded-xl text-sm opacity-50 cursor-not-allowed" />
          </Field>
        </div>
        <Msg k="profile" />
        <button onClick={saveProfile} disabled={loading.profile} className="btn-gold mt-4 px-5 py-2 rounded-lg text-sm">
          {loading.profile ? 'Saving…' : 'Save Changes'}
        </button>
      </Section>

      {/* Password */}
      <Section title="Change Password">
        <div className="space-y-3">
          <Field label="Current Password">
            <input type="password" value={pwForm.current} onChange={e => setPwForm(p => ({ ...p, current: e.target.value }))}
              className="input-dark w-full px-4 py-2.5 rounded-xl text-sm" placeholder="••••••••" />
          </Field>
          <Field label="New Password">
            <input type="password" value={pwForm.next} onChange={e => setPwForm(p => ({ ...p, next: e.target.value }))}
              className="input-dark w-full px-4 py-2.5 rounded-xl text-sm" placeholder="Min. 8 characters" />
          </Field>
          <Field label="Confirm New Password">
            <input type="password" value={pwForm.confirm} onChange={e => setPwForm(p => ({ ...p, confirm: e.target.value }))}
              className="input-dark w-full px-4 py-2.5 rounded-xl text-sm" placeholder="••••••••" />
          </Field>
        </div>
        <Msg k="pw" />
        <button onClick={savePassword} disabled={loading.pw} className="btn-gold mt-4 px-5 py-2 rounded-lg text-sm">
          {loading.pw ? 'Updating…' : 'Update Password'}
        </button>
      </Section>

      {/* FR-M10.3 + FR-M10.4: Preferences */}
      <Section title="Preferences">
        <div className="space-y-4">

          {/* Dark mode toggle */}
          <div className="flex items-center justify-between py-3 border-b"
            style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
            <div>
              <p className="text-sm" style={{ color: '#e8e8f0' }}>Dark Mode</p>
              <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>
                Toggle between dark and light interface theme
              </p>
            </div>
            <button onClick={toggleDark}
              className="relative flex-shrink-0 w-11 h-6 rounded-full transition-all duration-300"
              style={{ background: darkMode ? 'var(--gold)' : 'rgba(200,200,220,0.15)' }}>
              <div className="absolute top-0.5 w-5 h-5 rounded-full transition-all duration-300"
                style={{
                  background: '#0a0a0f',
                  left: darkMode ? 'calc(100% - 22px)' : '2px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
                }} />
            </button>
          </div>

          {/* Email notifications toggle */}
          <div className="flex items-center justify-between py-3">
            <div>
              <p className="text-sm" style={{ color: '#e8e8f0' }}>Email Notifications</p>
              <p className="text-xs mt-0.5" style={{ color: 'rgba(200,200,220,0.35)' }}>
                Receive scan completion and alert emails
              </p>
            </div>
            <button onClick={toggleNotifs}
              className="relative flex-shrink-0 w-11 h-6 rounded-full transition-all duration-300"
              style={{ background: emailNotifs ? 'var(--gold)' : 'rgba(200,200,220,0.15)' }}>
              <div className="absolute top-0.5 w-5 h-5 rounded-full transition-all duration-300"
                style={{
                  background: '#0a0a0f',
                  left: emailNotifs ? 'calc(100% - 22px)' : '2px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
                }} />
            </button>
          </div>

        </div>
      </Section>

      {/* 2FA */}
      <Section title="Two-Factor Authentication (TOTP)">
        <div className="flex items-center gap-3 mb-5">
          <div className={`w-2.5 h-2.5 rounded-full ${twoFA.enabled ? 'bg-green-400' : 'bg-gray-600'}`} />
          <span className="text-sm" style={{ color: twoFA.enabled ? '#86efac' : 'rgba(200,200,220,0.5)' }}>
            {twoFA.enabled ? '2FA is enabled' : '2FA is disabled'}
          </span>
        </div>

        {!twoFA.enabled && (
          <div className="space-y-4">
            {!twoFA.qr ? (
              <button onClick={setup2fa} disabled={loading['2fa']} className="btn-gold px-5 py-2 rounded-lg text-sm">
                {loading['2fa'] ? 'Generating…' : '⚛ Set Up 2FA'}
              </button>
            ) : (
              <>
                <p className="text-xs" style={{ color: 'rgba(200,200,220,0.5)' }}>
                  Scan this QR code with Google Authenticator, Authy, or any TOTP app.
                </p>
                <div className="flex justify-center">
                  <div className="p-3 rounded-xl" style={{ background: 'white' }}>
                    <img src={twoFA.qr} alt="QR Code" className="w-40 h-40" />
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-xs mb-1" style={{ color: 'rgba(200,200,220,0.4)' }}>Manual entry key:</p>
                  <code className="text-xs font-mono px-3 py-1.5 rounded"
                    style={{ background: 'rgba(200,169,110,0.08)', color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.2)' }}>
                    {twoFA.secret}
                  </code>
                </div>
                <Field label="Enter 6-digit code to confirm">
                  <input type="tel" inputMode="numeric" maxLength={6}
                    value={twoFA.code} onChange={e => setTwoFA(t => ({ ...t, code: e.target.value.replace(/\D/g,'') }))}
                    autoComplete="one-time-code"
                    className="input-dark w-full px-4 py-2.5 rounded-xl text-xl text-center tracking-widest font-mono"
                    placeholder="000000" />
                </Field>
                <button onClick={verify2fa} disabled={loading['2fa_verify'] || twoFA.code.length !== 6}
                  className="btn-gold px-5 py-2 rounded-lg text-sm">
                  {loading['2fa_verify'] ? 'Verifying…' : 'Enable 2FA'}
                </button>
              </>
            )}
          </div>
        )}

        {twoFA.enabled && (
          <div className="space-y-3">
            <p className="text-xs" style={{ color: 'rgba(200,200,220,0.45)' }}>
              To disable 2FA, enter your account password below.
            </p>
            <Field label="Account Password">
              <input type="password" value={twoFA.disablePw}
                onChange={e => setTwoFA(t => ({ ...t, disablePw: e.target.value }))}
                className="input-dark w-full px-4 py-2.5 rounded-xl text-sm" placeholder="••••••••" />
            </Field>
            <button onClick={disable2fa} disabled={loading['2fa_dis']}
              className="px-5 py-2 rounded-lg text-sm transition-all"
              style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
              {loading['2fa_dis'] ? 'Disabling…' : 'Disable 2FA'}
            </button>
          </div>
        )}
        <Msg k="2fa" />
      </Section>
    </div>
  )
}

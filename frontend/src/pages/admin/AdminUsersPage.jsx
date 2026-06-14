import React, { useState, useEffect } from 'react'
import { useAuth } from '../../context/AuthContext'
import { api } from '../../services/api'

const ROLES = ['analyst', 'admin']

function Modal({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md glass rounded-2xl p-6 animate-fadeUp">
        <div className="flex items-center justify-between mb-5">
          <h3 className="font-display font-semibold text-sm" style={{ color: '#f0f0f8' }}>{title}</h3>
          <button onClick={onClose} className="text-xs" style={{ color: 'rgba(200,200,220,0.4)' }}>✕</button>
        </div>
        {children}
      </div>
    </div>
  )
}

export default function AdminUsersPage() {
  const { user: me } = useAuth()
  const [users, setUsers]     = useState([])
  const [loading, setLoading] = useState(true)
  const [msg, setMsg]         = useState('')
  const [search, setSearch]   = useState('')

  // Modals
  const [createOpen, setCreateOpen]   = useState(false)
  const [editUser, setEditUser]       = useState(null)
  const [resetUser, setResetUser]     = useState(null)
  const [deleteUser, setDeleteUser]   = useState(null)

  // Forms
  const [createForm, setCreateForm]   = useState({ full_name: '', email: '', password: '', role: 'analyst' })
  const [editForm, setEditForm]       = useState({})
  const [newPassword, setNewPassword] = useState('')
  const [formErr, setFormErr]         = useState('')

  const load = () => {
    setLoading(true)
    api.adminUsers()
      .then(r => setUsers(r.data || []))
      .catch(() => setMsg('Failed to load users'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const filtered = users.filter(u =>
    u.full_name?.toLowerCase().includes(search.toLowerCase()) ||
    u.email?.toLowerCase().includes(search.toLowerCase())
  )

  const createUser = async () => {
    setFormErr('')
    if (!createForm.full_name || !createForm.email || !createForm.password)
      return setFormErr('All fields required')
    try {
      await api.adminCreateUser(createForm)
      setMsg('User created'); setCreateOpen(false)
      setCreateForm({ full_name: '', email: '', password: '', role: 'analyst' })
      load()
    } catch (e) { setFormErr(e.response?.data?.error || 'Failed') }
  }

  const updateUser = async () => {
    try {
      await api.adminUpdateUser(editUser.id, editForm)
      setMsg('User updated'); setEditUser(null); load()
    } catch (e) { setFormErr(e.response?.data?.error || 'Failed') }
  }

  const resetPassword = async () => {
    if (!newPassword || newPassword.length < 8) return setFormErr('Min 8 characters')
    try {
      await api.adminResetPassword(resetUser.id, newPassword)
      setMsg('Password reset'); setResetUser(null); setNewPassword('')
    } catch (e) { setFormErr(e.response?.data?.error || 'Failed') }
  }

  const toggleActive = async (u) => {
    await api.adminUpdateUser(u.id, { is_active: u.is_active ? 0 : 1 }).catch(() => {})
    load()
  }

  const confirmDelete = async () => {
    try {
      await api.adminDeleteUser(deleteUser.id)
      setMsg('User deleted'); setDeleteUser(null); load()
    } catch (e) { setMsg(e.response?.data?.error || 'Failed') }
  }

  const openEdit = (u) => { setEditForm({ full_name: u.full_name, role: u.role }); setEditUser(u); setFormErr('') }
  const openReset = (u) => { setResetUser(u); setNewPassword(''); setFormErr('') }

  return (
    <div className="space-y-5 animate-fadeUp">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="font-display font-semibold text-xl" style={{ color: '#f0f0f8' }}>User Management</h2>
          <p className="text-xs mt-1" style={{ color: 'rgba(200,200,220,0.4)' }}>
            {users.length} users · Create, edit, disable, or delete accounts
          </p>
        </div>
        <button onClick={() => { setCreateOpen(true); setFormErr('') }} className="btn-gold px-4 py-2 rounded-lg text-xs font-semibold">
          + New User
        </button>
      </div>

      {msg && (
        <div className="text-xs rounded-lg px-3 py-2"
          style={{ background: 'rgba(200,169,110,0.08)', border: '1px solid rgba(200,169,110,0.2)', color: 'var(--gold)' }}>
          {msg}
        </div>
      )}

      {/* Search */}
      <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by name or email…"
        className="input-dark w-full max-w-sm px-4 py-2.5 rounded-xl text-sm" />

      {/* Table */}
      <div className="glass rounded-2xl overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-xs" style={{ color: 'rgba(200,200,220,0.3)' }}>Loading…</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(255,255,255,0.02)' }}>
                  {['User', 'Email', 'Role', 'Status', '2FA', 'Created', 'Actions'].map(h => (
                    <th key={h} className="text-left px-4 py-3 font-medium"
                      style={{ color: 'rgba(200,200,220,0.45)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((u, i) => (
                  <tr key={u.id} className="border-b transition-all"
                    style={{ borderColor: 'rgba(255,255,255,0.04)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                          style={{ background: 'linear-gradient(135deg, var(--gold), var(--gold-dim))', color: '#0a0a0f' }}>
                          {u.full_name?.[0]?.toUpperCase()}
                        </div>
                        <span style={{ color: '#e8e8f0' }}>
                          {u.full_name}
                          {u.id === me?.id && <span className="ml-1 text-[9px]" style={{ color: 'var(--gold-dim)' }}>(you)</span>}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono" style={{ color: 'rgba(200,200,220,0.55)' }}>{u.email}</td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded text-[10px]"
                        style={{ background: u.role === 'admin' ? 'rgba(239,68,68,0.12)' : 'rgba(34,197,94,0.1)',
                          color: u.role === 'admin' ? '#fca5a5' : '#86efac' }}>
                        {u.role}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] ${u.is_active ? 'text-green-400' : 'text-red-400'}`}
                        style={{ background: u.is_active ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)' }}>
                        {u.is_active ? 'Active' : 'Disabled'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span style={{ color: u.totp_enabled ? '#86efac' : 'rgba(200,200,220,0.25)' }}>
                        {u.totp_enabled ? '✓' : '—'}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-[10px]" style={{ color: 'rgba(200,200,220,0.3)' }}>
                      {u.created_at?.slice(0, 10)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <button onClick={() => openEdit(u)} className="text-[10px] px-2 py-1 rounded transition-all"
                          style={{ color: 'var(--gold)', border: '1px solid rgba(200,169,110,0.2)' }}>Edit</button>
                        <button onClick={() => openReset(u)} className="text-[10px] px-2 py-1 rounded transition-all"
                          style={{ color: '#60a5fa', border: '1px solid rgba(96,165,250,0.2)' }}>Reset PW</button>
                        {u.id !== me?.id && (
                          <>
                            <button onClick={() => toggleActive(u)}
                              className="text-[10px] px-2 py-1 rounded transition-all"
                              style={{ color: u.is_active ? '#fde047' : '#86efac', border: `1px solid ${u.is_active ? 'rgba(234,179,8,0.2)' : 'rgba(34,197,94,0.2)'}` }}>
                              {u.is_active ? 'Disable' : 'Enable'}
                            </button>
                            <button onClick={() => setDeleteUser(u)}
                              className="text-[10px] px-2 py-1 rounded transition-all"
                              style={{ color: '#fca5a5', border: '1px solid rgba(239,68,68,0.2)' }}>Del</button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create Modal */}
      {createOpen && (
        <Modal title="Create New User" onClose={() => setCreateOpen(false)}>
          <div className="space-y-3">
            {[['Full Name', 'full_name', 'text', 'Jane Smith'], ['Email', 'email', 'email', 'jane@company.com'], ['Password', 'password', 'password', 'Min. 8 characters']].map(([l, k, t, ph]) => (
              <div key={k}>
                <label className="block text-xs mb-1" style={{ color: 'rgba(200,200,220,0.5)' }}>{l}</label>
                <input type={t} placeholder={ph} value={createForm[k]}
                  onChange={e => setCreateForm(f => ({ ...f, [k]: e.target.value }))}
                  className="input-dark w-full px-3 py-2.5 rounded-xl text-sm" />
              </div>
            ))}
            <div>
              <label className="block text-xs mb-1" style={{ color: 'rgba(200,200,220,0.5)' }}>Role</label>
              <select value={createForm.role} onChange={e => setCreateForm(f => ({ ...f, role: e.target.value }))}
                className="input-dark w-full px-3 py-2.5 rounded-xl text-sm">
                {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            {formErr && <div className="text-xs text-red-400">{formErr}</div>}
            <button onClick={createUser} className="btn-gold w-full py-2.5 rounded-xl text-sm font-semibold">Create User</button>
          </div>
        </Modal>
      )}

      {/* Edit Modal */}
      {editUser && (
        <Modal title={`Edit: ${editUser.full_name}`} onClose={() => setEditUser(null)}>
          <div className="space-y-3">
            <div>
              <label className="block text-xs mb-1" style={{ color: 'rgba(200,200,220,0.5)' }}>Full Name</label>
              <input value={editForm.full_name} onChange={e => setEditForm(f => ({ ...f, full_name: e.target.value }))}
                className="input-dark w-full px-3 py-2.5 rounded-xl text-sm" />
            </div>
            <div>
              <label className="block text-xs mb-1" style={{ color: 'rgba(200,200,220,0.5)' }}>Role</label>
              <select value={editForm.role} onChange={e => setEditForm(f => ({ ...f, role: e.target.value }))}
                className="input-dark w-full px-3 py-2.5 rounded-xl text-sm">
                {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            {formErr && <div className="text-xs text-red-400">{formErr}</div>}
            <button onClick={updateUser} className="btn-gold w-full py-2.5 rounded-xl text-sm font-semibold">Save Changes</button>
          </div>
        </Modal>
      )}

      {/* Reset Password Modal */}
      {resetUser && (
        <Modal title={`Reset Password: ${resetUser.full_name}`} onClose={() => setResetUser(null)}>
          <div className="space-y-3">
            <p className="text-xs" style={{ color: 'rgba(200,200,220,0.45)' }}>
              Set a new password for this user. They will need to use it on next login.
            </p>
            <input type="password" placeholder="New password (min. 8 chars)"
              value={newPassword} onChange={e => setNewPassword(e.target.value)}
              className="input-dark w-full px-3 py-2.5 rounded-xl text-sm" />
            {formErr && <div className="text-xs text-red-400">{formErr}</div>}
            <button onClick={resetPassword} className="btn-gold w-full py-2.5 rounded-xl text-sm font-semibold">Reset Password</button>
          </div>
        </Modal>
      )}

      {/* Delete Confirm */}
      {deleteUser && (
        <Modal title="Confirm Delete" onClose={() => setDeleteUser(null)}>
          <p className="text-sm mb-5" style={{ color: 'rgba(200,200,220,0.6)' }}>
            Permanently delete <strong style={{ color: '#e8e8f0' }}>{deleteUser.full_name}</strong>?
            This cannot be undone.
          </p>
          <div className="flex gap-3">
            <button onClick={() => setDeleteUser(null)}
              className="flex-1 py-2.5 rounded-xl text-sm"
              style={{ border: '1px solid var(--border)', color: 'rgba(200,200,220,0.5)' }}>
              Cancel
            </button>
            <button onClick={confirmDelete}
              className="flex-1 py-2.5 rounded-xl text-sm font-semibold"
              style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
              Delete
            </button>
          </div>
        </Modal>
      )}
    </div>
  )
}

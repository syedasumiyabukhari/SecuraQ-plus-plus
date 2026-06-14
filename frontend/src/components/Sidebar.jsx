import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NavItem = ({ to, icon, label, end = false, badge = null }) => (
  <NavLink to={to} end={end}
    className={({ isActive }) => `nav-item flex items-center gap-3 px-3 py-2.5 text-sm cursor-pointer
      ${isActive ? 'nav-active' : 'text-gray-400'}`}
  >
    <span className="text-base w-5 text-center">{icon}</span>
    <span className="flex-1">{label}</span>
    {badge && (
      <span className="text-[9px] px-1.5 py-0.5 rounded font-mono"
        style={{ background: 'rgba(234,179,8,0.12)', color: '#fde047', border: '1px solid rgba(234,179,8,0.2)' }}>
        {badge}
      </span>
    )}
  </NavLink>
)

export default function Sidebar({ collapsed, setCollapsed }) {
  const { user, logout, isAdmin } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate(isAdmin ? '/admin/login' : '/login')
  }

  return (
    <aside className="flex flex-col h-full" style={{ background: 'var(--surface-1)', borderRight: '1px solid var(--border)' }}>
      {/* Logo */}
      <div className="px-4 py-5 flex items-center gap-3" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-lg"
          style={{ background: isAdmin
            ? 'linear-gradient(135deg, #8b5cf6, #6d28d9)'
            : 'linear-gradient(135deg, var(--gold), var(--gold-dim))' }}>
          {isAdmin ? '🛡️' : '⚛️'}
        </div>
        {!collapsed && (
          <div>
            <div className="font-display font-bold text-sm"
              style={{ color: isAdmin ? '#a78bfa' : 'var(--gold-light)' }}>
              SecuraQ++
            </div>
            <div className="text-[10px]" style={{ color: 'var(--text-dim)' }}>
              {isAdmin ? 'Admin Portal' : 'QEGVD v2.0'}
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">

        {/* Workspace — visible to ALL roles */}
        <div className="mb-2">
          {!collapsed && (
            <div className="text-[10px] font-semibold uppercase tracking-wider px-3 mb-2"
              style={{ color: 'var(--text-dim)' }}>Workspace</div>
          )}
          <NavItem to="/dashboard" icon="◈" label="Dashboard" end />
          <NavItem to="/scan"      icon="⬡" label="Scan Console" />
          <NavItem to="/reports"   icon="◻" label="Reports" />
          <NavItem to="/patch"     icon="◈" label="Patch Engine" />
        </div>

        {/* Admin-only: ML Tools + System Health + Admin Portal */}
        {isAdmin && (
          <>
            <div className="mt-4">
              {!collapsed && (
                <div className="text-[10px] font-semibold uppercase tracking-wider px-3 mb-2"
                  style={{ color: 'var(--text-dim)' }}>Tools</div>
              )}
              <NavItem to="/health"   icon="⬡" label="System Health" />
              <NavItem to="/datasets" icon="◻" label="Dataset Manager" />
              <NavItem to="/models"   icon="⚛" label="Model Trainer" />
            </div>

            <div className="mt-4">
              {!collapsed && (
                <div className="text-[10px] font-semibold uppercase tracking-wider px-3 mb-2 flex items-center gap-2"
                  style={{ color: '#8b5cf6' }}>
                  <span>Admin Portal</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded font-mono"
                    style={{ background: 'rgba(139,92,246,0.15)', color: '#a78bfa', border: '1px solid rgba(139,92,246,0.3)' }}>
                    Admin
                  </span>
                </div>
              )}
              <NavItem to="/admin"        icon="⊠" label="Overview" end />
              <NavItem to="/admin/users"  icon="⬡" label="User Management" />
              <NavItem to="/admin/scans"  icon="◈" label="Scan Management" />
              <NavItem to="/admin/audit"  icon="◻" label="Audit Log" />
            </div>
          </>
        )}
      </nav>

      {/* User footer */}
      <div className="p-3" style={{ borderTop: '1px solid var(--border)' }}>
        <NavLink to="/profile"
          className="nav-item flex items-center gap-3 px-3 py-2.5 text-sm text-gray-400 w-full">
          <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
            style={{
              background: isAdmin
                ? 'linear-gradient(135deg,#8b5cf6,#6d28d9)'
                : 'linear-gradient(135deg,var(--gold),var(--gold-dim))',
              color: '#fff',
            }}>
            {user?.full_name?.[0]?.toUpperCase() || '?'}
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium truncate" style={{ color: 'var(--text)' }}>
                {user?.full_name}
              </div>
              <div className="text-[10px] truncate font-mono"
                style={{ color: isAdmin ? '#a78bfa' : 'var(--text-dim)' }}>
                {user?.role}
              </div>
            </div>
          )}
        </NavLink>
        <button onClick={handleLogout}
          className="nav-item flex items-center gap-3 px-3 py-2 text-sm text-gray-400 w-full mt-1 cursor-pointer">
          <span className="w-5 text-center text-base">↩</span>
          {!collapsed && <span>Sign out</span>}
        </button>
      </div>
    </aside>
  )
}

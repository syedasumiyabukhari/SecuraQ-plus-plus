import React, { useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from '../components/Sidebar'

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="flex h-screen overflow-hidden">
      <div style={{ width: collapsed ? 60 : 220, minWidth: collapsed ? 60 : 220, transition: 'width .2s' }}>
        <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
      </div>
      <main className="flex-1 overflow-y-auto flex flex-col" style={{ background: 'var(--surface-0)' }}>
        <Outlet />
      </main>
    </div>
  )
}

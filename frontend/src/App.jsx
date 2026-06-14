import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ToastProvider } from './context/ToastContext'
import { ProtectedRoute, AdminRoute } from './components/ProtectedRoute'

import LandingPage        from './pages/LandingPage'
import LoginPage          from './pages/LoginPage'
import AdminLoginPage     from './pages/AdminLoginPage'
import RegisterPage       from './pages/RegisterPage'
import MainLayout         from './layouts/MainLayout'
import DashboardPage      from './pages/DashboardPage'
import ScanConsolePage    from './pages/ScanConsolePage'
import ReportsPage        from './pages/ReportsPage'
import PatchEnginePage    from './pages/PatchEnginePage'
import ProfilePage        from './pages/ProfilePage'
import SystemHealthPage   from './pages/SystemHealthPage'
import DatasetManagerPage from './pages/DatasetManagerPage'
import ModelTrainerPage   from './pages/ModelTrainerPage'
import AdminDashboard     from './pages/admin/AdminDashboard'
import AdminUsersPage     from './pages/admin/AdminUsersPage'
import AdminAuditPage     from './pages/admin/AdminAuditPage'
import AdminScansPage     from './pages/admin/AdminScansPage'
import NotFoundPage       from './pages/NotFoundPage'

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <BrowserRouter>
          <Routes>
            {/* Public */}
            <Route path="/"            element={<LandingPage />} />
            <Route path="/login"       element={<LoginPage />} />
            <Route path="/register"    element={<RegisterPage />} />
            <Route path="/admin/login" element={<AdminLoginPage />} />

            {/* Protected — Analyst + Admin */}
            <Route element={<ProtectedRoute><MainLayout /></ProtectedRoute>}>
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/scan"      element={<ScanConsolePage />} />
              <Route path="/reports"   element={<ReportsPage />} />
              <Route path="/patch"     element={<PatchEnginePage />} />
              <Route path="/profile"   element={<ProfilePage />} />

              {/* Admin-only tools + portal */}
              <Route path="/health"    element={<AdminRoute><SystemHealthPage /></AdminRoute>} />
              <Route path="/datasets"  element={<AdminRoute><DatasetManagerPage /></AdminRoute>} />
              <Route path="/models"    element={<AdminRoute><ModelTrainerPage /></AdminRoute>} />
              <Route path="/admin"             element={<AdminRoute><AdminDashboard /></AdminRoute>} />
              <Route path="/admin/users"       element={<AdminRoute><AdminUsersPage /></AdminRoute>} />
              <Route path="/admin/audit"       element={<AdminRoute><AdminAuditPage /></AdminRoute>} />
              <Route path="/admin/scans"       element={<AdminRoute><AdminScansPage /></AdminRoute>} />
            </Route>

            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </BrowserRouter>
      </ToastProvider>
    </AuthProvider>
  )
}

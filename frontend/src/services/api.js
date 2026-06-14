import axios from 'axios'

const AUTH_API = '/api/auth'
const ADMIN_API = '/api/admin'
const SCAN_API = '/api'

function authHeader() {
  const token = localStorage.getItem('sqpp_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// ── Auth ──────────────────────────────────────────────────────────────────────
export const api = {
  // Auth
  register: (data) => axios.post(`${AUTH_API}/register`, data),
  login: (data) => axios.post(`${AUTH_API}/login`, data),
  login2fa: (data) => axios.post(`${AUTH_API}/login/2fa`, data),
  me: () => axios.get(`${AUTH_API}/me`, { headers: authHeader() }),
  updateProfile: (data) => axios.put(`${AUTH_API}/profile`, data, { headers: authHeader() }),
  updatePassword: (data) => axios.put(`${AUTH_API}/password`, data, { headers: authHeader() }),
  setup2fa: () => axios.post(`${AUTH_API}/2fa/setup`, {}, { headers: authHeader() }),
  verify2fa: (code) => axios.post(`${AUTH_API}/2fa/verify`, { code }, { headers: authHeader() }),
  disable2fa: (password) => axios.post(`${AUTH_API}/2fa/disable`, { password }, { headers: authHeader() }),

  // Admin
  adminStats: () => axios.get(`${ADMIN_API}/stats`, { headers: authHeader() }),
  adminUsers: () => axios.get(`${ADMIN_API}/users`, { headers: authHeader() }),
  adminCreateUser: (data) => axios.post(`${ADMIN_API}/users`, data, { headers: authHeader() }),
  adminUpdateUser: (id, data) => axios.put(`${ADMIN_API}/users/${id}`, data, { headers: authHeader() }),
  adminDeleteUser: (id) => axios.delete(`${ADMIN_API}/users/${id}`, { headers: authHeader() }),
  adminResetPassword: (id, new_password) => axios.post(`${ADMIN_API}/users/${id}/reset-password`, { new_password }, { headers: authHeader() }),
  adminAuditLog: () => axios.get(`${ADMIN_API}/audit`, { headers: authHeader() }),

  // Auth — forgot / reset password (FR-M2.5)
  forgotPassword: (email) => axios.post(`${AUTH_API}/forgot-password`, { email }),
  resetPassword:  (data)  => axios.post(`${AUTH_API}/reset-password`, data),

  // Scanning
  uploadFile: (formData) => axios.post(`${SCAN_API}/upload`, formData, { headers: { ...authHeader(), 'Content-Type': 'multipart/form-data' } }),
  startScan: (scanId) => axios.post(`${SCAN_API}/scan/start/${scanId}`, {}, { headers: authHeader() }),
  stopScan:  (scanId) => axios.post(`${SCAN_API}/scan/stop/${scanId}`,  {}, { headers: authHeader() }),
  getScanResults: (scanId) => axios.get(`${SCAN_API}/scan/results/${scanId}`, { headers: authHeader() }),
  listScans: () => axios.get(`${SCAN_API}/scans`, { headers: authHeader() }),
  deleteScan: (scanId) => axios.delete(`${SCAN_API}/scan/${scanId}`, { headers: authHeader() }),
  downloadReport: (scanId) => `${SCAN_API}/download-report/${scanId}`,
  health: () => axios.get(`${SCAN_API}/health`),

  // Auto-fix
  autoFixPreview: (scanId) =>
    axios.get(`${SCAN_API}/scan/auto-fix-preview/${scanId}`, { headers: authHeader() }),
  autoFixDownload: (scanId) =>
    axios.get(`${SCAN_API}/scan/auto-fix/${scanId}`, {
      headers: authHeader(),
      responseType: 'blob',
    }),

  // AI Improve
  aiImprove: (data) =>
    axios.post(`${SCAN_API}/scan/ai-improve`, data, { headers: authHeader() }),

  // Patch validation
  validatePatch: (originalScanId, formData) =>
    axios.post(`${SCAN_API}/scan/validate-patch/${originalScanId}`, formData, {
      headers: { ...authHeader(), 'Content-Type': 'multipart/form-data' },
    }),
  getComparison: (originalScanId, patchScanId) =>
    axios.get(`${SCAN_API}/scan/comparison/${originalScanId}/${patchScanId}`, {
      headers: authHeader(),
    }),

  // Charts
  saveCharts: (scanId, data) =>
    axios.post(`${SCAN_API}/scan/save-charts/${scanId}`, data, { headers: authHeader() }),
  getCharts: (scanId) =>
    axios.get(`${SCAN_API}/scan/charts/${scanId}`, { headers: authHeader() }),
}

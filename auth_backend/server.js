/**
 * SecuraQ++ Auth Backend — Express + sql.js
 * Port: 4000
 * Roles: admin | analyst
 */

require('dotenv').config()
const express    = require('express')
const cors       = require('cors')
const bcrypt     = require('bcryptjs')
const jwt        = require('jsonwebtoken')
const { authenticator } = require('otplib')
const QRCode     = require('qrcode')
const nodemailer = require('nodemailer')
const initSqlJs  = require('sql.js')
const fs         = require('fs')
const path       = require('path')

const app        = express()
const PORT       = 4000
const JWT_SECRET = process.env.JWT_SECRET || 'securaq-jwt-secret-2025-change-in-prod'
const DB_PATH    = path.join(__dirname, 'securaq.db')

// ─── Email OTP store (in-memory) ─────────────────────────────────────────────
const otpStore = {} // { email: { code, expiry } }

function generateOTP() {
  return Math.floor(100000 + Math.random() * 900000).toString()
}

function isStrongPassword(p) {
  return p.length >= 8 && /[A-Z]/.test(p) && /[0-9]/.test(p) && /[^A-Za-z0-9]/.test(p)
}

const transporter = nodemailer.createTransport({
  host:   process.env.SMTP_HOST   || 'smtp.gmail.com',
  port:   parseInt(process.env.SMTP_PORT || '587'),
  secure: process.env.SMTP_SECURE === 'true',
  auth: {
    user: process.env.SMTP_USER || '',
    pass: process.env.SMTP_PASS || '',
  },
})

async function sendOTPEmail(toEmail, otp) {
  if (!process.env.SMTP_USER) {
    console.log(`\n📧 [2FA OTP] To: ${toEmail}  Code: ${otp}  (configure SMTP_USER/SMTP_PASS in .env to send real emails)\n`)
    return
  }
  console.log(`\n📧 [OTP LOG] To: ${toEmail}  Code: ${otp}\n`)
  await transporter.sendMail({
    from: `"SecuraQ++" <${process.env.SMTP_USER}>`,
    to:   toEmail,
    subject: 'SecuraQ++ — Your Login Verification Code',
    html: `
      <div style="font-family:sans-serif;max-width:420px;margin:auto;padding:32px;background:#0f0f1a;color:#e8e8f0;border-radius:12px">
        <h2 style="color:#c8a96e;margin-top:0">SecuraQ++ Verification</h2>
        <p style="color:#aaa">Your one-time login code is:</p>
        <div style="font-size:36px;font-weight:bold;letter-spacing:10px;color:#c8a96e;padding:16px 0">${otp}</div>
        <p style="color:#888;font-size:13px">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
        <hr style="border-color:#333;margin:24px 0"/>
        <p style="color:#555;font-size:12px">If you didn't request this, you can ignore this email.</p>
      </div>
    `,
  })
}

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:4173'], credentials: true }))
app.use(express.json())

// ─── DB ──────────────────────────────────────────────────────────────────────
let db
async function initDB() {
  const SQL = await initSqlJs()
  if (fs.existsSync(DB_PATH)) {
    db = new SQL.Database(fs.readFileSync(DB_PATH))
    console.log('📂 Loaded existing database')
  } else {
    db = new SQL.Database()
    console.log('🆕 Created new database')
  }

  db.run(`CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name    TEXT NOT NULL,
    email        TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    totp_secret  TEXT,
    totp_enabled INTEGER DEFAULT 0,
    role         TEXT DEFAULT 'analyst',
    is_active    INTEGER DEFAULT 1,
    created_at   TEXT DEFAULT (datetime('now')),
    last_login   TEXT
  )`)

  db.run(`CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    action     TEXT,
    detail     TEXT,
    ip         TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  )`)

  // Seed default admin if none exists
  const adminCheck = db.exec(`SELECT id FROM users WHERE role='admin' LIMIT 1`)
  if (!adminCheck.length || !adminCheck[0].values.length) {
    const hash = await bcrypt.hash('Admin@1234', 12)
    db.run(`INSERT INTO users (full_name, email, password_hash, role) VALUES ('Admin', 'admin@securaqpp.local', '${hash}', 'admin')`)
    console.log('👤 Default admin created: admin@securaqpp.local / Admin@1234')
  }

  saveDB()
  console.log('✅ Database ready')
}

function saveDB() {
  fs.writeFileSync(DB_PATH, Buffer.from(db.export()))
}

function audit(userId, action, detail = '', ip = '') {
  try {
    db.run(`INSERT INTO audit_log (user_id, action, detail, ip) VALUES (?, ?, ?, ?)`,
      [userId, action, detail, ip])
    saveDB()
  } catch (_) {}
}

// ─── Middleware ───────────────────────────────────────────────────────────────
function requireAuth(req, res, next) {
  const auth = req.headers.authorization
  if (!auth?.startsWith('Bearer ')) return res.status(401).json({ error: 'No token' })
  try {
    req.user = jwt.verify(auth.slice(7), JWT_SECRET)
    next()
  } catch {
    res.status(401).json({ error: 'Invalid or expired token' })
  }
}

function requireAdmin(req, res, next) {
  requireAuth(req, res, () => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: 'Admin access required' })
    next()
  })
}

function getUser(id) {
  const r = db.exec(`SELECT id,full_name,email,totp_enabled,role,is_active,created_at,last_login FROM users WHERE id=?`, [id])
  if (!r.length || !r[0].values.length) return null
  const [uid,full_name,email,totp_enabled,role,is_active,created_at,last_login] = r[0].values[0]
  return { id:uid, full_name, email, totp_enabled:!!totp_enabled, role, is_active:!!is_active, created_at, last_login }
}

// ─── Auth Routes ──────────────────────────────────────────────────────────────

// Register
app.post('/api/auth/register', async (req, res) => {
  try {
    const { full_name, email, password, admin_code } = req.body
    if (!full_name || !email || !password)
      return res.status(400).json({ error: 'All fields required' })
    if (!isStrongPassword(password))
      return res.status(400).json({ error: 'Password must be at least 8 characters and include an uppercase letter, a number, and a special character' })

    const dup = db.exec(`SELECT id FROM users WHERE email=?`, [email])
    if (dup.length && dup[0].values.length)
      return res.status(409).json({ error: 'Email already registered' })

    const INVITE = process.env.ADMIN_INVITE_CODE
    const role = (INVITE && admin_code && admin_code === INVITE) ? 'admin' : 'analyst'

    const hash = await bcrypt.hash(password, 12)
    db.run(`INSERT INTO users (full_name, email, password_hash, role) VALUES (?, ?, ?, ?)`,
      [full_name, email, hash, role])
    saveDB()
    audit(null, 'REGISTER', `${email} role=${role}`, req.ip)
    res.status(201).json({ message: 'Account created successfully', role })
  } catch (err) {
    console.error(err)
    res.status(500).json({ error: 'Registration failed' })
  }
})

// Login step 1
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password, remember_me } = req.body
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' })

    const r = db.exec(`SELECT id,full_name,email,password_hash,totp_enabled,role,is_active FROM users WHERE email=?`, [email])
    if (!r.length || !r[0].values.length)
      return res.status(401).json({ error: 'No account found with that email address', field: 'email' })

    const [id,,userEmail,hash,,role,is_active] = r[0].values[0]
    if (!is_active) return res.status(403).json({ error: 'Account suspended. Contact admin.' })

    const valid = await bcrypt.compare(password, hash)
    if (!valid) return res.status(401).json({ error: 'Incorrect password', field: 'password' })

    db.run(`UPDATE users SET last_login=datetime('now') WHERE id=${id}`)
    saveDB()
    audit(id, 'LOGIN_ATTEMPT', `role=${role}`, req.ip)

    // Always require email OTP 2FA
    const otp = generateOTP()
    otpStore[userEmail] = { code: otp, expiry: Date.now() + 10 * 60 * 1000 }
    try { await sendOTPEmail(userEmail, otp) } catch (mailErr) {
      console.error('Email send failed:', mailErr.message)
      console.log(`[2FA OTP fallback] ${userEmail} → ${otp}`)
    }

    const tempToken = jwt.sign({ id, email: userEmail, step: '2fa_pending', remember_me: !!remember_me }, JWT_SECRET, { expiresIn: '10m' })
    return res.json({ requires2FA: true, tempToken })
  } catch (err) {
    console.error(err)
    res.status(500).json({ error: 'Login failed' })
  }
})

// Login step 2 — Email OTP 2FA
app.post('/api/auth/login/2fa', (req, res) => {
  try {
    const { tempToken, code } = req.body
    if (!tempToken || !code) return res.status(400).json({ error: 'Token and code required' })

    let decoded
    try { decoded = jwt.verify(tempToken, JWT_SECRET) }
    catch { return res.status(401).json({ error: 'Session expired. Please sign in again.' }) }
    if (decoded.step !== '2fa_pending') return res.status(401).json({ error: 'Invalid step' })

    const stored = otpStore[decoded.email]
    if (!stored) return res.status(401).json({ error: 'Code expired. Please sign in again.' })
    if (Date.now() > stored.expiry) {
      delete otpStore[decoded.email]
      return res.status(401).json({ error: 'Code expired. Please sign in again.' })
    }
    if (stored.code !== code.trim())
      return res.status(401).json({ error: 'Incorrect verification code' })

    delete otpStore[decoded.email]

    const r = db.exec(`SELECT id,full_name,email,role FROM users WHERE id=${decoded.id}`)
    if (!r.length || !r[0].values.length) return res.status(401).json({ error: 'User not found' })
    const [id,full_name,email,role] = r[0].values[0]

    audit(id, 'LOGIN', `role=${role}`, req.ip)
    const expiry = decoded.remember_me ? '7d' : '1d'
    const token = jwt.sign({ id, email, full_name, role }, JWT_SECRET, { expiresIn: expiry })
    res.json({ token, user: { id, email, full_name, role } })
  } catch (err) {
    res.status(500).json({ error: '2FA verify failed' })
  }
})

// FR-M2.5: Forgot password — send OTP to email
app.post('/api/auth/forgot-password', async (req, res) => {
  try {
    const { email } = req.body
    if (!email) return res.status(400).json({ error: 'Email required' })
    const r = db.exec(`SELECT id FROM users WHERE email=? AND is_active=1`, [email])
    // Always respond OK — don't leak whether email exists
    if (!r.length || !r[0].values.length) {
      return res.json({ message: 'If that email exists, a reset code has been sent.' })
    }
    const [id] = r[0].values[0]
    const otp = generateOTP()
    otpStore[email] = { code: otp, expiry: Date.now() + 10 * 60 * 1000, reset: true }
    try { await sendOTPEmail(email, otp) } catch (mailErr) {
      console.log(`[Password Reset OTP] ${email} → ${otp}`)
    }
    audit(id, 'FORGOT_PASSWORD', email, req.ip)
    const tempToken = jwt.sign({ id, email, step: 'reset_pending' }, JWT_SECRET, { expiresIn: '10m' })
    res.json({ message: 'Reset code sent', tempToken })
  } catch (err) {
    res.status(500).json({ error: 'Failed to send reset code' })
  }
})

// FR-M2.5: Reset password — verify OTP + update password
app.post('/api/auth/reset-password', async (req, res) => {
  try {
    const { tempToken, code, new_password } = req.body
    if (!tempToken || !code || !new_password)
      return res.status(400).json({ error: 'Token, code and new password required' })

    let decoded
    try { decoded = jwt.verify(tempToken, JWT_SECRET) }
    catch { return res.status(401).json({ error: 'Session expired. Request a new reset code.' }) }
    if (decoded.step !== 'reset_pending') return res.status(401).json({ error: 'Invalid token' })

    const stored = otpStore[decoded.email]
    if (!stored || !stored.reset) return res.status(401).json({ error: 'Code expired. Request a new one.' })
    if (Date.now() > stored.expiry) {
      delete otpStore[decoded.email]
      return res.status(401).json({ error: 'Code expired. Request a new one.' })
    }
    if (stored.code !== code.trim()) return res.status(401).json({ error: 'Incorrect reset code' })

    if (!isStrongPassword(new_password))
      return res.status(400).json({ error: 'Password must be at least 8 characters and include an uppercase letter, a number, and a special character' })

    delete otpStore[decoded.email]
    const hash = await bcrypt.hash(new_password, 12)
    db.run(`UPDATE users SET password_hash=? WHERE id=?`, [hash, decoded.id])
    saveDB()
    audit(decoded.id, 'PASSWORD_RESET', decoded.email, req.ip)
    res.json({ message: 'Password reset successfully. Please sign in.' })
  } catch (err) {
    res.status(500).json({ error: 'Password reset failed' })
  }
})

// Me
app.get('/api/auth/me', requireAuth, (req, res) => {
  const u = getUser(req.user.id)
  if (!u) return res.status(404).json({ error: 'User not found' })
  res.json(u)
})

// Update profile
app.put('/api/auth/profile', requireAuth, (req, res) => {
  try {
    const { full_name } = req.body
    if (!full_name || full_name.trim().length < 2)
      return res.status(400).json({ error: 'Name too short' })
    db.run(`UPDATE users SET full_name=? WHERE id=?`, [full_name, req.user.id])
    saveDB()
    res.json({ message: 'Profile updated' })
  } catch { res.status(500).json({ error: 'Update failed' }) }
})

// Change password
app.put('/api/auth/password', requireAuth, async (req, res) => {
  try {
    const { current_password, new_password } = req.body
    const r = db.exec(`SELECT password_hash FROM users WHERE id=${req.user.id}`)
    const hash = r[0].values[0][0]
    if (!await bcrypt.compare(current_password, hash))
      return res.status(401).json({ error: 'Current password incorrect' })
    if (!isStrongPassword(new_password))
      return res.status(400).json({ error: 'Password must be at least 8 characters and include an uppercase letter, a number, and a special character' })
    const newHash = await bcrypt.hash(new_password, 12)
    db.run(`UPDATE users SET password_hash=? WHERE id=?`, [newHash, req.user.id])
    saveDB()
    audit(req.user.id, 'PASSWORD_CHANGE', '', req.ip)
    res.json({ message: 'Password updated' })
  } catch { res.status(500).json({ error: 'Failed to update password' }) }
})

// 2FA setup
app.post('/api/auth/2fa/setup', requireAuth, async (req, res) => {
  try {
    const secret = authenticator.generateSecret()
    const url    = authenticator.keyuri(req.user.email, 'SecuraQ++', secret)
    const qr     = await QRCode.toDataURL(url)
    db.run(`UPDATE users SET totp_secret='${secret}' WHERE id=${req.user.id}`)
    saveDB()
    res.json({ secret, qrCode: qr, otpAuthUrl: url })
  } catch { res.status(500).json({ error: 'Failed to generate 2FA secret' }) }
})

// 2FA enable/confirm
app.post('/api/auth/2fa/verify', requireAuth, (req, res) => {
  try {
    const { code } = req.body
    const r = db.exec(`SELECT totp_secret FROM users WHERE id=${req.user.id}`)
    const secret = r[0].values[0][0]
    if (!secret) return res.status(400).json({ error: 'Run /2fa/setup first' })
    if (!authenticator.verify({ token: code, secret }))
      return res.status(400).json({ error: 'Invalid code' })
    db.run(`UPDATE users SET totp_enabled=1 WHERE id=${req.user.id}`)
    saveDB()
    audit(req.user.id, '2FA_ENABLED', '', req.ip)
    res.json({ message: '2FA enabled' })
  } catch { res.status(500).json({ error: '2FA enable failed' }) }
})

// 2FA disable
app.post('/api/auth/2fa/disable', requireAuth, async (req, res) => {
  try {
    const { password } = req.body
    const r = db.exec(`SELECT password_hash FROM users WHERE id=${req.user.id}`)
    const hash = r[0].values[0][0]
    if (!await bcrypt.compare(password, hash))
      return res.status(401).json({ error: 'Incorrect password' })
    db.run(`UPDATE users SET totp_enabled=0, totp_secret=NULL WHERE id=${req.user.id}`)
    saveDB()
    audit(req.user.id, '2FA_DISABLED', '', req.ip)
    res.json({ message: '2FA disabled' })
  } catch { res.status(500).json({ error: 'Failed to disable 2FA' }) }
})

// ─── Admin Routes ─────────────────────────────────────────────────────────────

// List all users
app.get('/api/admin/users', requireAdmin, (req, res) => {
  const r = db.exec(`SELECT id,full_name,email,role,is_active,totp_enabled,created_at,last_login FROM users ORDER BY created_at DESC`)
  if (!r.length) return res.json([])
  const [cols, ...rows] = [r[0].columns, ...r[0].values]
  res.json(rows.map(row => Object.fromEntries(cols.map((c,i) => [c, row[i]]))))
})

// Create user (admin)
app.post('/api/admin/users', requireAdmin, async (req, res) => {
  try {
    const { full_name, email, password, role = 'analyst' } = req.body
    if (!full_name || !email || !password) return res.status(400).json({ error: 'All fields required' })
    if (!['admin','analyst'].includes(role)) return res.status(400).json({ error: 'Invalid role' })
    const dup = db.exec(`SELECT id FROM users WHERE email=?`, [email])
    if (dup.length && dup[0].values.length) return res.status(409).json({ error: 'Email already exists' })
    const hash = await bcrypt.hash(password, 12)
    db.run(`INSERT INTO users (full_name,email,password_hash,role) VALUES (?,?,?,?)`,
      [full_name, email, hash, role])
    saveDB()
    audit(req.user.id, 'ADMIN_CREATE_USER', email, req.ip)
    res.status(201).json({ message: 'User created' })
  } catch { res.status(500).json({ error: 'Failed to create user' }) }
})

// Update user role / status
app.put('/api/admin/users/:id', requireAdmin, (req, res) => {
  try {
    const { role, is_active, full_name } = req.body
    const uid = parseInt(req.params.id)
    if (isNaN(uid)) return res.status(400).json({ error: 'Invalid ID' })
    if (role !== undefined) {
      if (!['admin','analyst'].includes(role)) return res.status(400).json({ error: 'Invalid role' })
      db.run(`UPDATE users SET role=? WHERE id=?`, [role, uid])
    }
    if (is_active !== undefined) db.run(`UPDATE users SET is_active=? WHERE id=?`, [is_active?1:0, uid])
    if (full_name) db.run(`UPDATE users SET full_name=? WHERE id=?`, [full_name, uid])
    saveDB()
    audit(req.user.id, 'ADMIN_UPDATE_USER', `id=${uid}`, req.ip)
    res.json({ message: 'User updated' })
  } catch { res.status(500).json({ error: 'Update failed' }) }
})

// Reset user password (admin)
app.post('/api/admin/users/:id/reset-password', requireAdmin, async (req, res) => {
  try {
    const { new_password } = req.body
    if (!new_password || new_password.length < 8)
      return res.status(400).json({ error: 'Password must be at least 8 characters' })
    const hash = await bcrypt.hash(new_password, 12)
    db.run(`UPDATE users SET password_hash=? WHERE id=?`, [hash, parseInt(req.params.id)])
    saveDB()
    audit(req.user.id, 'ADMIN_RESET_PASSWORD', `uid=${req.params.id}`, req.ip)
    res.json({ message: 'Password reset' })
  } catch { res.status(500).json({ error: 'Reset failed' }) }
})

// Delete user
app.delete('/api/admin/users/:id', requireAdmin, (req, res) => {
  try {
    const uid = parseInt(req.params.id)
    if (uid === req.user.id) return res.status(400).json({ error: 'Cannot delete yourself' })
    db.run(`DELETE FROM users WHERE id=${uid}`)
    saveDB()
    audit(req.user.id, 'ADMIN_DELETE_USER', `id=${uid}`, req.ip)
    res.json({ message: 'User deleted' })
  } catch { res.status(500).json({ error: 'Delete failed' }) }
})

// Audit log
app.get('/api/admin/audit', requireAdmin, (req, res) => {
  const limit = parseInt(req.query.limit) || 100
  const r = db.exec(`
    SELECT a.id, a.action, a.detail, a.ip, a.created_at, u.email
    FROM audit_log a
    LEFT JOIN users u ON a.user_id = u.id
    ORDER BY a.created_at DESC
    LIMIT ${limit}
  `)
  if (!r.length) return res.json([])
  const cols = r[0].columns
  res.json(r[0].values.map(row => Object.fromEntries(cols.map((c,i) => [c, row[i]]))))
})

// Stats for admin dashboard
app.get('/api/admin/stats', requireAdmin, (req, res) => {
  const totalUsers  = db.exec(`SELECT COUNT(*) FROM users`)[0].values[0][0]
  const activeUsers = db.exec(`SELECT COUNT(*) FROM users WHERE is_active=1`)[0].values[0][0]
  const adminCount  = db.exec(`SELECT COUNT(*) FROM users WHERE role='admin'`)[0].values[0][0]
  const recentLogins = db.exec(
    `SELECT COUNT(*) FROM audit_log WHERE action='LOGIN' AND created_at >= datetime('now','-24 hours')`
  )[0].values[0][0]
  res.json({ totalUsers, activeUsers, adminCount, recentLogins })
})

// ─── Start ────────────────────────────────────────────────────────────────────
initDB().then(() => {
  app.listen(PORT, () => {
    console.log(`🚀 SecuraQ++ Auth Backend on http://localhost:${PORT}`)
    console.log(`   Roles: admin | analyst`)
    console.log(`   Default admin: admin@securaqpp.local / Admin@1234`)
  })
})

# SecuraQ++ — Quantum-Enhanced Vulnerability Detection Platform

## Architecture

```
securaqpp/
├── ml_core/              ← QEGVD pipeline (copy of givingUpVersion_v2/)
│   ├── src/              ← stages 1-9 + inference
│   └── models/           ← trained .pt checkpoints (BO, FS, UAF)
│
├── scanning_backend/     ← FastAPI  (port 8000)
│   └── backend_api.py    ← upload, scan, WebSocket, report download
│
├── auth_backend/         ← Express (port 4000)
│   └── server.js         ← JWT auth, 2FA (TOTP), admin CRUD, audit log
│
└── frontend/             ← React + Vite (port 5173)
    └── src/
        ├── pages/        ← Dashboard, Scan, Reports, Patch, Profile, Health
        └── pages/admin/  ← Admin Dashboard, Users, Audit Log
```

---

## Quick Start (Windows)

### Step 1 — Copy ML Core

```
xcopy /E /I path\to\givingUpVersion_v2 securaqpp\ml_core
```

### Step 2 — Auth Backend

```
cd securaqpp\auth_backend
npm install
node server.js
```
Runs on http://localhost:4000
Default admin: `admin@securaqpp.local` / `Admin@1234`

### Step 3 — Scanning Backend

```
cd securaqpp\scanning_backend
pip install fastapi uvicorn python-multipart
# For full ML: pip install -r ..\ml_core\requirements.txt
python backend_api.py
```
Runs on http://localhost:8000

If ML dependencies are not installed, the backend runs in **demo mode** and returns
realistic example vulnerabilities.

### Step 4 — Frontend

```
cd securaqpp\frontend
npm install
npm run dev
```
Runs on http://localhost:5173

---

## Features

### User Portal
- **Dashboard** — scan stats, F1 charts, severity distribution, recent scans
- **Scan Console** — upload .c/.cpp, real-time WebSocket log, vulnerability results with
  CWE tags, confidence scores, code snippets, report download
- **Reports** — full scan history, filter by status/vulns, side-by-side detail panel
- **Patch Engine** — CWE-mapped before/after code fixes for BO, FS, UAF
- **System Health** — ML pipeline status, service checks, QEGVD stage table
- **Profile & 2FA** — name/password change, full TOTP 2FA setup with QR code

### Admin Portal (`/admin`)
- **Overview** — user counts, 24h logins, scan totals
- **User Management** — create/edit/disable/delete users, role management, password reset
- **Audit Log** — all auth events (LOGIN, REGISTER, ADMIN_*) with IP, timestamp, action

---

## FS Accuracy Note

The Format String (FS) classifier uses a threshold of `0.371` from calibration
(vs `0.46` for BO and `0.32` for UAF). If FS precision is too low in practice,
raise the threshold:

```python
# In results/calibration_matrix.json
"fs": 0.45   # raise from 0.371 to reduce false positives
```

Or retrain with stronger regularization in `src/stage7_fusion.py`:
```python
# FS MLP already uses Dropout(0.35) + smaller hidden dims
# Increase to Dropout(0.4) if still overfitting
```

---

## Detectors

| Classifier | CWE         | Threshold | Notes |
|------------|-------------|-----------|-------|
| BO         | CWE-121/122 | 0.461     | Buffer overflow — best accuracy |
| FS         | CWE-134     | 0.371     | Format string — may need threshold tuning |
| UAF        | CWE-416     | 0.317     | Use-after-free — highest F1 (0.94) |

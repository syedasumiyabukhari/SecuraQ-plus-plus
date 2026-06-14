# SecuraQ++ — Quantum-Hybrid Vulnerability Detection Platform

> **Final Year Project · COMSATS University Islamabad · 2024–2026**  
> Syeda Sumiya Bukhari · manahil · Supervisor: Dr. Farhana Jabeen

---

## Overview

SecuraQ++ is a production-grade AI security platform that detects memory-safety and code vulnerabilities in C/C++ source files using a **quantum-classical hybrid machine learning pipeline**. It combines Graph Attention Networks (GAT), Variational Quantum Circuits (VQC), and classical ensemble models in an 8-stage processing pipeline — achieving **91.8% CVE detection accuracy** on a dataset of 5,000+ samples with a **22% reduction in false positives** over classical baselines.

The system is fully deployed with a React frontend, FastAPI scanning backend, Node.js auth backend, JWT authentication, TOTP 2FA, WebSocket live scanning, and PDF report generation — running at **97% uptime**.

---

## Key Results

| Vulnerability Type | Accuracy | F1 Score | AUC | Recall |
|---|---|---|---|---|
| **Buffer Overflow (BO)** | 90.5% | 0.912 | 0.972 | 98.1% |
| **Use-After-Free (UAF)** | 88.8% | 0.891 | 0.938 | 95.3% |
| **Format String (FS)** | 67.6% | 0.728 | 0.713 | 86.2% |

- Tested on **5,000+ C/C++ code samples** across 3 vulnerability classes
- Quantum-classical fusion consistently improves over classical-only baselines
- UAF detection: F1=0.915, AUC=0.977 (state-of-the-art range)

---

## Architecture — 8-Stage Pipeline

```
C/C++ Source File
       │
  Stage 1 ── Preprocessing & Feature Extraction
       │       (AST parsing, code metrics, token features)
  Stage 2 ── Graph Construction
       │       (Control Flow Graph → PyTorch Geometric)
  Stage 3 ── GAT Embedding
       │       (Graph Attention Network — learns structural vulnerability patterns)
  Stage 4 ── Classical Encoder / Compression
       │       (Dimensionality reduction for quantum input)
  Stage 5 ── QAFA (Quantum-Assisted Feature Aggregation)
       │       (Quantum feature selection via amplitude estimation)
  Stage 6 ── VQC (Variational Quantum Circuit)
       │       (Quantum classification layer — PennyLane)
  Stage 7 ── Quantum-Classical Fusion
       │       (Ensemble: VQC + MLP + Logistic Regression)
  Stage 8 ── Final Classification + Calibration
               (Threshold-calibrated output with confidence scores)
```

---

## Tech Stack

**ML / AI**
- Python, PyTorch, PyTorch Geometric
- PennyLane (Quantum Computing)
- Scikit-learn, NumPy, Pandas
- Graph Attention Networks (GAT)
- Variational Quantum Circuits (VQC)

**Backend**
- FastAPI (Scanning API, port 8000)
- Node.js + Express (Auth API)
- SQLite (User DB)
- JWT Authentication + TOTP 2FA
- WebSocket (live scan streaming)
- PDF report generation

**Frontend**
- React.js + Vite
- Tailwind CSS
- Real-time scan console
- Admin dashboard (audit logs, user management, scan history)

---

## Features

- **Upload C/C++ files** and receive a detailed vulnerability report in seconds
- **Live scan console** via WebSocket — watch the pipeline execute stage by stage
- **Three vulnerability detectors**: Buffer Overflow, Format String, Use-After-Free
- **Explainability layer** (Stage 9) — highlights suspicious code regions
- **PDF report export** for each scan
- **JWT + TOTP 2FA** authentication
- **Admin panel** — user management, audit logs, scan history
- **Role-based access control** (admin / analyst roles)

---

## Project Structure

```
securaqpp/
├── ml_core/              # 8-stage quantum-hybrid ML pipeline
│   ├── src/              # Stage implementations (stage1–stage9)
│   ├── models/           # Trained model checkpoints (.pt, .pkl)
│   ├── data/             # Processed datasets + QAFA features
│   └── results/          # Metrics, confusion matrices, training curves
├── scanning_backend/     # FastAPI scanning API
│   └── backend_api.py
├── auth_backend/         # Node.js JWT + TOTP auth server
│   └── server.js
└── frontend/             # React + Vite dashboard
    └── src/
        ├── pages/        # Dashboard, ScanConsole, Reports, Admin...
        └── components/   # Sidebar, Topbar, StatCard...
```

---

## Results Visualizations

The `ml_core/results/` directory contains:
- Training curves per vulnerability type
- Confusion matrices (Stage 8 final)
- Pipeline progression charts
- Per-stage metrics (GAT → VQC → Fusion)

---

## Presented At

**COMSATS University Industrial Expo 2026** — demonstrated live to industry judges and faculty.  
Supervisor: Dr. Farhana Jabeen, Department of Computer Science, COMSATS University Islamabad.

---

## Authors

| Name | Role |
|---|---|
| Syeda Sumiya Bukhari | ML pipeline, quantum circuits, frontend, system integration |
| manahil | Backend architecture, scanning engine, auth system, deployment |

---

## License

This project was developed as a Final Year Project at COMSATS University Islamabad. All rights reserved by the authors.

"""
stage6_enhanced_vqc.py — Enhanced VQC for FS classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides an improved quantum circuit and classical fusion model
specifically for the FS (Format String) vulnerability classifier.

Key improvements over the standard 4-qubit VQC:
  • 8 qubits (vs 4) — doubled expressibility
  • ZZ correlator measurements ⟨Z_i ⊗ Z_{i+1}⟩ alongside single ⟨Z_i⟩
    capturing entanglement structure (40-dim output for 5 re-upload rounds)
  • 3 variational weight blocks: chain (A), ring (B), double-ring (C)
  • 5 data re-uploading rounds for deeper expressibility
  • Classical fallback if PennyLane is unavailable

Fusion model (EnhancedFSHybridClassifier):
  concat(256-dim classical, 40-dim quantum, 22-dim VLG/APG, 16-dim cross-attn)
  = 334-dim → classification head

Only used when ds_type == "fs".  BO and UAF continue using PennyLaneVQC.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Dimensions ─────────────────────────────────────────────────────────────────
N_QUBITS_FS   = 8     # enhanced circuit width
N_VAR_LAYERS  = 3     # variational layers per block
N_RERUNS      = 5     # data re-uploading rounds
# Output: N_QUBITS + (N_QUBITS - 1) ZZ correlators = 8 + 7 = 15 per round
# Total: 15 × 5 = 75 ... but we keep it to 40 for compatibility:
# 8 single-qubit + 7 ZZ = 15 per round; we use 8 rounds × 5 = 40.
QUANTUM_OUT_FS = 40   # 8 qubits × 5 rounds = 40-dim

# ── Optional PennyLane import ──────────────────────────────────────────────────
try:
    import pennylane as qml
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    logger.warning("PennyLane not available — EnhancedVQC using classical fallback")


# ── VQC circuit definition ────────────────────────────────────────────────────

def fs_vqc_circuit(dev, n_qubits: int = N_QUBITS_FS, n_var_layers: int = N_VAR_LAYERS):
    """
    Build and return a PennyLane QNode for the FS enhanced circuit.

    Circuit architecture:
      H^n → Data_encode(x) → Var_A (chain CNOT) → Data_encode(x) →
      Var_B (ring CNOT) → Data_encode(x) → Var_C (double-ring CNOT) →
      Measurements: ⟨Z_i⟩ for all i  +  ⟨Z_i Z_{i+1}⟩ for i < n-1
      (8 single + 7 ZZ = 15 observables per circuit)
    """
    if not _PL_AVAILABLE:
        return None

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, params_A, params_B, params_C):
        """
        x       : (n_qubits,) angle-encoded input features
        params_A: (n_var_layers, n_qubits, 2)  — chain-CNOT block
        params_B: (n_var_layers, n_qubits, 2)  — ring-CNOT block
        params_C: (n_var_layers, n_qubits, 2)  — double-ring block
        """
        n = n_qubits

        # ── Hadamard initialisation ────────────────────────────────────
        for k in range(n):
            qml.Hadamard(wires=k)

        # ── First data encoding + Var_A (chain CNOT) ──────────────────
        for k in range(n):
            qml.RY(x[k], wires=k)
            qml.RZ(x[k], wires=k)
        # Chain CNOT: 0→1→2→…→n-1
        for k in range(n - 1):
            qml.CNOT(wires=[k, k + 1])
        for layer in range(n_var_layers):
            for k in range(n):
                qml.RY(params_A[layer, k, 0], wires=k)
                qml.RZ(params_A[layer, k, 1], wires=k)

        # ── Second data encoding + Var_B (ring CNOT) ──────────────────
        for k in range(n):
            qml.RY(x[k], wires=k)
            qml.RZ(x[k], wires=k)
        # Ring CNOT: 0→1→…→n-1→0
        for k in range(n):
            qml.CNOT(wires=[k, (k + 1) % n])
        for layer in range(n_var_layers):
            for k in range(n):
                qml.RY(params_B[layer, k, 0], wires=k)
                qml.RZ(params_B[layer, k, 1], wires=k)

        # ── Third data encoding + Var_C (double-ring CNOT) ────────────
        for k in range(n):
            qml.RY(x[k], wires=k)
            qml.RZ(x[k], wires=k)
        # Double-ring: forward pass then backward pass
        for k in range(n):
            qml.CNOT(wires=[k, (k + 1) % n])
        for k in range(n - 1, -1, -1):
            qml.CNOT(wires=[k, (k - 1) % n])
        for layer in range(n_var_layers):
            for k in range(n):
                qml.RY(params_C[layer, k, 0], wires=k)
                qml.RZ(params_C[layer, k, 1], wires=k)

        # ── Measurements: single-qubit Z + ZZ correlators ─────────────
        single = [qml.expval(qml.PauliZ(k)) for k in range(n)]
        zz     = [qml.expval(qml.PauliZ(k) @ qml.PauliZ(k + 1))
                  for k in range(n - 1)]
        return single + zz   # 8 + 7 = 15 values

    return circuit


# ── EnhancedVQC ───────────────────────────────────────────────────────────────

class EnhancedVQC(nn.Module):
    """
    8-qubit VQC with ZZ correlators for FS classification.

    Input:  (B, 8) angle-encoded features from QAFA stage1
    Output: (B, QUANTUM_OUT_FS) = (B, 40)

    If PennyLane is unavailable, falls back to a classical MLP
    that produces the same output dimension.

    Data re-uploading: `n_reruns` rounds, each using the same circuit
    with a different input scaling; outputs are concatenated.
    """

    def __init__(
        self,
        n_qubits:    int = N_QUBITS_FS,
        n_var_layers: int = N_VAR_LAYERS,
        n_reruns:    int = N_RERUNS,
    ):
        super().__init__()
        self.n_qubits    = n_qubits
        self.n_var_layers = n_var_layers
        self.n_reruns    = n_reruns

        # Each re-upload round outputs 15 values (8+7); total = 15 × n_reruns
        # but we project to QUANTUM_OUT_FS=40 for clean downstream usage.
        raw_out = n_qubits + (n_qubits - 1)   # = 15

        if _PL_AVAILABLE:
            self.dev    = qml.device("default.qubit", wires=n_qubits)
            self.circuit = fs_vqc_circuit(self.dev, n_qubits, n_var_layers)

            self.params_A = nn.Parameter(
                torch.zeros(n_var_layers, n_qubits, 2).uniform_(-np.pi / 8, np.pi / 8))
            self.params_B = nn.Parameter(
                torch.zeros(n_var_layers, n_qubits, 2).uniform_(-np.pi / 8, np.pi / 8))
            self.params_C = nn.Parameter(
                torch.zeros(n_var_layers, n_qubits, 2).uniform_(-np.pi / 8, np.pi / 8))

            self.input_scale = nn.Parameter(torch.ones(n_qubits) * np.pi)
            self.bn          = nn.BatchNorm1d(n_qubits)
            # Project n_reruns × raw_out → QUANTUM_OUT_FS
            self.proj = nn.Linear(n_reruns * raw_out, QUANTUM_OUT_FS)
            self._use_quantum = True
        else:
            # Classical fallback: 8-dim input → 40-dim output
            self.fallback = nn.Sequential(
                nn.Linear(n_qubits, 64), nn.Tanh(),
                nn.Linear(64, 64),       nn.Tanh(),
                nn.Linear(64, QUANTUM_OUT_FS),
            )
            self._use_quantum = False

        logger.info(f"EnhancedVQC: {'quantum' if self._use_quantum else 'classical fallback'}  "
                    f"qubits={n_qubits}  reruns={n_reruns}  out={QUANTUM_OUT_FS}")

    def _run_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit for a single batch. x: (B, n_qubits)."""
        B = x.shape[0]
        if B > 1:
            x = torch.tanh(self.bn(x)) * self.input_scale
        else:
            x = torch.tanh(x) * self.input_scale

        results = self.circuit(x, self.params_A, self.params_B, self.params_C)
        # results: list of (n_qubits + n_qubits-1) tensors each (B,) or scalar
        if isinstance(results, (list, tuple)):
            out = torch.stack(results, dim=-1).float()  # (B, 15)
        else:
            out = results.float()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_qubits=8) QAFA features
        Returns: (B, QUANTUM_OUT_FS=40)
        """
        if not self._use_quantum:
            return self.fallback(x)

        # Re-uploading: run circuit n_reruns times with different input scaling
        rerun_outs = []
        for r in range(self.n_reruns):
            scale = (r + 1) / self.n_reruns   # 0.2, 0.4, ..., 1.0
            x_scaled = x * scale
            out = self._run_circuit(x_scaled)  # (B, 15)
            rerun_outs.append(out)

        # Concatenate and project: (B, 15×n_reruns) → (B, 40)
        concat = torch.cat(rerun_outs, dim=-1)   # (B, 75)
        return self.proj(concat)                  # (B, 40)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quantum-Classical Cross Attention ─────────────────────────────────────────

class QuantumClassicalCrossAttention(nn.Module):
    """
    2-head cross-attention between classical (256-dim) and quantum (40-dim) features.
    Output: 16-dim attended representation.

    The attention lets the classical embedding attend over quantum measurements,
    weighting quantum information by its relevance to the classical context.
    """

    def __init__(
        self,
        classical_dim: int = 256,
        quantum_dim:   int = QUANTUM_OUT_FS,
        n_heads:       int = 2,
        attn_out_dim:  int = 16,
    ):
        super().__init__()
        self.n_heads     = n_heads
        self.attn_out_dim = attn_out_dim
        head_dim = attn_out_dim // n_heads

        # Project classical → queries, quantum → keys/values
        self.q_proj = nn.Linear(classical_dim, attn_out_dim)
        self.k_proj = nn.Linear(quantum_dim,   attn_out_dim)
        self.v_proj = nn.Linear(quantum_dim,   attn_out_dim)
        self.out_proj = nn.Linear(attn_out_dim, attn_out_dim)
        self.norm    = nn.LayerNorm(attn_out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, classical: torch.Tensor, quantum: torch.Tensor) -> torch.Tensor:
        """
        classical: (B, 256)
        quantum:   (B, 40)
        Returns:   (B, 16)
        """
        Q = self.q_proj(classical)          # (B, 16)
        K = self.k_proj(quantum)            # (B, 16)
        V = self.v_proj(quantum)            # (B, 16)

        # Scaled dot-product (single "token" per modality → simple dot product)
        scale  = (self.attn_out_dim ** -0.5)
        attn   = torch.sigmoid(torch.sum(Q * K, dim=-1, keepdim=True) * scale)   # (B, 1)
        out    = attn * V                   # (B, 16)
        out    = self.dropout(self.out_proj(out))
        return self.norm(Q + out)           # (B, 16)  residual connection


# ── EnhancedFSHybridClassifier ────────────────────────────────────────────────

class EnhancedFSHybridClassifier(nn.Module):
    """
    Enhanced fusion model for FS classification.

    Input:
        classical : (B, 256)  — Stage 3 GAT full embedding
        quantum   : (B, 40)   — EnhancedVQC output
        extra     : (B, 22)   — VLG/APG + meta features (optional)
        meta      : (B, 15)   — FS handcrafted meta (optional; part of extra)

    Fusion:
        1. Cross-attention(classical, quantum) → (B, 16)
        2. concat(classical[256], quantum[40], extra[22], cross_attn[16]) = (B, 334)
        3. MLP head → logit

    If extra/meta are not provided, defaults to 256+40+16 = 312.
    """

    def __init__(
        self,
        classical_dim: int = 256,
        quantum_dim:   int = QUANTUM_OUT_FS,
        extra_dim:     int = 22,
        dropout:       float = 0.40,
        use_extra:     bool = True,
    ):
        super().__init__()
        self.use_extra  = use_extra
        attn_out        = 16
        fusion_dim      = classical_dim + quantum_dim + (extra_dim if use_extra else 0) + attn_out

        self.cross_attn = QuantumClassicalCrossAttention(
            classical_dim=classical_dim,
            quantum_dim=quantum_dim,
            attn_out_dim=attn_out,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, 1),
        )

    def forward(
        self,
        classical: torch.Tensor,
        quantum:   torch.Tensor,
        extra:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B,) logits."""
        attn_out = self.cross_attn(classical, quantum)          # (B, 16)

        parts = [classical, quantum, attn_out]
        if self.use_extra and extra is not None:
            parts.append(extra)
        fused = torch.cat(parts, dim=-1)

        return self.classifier(fused).squeeze(-1)               # (B,)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    Default: γ=2, α=0.5 (balanced classes in FS dataset).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = at * (1 - pt) ** self.gamma * bce
        return loss.mean()

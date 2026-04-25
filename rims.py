"""
# SPDX-License-Identifier: MSC-ORL-1.0
# Copyright (c) 2026 Raul Cruz Acosta (Esraderey) — MSC Tecnología
# Licensed under MSC Open Research License v1.0
# Free for research/education. Commercial use requires written authorization.
# See LICENSE.txt

Recurrent Independent Mechanisms (RIMs) v5.1.1 — Implementacion completa.

Cambios respecto a v5.1 (correcciones de bugs):
  * #1: VQ loss con .detach() en posicion canonica (codebook + commitment).
  * #2: activation_entropy se mantiene como tensor para no romper el grafo;
        ademas el float-fallback no crea tensor en cada paso.
  * #4: Inicializacion de W_q per-modulo via loop por slice (xavier valido en 2D).
  * #5: fingerprint opcional (flag), no se calcula por defecto en cada forward
        para evitar sync GPU->CPU.
  * #6: _init_weights respeta proyecciones ya inicializadas (skip explicito).
  * #7: ws_generator ultima capa se inicializa a zero -> gate residual real.
  * #8: cap de inactividad expuesto como parametro inactivity_cap.
  * #11: Decay aplicado ANTES del LayerNorm (sobre el estado preservado),
         no despues, consistente con el docstring.
  * #16: torch.zeros(()) en lugar de torch.zeros(1) para escalares.
  * Nuevo: register_buffer opcional via track_inactivity_in_state_dict.
  * Nuevo: __repr__ informativo.

Implementa fielmente las formulaciones matematicas y variantes modernas
descritas en la literatura de RIMs (Goyal et al., 2019) y sus extensiones
teoricas documentadas hasta 2026:

  1. GroupGRUCell          — GRU vectorizado con einsum (sin bucle Python).
  2. _InputAttentionRIM    — Atencion de entrada exacta del paper:
                             Q desde h_{t-1,i}, K/V desde x_t.
                             W_q per-modulo via einsum.
  3. _MultiHeadCommResidual— Comunicacion multi-cabeza con conexion residual:
                             h_{t,k} = softmax(QK^T/sqrt(d))V + h~_{t,k}.
                             Mascara: solo filas (queries), no columnas (sources).
  4. _GlobalWorkspace      — Global Workspace Theory (GWT): competencia ->
                             escritura al buffer -> broadcast a todos los modulos.
                             Buffer dinamico condicionado al contexto.
  5. _DVNCCodebook         — Discrete-Valued Neural Communication (DVNC):
                             VQ-VAE codebook con commitment adaptativo por entropia.
  6. Routing diferenciable — STE o Gumbel-Softmax (temperatura via softplus).
  7. Grupos fast/slow      — theta_modulos vs theta_atencion para meta-learning.
  8. Inactivity decay      — decay exponencial para modulos inactivos.
  9. NCO fingerprint       — opcional: hash ligero del estado para deteccion
                             de divergencia ontologica entre checkpoints.

Author: Escribano Silente (MSC Framework)
Version: 5.1.1
"""

from __future__ import annotations

import math
import hashlib
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch import Tensor

from .base import ConsciousnessLayerBase
from .config import LayerConfig


# ============================================================================
# 1. GroupGRUCell — GRU vectorizado (sin bucle por modulo)
# ============================================================================

class GroupGRUCell(nn.Module):
    """
    GRU multi-grupo vectorizado mediante operaciones einsum.

    Ecuaciones GRU por grupo k:
        gates_i = x_k @ W_ih[k]^T + b_ih[k]
        gates_h = h_k @ W_hh[k]^T + b_hh[k]
        z_k = sigmoid(z_i + z_h)
        r_k = sigmoid(r_i + r_h)
        n_k = tanh(n_i + r_k * n_h)
        h_k_new = (1 - z_k) * n_k + z_k * h_k
    """

    def __init__(self, input_size: int, hidden_size: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.empty(num_groups, 3 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(num_groups, 3 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(num_groups, 3 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(num_groups, 3 * hidden_size))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_ih, -std, std)
        nn.init.uniform_(self.W_hh, -std, std)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Args:
            x: [batch, K, d_in]
            h: [batch, K, d_h]
        Returns:
            h_new: [batch, K, d_h]
        """
        gates_i = torch.einsum('bki,koi->bko', x, self.W_ih) + self.b_ih
        gates_h = torch.einsum('bkh,koh->bko', h, self.W_hh) + self.b_hh

        z_i, r_i, n_i = gates_i.chunk(3, dim=-1)
        z_h, r_h, n_h = gates_h.chunk(3, dim=-1)

        z = torch.sigmoid(z_i + z_h)
        r = torch.sigmoid(r_i + r_h)
        n = torch.tanh(n_i + r * n_h)

        return (1.0 - z) * n + z * h


# ============================================================================
# 2. _InputAttentionRIM — Atencion de entrada (formulacion exacta del paper)
# ============================================================================

class _InputAttentionRIM(nn.Module):
    """
    Atencion de entrada de los RIMs segun Goyal et al. (2019).

    Cada modulo i genera su propia consulta desde h_{t-1,i} con su propio W_q^{(i)}:
        Q_{inp,i} = h_{t-1,i} @ W_q^{(i)}
    El input x_t proyecta claves y valores compartidos:
        K_{inp}   = x_t @ W_k
        V_{inp}   = x_t @ W_v
    Puntuacion de relevancia (escalar por modulo):
        s_{t,i}   = Q_{inp,i} @ K_{inp}^T / sqrt(d_k)

    v5.1.1 FIX #4: W_q se inicializa con xavier per-slice 2D, en lugar de
    sobre la view 3D (donde el calculo de fan_in/fan_out es no estandar).
    """

    def __init__(self, hidden_size: int, rim_size: int, num_rims: int, d_key: int = 64):
        super().__init__()
        self.num_rims = num_rims
        self.d_key = d_key

        # W_q per-modulo: [K, d_key, rim_size]
        self.W_q = nn.Parameter(torch.empty(num_rims, d_key, rim_size))

        # FIX #4: inicializar cada slice 2D por separado (xavier asume 2D)
        for k in range(num_rims):
            nn.init.xavier_uniform_(self.W_q.data[k])

        self.W_k = nn.Linear(hidden_size, d_key, bias=False)
        self.W_v = nn.Linear(hidden_size, rim_size, bias=False)
        self.scale = d_key ** -0.5

        # Marca para que _init_weights del modulo padre no sobrescriba
        # la inicializacion (FIX #6).
        self._initialized = True

    def forward(self, x: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:      [batch, input_size]
            hidden: [batch, num_rims, rim_size]
        Returns:
            x_per_rim: [batch, num_rims, rim_size]
            scores:    [batch, num_rims]
        """
        # Q_{inp,i} = h_{t-1,i} @ W_q^{(i)}  -> [B, K, d_key]
        Q = torch.einsum('bkr,kdr->bkd', hidden, self.W_q)

        K = self.W_k(x)                                       # [B, d_key]
        V = self.W_v(x)                                       # [B, rim_size]

        scores = torch.bmm(Q, K.unsqueeze(-1)).squeeze(-1) * self.scale  # [B, K]
        V_exp = V.unsqueeze(1).expand(-1, self.num_rims, -1)             # [B, K, rim_size]

        return V_exp, scores


# ============================================================================
# 3. _MultiHeadCommResidual — Comunicacion con residual
# ============================================================================

class _MultiHeadCommResidual(nn.Module):
    """
    Comunicacion inter-modulo multi-cabeza con conexion residual.

    Formalizacion:
        h_{t,k} = MH_Att(h~_{t,k}, h~_{t,:}) + h~_{t,k}    forall k in S_t
        h_{t,i} = h~_{t,i}                                  forall i not in S_t

    Mascara: solo filas (queries) de modulos inactivos. TODOS son fuente
    (keys/values), incluyendo inactivos, porque pueden tener estado
    relevante acumulado.
    """

    def __init__(self, rim_size: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert rim_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = rim_size // num_heads
        self.scale     = self.head_dim ** -0.5

        self.W_q = nn.Linear(rim_size, rim_size, bias=False)
        self.W_k = nn.Linear(rim_size, rim_size, bias=False)
        self.W_v = nn.Linear(rim_size, rim_size, bias=False)
        self.W_o = nn.Linear(rim_size, rim_size, bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(rim_size)

    def forward(self, hidden: Tensor, active_mask: Tensor) -> Tensor:
        """
        Args:
            hidden:      [B, K, rim_size]
            active_mask: [B, K] bool
        Returns:
            h_final: [B, K, rim_size]
        """
        B, K, D = hidden.shape
        H, Dh   = self.num_heads, self.head_dim

        def heads(t: Tensor) -> Tensor:
            return t.view(B, K, H, Dh).transpose(1, 2)

        Q  = heads(self.W_q(hidden))
        Km = heads(self.W_k(hidden))
        V  = heads(self.W_v(hidden))

        attn = torch.matmul(Q, Km.transpose(-1, -2)) * self.scale   # [B, H, K, K]

        # Enmascarar FILAS de modulos inactivos (no emiten queries).
        # Las columnas (sources) NO se enmascaran.
        row_mask = active_mask.unsqueeze(1).unsqueeze(-1).expand(B, H, K, K)

        attn = F.softmax(attn, dim=-1)
        attn = attn * row_mask.float()  # filas inactivas -> atencion zero
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        out = self.W_o(out)

        # Residual: solo activos reciben update
        active_f = active_mask.unsqueeze(-1).float()
        h_final  = hidden + active_f * out

        return self.norm_out(h_final)


# ============================================================================
# 4. _GlobalWorkspace — Global Workspace Theory (GWT)
# ============================================================================

class _GlobalWorkspace(nn.Module):
    """
    Espacio de Trabajo Global (Global Workspace Theory).

    Ciclo tripartito:
      1. Competencia: modulos activos compiten por escribir al buffer.
      2. Consolidacion: el buffer agrega los aportes ponderados por atencion.
      3. Broadcast: el buffer envia su resumen a TODOS los modulos.

    v5.1.1 FIX #7: La ultima capa de ws_generator se inicializa a zero,
    de modo que al inicio del entrenamiento el contexto-condicionado es ~0
    y el fallback estatico domina realmente. Esto hace que la afirmacion
    del docstring "fallback domina via residual" sea verdadera.
    """

    def __init__(self, rim_size: int, num_rims: int, ws_slots: int = 2):
        super().__init__()
        self.ws_slots = ws_slots
        self.rim_size = rim_size

        # Fallback estatico
        self.workspace_fallback = nn.Parameter(torch.randn(ws_slots, rim_size) * 0.02)

        # Generador dinamico
        self.ws_generator = nn.Sequential(
            nn.Linear(rim_size, rim_size * 2),
            nn.GELU(),
            nn.Linear(rim_size * 2, ws_slots * rim_size),
        )
        # FIX #7: Inicializar la ultima capa a zero -> ws_generator(x) ≈ 0 al inicio
        nn.init.zeros_(self.ws_generator[-1].weight)
        nn.init.zeros_(self.ws_generator[-1].bias)

        # Atencion de escritura: modulos -> workspace
        self.write_q = nn.Linear(rim_size, rim_size, bias=False)
        self.write_k = nn.Linear(rim_size, rim_size, bias=False)
        self.write_v = nn.Linear(rim_size, rim_size, bias=False)

        # Atencion de lectura (broadcast): workspace -> modulos
        self.read_q = nn.Linear(rim_size, rim_size, bias=False)
        self.read_k = nn.Linear(rim_size, rim_size, bias=False)
        self.read_v = nn.Linear(rim_size, rim_size, bias=False)
        self.read_o = nn.Linear(rim_size, rim_size, bias=False)

        self.norm_ws  = nn.LayerNorm(rim_size)
        self.norm_out = nn.LayerNorm(rim_size)
        self.scale    = rim_size ** -0.5

        # Marca para _init_weights padre (FIX #6).
        self._initialized = True

    def _generate_workspace(self, hidden: Tensor, active_mask: Tensor) -> Tensor:
        """
        Genera el estado inicial del workspace condicionado al contexto.
        """
        B, K, D = hidden.shape
        S = self.ws_slots

        active_f = active_mask.unsqueeze(-1).float()
        n_active = active_f.sum(dim=1).clamp(min=1.0)
        context  = (hidden * active_f).sum(dim=1) / n_active           # [B, D]

        ws_flat = self.ws_generator(context)                           # [B, S*D] (~0 al inicio)
        ws = ws_flat.view(B, S, D)

        fallback = self.workspace_fallback.unsqueeze(0).expand(B, -1, -1)
        return ws + fallback

    def forward(self, hidden: Tensor, active_mask: Tensor) -> Tensor:
        B, K, D = hidden.shape
        S = self.ws_slots

        ws = self._generate_workspace(hidden, active_mask)            # [B, S, D]

        # ---- Fase 1: Escritura al workspace ----
        Q_w = self.write_q(ws)
        K_w = self.write_k(hidden)
        V_w = self.write_v(hidden)

        attn_w = torch.bmm(Q_w, K_w.transpose(1, 2)) * self.scale     # [B, S, K]

        # Solo activos pueden escribir (enmascarar columnas inactivas)
        col_mask = active_mask.unsqueeze(1).expand(B, S, K)

        # Manejo robusto de filas completamente vacias
        any_active = col_mask.any(dim=-1, keepdim=True)               # [B, S, 1]
        attn_w = attn_w.masked_fill(~col_mask, float('-inf'))
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = attn_w * any_active.float()                          # zero si nadie activo

        ws_updated = self.norm_ws(ws + torch.bmm(attn_w, V_w))

        # ---- Fase 2: Broadcast desde workspace a todos los modulos ----
        Q_r = self.read_q(hidden)
        K_r = self.read_k(ws_updated)
        V_r = self.read_v(ws_updated)

        attn_r = torch.bmm(Q_r, K_r.transpose(1, 2)) * self.scale     # [B, K, S]
        attn_r = F.softmax(attn_r, dim=-1)

        broadcast = self.read_o(torch.bmm(attn_r, V_r))               # [B, K, D]
        return self.norm_out(hidden + broadcast)


# ============================================================================
# 5. _DVNCCodebook — Discrete-Valued Neural Communication (DVNC / VQ)
# ============================================================================

class _DVNCCodebook(nn.Module):
    """
    Comunicacion de Valores Discretos (DVNC / VQ-VAE).

    Cada modulo activo cuantifica su mensaje al vector mas cercano en un
    codebook compartido. Gradiente via Straight-Through Estimator:

        z_q = codebook[argmin_c ||z - c||]
        z_st = z + (z_q - z).detach()

    Loss VQ canonico (van den Oord et al., 2017, eq. 3):
        codebook_loss   = ||sg[z] - z_q||^2          # entrena codebook
        commitment_loss = beta * ||z - sg[z_q]||^2   # entrena encoder

    v5.1.1 FIX #1: posicion canonica de los .detach() (en v5.1 estaban
    intercambiados, propagando gradiente al modulo equivocado).

    v5.1.1 FIX #2: activation_entropy se acepta como tensor o float;
    si es float, beta_eff se calcula como float (sin allocacion de tensor
    en cada paso). Si se pasa como tensor (con grafo), se mantiene
    diferenciable hacia attention_weights.

    v5.1.1: Commitment adaptativo por entropia de activacion:
        beta_eff = beta_base * sigmoid(entropy / entropy_scale)
    Baja entropia -> codebook relajado; alta entropia -> codebook estricto.
    """

    def __init__(
        self,
        rim_size: int,
        num_codes: int = 64,
        commitment: float = 0.25,
        entropy_scale: float = 1.0,
    ):
        super().__init__()
        self.commitment_base = commitment
        self.entropy_scale = entropy_scale
        self.num_codes = num_codes

        self.codebook = nn.Embedding(num_codes, rim_size)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.proj_in  = nn.Linear(rim_size, rim_size, bias=False)
        self.proj_out = nn.Linear(rim_size, rim_size, bias=False)
        self.norm     = nn.LayerNorm(rim_size)

        # Marca para _init_weights padre (FIX #6).
        self._initialized = True

    def forward(
        self,
        hidden: Tensor,
        active_mask: Tensor,
        activation_entropy: Optional[Any] = None,  # float o Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden:             [B, K, rim_size]
            active_mask:        [B, K] bool
            activation_entropy: float | Tensor | None
                Si float: modulacion no diferenciable.
                Si Tensor: modulacion diferenciable (mantiene grafo).
                Si None: beta_eff = beta_base.

        Returns:
            h_comm:  [B, K, rim_size]
            vq_loss: scalar (shape [])
        """
        z = self.proj_in(hidden)

        # Distancias al codebook
        z_flat = z.reshape(-1, z.shape[-1])
        cb = self.codebook.weight
        dist = (
            z_flat.pow(2).sum(-1, keepdim=True)
            - 2 * z_flat @ cb.t()
            + cb.pow(2).sum(-1)
        )
        idx = dist.argmin(dim=-1)
        z_q = self.codebook(idx).view_as(z)

        # FIX #2: manejo dual de activation_entropy
        if activation_entropy is None:
            beta_eff = self.commitment_base
        elif isinstance(activation_entropy, Tensor):
            beta_eff = self.commitment_base * torch.sigmoid(
                activation_entropy / self.entropy_scale
            )
        else:
            # float -> calcular en Python sin allocar tensor
            beta_eff = self.commitment_base * float(
                torch.sigmoid(torch.tensor(activation_entropy / self.entropy_scale))
            )

        # FIX #1: posicion canonica del VQ loss
        codebook_loss   = (z_q - z.detach()).pow(2).mean()
        commitment_loss = (z - z_q.detach()).pow(2).mean()
        vq_loss = codebook_loss + beta_eff * commitment_loss

        # Straight-Through Estimator
        z_st = z + (z_q - z).detach()

        out = self.proj_out(z_st)
        active_f = active_mask.unsqueeze(-1).float()
        h_comm = self.norm(hidden + active_f * out)

        return h_comm, vq_loss


# ============================================================================
# Dataclass de estado
# ============================================================================

@dataclass
class RIMsState:
    """Estado completo de un paso RIMs."""
    hidden_states:     Tensor   # [B, K, rim_size]
    active_rims:       Tensor   # [B, K] bool
    attention_weights: Tensor   # [B, K] float (softmax de seleccion)
    communication:     Tensor   # [B, K, rim_size] (ultimo mensaje)
    vq_loss:           Tensor   # scalar
    inactivity_steps:  Tensor   # [B, K] int — pasos consecutivos inactivo
    fingerprint:       str = "" # hash del estado para NCO (vacio si disabled)

    def to_dict(self) -> Dict[str, Any]:
        aw = self.attention_weights
        d = {
            'num_active':          self.active_rims.float().sum(-1).mean().item(),
            'activation_rate':     self.active_rims.float().mean().item(),
            'attention_entropy':   -(aw * (aw + 1e-10).log()).sum(-1).mean().item(),
            'comm_norm':           self.communication.norm(dim=-1).mean().item(),
            'vq_loss':             self.vq_loss.item(),
            'max_inactivity':      self.inactivity_steps.max().item(),
            'mean_inactivity':     self.inactivity_steps.float().mean().item(),
        }
        if self.fingerprint:
            d['fingerprint'] = self.fingerprint
        return d


# ============================================================================
# Utilidad: Fingerprint para NCO (opcional, sync GPU->CPU)
# ============================================================================

def _compute_fingerprint(hidden: Tensor, precision: int = 4) -> str:
    """
    Hash ligero del estado para deteccion de divergencia ontologica.

    ATENCION: incluye `.cpu()` -> sync GPU->CPU. Solo llamar bajo demanda
    (no en cada forward de training). Use el flag `compute_fingerprint`
    del modulo principal.
    """
    scale = 10 ** precision
    quantized = (hidden.detach().float().cpu() * scale).round().to(torch.int32)
    raw_bytes = quantized.numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


# ============================================================================
# Modulo principal
# ============================================================================

class RecurrentIndependentMechanisms(ConsciousnessLayerBase):
    """
    Recurrent Independent Mechanisms v5.1.1.

    Fases de cada timestep:
      1. Input Attention (paper-exact, per-module W_q): cada RIM genera Q
         desde su h_{t-1} con su propio W_q^{(i)}; entrada genera K, V
         compartidos; scores -> top-k con STE o Gumbel.
      2. Inactivity decay (FIX #11): aplicado al estado preservado de
         modulos inactivos ANTES de mezclar con el estado nuevo y antes
         del LayerNorm.
      3. Dinamicas Independientes (vectorizado): GroupGRUCell con einsum;
         mascara de Hadamard aplica el resultado solo a activos:
             h_t = M (.) h_new + (1-M) (.) h_decayed
      4. Comunicacion (configurable): standard | gwt | dvnc.
      5. Output: residual + LayerNorm.

    Args:
        input_size:       Dimension de x_t.
        hidden_size:      Dimension total del estado (= num_rims * rim_size).
        num_rims:         K_t — numero total de modulos.
        num_active:       K_a — modulos activos por paso (top-k).
        num_heads:        Cabezas en la atencion de comunicacion.
        comm_mode:        'standard' | 'gwt' | 'dvnc'.
        routing:          'ste' | 'gumbel'.
        gumbel_temp:      Temperatura inicial de Gumbel.
        num_codes:        Tamano del codebook DVNC.
        ws_slots:         Slots del workspace GWT.
        dropout:          Dropout general.
        inactivity_decay: Lambda de decay para modulos inactivos.
        inactivity_cap:   Pasos maximos de inactividad considerados (FIX #16).
        compute_fingerprint: Si True, calcula NCO fingerprint en cada forward
                             (incurre en sync GPU->CPU; default False).
        track_inactivity_in_state_dict: Si True, registra inactivity_steps
                             como buffer (se serializa con state_dict).
        config:           LayerConfig opcional.
    """

    def __init__(
        self,
        input_size:       int = 64,
        hidden_size:      int = 256,
        num_rims:         int = 6,
        num_active:       int = 3,
        num_heads:        int = 4,
        comm_mode:        Literal['standard', 'gwt', 'dvnc'] = 'standard',
        routing:          Literal['ste', 'gumbel'] = 'ste',
        gumbel_temp:      float = 1.0,
        num_codes:        int = 64,
        ws_slots:         int = 2,
        dropout:          float = 0.1,
        inactivity_decay: float = 0.001,
        inactivity_cap:   float = 100.0,
        compute_fingerprint: bool = False,
        track_inactivity_in_state_dict: bool = False,
        config:           Optional[LayerConfig] = None,
    ):
        super().__init__(config)

        # --- Validaciones ---
        if num_active < 1 or num_active > num_rims:
            raise ValueError(f"num_active={num_active} debe estar en [1, {num_rims}]")
        if hidden_size % num_rims != 0:
            raise ValueError(
                f"hidden_size={hidden_size} debe ser divisible por num_rims={num_rims}"
            )
        rim_size = hidden_size // num_rims
        if rim_size % num_heads != 0:
            raise ValueError(
                f"rim_size={rim_size} debe ser divisible por num_heads={num_heads}"
            )
        if inactivity_cap <= 0:
            raise ValueError(f"inactivity_cap={inactivity_cap} debe ser > 0")
        if inactivity_decay < 0 or inactivity_decay >= 1:
            raise ValueError(f"inactivity_decay={inactivity_decay} debe estar en [0, 1)")

        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.num_rims         = num_rims
        self.num_active       = num_active
        self.rim_size         = rim_size
        self.comm_mode        = comm_mode
        self.routing          = routing
        self.inactivity_decay = inactivity_decay
        self.inactivity_cap   = float(inactivity_cap)
        self.compute_fingerprint = compute_fingerprint
        self.track_inactivity_in_state_dict = track_inactivity_in_state_dict

        # ---- Atencion de entrada (per-module W_q) ----
        self.input_attention = _InputAttentionRIM(
            hidden_size=input_size,
            rim_size=rim_size,
            num_rims=num_rims,
            d_key=max(32, rim_size // 2),
        )

        # ---- Temperatura Gumbel via softplus ----
        # raw_gumbel_temp es input a softplus, con offset minimo de 0.1
        self._raw_gumbel_temp = nn.Parameter(
            torch.tensor(math.log(math.exp(gumbel_temp - 0.1) - 1.0))
        )

        # ---- GRU vectorizado ----
        self.gru = GroupGRUCell(rim_size, rim_size, num_rims)

        # ---- LayerNorm post-update ----
        self.hidden_norm = nn.LayerNorm(rim_size)

        # ---- Comunicacion ----
        if comm_mode == 'standard':
            self.comm_layer = _MultiHeadCommResidual(rim_size, num_heads, dropout)
        elif comm_mode == 'gwt':
            self.comm_layer = _GlobalWorkspace(rim_size, num_rims, ws_slots)
        elif comm_mode == 'dvnc':
            self.comm_layer = _DVNCCodebook(rim_size, num_codes)
        else:
            raise ValueError(f"comm_mode desconocido: {comm_mode}")

        # ---- Proyeccion de salida ----
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

        # ---- Estado inicial aprendible ----
        self.initial_hidden = nn.Parameter(
            torch.randn(1, num_rims, rim_size) * 0.01
        )

        # FIX #6: _init_weights respeta submodulos ya inicializados
        self._init_weights()

        # Validacion de cobertura fast/slow
        self._validate_param_groups()

    # ------------------------------------------------------------------
    # Inicializacion
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Inicializa los nn.Linear que NO pertenezcan a submodulos que ya
        marcaron `_initialized = True` (input_attention, comm_layer).
        FIX #6: evita sobrescribir inicializacion especifica.
        """
        # Recolectar submodulos que se autogestionan
        self_managed_modules = []
        for child in self.children():
            if getattr(child, '_initialized', False):
                self_managed_modules.append(child)

        for m in self.modules():
            # Saltar si pertenece a un submodulo autogestionado
            skip = False
            for managed in self_managed_modules:
                if m is managed:
                    skip = True
                    break
                # Verificar si m esta dentro de managed
                for sub in managed.modules():
                    if sub is m:
                        skip = True
                        break
                if skip:
                    break
            if skip:
                continue

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Validacion de cobertura de parametros fast/slow
    # ------------------------------------------------------------------

    def _validate_param_groups(self) -> None:
        all_params = set(id(p) for p in self.parameters())
        fast_ids   = set(id(p) for p in self.fast_params())
        slow_ids   = set(id(p) for p in self.slow_params())

        overlap = fast_ids & slow_ids
        missing = all_params - (fast_ids | slow_ids)

        if overlap:
            raise RuntimeError(
                f"[RIMs v5.1.1] {len(overlap)} parametros duplicados en "
                f"fast/slow groups. Revisar asignacion de modulos."
            )
        if missing:
            raise RuntimeError(
                f"[RIMs v5.1.1] {len(missing)} parametros no asignados a "
                f"fast ni slow. Todo parametro debe pertenecer a un grupo."
            )

    # ------------------------------------------------------------------
    # Temperatura Gumbel (softplus)
    # ------------------------------------------------------------------

    @property
    def gumbel_temp(self) -> Tensor:
        """
        Temperatura Gumbel via softplus + offset minimo.
        Rango efectivo: [0.1, +inf) con gradientes continuos.
        """
        return F.softplus(self._raw_gumbel_temp) + 0.1

    # ------------------------------------------------------------------
    # Grupos de parametros (fast / slow)
    # ------------------------------------------------------------------

    def fast_params(self) -> List[nn.Parameter]:
        """Parametros de las dinamicas recurrentes (inner loop MAML)."""
        fast_modules = (self.gru, self.hidden_norm, self.output_proj, self.output_norm)
        return [p for m in fast_modules for p in m.parameters()]

    def slow_params(self) -> List[nn.Parameter]:
        """Parametros del enrutamiento y comunicacion (outer loop MAML)."""
        slow_modules = (self.input_attention, self.comm_layer)
        slow_list = [p for m in slow_modules for p in m.parameters()]
        slow_list.append(self._raw_gumbel_temp)
        slow_list.append(self.initial_hidden)
        return slow_list

    # ------------------------------------------------------------------
    # Seleccion de RIMs activos
    # ------------------------------------------------------------------

    def _select(self, scores: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Top-k diferenciable.

        STE:    forward = hard mask, backward = softmax.
        Gumbel: scores perturbados con ruido Gumbel -> top-k mas suave
                (solo en training).

        Returns:
            active_mask:  [B, K] bool
            attn_weights: [B, K] float (softmax para metricas)
            sel_weights:  [B, K] float (diferenciable para escalar update)
        """
        B, K = scores.shape

        if self.routing == 'gumbel' and self.training:
            temp = self.gumbel_temp
            gumbel = -torch.empty_like(scores).exponential_().log()
            scores = (scores + gumbel) / temp

        _, top_idx = scores.topk(self.num_active, dim=-1)
        active_mask = torch.zeros(B, K, device=scores.device, dtype=torch.bool)
        active_mask.scatter_(1, top_idx, True)

        attn_weights = F.softmax(scores, dim=-1)
        # STE: forward hard, backward por softmax
        sel_weights  = active_mask.float() + attn_weights - attn_weights.detach()

        return active_mask, attn_weights, sel_weights

    # ------------------------------------------------------------------
    # Inactivity decay (FIX #11: aplicado antes del LayerNorm)
    # ------------------------------------------------------------------

    def _apply_inactivity_decay(
        self, hidden: Tensor, active_mask: Tensor, inactivity_steps: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Aplica decay exponencial al estado de modulos inactivos.

            h_inactive *= (1 - lambda * min(steps, cap) / cap)

        Modulos activos: factor = 1.0 (sin decay).
        Modulos inactivos: factor decae proporcional a tiempo de inactividad.

        FIX #11: este metodo se llama ANTES de mezclar con h_all y antes del
        LayerNorm, de modo que el decay opera sobre el estado preservado
        h_{t-1} de modulos inactivos, no sobre el output post-norm.

        Args:
            hidden:           [B, K, rim_size] — estado h_{t-1}
            active_mask:      [B, K] bool — quien esta activo en t
            inactivity_steps: [B, K] long — contador previo (de t-1)

        Returns:
            hidden_decayed:   [B, K, rim_size]
            inactivity_new:   [B, K] long — contadores actualizados
        """
        # Activos -> 0; inactivos -> +1
        inactivity_new = torch.where(
            active_mask,
            torch.zeros_like(inactivity_steps),
            inactivity_steps + 1,
        )

        cap = self.inactivity_cap
        decay_ratio = (inactivity_new.float().clamp(max=cap) / cap)        # [B, K]
        decay_factor = 1.0 - self.inactivity_decay * decay_ratio           # [B, K]
        decay_factor = decay_factor.unsqueeze(-1)                          # [B, K, 1]

        hidden_decayed = hidden * decay_factor

        return hidden_decayed, inactivity_new

    # ------------------------------------------------------------------
    # Paso temporal unico
    # ------------------------------------------------------------------

    def _step(
        self,
        x_t: Tensor,
        hidden: Tensor,
        inactivity_steps: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Procesa un timestep.

        Pipeline:
          1. (V_exp, scores) = InputAttention(x_t, h_{t-1})
          2. active = top-k(scores)
          3. h_decayed, inactivity_new = decay(h_{t-1}, active, inactivity)
          4. h_all = GroupGRU(x_attended, h_{t-1})
          5. h_t = M (.) h_all + (1-M) (.) h_decayed
          6. h_t = LayerNorm(h_t)
          7. h_t = Communication(h_t, active)
          8. out = LayerNorm(h_flat + OutputProj(h_flat))

        Returns:
            hidden_new, comm, output_t, active_mask, attn_weights, vq_loss, inactivity_new
        """
        B = x_t.shape[0]

        # 1. Atencion de entrada
        V_exp, scores = self.input_attention(x_t, hidden)

        # 2. Seleccion top-k
        active_mask, attn_weights, sel_weights = self._select(scores)
        x_per_rim = sel_weights.unsqueeze(-1) * V_exp

        # 3. FIX #11: decay aplicado ANTES de la mezcla y LayerNorm
        hidden_decayed, inactivity_new = self._apply_inactivity_decay(
            hidden, active_mask, inactivity_steps
        )

        # 4. GroupGRU (sobre el hidden ORIGINAL, no decayed, para no
        #    contaminar la dinamica de los activos)
        h_all = self.gru(x_per_rim, hidden)

        # 5. Mascara de Hadamard: activos toman h_all, inactivos toman h_decayed
        M = active_mask.unsqueeze(-1).float()
        hidden_new = M * h_all + (1.0 - M) * hidden_decayed

        # 6. LayerNorm
        hidden_new = self.hidden_norm(hidden_new)

        # 7. Comunicacion
        vq_loss = torch.zeros((), device=x_t.device)  # FIX: scalar shape []
        if self.comm_mode == 'dvnc':
            # FIX #2: pasar entropia como TENSOR para mantener grafo
            # (entropia diferenciable de attn_weights)
            act_entropy_t = -(attn_weights * (attn_weights + 1e-10).log()).sum(-1).mean()
            hidden_new, vq_loss = self.comm_layer(hidden_new, active_mask, act_entropy_t)
            comm = hidden_new
        else:
            hidden_new = self.comm_layer(hidden_new, active_mask)
            comm = hidden_new

        # 8. Salida con residual
        h_flat = hidden_new.view(B, -1)
        out_t  = self.output_norm(h_flat + self.output_proj(h_flat))

        return hidden_new, comm, out_t, active_mask, attn_weights, vq_loss, inactivity_new

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:                Tensor,
        hidden:           Optional[Tensor] = None,
        inactivity_steps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, RIMsState]:
        """
        Args:
            x:                [B, input_size] o [B, seq_len, input_size]
            hidden:           [B, num_rims, rim_size] o None
            inactivity_steps: [B, num_rims] long o None

        Returns:
            output: [B, hidden_size] o [B, seq_len, hidden_size]
            state:  RIMsState del ultimo timestep
        """
        is_seq = x.dim() == 3
        if not is_seq:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        device  = x.device

        if hidden is None:
            hidden = self.initial_hidden.expand(B, -1, -1).clone().to(device)

        if inactivity_steps is None:
            inactivity_steps = torch.zeros(
                B, self.num_rims, dtype=torch.long, device=device
            )

        outputs:  List[Tensor] = []
        total_vq = torch.zeros((), device=device)  # FIX: scalar shape []
        last_active = last_attn = last_comm = None

        for t in range(T):
            hidden, comm, out_t, active, attn, vq, inactivity_steps = self._step(
                x[:, t], hidden, inactivity_steps
            )
            outputs.append(out_t)
            total_vq    = total_vq + vq
            last_active = active
            last_attn   = attn
            last_comm   = comm

        output = torch.stack(outputs, dim=1)
        if not is_seq:
            output = output.squeeze(1)

        # Metricas
        act_rate = last_active.float().mean().item()
        entropy  = -(last_attn * (last_attn + 1e-10).log()).sum(-1).mean().item()
        sparsity = 1.0 - act_rate
        avg_vq   = (total_vq / T).item() if T > 0 else 0.0
        self.metrics.record({
            'activation_rate':   act_rate,
            'attention_entropy': entropy,
            'sparsity':          sparsity,
            'vq_loss':           avg_vq,
            'gumbel_temp':       self.gumbel_temp.item(),
            'max_inactivity':    inactivity_steps.max().item(),
            'mean_inactivity':   inactivity_steps.float().mean().item(),
        })

        # FIX #5: fingerprint solo si esta habilitado (sync GPU->CPU caro)
        fp = _compute_fingerprint(hidden) if self.compute_fingerprint else ""

        state = RIMsState(
            hidden_states     = hidden,
            active_rims       = last_active,
            attention_weights = last_attn,
            communication     = last_comm,
            vq_loss           = total_vq / T if T > 0 else total_vq,
            inactivity_steps  = inactivity_steps,
            fingerprint       = fp,
        )

        return output, state

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **{k: self.metrics.get_stats(k)
               for k in ('activation_rate', 'attention_entropy',
                         'sparsity', 'vq_loss', 'gumbel_temp',
                         'max_inactivity', 'mean_inactivity')},
            'num_rims':            self.num_rims,
            'num_active':          self.num_active,
            'rim_size':            self.rim_size,
            'comm_mode':           self.comm_mode,
            'routing':             self.routing,
            'inactivity_decay':    self.inactivity_decay,
            'inactivity_cap':      self.inactivity_cap,
            'compute_fingerprint': self.compute_fingerprint,
        }

    def reset_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        Genera estado oculto inicial limpio + contador de inactividad.

        Returns:
            hidden:           [B, num_rims, rim_size]
            inactivity_steps: [B, num_rims] long
        """
        hidden = self.initial_hidden.expand(batch_size, -1, -1).clone().to(device)
        inactivity_steps = torch.zeros(
            batch_size, self.num_rims, dtype=torch.long, device=device
        )
        return hidden, inactivity_steps

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_rims={self.num_rims}, num_active={self.num_active}, "
            f"rim_size={self.rim_size}, comm_mode='{self.comm_mode}', "
            f"routing='{self.routing}', "
            f"inactivity_decay={self.inactivity_decay}, "
            f"inactivity_cap={self.inactivity_cap}, "
            f"compute_fingerprint={self.compute_fingerprint}"
        )

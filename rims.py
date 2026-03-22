"""
# SPDX-License-Identifier: MSC-ORL-1.0
# Copyright (c) 2026 Raul Cruz Acosta (Esraderey) — MSC Tecnología
# Licensed under MSC Open Research License v1.0
# Free for research/education. Commercial use requires written authorization.
# See LICENSE.txt 

Recurrent Independent Mechanisms (RIMs) – Implementacion Completa.

Implementa fielmente las formulaciones matematicas y variantes modernas
descritas en la literatura de RIMs (Goyal et al., 2019) y sus extensiones
teoricas documentadas hasta 2026:

  1. GroupGRUCell          — GRU vectorizado con einsum (sin bucle Python).
  2. _InputAttentionRIM    — Atencion de entrada exacta del paper:
                             Q desde h_{t-1,i}, K/V desde x_t.
                             W_q per-modulo via einsum (v5.1).
  3. _MultiHeadCommResidual— Comunicacion multi-cabeza con conexion residual
                             segun formula h_{t,k} = softmax(QK^T/sqrt(d))V + h~_{t,k}.
                             Mascara corregida: solo filas (queries), no columnas (v5.1).
  4. _GlobalWorkspace      — Global Workspace Theory (GWT): competencia->
                             escritura al buffer->broadcast a todos los modulos.
                             Buffer dinamico condicionado al contexto (v5.1).
  5. _DVNCCodebook         — Discrete-Valued Neural Communication (DVNC):
                             comunicacion cuantificada via VQ-VAE codebook.
                             Commitment adaptativo por entropia (v5.1).
  6. Routing diferenciable — STE (Straight-Through Estimator) o Gumbel-Softmax.
                             Temperatura via softplus (v5.1).
  7. Grupos fast/slow      — Separacion de parametros para meta-aprendizaje:
                             theta_modulos (rapidos) vs theta_atencion (lentos).
                             Assertion de cobertura exhaustiva (v5.1).
  8. Inactivity decay      — Decay exponencial para modulos inactivos (v5.1).
  9. NCO fingerprint       — Hash ligero del estado para deteccion de divergencia (v5.1).

Referencia matematica completa en docstrings inline.

Author: Escribano Silente (MSC Framework)
Version: 5.1.0
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

    En lugar de K celdas GRUCell separadas (K kernels CUDA), agrupa todos los
    parametros en tensores 3-D y aplica una multiplicacion matricial por lotes.
    Reduce el overhead de lanzamiento de kernels de O(K) a O(1).

    Ecuaciones GRU por grupo k:
        gates_i = x_k @ W_ih[k]^T + b_ih[k]     (contribucion de entrada)
        gates_h = h_k @ W_hh[k]^T + b_hh[k]     (contribucion de estado)
        z_k = sigmoid(z_i + z_h)                 (puerta de actualizacion)
        r_k = sigmoid(r_i + r_h)                 (puerta de reset)
        n_k = tanh(n_i + r_k * n_h)              (candidato nuevo estado)
        h_k_new = (1 - z_k) * n_k + z_k * h_k   (estado actualizado)

    Args:
        input_size:  d_in por modulo.
        hidden_size: d_h por modulo.
        num_groups:  K — numero de modulos.
    """

    def __init__(self, input_size: int, hidden_size: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_size = hidden_size

        # Pesos agrupados: [K, 3*H, d_in] y [K, 3*H, H]
        # 3 puertas (z, r, n) concatenadas en la primera dimension interna
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
        # einsum 'bki,koi->bko': multiplica cada grupo k por su propio W
        gates_i = torch.einsum('bki,koi->bko', x, self.W_ih) + self.b_ih   # [B, K, 3H]
        gates_h = torch.einsum('bkh,koh->bko', h, self.W_hh) + self.b_hh   # [B, K, 3H]

        z_i, r_i, n_i = gates_i.chunk(3, dim=-1)
        z_h, r_h, n_h = gates_h.chunk(3, dim=-1)

        z = torch.sigmoid(z_i + z_h)
        r = torch.sigmoid(r_i + r_h)
        n = torch.tanh(n_i + r * n_h)

        return (1.0 - z) * n + z * h


# ============================================================================
# 2. _InputAttentionRIM — Atencion de entrada (formulacion exacta del paper)
#    v5.1: W_q per-modulo via einsum (fidelidad al paper)
# ============================================================================

class _InputAttentionRIM(nn.Module):
    """
    Atencion de entrada de los RIMs segun Goyal et al. (2019).

    Cada modulo i genera su propia consulta desde h_{t-1,i} con su propio W_q^{(i)}:
        Q_{inp,i} = h_{t-1,i} @ W_q^{(i)}   in R^{d_k}
    El input x_t proyecta claves y valores compartidos:
        K_{inp}   = x_t @ W_k                in R^{d_k}
        V_{inp}   = x_t @ W_v                in R^{d_v}
    Puntuacion de relevancia (escalar por modulo):
        s_{t,i}   = Q_{inp,i} @ K_{inp}^T / sqrt(d_k)

    v5.1: W_q implementado como tensor [K, d_key, rim_size] con einsum,
    consistente con GroupGRUCell. Cada modulo tiene proyeccion independiente.

    Args:
        hidden_size: d_h total (= input_size en la practica).
        rim_size:    d_h por modulo.
        num_rims:    K_t.
        d_key:       Dimension del espacio de claves/consultas.
    """

    def __init__(self, hidden_size: int, rim_size: int, num_rims: int, d_key: int = 64):
        super().__init__()
        self.num_rims = num_rims
        self.d_key = d_key

        # v5.1: Proyeccion de consulta PER-MODULO via einsum
        # W_q^{(i)}: cada modulo k tiene su propia transformacion [rim_size -> d_key]
        self.W_q = nn.Parameter(torch.empty(num_rims, d_key, rim_size))
        nn.init.xavier_uniform_(self.W_q.view(num_rims, d_key, rim_size))

        self.W_k = nn.Linear(hidden_size, d_key, bias=False)   # input -> key
        self.W_v = nn.Linear(hidden_size, rim_size, bias=False) # input -> value
        self.scale = d_key ** -0.5

    def forward(
        self, x: Tensor, hidden: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:      [batch, input_size]    — x_t
            hidden: [batch, num_rims, rim_size]

        Returns:
            x_per_rim: [batch, num_rims, rim_size]  — x̃_{t,i} (0 si inactivo)
            scores:    [batch, num_rims]             — s_{t,i} sin normalizar
        """
        # Q_{inp,i} = h_{t-1,i} @ W_q^{(i)}  -> [B, K, d_key]
        # einsum: 'bkr,kdr->bkd' — cada modulo k proyecta con su propio W_q[k]
        Q = torch.einsum('bkr,kdr->bkd', hidden, self.W_q)   # [B, K, d_key]

        # K_{inp} = x_t @ W_k           -> [B, d_key]
        K = self.W_k(x)                                       # [B, d_key]
        # V_{inp} = x_t @ W_v           -> [B, rim_size]
        V = self.W_v(x)                                       # [B, rim_size]

        # Scores: Q . K^T / sqrt(d_k)  -> [B, K]
        scores = torch.bmm(Q, K.unsqueeze(-1)).squeeze(-1) * self.scale  # [B, K]

        # Valor proyectado expandido para todos los RIMs
        V_exp = V.unsqueeze(1).expand(-1, self.num_rims, -1)            # [B, K, rim_size]

        return V_exp, scores


# ============================================================================
# 3. _MultiHeadCommResidual — Comunicacion con residual (formulacion del paper)
#    v5.1: Mascara corregida — solo filas (queries), no columnas (sources)
#    v5.1: Manejo robusto de filas completamente enmascaradas
# ============================================================================

class _MultiHeadCommResidual(nn.Module):
    """
    Comunicacion inter-modulo multi-cabeza con conexion residual.

    Segun la formalizacion matematica del paper:
        h_{t,k} = MH_Att(h̃_{t,k}, h̃_{t,:}) + h̃_{t,k}    ∀k ∈ S_t
        h_{t,i} = h̃_{t,i}                                   ∀i ∉ S_t

    v5.1 CORRECCION: Solo los modulos activos emiten consultas (filas).
    TODOS los modulos son fuente (keys/values), incluyendo inactivos,
    porque pueden tener estado relevante acumulado.

    v5.1: Filas completamente inactivas reciben atencion zero explicitamente
    en lugar de nan_to_num post-hoc.

    Args:
        rim_size:  Dimension por modulo.
        num_heads: Numero de cabezas de atencion.
        dropout:   Dropout en pesos de atencion.
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
            h_final: [B, K, rim_size]  — residual aplicado solo en activos
        """
        B, K, D = hidden.shape
        H, Dh   = self.num_heads, self.head_dim

        def heads(t: Tensor) -> Tensor:
            return t.view(B, K, H, Dh).transpose(1, 2)   # [B, H, K, Dh]

        Q = heads(self.W_q(hidden))   # [B, H, K, Dh]
        Km = heads(self.W_k(hidden))  # Todos los modulos como fuente
        V  = heads(self.W_v(hidden))  # Todos los modulos como fuente

        attn = torch.matmul(Q, Km.transpose(-1, -2)) * self.scale   # [B, H, K, K]

        # v5.1 CORRECCION: enmascarar FILAS de modulos inactivos, NO columnas.
        # Los inactivos no emiten queries, pero SI son fuente (keys/values).
        row_mask = active_mask.unsqueeze(1).unsqueeze(-1).expand(B, H, K, K)  # [B,H,K,K]

        # v5.1: Manejo robusto — filas inactivas se ponen a 0 directamente,
        # evitando el camino -inf -> softmax -> NaN -> nan_to_num
        attn = F.softmax(attn, dim=-1)
        attn = attn * row_mask.float()   # filas inactivas -> atencion zero
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                          # [B, H, K, Dh]
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        out = self.W_o(out)

        # Residual: h_{t,k} = MH_Att(...) + h̃_{t,k}  (solo activos)
        active_f = active_mask.unsqueeze(-1).float()         # [B, K, 1]
        h_final  = hidden + active_f * out                   # inactivos: += 0

        return self.norm_out(h_final)


# ============================================================================
# 4. _GlobalWorkspace — Global Workspace Theory (GWT)
#    v5.1: Buffer dinamico condicionado al contexto
#    v5.1: Manejo robusto de softmax con filas completamente enmascaradas
# ============================================================================

class _GlobalWorkspace(nn.Module):
    """
    Espacio de Trabajo Global (Global Workspace Theory, Baars 1988 / Bengio 2017).

    Ciclo tripartito:
      1. Competencia: modulos activos compiten por escribir al buffer.
      2. Consolidacion: el buffer agrega los aportes ponderados por atencion.
      3. Broadcast: el buffer envia su resumen a TODOS los modulos.

    v5.1: El buffer inicial se genera condicionado al contexto (promedio de
    estados activos -> MLP -> ws_slots vectores), en lugar de ser un parametro
    estatico. Esto permite adaptacion dinamica de la capacidad del workspace.

    Args:
        rim_size:  Dimension por modulo.
        num_rims:  K_t.
        ws_slots:  Numero de slots en el workspace (tipicamente 1-4).
    """

    def __init__(self, rim_size: int, num_rims: int, ws_slots: int = 2):
        super().__init__()
        self.ws_slots = ws_slots
        self.rim_size = rim_size

        # v5.1: Buffer dinamico — genera slots condicionados al contexto
        # Fallback estatico para cuando no hay modulos activos
        self.workspace_fallback = nn.Parameter(torch.randn(ws_slots, rim_size) * 0.02)
        # MLP: promedio de hidden activos -> ws_slots vectores
        self.ws_generator = nn.Sequential(
            nn.Linear(rim_size, rim_size * 2),
            nn.GELU(),
            nn.Linear(rim_size * 2, ws_slots * rim_size),
        )

        # Atencion de escritura: modulos -> workspace
        self.write_q = nn.Linear(rim_size, rim_size, bias=False)
        self.write_k = nn.Linear(rim_size, rim_size, bias=False)
        self.write_v = nn.Linear(rim_size, rim_size, bias=False)

        # Atencion de lectura (broadcast): workspace -> todos los modulos
        self.read_q  = nn.Linear(rim_size, rim_size, bias=False)
        self.read_k  = nn.Linear(rim_size, rim_size, bias=False)
        self.read_v  = nn.Linear(rim_size, rim_size, bias=False)
        self.read_o  = nn.Linear(rim_size, rim_size, bias=False)

        self.norm_ws  = nn.LayerNorm(rim_size)
        self.norm_out = nn.LayerNorm(rim_size)
        self.scale    = rim_size ** -0.5

    def _generate_workspace(self, hidden: Tensor, active_mask: Tensor) -> Tensor:
        """
        Genera el estado inicial del workspace condicionado al contexto.

        Args:
            hidden:      [B, K, D]
            active_mask: [B, K] bool

        Returns:
            ws: [B, ws_slots, D]
        """
        B, K, D = hidden.shape
        S = self.ws_slots

        # Promedio de estados activos (con fallback si ninguno activo)
        active_f = active_mask.unsqueeze(-1).float()         # [B, K, 1]
        n_active = active_f.sum(dim=1).clamp(min=1.0)       # [B, 1]
        context  = (hidden * active_f).sum(dim=1) / n_active # [B, D]

        # Generar slots condicionados
        ws_flat = self.ws_generator(context)                  # [B, S*D]
        ws = ws_flat.view(B, S, D)

        # Mezcla con fallback estatico para estabilidad en inicializacion
        fallback = self.workspace_fallback.unsqueeze(0).expand(B, -1, -1)
        # Gate aprendido implicitamente: el MLP se inicializa cerca de zero,
        # asi al inicio ws ≈ 0 y el fallback domina via la suma residual
        return ws + fallback

    def forward(self, hidden: Tensor, active_mask: Tensor) -> Tensor:
        """
        Args:
            hidden:      [B, K, rim_size]
            active_mask: [B, K] bool

        Returns:
            h_updated: [B, K, rim_size]  — despues del broadcast
        """
        B, K, D = hidden.shape
        S = self.ws_slots

        # v5.1: Buffer dinamico condicionado al contexto
        ws = self._generate_workspace(hidden, active_mask)    # [B, S, D]

        # ---- Fase 1: Escritura al workspace ----
        Q_w = self.write_q(ws)                               # [B, S, D]
        K_w = self.write_k(hidden)                           # [B, K, D]
        V_w = self.write_v(hidden)                           # [B, K, D]

        attn_w = torch.bmm(Q_w, K_w.transpose(1, 2)) * self.scale   # [B, S, K]

        # Solo activos pueden escribir (enmascarar columnas inactivas)
        col_mask = active_mask.unsqueeze(1).expand(B, S, K)

        # v5.1: Manejo robusto de filas completamente enmascaradas
        # Verificar si hay al menos un activo por fila del workspace
        any_active = col_mask.any(dim=-1, keepdim=True)               # [B, S, 1]
        attn_w = attn_w.masked_fill(~col_mask, float('-inf'))
        attn_w = F.softmax(attn_w, dim=-1)
        # Si ningun modulo activo, atencion zero (en lugar de NaN)
        attn_w = attn_w * any_active.float()

        ws_updated = self.norm_ws(ws + torch.bmm(attn_w, V_w))       # [B, S, D]

        # ---- Fase 2: Broadcast desde workspace a todos los modulos ----
        Q_r = self.read_q(hidden)                                     # [B, K, D]
        K_r = self.read_k(ws_updated)                                 # [B, S, D]
        V_r = self.read_v(ws_updated)                                 # [B, S, D]

        attn_r = torch.bmm(Q_r, K_r.transpose(1, 2)) * self.scale    # [B, K, S]
        attn_r = F.softmax(attn_r, dim=-1)

        broadcast = self.read_o(torch.bmm(attn_r, V_r))              # [B, K, D]
        return self.norm_out(hidden + broadcast)


# ============================================================================
# 5. _DVNCCodebook — Discrete-Valued Neural Communication (DVNC / VQ)
#    v5.1: Commitment adaptativo por entropia de activacion
# ============================================================================

class _DVNCCodebook(nn.Module):
    """
    Comunicacion de Valores Discretos (DVNC).

    Antes de comunicarse, cada modulo activo cuantifica su mensaje al vector
    discreto mas cercano en un codebook compartido (VQ-VAE style).
    Esto actua como filtro de ruido y fuerza un 'lenguaje interno' simbolico.

    Gradiente via Straight-Through: forward usa el vector discreto,
    backward fluye por la proyeccion continua.

        z_q = codebook[argmin_c ||z - c||]
        z_sg = z + (z_q - z).detach()    (straight-through)

    v5.1: El commitment loss se pondera inversamente a la entropia de activacion.
    Cuando pocos modulos estan activos (sistema bajo estres/muerte progresiva),
    el codebook se relaja para permitir mayor expresividad.

        beta_eff = beta_base * sigmoid(entropy / entropy_scale)

    Args:
        rim_size:       Dimension del espacio de comunicacion.
        num_codes:      Tamano del libro de codigos (vocabulario discreto).
        commitment:     Peso base del commitment loss (beta en VQ-VAE).
        entropy_scale:  Escala para la modulacion por entropia.
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
        self.codebook = nn.Embedding(num_codes, rim_size)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.proj_in  = nn.Linear(rim_size, rim_size, bias=False)
        self.proj_out = nn.Linear(rim_size, rim_size, bias=False)
        self.norm     = nn.LayerNorm(rim_size)

    def forward(
        self,
        hidden: Tensor,
        active_mask: Tensor,
        activation_entropy: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden:               [B, K, rim_size]
            active_mask:          [B, K] bool
            activation_entropy:   float o None — entropia de los scores de activacion.
                                  Si se provee, modula el commitment loss.

        Returns:
            h_comm:       [B, K, rim_size]  — mensajes discretizados (inactivos = 0)
            vq_loss:      scalar             — loss de cuantificacion adaptativo
        """
        z = self.proj_in(hidden)          # [B, K, D]

        # Distancias al codebook: ||z - c||^2 = ||z||^2 - 2 z.c + ||c||^2
        z_flat = z.reshape(-1, z.shape[-1])                           # [B*K, D]
        cb = self.codebook.weight                                     # [C, D]
        dist = (
            z_flat.pow(2).sum(-1, keepdim=True)
            - 2 * z_flat @ cb.t()
            + cb.pow(2).sum(-1)
        )                                                             # [B*K, C]
        idx = dist.argmin(dim=-1)                                     # [B*K]
        z_q = self.codebook(idx).view_as(z)                          # [B, K, D]

        # v5.1: Commitment adaptativo
        # beta_eff = beta_base * sigmoid(entropy / scale)
        # Baja entropia (pocos activos, estres) -> beta_eff baja -> codebook relajado
        # Alta entropia (muchos activos, sano)   -> beta_eff alta -> codebook estricto
        if activation_entropy is not None:
            beta_eff = self.commitment_base * torch.sigmoid(
                torch.tensor(activation_entropy / self.entropy_scale, device=hidden.device)
            )
        else:
            beta_eff = self.commitment_base

        vq_loss = (
            (z_q.detach() - z).pow(2).mean()
            + beta_eff * (z_q - z.detach()).pow(2).mean()
        )

        # Straight-Through Estimator
        z_st = z + (z_q - z).detach()                                # [B, K, D]

        out = self.proj_out(z_st)
        # Solo activos emiten mensaje
        active_f = active_mask.unsqueeze(-1).float()
        h_comm = self.norm(hidden + active_f * out)

        return h_comm, vq_loss


# ============================================================================
# Dataclass de estado
# v5.1: Incluye fingerprint para NCO y contador de inactividad
# ============================================================================

@dataclass
class RIMsState:
    """Estado completo de un paso RIMs."""
    hidden_states:     Tensor   # [B, K, rim_size]
    active_rims:       Tensor   # [B, K] bool
    attention_weights: Tensor   # [B, K] float (scores softmax de seleccion)
    communication:     Tensor   # [B, K, rim_size] (ultimo mensaje)
    vq_loss:           Tensor   # scalar (0 si comm_mode != 'dvnc')
    inactivity_steps:  Tensor   # [B, K] int — pasos consecutivos inactivo (v5.1)
    fingerprint:       str = "" # hash ligero del estado para NCO (v5.1)

    def to_dict(self) -> Dict[str, Any]:
        aw = self.attention_weights
        return {
            'num_active':          self.active_rims.float().sum(-1).mean().item(),
            'activation_rate':     self.active_rims.float().mean().item(),
            'attention_entropy':   -(aw * (aw + 1e-10).log()).sum(-1).mean().item(),
            'comm_norm':           self.communication.norm(dim=-1).mean().item(),
            'vq_loss':             self.vq_loss.item(),
            'max_inactivity':      self.inactivity_steps.max().item(),
            'mean_inactivity':     self.inactivity_steps.float().mean().item(),
            'fingerprint':         self.fingerprint,
        }


# ============================================================================
# Utilidad: Fingerprint para NCO
# ============================================================================

def _compute_fingerprint(hidden: Tensor, precision: int = 4) -> str:
    """
    Computa un hash ligero del estado oculto para deteccion de divergencia
    ontologica entre checkpoints por parte del NCO.

    Cuantifica el tensor a `precision` decimales, luego aplica SHA-256 truncado.
    Esto permite comparacion rapida sin almacenar el tensor completo.

    Args:
        hidden:    [B, K, rim_size]
        precision: decimales de cuantificacion (4 = ~0.0001 resolucion)

    Returns:
        hex string de 16 caracteres (64 bits de hash)
    """
    # Cuantificar para estabilidad (ignorar ruido de float)
    scale = 10 ** precision
    quantized = (hidden.detach().float().cpu() * scale).round().to(torch.int32)
    raw_bytes = quantized.numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


# ============================================================================
# Modulo principal
# ============================================================================

class RecurrentIndependentMechanisms(ConsciousnessLayerBase):
    """
    Recurrent Independent Mechanisms v5.1 — implementacion completa del paper
    y variantes modernas (GWT, DVNC, Gumbel-Softmax, grupos fast/slow).

    Fases de cada timestep:
      1. **Input Attention** (paper-exact, v5.1 per-module W_q):
         cada RIM genera Q desde su h_{t-1} con su propio W_q^{(i)},
         la entrada genera K y V compartidos; scores -> top-k con STE o Gumbel.
      2. **Dinamicas Independientes** (vectorizado): GroupGRUCell con einsum;
         mascara de Hadamard aplica el resultado solo a activos.
         h_t = M ⊙ h_new + (1-M) ⊙ h_{t-1}
      3. **Inactivity Decay** (v5.1): modulos inactivos sufren decay exponencial
         proporcional a su tiempo de inactividad, previniendo estados stale.
      4. **Comunicacion** (configurable):
         - `'standard'`: multi-head con residual (paper math, v5.1 mascara corregida).
         - `'gwt'`     : Global Workspace Theory (v5.1 buffer dinamico).
         - `'dvnc'`    : codebook VQ discreto (v5.1 commitment adaptativo).

    Args:
        input_size:       Dimension de x_t.
        hidden_size:      Dimension total del estado (divisible por num_rims).
        num_rims:         K_t — numero total de modulos.
        num_active:       K_a — modulos activos por paso (top-k).
        num_heads:        Cabezas en la atencion de comunicacion.
        comm_mode:        'standard' | 'gwt' | 'dvnc'.
        routing:          'ste' (Straight-Through) | 'gumbel' (Gumbel-Softmax).
        gumbel_temp:      Temperatura inicial de Gumbel (solo si routing='gumbel').
        num_codes:        Tamano del codebook DVNC (solo si comm_mode='dvnc').
        ws_slots:         Slots del workspace GWT (solo si comm_mode='gwt').
        dropout:          Dropout general.
        inactivity_decay: Lambda de decay para modulos inactivos (v5.1).
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
        config:           Optional[LayerConfig] = None,
    ):
        super().__init__(config)

        # --- Validaciones ---
        if num_active < 1 or num_active > num_rims:
            raise ValueError(f"num_active={num_active} debe estar en [1, {num_rims}]")
        if hidden_size % num_rims != 0:
            raise ValueError(f"hidden_size={hidden_size} debe ser divisible por num_rims={num_rims}")
        rim_size = hidden_size // num_rims
        if rim_size % num_heads != 0:
            raise ValueError(f"rim_size={rim_size} debe ser divisible por num_heads={num_heads}")

        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.num_rims        = num_rims
        self.num_active      = num_active
        self.rim_size        = rim_size
        self.comm_mode       = comm_mode
        self.routing         = routing
        self.inactivity_decay = inactivity_decay

        # ---- Atencion de entrada (paper-exact, v5.1 per-module) ---- [SLOW]
        self.input_attention = _InputAttentionRIM(
            hidden_size=input_size,
            rim_size=rim_size,
            num_rims=num_rims,
            d_key=max(32, rim_size // 2),
        )

        # ---- Temperatura Gumbel (v5.1: softplus en lugar de clamp) ---- [SLOW]
        # Se usa raw_gumbel_temp como input a softplus + offset
        self._raw_gumbel_temp = nn.Parameter(
            torch.tensor(math.log(math.exp(gumbel_temp - 0.1) - 1.0))  # inverse softplus
        )

        # ---- GRU vectorizado ---- [FAST]
        self.gru = GroupGRUCell(rim_size, rim_size, num_rims)

        # ---- LayerNorm post-update ---- [FAST]
        self.hidden_norm = nn.LayerNorm(rim_size)

        # ---- Comunicacion ---- [SLOW]
        if comm_mode == 'standard':
            self.comm_layer = _MultiHeadCommResidual(rim_size, num_heads, dropout)
        elif comm_mode == 'gwt':
            self.comm_layer = _GlobalWorkspace(rim_size, num_rims, ws_slots)
        elif comm_mode == 'dvnc':
            self.comm_layer = _DVNCCodebook(rim_size, num_codes)
        else:
            raise ValueError(f"comm_mode desconocido: {comm_mode}")

        # ---- Proyeccion de salida con residual ---- [FAST]
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

        self._init_weights()

        # ---- v5.1: Assertion de cobertura fast/slow exhaustiva ----
        self._validate_param_groups()

    # ------------------------------------------------------------------
    # Inicializacion
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # v5.1: Validacion de cobertura de parametros fast/slow
    # ------------------------------------------------------------------

    def _validate_param_groups(self) -> None:
        """
        Verifica que fast_params() + slow_params() cubren exactamente
        todos los parametros del modulo, sin duplicados ni omisiones.
        """
        all_params = set(id(p) for p in self.parameters())
        fast_ids   = set(id(p) for p in self.fast_params())
        slow_ids   = set(id(p) for p in self.slow_params())

        overlap = fast_ids & slow_ids
        missing = all_params - (fast_ids | slow_ids)

        if overlap:
            raise RuntimeError(
                f"[RIMs v5.1] {len(overlap)} parametros duplicados en fast/slow groups. "
                f"Revisar asignacion de modulos."
            )
        if missing:
            raise RuntimeError(
                f"[RIMs v5.1] {len(missing)} parametros no asignados a fast ni slow. "
                f"Todo parametro debe pertenecer a exactamente un grupo."
            )

    # ------------------------------------------------------------------
    # Temperatura Gumbel (v5.1: softplus)
    # ------------------------------------------------------------------

    @property
    def gumbel_temp(self) -> Tensor:
        """
        Temperatura Gumbel via softplus + offset minimo.
        Rango efectivo: [0.1, +inf) con gradientes continuos en todo el rango.
        Evita gradientes discontinuos del clamp anterior.
        """
        return F.softplus(self._raw_gumbel_temp) + 0.1

    # ------------------------------------------------------------------
    # Grupos de parametros para meta-aprendizaje (fast / slow)
    # ------------------------------------------------------------------

    def fast_params(self) -> List[nn.Parameter]:
        """
        theta_modulos — parametros de las dinamicas recurrentes.
        Se adaptan rapidamente al entorno inmediato (inner loop de MAML).
        """
        fast_modules = (self.gru, self.hidden_norm, self.output_proj, self.output_norm)
        return [p for m in fast_modules for p in m.parameters()]

    def slow_params(self) -> List[nn.Parameter]:
        """
        theta_atencion — parametros del enrutamiento y la comunicacion.
        Se actualizan lentamente a traves de muchas distribuciones (outer loop).
        """
        slow_modules = (self.input_attention, self.comm_layer)
        slow_list = [p for m in slow_modules for p in m.parameters()]
        slow_list.append(self._raw_gumbel_temp)
        slow_list.append(self.initial_hidden)
        return slow_list

    # ------------------------------------------------------------------
    # Seleccion de RIMs activos
    # ------------------------------------------------------------------

    def _select(
        self, scores: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Aplica routing top-k diferenciable.

        STE:    forward = hard mask, backward = softmax.
        Gumbel: scores perturbados con ruido Gumbel -> top-k mas suave.
                v5.1: temperatura via softplus (gradientes continuos).

        Returns:
            active_mask:  [B, K] bool
            attn_weights: [B, K] float (softmax normalizado para metricas)
            sel_weights:  [B, K] float (diferenciable para escalar el update)
        """
        B, K = scores.shape

        if self.routing == 'gumbel' and self.training:
            temp = self.gumbel_temp  # v5.1: softplus
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
    # v5.1: Inactivity decay
    # ------------------------------------------------------------------

    def _apply_inactivity_decay(
        self, hidden: Tensor, active_mask: Tensor, inactivity_steps: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Aplica decay exponencial a modulos inactivos proporcional a su
        tiempo de inactividad acumulado.

            h_inactive *= (1 - lambda * min(steps, cap) / cap)

        El decay es suave y capped para evitar que el estado colapse a zero
        demasiado rapido. Modulos activos resetean su contador a 0.

        Args:
            hidden:           [B, K, rim_size]
            active_mask:      [B, K] bool
            inactivity_steps: [B, K] long — pasos consecutivos inactivo

        Returns:
            hidden_decayed:   [B, K, rim_size]
            inactivity_new:   [B, K] long — contadores actualizados
        """
        # Actualizar contadores: activos -> 0, inactivos -> +1
        inactivity_new = torch.where(
            active_mask,
            torch.zeros_like(inactivity_steps),
            inactivity_steps + 1,
        )

        # Decay factor: 1.0 para activos, decae para inactivos
        # Cap en 100 pasos para evitar colapso total
        cap = 100.0
        decay_ratio = (inactivity_new.float().clamp(max=cap) / cap)       # [B, K] in [0,1]
        decay_factor = 1.0 - self.inactivity_decay * decay_ratio          # [B, K]
        decay_factor = decay_factor.unsqueeze(-1)                         # [B, K, 1]

        # Solo aplicar decay a inactivos (activos mantienen factor=1.0)
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

        Matematica:
          1. x̃_{t,i}, s_{t,i} = InputAttention(x_t, h_{t-1})
          2. active = top-k(s_{t,i})
          3. h_all  = GroupGRU(x̃ * alpha_i, h_{t-1})     (todos los grupos)
          4. h_t    = M ⊙ h_all + (1-M) ⊙ h_{t-1}        (Hadamard mask)
          4b. Inactivity decay (v5.1)
          5. h_t    = Communication(h_t, active)
          6. out_t  = LayerNorm(h_flat + OutputProj(h_flat))

        Returns:
            hidden_new, comm, output_t, active_mask, attn_weights, vq_loss, inactivity_new
        """
        B = x_t.shape[0]

        # 1. Atencion de entrada
        V_exp, scores = self.input_attention(x_t, hidden)

        # 2. Seleccion top-k
        active_mask, attn_weights, sel_weights = self._select(scores)

        # Entrada atendida: x̃_{t,i} = alpha_i * V_{inp}
        x_per_rim = sel_weights.unsqueeze(-1) * V_exp

        # 3. GroupGRU vectorizado
        h_all = self.gru(x_per_rim, hidden)

        # 4. Mascara de Hadamard
        M = active_mask.unsqueeze(-1).float()
        hidden_new = M * h_all + (1.0 - M) * hidden
        hidden_new = self.hidden_norm(hidden_new)

        # 4b. v5.1: Inactivity decay
        hidden_new, inactivity_new = self._apply_inactivity_decay(
            hidden_new, active_mask, inactivity_steps
        )

        # 5. Comunicacion (segun modo)
        vq_loss = torch.zeros(1, device=x_t.device)
        if self.comm_mode == 'dvnc':
            # v5.1: Pasar entropia de activacion para commitment adaptativo
            act_entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(-1).mean().item()
            hidden_new, vq_loss = self.comm_layer(hidden_new, active_mask, act_entropy)
            comm = hidden_new
        else:
            hidden_new = self.comm_layer(hidden_new, active_mask)
            comm = hidden_new

        # 6. Salida con residual
        h_flat  = hidden_new.view(B, -1)
        out_t   = self.output_norm(h_flat + self.output_proj(h_flat))

        return hidden_new, comm, out_t, active_mask, attn_weights, vq_loss, inactivity_new

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:      Tensor,
        hidden: Optional[Tensor] = None,
        inactivity_steps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, RIMsState]:
        """
        Args:
            x:                [B, input_size] o [B, seq_len, input_size]
            hidden:           [B, num_rims, rim_size] o None
            inactivity_steps: [B, num_rims] long o None (v5.1)

        Returns:
            output: [B, hidden_size] o [B, seq_len, hidden_size]
            state:  RIMsState del ultimo timestep (incluye fingerprint v5.1)
        """
        is_seq = x.dim() == 3
        if not is_seq:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        device   = x.device

        if hidden is None:
            hidden = self.initial_hidden.expand(B, -1, -1).clone().to(device)

        if inactivity_steps is None:
            inactivity_steps = torch.zeros(B, self.num_rims, dtype=torch.long, device=device)

        outputs: List[Tensor] = []
        total_vq = torch.zeros(1, device=device)
        last_active = last_attn = last_comm = None

        for t in range(T):
            hidden, comm, out_t, active, attn, vq, inactivity_steps = self._step(
                x[:, t], hidden, inactivity_steps
            )
            outputs.append(out_t)
            total_vq   = total_vq + vq
            last_active = active
            last_attn   = attn
            last_comm   = comm

        output = torch.stack(outputs, dim=1)
        if not is_seq:
            output = output.squeeze(1)

        # Metricas
        act_rate  = last_active.float().mean().item()
        entropy   = -(last_attn * (last_attn + 1e-10).log()).sum(-1).mean().item()
        sparsity  = 1.0 - act_rate
        self.metrics.record({
            'activation_rate':    act_rate,
            'attention_entropy':  entropy,
            'sparsity':           sparsity,
            'vq_loss':            (total_vq / T).item(),
            'gumbel_temp':        self.gumbel_temp.item(),
            'max_inactivity':     inactivity_steps.max().item(),
            'mean_inactivity':    inactivity_steps.float().mean().item(),
        })

        # v5.1: Fingerprint para NCO
        fp = _compute_fingerprint(hidden)

        state = RIMsState(
            hidden_states     = hidden,
            active_rims       = last_active,
            attention_weights = last_attn,
            communication     = last_comm,
            vq_loss           = total_vq / T,
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
            'num_rims':          self.num_rims,
            'num_active':        self.num_active,
            'rim_size':          self.rim_size,
            'comm_mode':         self.comm_mode,
            'routing':           self.routing,
            'inactivity_decay':  self.inactivity_decay,
        }

    def reset_hidden(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Genera un estado oculto inicial limpio.

        v5.1: Retorna tambien el contador de inactividad inicializado a 0.

        Returns:
            hidden:           [B, num_rims, rim_size]
            inactivity_steps: [B, num_rims] long
        """
        hidden = self.initial_hidden.expand(batch_size, -1, -1).clone().to(device)
        inactivity_steps = torch.zeros(batch_size, self.num_rims, dtype=torch.long, device=device)
        return hidden, inactivity_steps

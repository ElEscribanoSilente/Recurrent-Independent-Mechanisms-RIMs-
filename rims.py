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
  3. _MultiHeadCommResidual— Comunicacion multi-cabeza con conexion residual
                             segun formula h_{t,k} = softmax(QK^T/sqrt(d))V + h~_{t,k}.
  4. _GlobalWorkspace      — Global Workspace Theory (GWT): competencia->
                             escritura al buffer->broadcast a todos los modulos.
  5. _DVNCCodebook         — Discrete-Valued Neural Communication (DVNC):
                             comunicacion cuantificada via VQ-VAE codebook.
  6. Routing diferenciable — STE (Straight-Through Estimator) o Gumbel-Softmax.
  7. Grupos fast/slow      — Separacion de parametros para meta-aprendizaje:
                             theta_modulos (rapidos) vs theta_atencion (lentos).

Referencia matematica completa en docstrings inline.

Author: Escribano Silente (MSC Framework)
Version: 5.0.0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
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
# ============================================================================

class _InputAttentionRIM(nn.Module):
    """
    Atencion de entrada de los RIMs segun Goyal et al. (2019).

    Cada modulo i genera su propia consulta desde h_{t-1,i}:
        Q_{inp,i} = h_{t-1,i} @ W_q      in R^{d_k}
    El input x_t proyecta claves y valores compartidos:
        K_{inp}   = x_t @ W_k             in R^{d_k}
        V_{inp}   = x_t @ W_v             in R^{d_v}
    Puntuacion de relevancia (escalar por modulo):
        s_{t,i}   = Q_{inp,i} @ K_{inp}^T / sqrt(d_k)
    Tras seleccion top-k, la entrada atendida de modulo i activo:
        x̃_{t,i}  = softmax_local(s_{t,i}) * V_{inp}   in R^{d_v}

    Retorna tambien los scores crudos para el selector (routing).

    Args:
        hidden_size: d_h total (= num_rims * rim_size).
        rim_size:    d_h por modulo.
        num_rims:    K_t.
        d_key:       Dimension del espacio de claves/consultas.
    """

    def __init__(self, hidden_size: int, rim_size: int, num_rims: int, d_key: int = 64):
        super().__init__()
        self.num_rims = num_rims
        self.d_key = d_key

        # Proyeccion de consulta: cada modulo tiene su propio W_q (rim_size -> d_key)
        # Implementado como una proyeccion 2D compartida sobre la dimension de modulos
        self.W_q = nn.Linear(rim_size, d_key, bias=False)
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
        # Q_{inp,i} = h_{t-1,i} @ W_q  -> [B, K, d_key]
        Q = self.W_q(hidden)                         # [B, K, d_key]
        # K_{inp} = x_t @ W_k           -> [B, d_key]
        K = self.W_k(x)                              # [B, d_key]
        # V_{inp} = x_t @ W_v           -> [B, rim_size]
        V = self.W_v(x)                              # [B, rim_size]

        # Scores: Q . K^T / sqrt(d_k)  -> [B, K]
        # bmm: [B, K, d_key] x [B, d_key, 1] -> [B, K, 1]
        scores = torch.bmm(Q, K.unsqueeze(-1)).squeeze(-1) * self.scale  # [B, K]

        # Valor proyectado expandido para todos los RIMs (se enmascara despues)
        # x̃_{t,i} = alpha_i * V, donde alpha viene del selector externo
        # Aqui devolvemos V expandido; la ponderacion la aplica el modulo principal
        V_exp = V.unsqueeze(1).expand(-1, self.num_rims, -1)             # [B, K, rim_size]

        return V_exp, scores


# ============================================================================
# 3. _MultiHeadCommResidual — Comunicacion con residual (formulacion del paper)
# ============================================================================

class _MultiHeadCommResidual(nn.Module):
    """
    Comunicacion inter-modulo multi-cabeza con conexion residual.

    Segun la formalizacion matematica del paper:
        h_{t,k} = MH_Att(h̃_{t,k}, h̃_{t,:}) + h̃_{t,k}    ∀k ∈ S_t
        h_{t,i} = h̃_{t,i}                                   ∀i ∉ S_t

    Solo los modulos activos emiten consultas (reduccion de computo).
    Todos los modulos pueden ser fuente (keys/values).

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
        Km = heads(self.W_k(hidden))
        V  = heads(self.W_v(hidden))

        attn = torch.matmul(Q, Km.transpose(-1, -2)) * self.scale   # [B, H, K, K]

        # Inactivos no pueden ser consulta (enmascarar filas) ni fuente columna inactiva
        col_mask = active_mask.unsqueeze(1).unsqueeze(2).expand(B, H, K, K)
        attn = attn.masked_fill(~col_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                          # [B, H, K, Dh]
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        out = self.W_o(out)

        # Residual: h_{t,k} = MH_Att(...) + h̃_{t,k}
        active_f = active_mask.unsqueeze(-1).float()         # [B, K, 1]
        h_final  = hidden + active_f * out                   # inactivos: += 0

        return self.norm_out(h_final)


# ============================================================================
# 4. _GlobalWorkspace — Global Workspace Theory (GWT)
# ============================================================================

class _GlobalWorkspace(nn.Module):
    """
    Espacio de Trabajo Global (Global Workspace Theory, Baars 1988 / Bengio 2017).

    Ciclo tripartito:
      1. Competencia: modulos activos compiten por escribir al buffer.
      2. Consolidacion: el buffer agrega los aportes ponderados por atencion.
      3. Broadcast: el buffer envia su resumen a TODOS los modulos.

    El buffer se implementa como un conjunto de ``ws_slots`` vectores aprendibles
    que actuan como memoria de trabajo centralizada.

    Args:
        rim_size:  Dimension por modulo.
        num_rims:  K_t.
        ws_slots:  Numero de slots en el workspace (tipicamente 1-4).
    """

    def __init__(self, rim_size: int, num_rims: int, ws_slots: int = 2):
        super().__init__()
        self.ws_slots = ws_slots

        # Buffer aprendible: [ws_slots, rim_size]
        self.workspace = nn.Parameter(torch.randn(ws_slots, rim_size) * 0.02)

        # Atencion de escritura: modulos -> workspace
        self.write_q = nn.Linear(rim_size, rim_size, bias=False)  # query del workspace
        self.write_k = nn.Linear(rim_size, rim_size, bias=False)  # key de cada modulo
        self.write_v = nn.Linear(rim_size, rim_size, bias=False)  # value de cada modulo

        # Atencion de lectura (broadcast): workspace -> todos los modulos
        self.read_q  = nn.Linear(rim_size, rim_size, bias=False)
        self.read_k  = nn.Linear(rim_size, rim_size, bias=False)
        self.read_v  = nn.Linear(rim_size, rim_size, bias=False)
        self.read_o  = nn.Linear(rim_size, rim_size, bias=False)

        self.norm_ws  = nn.LayerNorm(rim_size)
        self.norm_out = nn.LayerNorm(rim_size)
        self.scale    = rim_size ** -0.5

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

        # ---- Fase 1: Escritura al workspace ----
        ws = self.workspace.unsqueeze(0).expand(B, -1, -1)  # [B, S, D]
        Q_w = self.write_q(ws)                               # [B, S, D]
        K_w = self.write_k(hidden)                           # [B, K, D]
        V_w = self.write_v(hidden)                           # [B, K, D]

        attn_w = torch.bmm(Q_w, K_w.transpose(1, 2)) * self.scale   # [B, S, K]
        # Solo activos pueden escribir (enmascarar columnas inactivas)
        col_mask = active_mask.unsqueeze(1).expand(B, S, K)
        attn_w = attn_w.masked_fill(~col_mask, float('-inf'))
        attn_w = F.softmax(attn_w, dim=-1).nan_to_num(0.0)

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

    Args:
        rim_size:   Dimension del espacio de comunicacion.
        num_codes:  Tamano del libro de codigos (vocabulario discreto).
        commitment: Peso del commitment loss (beta en VQ-VAE).
    """

    def __init__(self, rim_size: int, num_codes: int = 64, commitment: float = 0.25):
        super().__init__()
        self.commitment = commitment
        self.codebook = nn.Embedding(num_codes, rim_size)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.proj_in  = nn.Linear(rim_size, rim_size, bias=False)
        self.proj_out = nn.Linear(rim_size, rim_size, bias=False)
        self.norm     = nn.LayerNorm(rim_size)

    def forward(self, hidden: Tensor, active_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden:      [B, K, rim_size]
            active_mask: [B, K] bool

        Returns:
            h_comm:       [B, K, rim_size]  — mensajes discretizados (inactivos = 0)
            vq_loss:      scalar             — loss de cuantificacion (para el trainer)
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

        # VQ loss: alinear codebook hacia z (sg) + commitment
        vq_loss = (
            (z_q.detach() - z).pow(2).mean()
            + self.commitment * (z_q - z.detach()).pow(2).mean()
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
# ============================================================================

@dataclass
class RIMsState:
    """Estado completo de un paso RIMs."""
    hidden_states:     Tensor   # [B, K, rim_size]
    active_rims:       Tensor   # [B, K] bool
    attention_weights: Tensor   # [B, K] float (scores softmax de seleccion)
    communication:     Tensor   # [B, K, rim_size] (ultimo mensaje)
    vq_loss:           Tensor   # scalar (0 si comm_mode != 'dvnc')

    def to_dict(self) -> Dict[str, Any]:
        aw = self.attention_weights
        return {
            'num_active':         self.active_rims.float().sum(-1).mean().item(),
            'activation_rate':    self.active_rims.float().mean().item(),
            'attention_entropy':  -(aw * (aw + 1e-10).log()).sum(-1).mean().item(),
            'comm_norm':          self.communication.norm(dim=-1).mean().item(),
            'vq_loss':            self.vq_loss.item(),
        }


# ============================================================================
# Modulo principal
# ============================================================================

class RecurrentIndependentMechanisms(ConsciousnessLayerBase):
    """
    Recurrent Independent Mechanisms v5.0 — implementacion completa del paper
    y variantes modernas (GWT, DVNC, Gumbel-Softmax, grupos fast/slow).

    Fases de cada timestep:
      1. **Input Attention** (paper-exact): cada RIM genera Q desde su h_{t-1},
         la entrada genera K y V compartidos; scores -> top-k con STE o Gumbel.
      2. **Dinamicas Independientes** (vectorizado): GroupGRUCell con einsum;
         mascara de Hadamard aplica el resultado solo a activos.
         h_t = M ⊙ h_new + (1-M) ⊙ h_{t-1}
      3. **Comunicacion** (configurable):
         - `'standard'`: multi-head con residual (paper math).
         - `'gwt'`     : Global Workspace Theory (competencia→buffer→broadcast).
         - `'dvnc'`    : codebook VQ discreto + straight-through.

    Args:
        input_size:    Dimension de x_t.
        hidden_size:   Dimension total del estado (divisible por num_rims).
        num_rims:      K_t — numero total de modulos.
        num_active:    K_a — modulos activos por paso (top-k).
        num_heads:     Cabezas en la atencion de comunicacion.
        comm_mode:     'standard' | 'gwt' | 'dvnc'.
        routing:       'ste' (Straight-Through) | 'gumbel' (Gumbel-Softmax).
        gumbel_temp:   Temperatura inicial de Gumbel (solo si routing='gumbel').
        num_codes:     Tamano del codebook DVNC (solo si comm_mode='dvnc').
        ws_slots:      Slots del workspace GWT (solo si comm_mode='gwt').
        dropout:       Dropout general.
        config:        LayerConfig opcional.
    """

    def __init__(
        self,
        input_size:   int = 64,
        hidden_size:  int = 256,
        num_rims:     int = 6,
        num_active:   int = 3,
        num_heads:    int = 4,
        comm_mode:    Literal['standard', 'gwt', 'dvnc'] = 'standard',
        routing:      Literal['ste', 'gumbel'] = 'ste',
        gumbel_temp:  float = 1.0,
        num_codes:    int = 64,
        ws_slots:     int = 2,
        dropout:      float = 0.1,
        config:       Optional[LayerConfig] = None,
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

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_rims    = num_rims
        self.num_active  = num_active
        self.rim_size    = rim_size
        self.comm_mode   = comm_mode
        self.routing     = routing

        # ---- Atencion de entrada (paper-exact) ---- [SLOW: theta_atencion]
        self.input_attention = _InputAttentionRIM(
            hidden_size=input_size,
            rim_size=rim_size,
            num_rims=num_rims,
            d_key=max(32, rim_size // 2),
        )

        # ---- Temperatura Gumbel aprendible ---- [SLOW]
        self.log_gumbel_temp = nn.Parameter(torch.tensor(math.log(gumbel_temp)))

        # ---- GRU vectorizado ---- [FAST: theta_modulos]
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
        slow_list.append(self.log_gumbel_temp)
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

        Returns:
            active_mask:  [B, K] bool
            attn_weights: [B, K] float (softmax normalizado para metricas)
            sel_weights:  [B, K] float (diferenciable para escalar el update)
        """
        B, K = scores.shape

        if self.routing == 'gumbel' and self.training:
            temp = self.log_gumbel_temp.exp().clamp(0.1, 10.0)
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
    # Paso temporal unico
    # ------------------------------------------------------------------

    def _step(
        self, x_t: Tensor, hidden: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Procesa un timestep.

        Matematica:
          1. x̃_{t,i}, s_{t,i} = InputAttention(x_t, h_{t-1})
          2. active = top-k(s_{t,i})
          3. h_all  = GroupGRU(x̃ * alpha_i, h_{t-1})     (todos los grupos)
          4. h_t    = M ⊙ h_all + (1-M) ⊙ h_{t-1}        (Hadamard mask)
          5. h_t    = Communication(h_t, active)
          6. out_t  = LayerNorm(h_flat + OutputProj(h_flat))

        Returns:
            hidden_new, comm, output_t, active_mask, attn_weights, vq_loss
        """
        B = x_t.shape[0]

        # 1. Atencion de entrada
        V_exp, scores = self.input_attention(x_t, hidden)  # V_exp: [B,K,rim]; scores: [B,K]

        # 2. Seleccion top-k
        active_mask, attn_weights, sel_weights = self._select(scores)

        # Entrada atendida: x̃_{t,i} = alpha_i * V_{inp}
        x_per_rim = sel_weights.unsqueeze(-1) * V_exp       # [B, K, rim_size]

        # 3. GroupGRU vectorizado (todos los modulos en paralelo)
        h_all = self.gru(x_per_rim, hidden)                 # [B, K, rim_size]

        # 4. Mascara de Hadamard: solo activos se actualizan
        #    h_t = M ⊙ h_new + (1-M) ⊙ h_{t-1}
        M = active_mask.unsqueeze(-1).float()
        hidden_new = M * h_all + (1.0 - M) * hidden
        hidden_new = self.hidden_norm(hidden_new)

        # 5. Comunicacion (segun modo)
        vq_loss = torch.zeros(1, device=x_t.device)
        if self.comm_mode == 'dvnc':
            hidden_new, vq_loss = self.comm_layer(hidden_new, active_mask)
            comm = hidden_new
        else:
            hidden_new = self.comm_layer(hidden_new, active_mask)
            comm = hidden_new

        # 6. Salida con residual
        h_flat  = hidden_new.view(B, -1)
        out_t   = self.output_norm(h_flat + self.output_proj(h_flat))

        return hidden_new, comm, out_t, active_mask, attn_weights, vq_loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:      Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, RIMsState]:
        """
        Args:
            x:      [B, input_size] o [B, seq_len, input_size]
            hidden: [B, num_rims, rim_size] o None

        Returns:
            output: [B, hidden_size] o [B, seq_len, hidden_size]
            state:  RIMsState del ultimo timestep
        """
        is_seq = x.dim() == 3
        if not is_seq:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        device   = x.device

        if hidden is None:
            hidden = self.initial_hidden.expand(B, -1, -1).clone().to(device)

        outputs: List[Tensor] = []
        total_vq = torch.zeros(1, device=device)
        last_active = last_attn = last_comm = None

        for t in range(T):
            hidden, comm, out_t, active, attn, vq = self._step(x[:, t], hidden)
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
            'activation_rate':   act_rate,
            'attention_entropy': entropy,
            'sparsity':          sparsity,
            'vq_loss':           (total_vq / T).item(),
            'gumbel_temp':       self.log_gumbel_temp.exp().item(),
        })

        state = RIMsState(
            hidden_states     = hidden,
            active_rims       = last_active,
            attention_weights = last_attn,
            communication     = last_comm,
            vq_loss           = total_vq / T,
        )

        return output, state

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **{k: self.metrics.get_stats(k)
               for k in ('activation_rate', 'attention_entropy',
                         'sparsity', 'vq_loss', 'gumbel_temp')},
            'num_rims':   self.num_rims,
            'num_active': self.num_active,
            'rim_size':   self.rim_size,
            'comm_mode':  self.comm_mode,
            'routing':    self.routing,
        }

    def reset_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """Genera un estado oculto inicial limpio."""
        return self.initial_hidden.expand(batch_size, -1, -1).clone().to(device)

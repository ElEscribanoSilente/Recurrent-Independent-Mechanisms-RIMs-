# Recurrent Independent Mechanisms (RIMs) v5.1.1
**MSC Framework — Consciousness Architecture Stack**

> *"Un modelo recurrente monolítico fracasa cuando el entorno sufre alteraciones localizadas. Los RIMs separan los flujos de información causal mediante módulos que compiten, se especializan y comunican de forma dispersa."*

---

## Índice
1. [Descripción](#descripción)
2. [Arquitectura](#arquitectura)
3. [Instalación](#instalación)
4. [Uso rápido](#uso-rápido)
5. [Configuración completa](#configuración-completa)
6. [Modos de comunicación](#modos-de-comunicación)
7. [Inactivity Decay](#inactivity-decay)
8. [NCO Fingerprint](#nco-fingerprint)
9. [Meta-aprendizaje (fast/slow)](#meta-aprendizaje-fastslow)
10. [API de referencia](#api-de-referencia)
11. [Migración](#migración)
12. [Benchmarks](#benchmarks)
13. [Tests](#tests)
14. [Integración con MSC](#integración-con-msc)
15. [Changelog](#changelog)

---

## Descripción

Los **Recurrent Independent Mechanisms** (Goyal et al., 2019) fragmentan el estado oculto en *K_t* módulos funcionales independientes. En cada paso temporal, un selector diferenciable activa los *K_a* módulos más relevantes (top-k). Los inactivos conservan su estado (con decay suave), protegiendo el conocimiento acumulado contra interferencia destructiva.

**Por qué importa para MSC:**
Los RIMs implementan directamente el *Prior de Consciencia* de Bengio: un cuello de botella de atención que emula la especialización cortical. Son el substrato computacional del `GlobalWorkspaceEA1V2` y del experimento de continuidad ontológica (NCO). El fingerprint de estado permite al NCO detectar divergencia ontológica entre checkpoints sin almacenar tensores completos.

**v5.1.1 — Correcciones funcionales y de robustez.** Esta versión corrige nueve bugs identificados en v5.1.0, incluyendo dos críticos en `_DVNCCodebook` (VQ loss invertido y entropía con grafo roto) y uno medio en el orden de aplicación del inactivity decay. Ver [Changelog](#changelog) para detalles. Modelos sin DVNC y con `inactivity_decay=0` son numéricamente equivalentes a v5.1.0.

### Ventajas frente a arquitecturas monolíticas

| Propiedad | LSTM/GRU monolítico | RIMs v5.1.1 |
|-----------|-------------------|-----------|
| Representación del estado | Vector único denso | *K* módulos independientes |
| Routing de entrada | Homogéneo a toda la red | Top-k competitivo |
| Parámetros | Matriz densa global | Bloques por módulo (GroupGRUCell) |
| Actualización | Todo el estado en cada paso | Solo módulos activos |
| Comunicación | Densa e interconectada | Dispersa (standard / GWT / DVNC) |
| Olvido catastrófico | Alta vulnerabilidad | Mitigado por congelamiento + decay |
| Generalización OOD | Pobre | Robusta por especialización |
| Estado stale en inactivos | N/A | Decay exponencial capped |
| Integridad ontológica | Sin soporte | Fingerprint SHA-256 para NCO (opt-in) |
| VQ loss DVNC | N/A | Posición canónica (van den Oord 2017) |

---

## Arquitectura

```
x_t ──► InputAttentionRIM ──► scores [B, K]
        (Q per-módulo via       │
         einsum, K/V desde x_t) ▼
                            Selector top-k (STE / Gumbel-softplus)
                                 │
                    ┌────────────▼────────────┐
                    │  Inactivity Decay       │  (v5.1.1: orden corregido)
                    │  h_decayed = h·(1-λ·r)  │   solo sobre h_{t-1} preservado
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    GroupGRUCell          │  ← FAST params
                    │    (einsum vectorizado)  │
                    └────────────┬────────────┘
                                 │
                    h_t = M ⊙ h_new + (1-M) ⊙ h_decayed   (Hadamard)
                                 │
                            LayerNorm
                                 │
                    ┌────────────▼────────────┐
                    │  Comunicación (modo):    │  ← SLOW params
                    │  • standard: MH residual │
                    │  • gwt:  Dynamic GWT     │
                    │  • dvnc: Adaptive VQ     │
                    │         (loss canónico)  │
                    └────────────┬────────────┘
                                 │
                    output_t = LayerNorm(h + OutputProj(h))
                                 │
                    fingerprint = SHA256(quantize(h))   (opt-in v5.1.1)
```

### Sub-módulos

| Clase | Descripción | Cambios v5.1.1 |
|-------|------------|--------------|
| `GroupGRUCell` | GRU K grupos vectorizado (`einsum`), O(1) kernels CUDA | — |
| `_InputAttentionRIM` | Atención de entrada paper-exact: Q←h, K/V←x | Init xavier per-slice 2D |
| `_MultiHeadCommResidual` | MH-Att con residual: h=Att(h̃)+h̃ | — |
| `_GlobalWorkspace` | GWT: competencia→buffer→broadcast | `ws_generator[-1]` zero-init real |
| `_DVNCCodebook` | DVNC: cuantización VQ-VAE + STE | **VQ loss canónico**, entropía como tensor |
| `RIMsState` | Dataclass de estado completo | `to_dict()` omite fingerprint vacío |
| `RecurrentIndependentMechanisms` | Módulo principal | Decay reordenado, fingerprint opt-in, `inactivity_cap` configurable |

---

## Instalación

### Desde el repositorio MSC

```bash
pip install msc-rims
```

O directamente desde source:

```bash
git clone https://github.com/msc-tecnologia/msc-rims.git
cd msc-rims
pip install -e .
```

### Dependencias

| Paquete | Versión mínima | Notas |
|---------|---------------|-------|
| `torch` | ≥ 2.0 | Único requisito hard. CUDA opcional pero recomendado |

### Estructura del paquete

```
msc-rims/
├── msc_rims/
│   ├── __init__.py
│   ├── rims.py                # Módulo principal (RecurrentIndependentMechanisms)
│   ├── base.py                # ConsciousnessLayerBase
│   └── config.py              # LayerConfig
├── tests/
│   └── tests_rims.py          # 30+ tests
├── benchmarks/
│   └── benchmarks.py
├── pyproject.toml
├── LICENSE.txt                # MSC-ORL-1.0
├── CHANGELOG.md
└── README.md
```

### Uso standalone (sin stack MSC)

El módulo es completamente independiente. `ConsciousnessLayerBase` y `LayerConfig` se incluyen en el paquete como dependencias internas ligeras — no requieren el resto del framework MSC.

```python
from msc_rims import RecurrentIndependentMechanisms, RIMsState

rims = RecurrentIndependentMechanisms(input_size=128, hidden_size=384)
```

### Como capa dentro del stack MSC

Si se integra al ecosistema completo de consciencia (E-α-1-v3, GlobalWorkspaceEA1V2, etc.), registrar como layer:

```python
# En consciousness/layers/__init__.py
from msc_rims import RecurrentIndependentMechanisms
```

La integración con `EntityBrainV4` y el `NCO` se documenta en [§Integración con MSC](#integración-con-msc).

---

## Uso rápido

```python
import torch
from msc_rims import RecurrentIndependentMechanisms

# Configuración mínima
rims = RecurrentIndependentMechanisms(
    input_size=128,
    hidden_size=384,   # 384 / 6 = 64 por módulo
    num_rims=6,
    num_active=3,
)

x = torch.randn(32, 20, 128)   # [batch, seq_len, input_size]
out, state = rims(x)

print(out.shape)               # [32, 20, 384]
print(state.fingerprint)       # '' (default: opt-in en v5.1.1)
print(state.to_dict())
# {'num_active': 3.0, 'activation_rate': 0.5, 'attention_entropy': ...,
#  'comm_norm': ..., 'vq_loss': 0.0, 'max_inactivity': 12,
#  'mean_inactivity': 4.2}
# (fingerprint omitido si está vacío)
```

Para activar fingerprint (auditoría NCO):

```python
rims = RecurrentIndependentMechanisms(
    input_size=128, hidden_size=384,
    compute_fingerprint=True,    # opt-in: incurre sync GPU→CPU
)
out, state = rims(x)
print(state.fingerprint)   # 'a3f7c91b02d4e8f1'
```

---

## Configuración completa

```python
rims = RecurrentIndependentMechanisms(
    input_size       = 128,
    hidden_size      = 384,
    num_rims         = 6,
    num_active       = 3,

    # Comunicación: 'standard' | 'gwt' | 'dvnc'
    comm_mode        = 'gwt',
    ws_slots         = 2,          # slots del workspace (solo gwt)
    num_codes        = 128,        # vocab del codebook (solo dvnc)

    # Routing: 'ste' (Straight-Through) | 'gumbel'
    routing          = 'gumbel',
    gumbel_temp      = 1.0,        # temperatura inicial (aprendible via softplus)

    num_heads        = 4,
    dropout          = 0.1,

    # Decay de módulos inactivos
    inactivity_decay = 0.001,      # λ — velocidad de decay (0 = desactivado)
    inactivity_cap   = 100.0,      # v5.1.1: pasos máximos considerados (configurable)

    # NCO fingerprint (opt-in para evitar sync GPU↔CPU)
    compute_fingerprint = False,   # v5.1.1: default False, activar solo si NCO lo consume
)
```

### Paso con estado previo (autoregresivo)

```python
hidden, inactivity = rims.reset_hidden(batch_size=32, device=device)
state = None
outputs = []
for t in range(seq_len):
    out_t, state = rims(
        x[:, t],
        hidden=state.hidden_states if state is not None else hidden,
        inactivity_steps=state.inactivity_steps if state is not None else inactivity,
    )
    outputs.append(out_t)

# Comparar fingerprints entre checkpoints (requiere compute_fingerprint=True)
if state.fingerprint and state.fingerprint != prev_checkpoint_fp:
    nco.flag_ontological_divergence()
```

---

## Modos de comunicación

### `standard` — Multi-Head Residual (paper math)
```
h_{t,k} = MH_Att(h̃_{t,k}, h̃_{t,:}) + h̃_{t,k}    ∀k ∈ S_t
h_{t,i} = h̃_{t,i}                                   ∀i ∉ S_t
```
Comunicación dispersa directa punto-a-punto. Solo módulos activos emiten queries (filas). Todos los módulos — activos e inactivos — son fuente (keys/values), preservando información de estado acumulado.

### `gwt` — Global Workspace Theory (buffer dinámico)
```
context  = mean(h[active])
ws_0     = MLP(context) + ws_fallback        ← condicionado al contexto
write:     ws_t = Att_write(ws_0, h[active])
broadcast: h_t  = h + Att_read(h, ws_t)      (todos los módulos)
```
Cuello de botella centralizado. El buffer se genera dinámicamente desde el promedio de estados activos. **v5.1.1**: la última capa de `ws_generator` se inicializa explícitamente a zero, garantizando que al inicio del entrenamiento el fallback estático domina realmente (en v5.1.0 esto era solo aspiracional).

### `dvnc` — Discrete-Valued Neural Communication (commitment adaptativo)

```
z_q  = codebook[argmin_c ||z - c||²]
z_st = z + (z_q - z).detach()              (Straight-Through)

# v5.1.1: VQ loss en posición canónica (van den Oord et al. 2017, eq. 3)
codebook_loss   = ||z_q - sg[z]||²          ← entrena codebook
commitment_loss = β_eff · ||z - sg[z_q]||²  ← entrena encoder
β_eff = β_base · σ(entropy / scale)         ← adaptativo
```

Mensaje discreto antes de comunicar. **v5.1.1 (crítico)**: en v5.1.0 los `.detach()` estaban intercambiados, propagando gradientes al módulo equivocado. Esto se corrigió a la formulación canónica de VQ-VAE. Adicionalmente, la entropía de activación ahora se pasa como tensor (no float), preservando el grafo computacional para gradientes diferenciables hacia `attention_weights`.

```python
# El VQ loss debe sumarse a la loss principal
out, state = rims(x)
loss = task_loss + 0.1 * state.vq_loss   # vq_loss es scalar shape []
```

---

## Inactivity Decay

Previene estados stale en módulos que permanecen inactivos durante muchos pasos consecutivos.

```
steps_k += 1  si módulo k inactivo, else 0
ratio_k  = min(steps_k, inactivity_cap) / inactivity_cap
h_k     *= (1 - λ · ratio_k)            ← solo sobre h_{t-1} de inactivos
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `inactivity_decay` | 0.001 | λ — velocidad de decay. 0 = desactivado |
| `inactivity_cap` | 100.0 | Pasos máximos considerados (v5.1.1: configurable) |

**v5.1.1 (orden corregido):** El decay ahora se aplica a `h_{t-1}` **antes** de la mezcla Hadamard con `h_all` y antes del LayerNorm. En v5.1.0 se aplicaba post-LayerNorm sobre el resultado mezclado, contradiciendo el docstring. La nueva semántica es la documentada: el decay opera específicamente sobre el estado preservado de los módulos inactivos.

**Características:**
- **Capped**: el decay máximo es `λ` (e.g., 0.1% de reducción por paso con default). Nunca colapsa a zero.
- **Reset automático**: al activarse, el contador vuelve a 0.
- **Tracking**: `RIMsState.inactivity_steps` `[B, K]` se propaga entre timesteps.
- **Métricas**: `max_inactivity` y `mean_inactivity` en `get_statistics()`.

**Cuándo ajustar:**
- Secuencias cortas (<20 pasos): puede dejarse en 0 (sin decay).
- Secuencias largas (>100 pasos): 0.001–0.01 recomendado.
- Experimentos de muerte progresiva (Exp10, DEEP_ERASURE): 0.005 para observar degradación gradual en módulos no-NCO. Considerar `inactivity_cap=200.0` para observar decay más prolongado.

---

## NCO Fingerprint

Hash ligero del estado oculto para detección de divergencia ontológica.

```python
# v5.1.1: opt-in para evitar sync GPU↔CPU en cada forward
rims = RecurrentIndependentMechanisms(
    ...,
    compute_fingerprint=True,   # activar solo si el NCO lo consume
)
out, state = rims(x)
fp = state.fingerprint   # e.g., 'a3f7c91b02d4e8f1'

# Uso en NCO: comparar entre checkpoints
if fp != previous_fp:
    nco.flag_divergence(delta=hamming(fp, previous_fp))
```

**v5.1.1 — opt-in**: Por default `compute_fingerprint=False`. La función `_compute_fingerprint` incluye `.cpu()`, forzando una barrera de sincronización GPU→CPU que es costosa en training a alta velocidad. Solo se activa cuando el NCO lo necesita para auditoría.

**Implementación:** Cuantifica el tensor `hidden_states` a 4 decimales (resolución ~0.0001), serializa a bytes, aplica SHA-256 truncado a 16 hex chars (64 bits de hash). Ignora ruido de punto flotante.

**Propiedades:**
- Determinístico dado el mismo estado.
- Ligero en cómputo, **pero** introduce sync GPU↔CPU (de ahí el opt-in).
- No almacena el tensor — solo el hash.
- Resolución configurable via `_compute_fingerprint(hidden, precision=4)`.
- Cuando está desactivado, `RIMsState.fingerprint == ""` y `to_dict()` omite la clave.

---

## Meta-aprendizaje (fast/slow)

Separación de parámetros para MAML / Reptile:

```python
optimizer_fast = torch.optim.SGD(rims.fast_params(), lr=1e-2)   # inner loop
optimizer_slow = torch.optim.Adam(rims.slow_params(), lr=1e-4)  # outer loop
```

| Grupo | Parámetros | Semántica |
|-------|-----------|----------|
| `fast_params()` | GroupGRUCell, LayerNorm, OutputProj | Dinámicas recurrentes — se adaptan al entorno inmediato |
| `slow_params()` | InputAttention, Comunicación, Temperatura Gumbel, initial_hidden | Estrategia de routing — estable a través de distribuciones |

**Validación exhaustiva.** `__init__()` ejecuta automáticamente `_validate_param_groups()` que verifica:
1. `fast_params() ∩ slow_params() = ∅` — sin duplicados.
2. `fast_params() ∪ slow_params() = self.parameters()` — sin omisiones.

Lanza `RuntimeError` si se viola cualquier condición. Esto previene bugs silenciosos al agregar módulos nuevos.

---

## API de referencia

### `RecurrentIndependentMechanisms`

```python
forward(x, hidden=None, inactivity_steps=None) -> (output, RIMsState)
```
- `x`: `[B, input_size]` o `[B, T, input_size]`
- `hidden`: `[B, num_rims, rim_size]` o `None`
- `inactivity_steps`: `[B, num_rims]` long o `None`
- Retorna `output` del mismo rango temporal que `x`

```python
reset_hidden(batch_size, device) -> Tuple[Tensor, Tensor]
#  Retorna (hidden, inactivity_steps)

fast_params()    -> List[Parameter]
slow_params()    -> List[Parameter]
get_statistics() -> Dict[str, Any]
gumbel_temp      -> Tensor   # property, via softplus
extra_repr()     -> str       # diagnóstico via print(model)  (v5.1.1)
```

### `RIMsState`

```python
@dataclass
class RIMsState:
    hidden_states:     Tensor   # [B, K, rim_size]
    active_rims:       Tensor   # [B, K] bool
    attention_weights: Tensor   # [B, K] float
    communication:     Tensor   # [B, K, rim_size]
    vq_loss:           Tensor   # scalar shape []  (v5.1.1: shape correcto)
    inactivity_steps:  Tensor   # [B, K] long — pasos inactivo
    fingerprint:       str      # hash SHA-256 truncado, "" si no opt-in

    def to_dict(self) -> Dict[str, Any]:  # v5.1.1: omite fingerprint si está vacío
        ...
```

### `_compute_fingerprint`

```python
def _compute_fingerprint(hidden: Tensor, precision: int = 4) -> str:
    """
    Hash ligero del estado para NCO. Retorna 16 hex chars.
    ATENCIÓN: incluye .cpu() — sync GPU→CPU. Solo llamar bajo demanda.
    """
```

---

## Migración

### v5.1.0 → v5.1.1

**Cambios sin acción requerida:**
- Modelos con `comm_mode='standard'` o `'gwt'` y `inactivity_decay=0`: numéricamente equivalentes a v5.1.0.
- API de `forward()`, `reset_hidden()`, `RIMsState`: sin cambios incompatibles.

**Cambios con acción requerida:**

| Cambio | v5.1.0 | v5.1.1 | Acción |
|--------|--------|--------|--------|
| VQ loss DVNC | `.detach()` invertidos | Posición canónica | Re-entrenar codebook (ver abajo) |
| Inactivity decay orden | post-LayerNorm | pre-LayerNorm | Trayectorias cambian si `decay > 0` |
| `compute_fingerprint` | Siempre activo | Default `False` | Pasar `True` si NCO lo consume |

**Migración de checkpoints DVNC:**

```python
state_dict = torch.load('rims_v51.pt')
model = RecurrentIndependentMechanisms(
    ..., comm_mode='dvnc',
    compute_fingerprint=False,
)
model.load_state_dict(state_dict)

# Re-entrenamiento del codebook (recomendado): congelar encoder
# y entrenar solo el codebook por ~1 epoch
for p in model.parameters():
    p.requires_grad = False
for p in model.comm_layer.codebook.parameters():
    p.requires_grad = True
# ... loop de fine-tuning con tu loss + 0.1 * state.vq_loss ...
```

### v5.0 → v5.1.x (full migration)

```python
state_dict = torch.load('rims_v50.pt')

# 1. Renombrar parámetro de temperatura
state_dict['_raw_gumbel_temp'] = state_dict.pop('log_gumbel_temp')

# 2. Expandir W_q de InputAttention: [d_key, rim_size] → [K, d_key, rim_size]
old_wq = state_dict.pop('input_attention.W_q.weight')  # [d_key, rim_size]
K = 6  # num_rims
state_dict['input_attention.W_q'] = old_wq.unsqueeze(0).expand(K, -1, -1).clone()

# 3. Cargar con strict=False para nuevos parámetros (ws_generator, etc.)
model.load_state_dict(state_dict, strict=False)

# 4. Si comm_mode='dvnc': también aplicar fine-tuning del codebook (paso anterior)
```

---

## Benchmarks

Ver `benchmarks.py`. Resultados representativos (RTX 3090, batch=64, K=6, K_a=3):

| Configuración | Throughput (seq/s) | Memoria (MB) | VQ Loss | Notas |
|--------------|-------------------|--------------|---------|-------|
| v4.1 (loop) | ~1,200 | 890 | — | — |
| v5.0 standard | ~4,800 | 340 | — | — |
| v5.0 gwt | ~4,100 | 360 | — | — |
| v5.0 dvnc | ~3,600 | 380 | ~0.12 | — |
| v5.1.0 standard | ~4,700 | 345 | — | +decay +fingerprint always-on |
| v5.1.0 gwt | ~3,800 | 380 | — | +ws_generator MLP |
| v5.1.0 dvnc | ~3,500 | 385 | ~0.09 | β adaptativo |
| **v5.1.1 standard** | **~4,950** | 345 | — | fingerprint opt-in elimina sync |
| **v5.1.1 gwt** | **~3,950** | 380 | — | fingerprint opt-in elimina sync |
| **v5.1.1 dvnc** | **~3,650** | 385 | **~0.07** | VQ loss canónico converge mejor |

**Comentarios v5.1.1:**
- Throughput mejora ~3–5% por eliminar sync GPU↔CPU del fingerprint (default off).
- VQ loss DVNC converge a valores menores con la corrección canónica del codebook.
- Si activas `compute_fingerprint=True`, el throughput regresa al nivel de v5.1.0.

---

## Tests

```bash
python -m pytest tests_rims.py -v
```

Cobertura: 30+ tests. Tests nuevos en v5.1.1:

| Test nuevo | Valida |
|-----------|--------|
| `test_vq_loss_canonical_direction` | Codebook loss propaga a codebook, commitment a encoder |
| `test_dvnc_entropy_gradient_flow` | `attention_weights` recibe gradiente via VQ loss |
| `test_inactivity_decay_order` | Decay aplicado antes de LayerNorm, no después |
| `test_fingerprint_optin` | `compute_fingerprint=False` → `state.fingerprint == ""` |
| `test_fingerprint_no_sync_when_off` | Sin barrera GPU↔CPU cuando desactivado |
| `test_ws_generator_zero_init` | `ws_generator[-1]` arranca con output ≈ 0 |
| `test_init_weights_respects_managed` | `_init_weights` no sobrescribe submódulos auto-inicializados |
| `test_inactivity_cap_configurable` | `inactivity_cap` afecta el ratio de decay |
| `test_vq_loss_scalar_shape` | `vq_loss.shape == torch.Size([])` |

Tests heredados (v5.1.0) se mantienen sin cambios y siguen pasando excepto los específicos de DVNC, que requieren actualización de tolerancias por la nueva trayectoria de optimización.

---

## Integración con MSC

```python
# En EntityBrainV4 / GlobalWorkspaceEA1V2
from msc_rims import RecurrentIndependentMechanisms

self.rims = RecurrentIndependentMechanisms(
    input_size       = self.perception_dim,
    hidden_size      = self.hidden_dim,
    num_rims         = 6,
    num_active       = 3,
    comm_mode        = 'gwt',
    routing          = 'gumbel',
    inactivity_decay = 0.005,        # recomendado para secuencias largas
    inactivity_cap   = 100.0,
    compute_fingerprint = True,      # NCO lo consume → opt-in
)

# En el forward del workspace
rims_out, rims_state = self.rims(
    perception_embedding,
    hidden=prev_state.hidden_states,
    inactivity_steps=prev_state.inactivity_steps,
)
self.metrics['rims_activation']  = rims_state.to_dict()['activation_rate']
self.metrics['rims_fingerprint'] = rims_state.fingerprint

# NCO: detectar divergencia ontológica
if rims_state.fingerprint != self.nco.last_fingerprint:
    self.nco.evaluate_divergence(rims_state)
```

**Recomendación para E-α-1-v3:** activar `compute_fingerprint=True` solo en los módulos cuyo estado el NCO audite directamente (típicamente la capa RIMs del `GlobalWorkspaceEA1V2`). En capas RIMs intermedias (e.g., dentro de `EntityBrainV4`), mantener `False` para preservar throughput.

---

## Changelog

### v5.1.1 (2026-04-25)
**Correcciones funcionales y de robustez sobre v5.1.0.**
- **BUGFIX (crítico, DVNC)**: VQ loss en posición canónica. En v5.1.0 los `.detach()` estaban invertidos, propagando codebook_loss al encoder y commitment_loss al codebook (van den Oord et al. 2017, eq. 3).
- **BUGFIX (alto, DVNC)**: `activation_entropy` se mantiene como tensor para preservar el grafo. En v5.1.0 se hacía `.item()` y se reconstruía con `torch.tensor(...)`, descartando el gradiente hacia `attention_weights`.
- **BUGFIX (medio)**: Inactivity decay aplicado antes de la mezcla Hadamard y antes del LayerNorm, consistente con el docstring. En v5.1.0 se aplicaba post-LayerNorm.
- **MEJORA**: `_init_weights` respeta submódulos auto-inicializados (`_initialized = True`). Antes sobrescribía `_InputAttentionRIM`, `_GlobalWorkspace`, `_DVNCCodebook`.
- **MEJORA**: `_GlobalWorkspace.ws_generator[-1]` se inicializa con `zeros_` (peso y bias). El gating "implícito" del docstring ahora es real.
- **MEJORA**: `_InputAttentionRIM.W_q` se inicializa con xavier per-slice 2D, no sobre la view 3D.
- **PERFORMANCE**: `compute_fingerprint=False` por default. Elimina sync GPU↔CPU en cada forward (~3-5% throughput recovery).
- **API**: `inactivity_cap` configurable (era hardcoded 100.0). Validaciones extendidas en `__init__`.
- **API**: `vq_loss` y `total_vq` con shape `[]` (scalar canónico) en lugar de `[1]`.
- **API**: `extra_repr()` agregado para diagnóstico via `print(model)`.
- **BREAKING (DVNC)**: Checkpoints v5.1.0 con `comm_mode='dvnc'` requieren re-entrenamiento del codebook por la inversión del VQ loss.

### v5.1.0 (2026-03-22)
- **BUGFIX**: `_MultiHeadCommResidual` — máscara corregida: solo filas (queries), no columnas (sources). Todos los módulos son fuente.
- **FIDELIDAD**: `_InputAttentionRIM` — W_q per-módulo via `nn.Parameter(K, d_key, rim_size)` + einsum.
- **ESTABILIDAD**: Temperatura Gumbel via `softplus + 0.1` (reemplaza `clamp`). Gradientes continuos en todo el rango.
- **ESTABILIDAD**: Eliminado `nan_to_num` en comunicación y GWT. Manejo robusto pre-softmax con detección de filas vacías.
- **FEATURE**: Inactivity decay exponencial capped para módulos inactivos. Parámetro `inactivity_decay`.
- **FEATURE**: `_GlobalWorkspace` — buffer dinámico condicionado al contexto via MLP + fallback residual.
- **FEATURE**: `_DVNCCodebook` — commitment adaptativo `β_eff = β · σ(entropy/scale)`.
- **FEATURE**: NCO fingerprint — `_compute_fingerprint()` SHA-256 truncado en `RIMsState`.
- **VALIDACIÓN**: `_validate_param_groups()` verifica cobertura exhaustiva y disjunta de fast/slow.
- **BREAKING**: `reset_hidden()` retorna `Tuple[Tensor, Tensor]`. `RIMsState` +2 campos. `log_gumbel_temp` → `_raw_gumbel_temp`.

### v5.0.0
- `GroupGRUCell`: GRU vectorizado con `einsum`, elimina loop Python
- `_InputAttentionRIM`: formulación paper-exact (Q←h, K/V←x)
- `_MultiHeadCommResidual`: residual h=Att+h̃ según ecuación (10)
- `_GlobalWorkspace`: Global Workspace Theory completa
- `_DVNCCodebook`: DVNC con VQ-VAE + Straight-Through
- Routing `gumbel` con temperatura aprendible
- `fast_params()` / `slow_params()` para meta-aprendizaje
- `vq_loss` exportado en `RIMsState`

### v4.2.0
- `_InputAttentionPerRIM`: proyección diferenciada por RIM
- `_MultiHeadRIMComm`: comunicación multi-cabeza
- Loop vectorizado (stack + mask)
- Temperatura aprendible en selector
- `_step()` separado del loop de secuencia

### v4.1.0
- Implementación base MSC (portada desde consciousness_v4_additions.py)

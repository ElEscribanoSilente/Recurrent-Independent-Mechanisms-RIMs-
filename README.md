# Recurrent Independent Mechanisms (RIMs) v5.1
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
11. [Migración v5.0 → v5.1](#migración-v50--v51)
12. [Benchmarks](#benchmarks)
13. [Tests](#tests)
14. [Integración con MSC](#integración-con-msc)
15. [Changelog](#changelog)

---

## Descripción

Los **Recurrent Independent Mechanisms** (Goyal et al., 2019) fragmentan el estado oculto en *K_t* módulos funcionales independientes. En cada paso temporal, un selector diferenciable activa los *K_a* módulos más relevantes (top-k). Los inactivos conservan su estado (con decay suave v5.1), protegiendo el conocimiento acumulado contra interferencia destructiva.

**Por qué importa para MSC:**
Los RIMs implementan directamente el *Prior de Consciencia* de Bengio: un cuello de botella de atención que emula la especialización cortical. Son el substrato computacional del `GlobalWorkspaceEA1V2` y del experimento de continuidad ontológica (NCO). El fingerprint de estado (v5.1) permite al NCO detectar divergencia ontológica entre checkpoints sin almacenar tensores completos.

### Ventajas frente a arquitecturas monolíticas

| Propiedad | LSTM/GRU monolítico | RIMs v5.1 |
|-----------|-------------------|-----------|
| Representación del estado | Vector único denso | *K* módulos independientes |
| Routing de entrada | Homogéneo a toda la red | Top-k competitivo |
| Parámetros | Matriz densa global | Bloques por módulo (GroupGRUCell) |
| Actualización | Todo el estado en cada paso | Solo módulos activos |
| Comunicación | Densa e interconectada | Dispersa (standard / GWT / DVNC) |
| Olvido catastrófico | Alta vulnerabilidad | Mitigado por congelamiento + decay |
| Generalización OOD | Pobre | Robusta por especialización |
| Estado stale en inactivos | N/A | Decay exponencial capped (v5.1) |
| Integridad ontológica | Sin soporte | Fingerprint SHA-256 para NCO (v5.1) |

---

## Arquitectura

```
x_t ──► InputAttentionRIM ──► scores [B, K]
        (Q per-módulo via       │
         einsum, K/V desde x_t) ▼
                            Selector top-k (STE / Gumbel-softplus)
                                 │
                    ┌────────────▼────────────┐
                    │    GroupGRUCell          │  ← FAST params
                    │    (einsum vectorizado)  │
                    └────────────┬────────────┘
                                 │
                    h_t = M ⊙ h_new + (1-M) ⊙ h_{t-1}   (Hadamard)
                                 │
                    ┌────────────▼────────────┐
                    │  Inactivity Decay       │  (v5.1)
                    │  h *= (1 - λ·ratio)     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Comunicación (modo):    │  ← SLOW params
                    │  • standard: MH residual │
                    │  • gwt:  Dynamic GWT     │
                    │  • dvnc: Adaptive VQ     │
                    └────────────┬────────────┘
                                 │
                    output_t = LayerNorm(h + OutputProj(h))
                                 │
                    fingerprint = SHA256(quantize(h))   (v5.1)
```

### Sub-módulos

| Clase | Descripción | Cambios v5.1 |
|-------|------------|--------------|
| `GroupGRUCell` | GRU K grupos vectorizado (`einsum`), O(1) kernels CUDA | — |
| `_InputAttentionRIM` | Atención de entrada paper-exact: Q←h, K/V←x | W_q per-módulo via einsum |
| `_MultiHeadCommResidual` | MH-Att con residual: h=Att(h̃)+h̃ | Máscara corregida: solo filas, no columnas |
| `_GlobalWorkspace` | GWT: competencia→buffer→broadcast | Buffer dinámico condicionado al contexto |
| `_DVNCCodebook` | DVNC: cuantización VQ-VAE + STE | Commitment adaptativo por entropía |
| `RIMsState` | Dataclass de estado completo | +`inactivity_steps`, +`fingerprint` |
| `RecurrentIndependentMechanisms` | Módulo principal | Inactivity decay, softplus temp, validación fast/slow |

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
│   └── tests_rims.py          # 30 tests (v5.1)
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
print(state.fingerprint)       # 'a3f7c91b02d4e8f1' (hash NCO)
print(state.to_dict())
# {'num_active': 3.0, 'activation_rate': 0.5, 'attention_entropy': ...,
#  'comm_norm': ..., 'vq_loss': 0.0, 'max_inactivity': 12,
#  'mean_inactivity': 4.2, 'fingerprint': 'a3f7c91b02d4e8f1'}
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

    # v5.1: Decay de módulos inactivos
    inactivity_decay = 0.001,      # λ — velocidad de decay (0 = desactivado)
)
```

### Paso con estado previo (autoregresivo)

```python
# v5.1: reset_hidden retorna (hidden, inactivity_steps)
hidden, inactivity = rims.reset_hidden(batch_size=32, device=device)
outputs = []
for t in range(seq_len):
    out_t, state = rims(
        x[:, t],
        hidden=state.hidden_states if t > 0 else hidden,
        inactivity_steps=state.inactivity_steps if t > 0 else inactivity,
    )
    outputs.append(out_t)

# Comparar fingerprints entre checkpoints
if state.fingerprint != prev_checkpoint_fp:
    nco.flag_ontological_divergence()
```

---

## Modos de comunicación

### `standard` — Multi-Head Residual (paper math)
```
h_{t,k} = MH_Att(h̃_{t,k}, h̃_{t,:}) + h̃_{t,k}    ∀k ∈ S_t
h_{t,i} = h̃_{t,i}                                   ∀i ∉ S_t
```
Comunicación dispersa directa punto-a-punto. **v5.1**: Solo módulos activos emiten queries (filas). Todos los módulos — activos e inactivos — son fuente (keys/values), preservando información de estado acumulado.

### `gwt` — Global Workspace Theory (buffer dinámico v5.1)
```
context  = mean(h[active])
ws_0     = MLP(context) + ws_fallback        ← v5.1: condicionado al contexto
write:     ws_t = Att_write(ws_0, h[active])
broadcast: h_t  = h + Att_read(h, ws_t)      (todos los módulos)
```
Cuello de botella centralizado. El buffer se genera dinámicamente desde el promedio de estados activos, adaptando la capacidad del workspace al contexto de cada batch/timestep. El fallback estático garantiza estabilidad durante las primeras épocas de entrenamiento.

### `dvnc` — Discrete-Valued Neural Communication (commitment adaptativo v5.1)
```
z_q  = codebook[argmin_c ||z - c||²]
z_st = z + (z_q - z).detach()              (Straight-Through)

β_eff = β_base · σ(entropy / scale)         ← v5.1: adaptativo
```
Mensaje discreto antes de comunicar. **v5.1**: El commitment loss se modula por la entropía de activación. Bajo estrés (pocos módulos activos, muerte progresiva), β baja automáticamente → el codebook se relaja para permitir mayor expresividad. En régimen sano, β alta → vocabulario estricto.

```python
# El VQ loss (ahora adaptativo) debe sumarse a la loss principal
out, state = rims(x)
loss = task_loss + 0.1 * state.vq_loss
```

---

## Inactivity Decay

**Nuevo en v5.1.** Previene estados stale en módulos que permanecen inactivos durante muchos pasos consecutivos.

```
steps_k += 1  si módulo k inactivo, else 0
ratio_k  = min(steps_k, 100) / 100
h_k     *= (1 - λ · ratio_k)
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `inactivity_decay` | 0.001 | λ — velocidad de decay. 0 = desactivado |

**Características:**
- **Capped a 100 pasos**: el decay máximo es `λ` (e.g., 0.1% de reducción por paso con default). Nunca colapsa a zero.
- **Reset automático**: al activarse, el contador vuelve a 0.
- **Tracking**: `RIMsState.inactivity_steps` `[B, K]` se propaga entre timesteps.
- **Métricas**: `max_inactivity` y `mean_inactivity` en `get_statistics()`.

**Cuándo ajustar:**
- Secuencias cortas (<20 pasos): puede dejarse en 0 (sin decay).
- Secuencias largas (>100 pasos): 0.001–0.01 recomendado.
- Experimentos de muerte progresiva (Exp10, DEEP_ERASURE): 0.005 para observar degradación gradual en módulos no-NCO.

---

## NCO Fingerprint

**Nuevo en v5.1.** Hash ligero del estado oculto para detección de divergencia ontológica.

```python
# Automático — se computa al final de forward()
out, state = rims(x)
fp = state.fingerprint   # e.g., 'a3f7c91b02d4e8f1'

# Uso en NCO: comparar entre checkpoints
if fp != previous_fp:
    nco.flag_divergence(delta=hamming(fp, previous_fp))
```

**Implementación:** Cuantifica el tensor `hidden_states` a 4 decimales (resolución ~0.0001), serializa a bytes, aplica SHA-256 truncado a 16 hex chars (64 bits de hash). Ignora ruido de punto flotante.

**Propiedades:**
- Determinístico dado el mismo estado.
- Ligero (~0.1ms overhead por paso).
- No almacena el tensor — solo el hash.
- Resolución configurable via `_compute_fingerprint(hidden, precision=4)`.

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

**v5.1: Validación exhaustiva.** `__init__()` ejecuta automáticamente `_validate_param_groups()` que verifica:
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
- `inactivity_steps`: `[B, num_rims]` long o `None` (v5.1)
- Retorna `output` del mismo rango temporal que `x`

```python
reset_hidden(batch_size, device) -> Tuple[Tensor, Tensor]
#  Retorna (hidden, inactivity_steps)  — BREAKING CHANGE v5.1

fast_params()    -> List[Parameter]
slow_params()    -> List[Parameter]
get_statistics() -> Dict[str, Any]
gumbel_temp      -> Tensor   # property, via softplus (v5.1)
```

### `RIMsState`

```python
@dataclass
class RIMsState:
    hidden_states:     Tensor   # [B, K, rim_size]
    active_rims:       Tensor   # [B, K] bool
    attention_weights: Tensor   # [B, K] float
    communication:     Tensor   # [B, K, rim_size]
    vq_loss:           Tensor   # scalar (0 si comm_mode != 'dvnc')
    inactivity_steps:  Tensor   # [B, K] long — pasos inactivo (v5.1)
    fingerprint:       str      # hash SHA-256 truncado (v5.1)

    def to_dict(self) -> Dict[str, Any]: ...
```

### `_compute_fingerprint`

```python
def _compute_fingerprint(hidden: Tensor, precision: int = 4) -> str:
    """Hash ligero del estado para NCO. Retorna 16 hex chars."""
```

---

## Migración v5.0 → v5.1

### Breaking changes

| Cambio | v5.0 | v5.1 | Acción requerida |
|--------|------|------|-----------------|
| `reset_hidden()` | Retorna `Tensor` | Retorna `Tuple[Tensor, Tensor]` | Desempaquetar: `h, inact = rims.reset_hidden(...)` |
| `forward()` | `(x, hidden)` | `(x, hidden, inactivity_steps)` | Pasar `inactivity_steps` en loops autoregresivos |
| `RIMsState` | 5 campos | 7 campos (+`inactivity_steps`, +`fingerprint`) | Actualizar destructuring si aplica |
| `log_gumbel_temp` | Parámetro directo | `_raw_gumbel_temp` (interno) | Renombrar en checkpoints |
| `_DVNCCodebook.forward()` | `(hidden, active_mask)` | `(hidden, active_mask, activation_entropy)` | Solo afecta si se llama directamente |

### Migración de checkpoints

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
| v5.1 standard | ~4,700 | 345 | — | +decay +fingerprint overhead |
| v5.1 gwt (dynamic) | ~3,800 | 380 | — | +ws_generator MLP |
| v5.1 dvnc (adaptive) | ~3,500 | 385 | ~0.09 | β adaptativo reduce VQ loss |

El overhead de v5.1 es ~2-7% en throughput, compensado por mayor estabilidad numérica y mejor convergencia del codebook DVNC.

---

## Tests

```bash
python -m pytest tests_rims.py -v
```

Cobertura: 30 tests (22 originales + 8 nuevos v5.1):

| Test nuevo | Valida |
|-----------|--------|
| `test_comm_mask_rows_not_cols` | Módulos inactivos son fuente en comunicación |
| `test_per_module_wq` | W_q shapes [K, d_key, rim_size] y gradientes por módulo |
| `test_gumbel_softplus_gradient` | Gradiente continuo de temperatura en todo el rango |
| `test_softmax_no_nan` | Sin NaN en comunicación con 0 activos edge case |
| `test_inactivity_decay_counter` | Contadores incrementan/resetean correctamente |
| `test_inactivity_decay_norm` | Norma del estado decrece bajo inactividad prolongada |
| `test_fingerprint_deterministic` | Mismo estado → mismo hash |
| `test_fast_slow_exhaustive` | `fast ∪ slow = all`, `fast ∩ slow = ∅` |

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
    inactivity_decay = 0.005,   # v5.1: recomendado para secuencias largas
)

# En el forward del workspace (v5.1: propagar inactivity_steps)
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

---

## Changelog

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

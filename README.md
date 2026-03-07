# Recurrent Independent Mechanisms (RIMs) v5.0
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
7. [Meta-aprendizaje (fast/slow)](#meta-aprendizaje-fastslow)
8. [API de referencia](#api-de-referencia)
9. [Benchmarks](#benchmarks)
10. [Tests](#tests)
11. [Integración con MSC](#integración-con-msc)
12. [Changelog](#changelog)

---

## Descripción

Los **Recurrent Independent Mechanisms** (Goyal et al., 2019) fragmentan el estado oculto en *K_t* módulos funcionales independientes. En cada paso temporal, un selector diferenciable activa los *K_a* módulos más relevantes (top-k). Los inactivos conservan su estado intacto, protegiendo el conocimiento acumulado contra interferencia destructiva.

**Por qué importa para MSC:**
Los RIMs implementan directamente el *Prior de Consciencia* de Bengio: un cuello de botella de atención que emula la especialización cortical. Son el substrato computacional del `GlobalWorkspaceEA1V2` y del experimento de continuidad ontológica (NCO).

### Ventajas frente a arquitecturas monolíticas

| Propiedad | LSTM/GRU monolítico | RIMs v5.0 |
|-----------|-------------------|-----------|
| Representación del estado | Vector único denso | *K* módulos independientes |
| Routing de entrada | Homogéneo a toda la red | Top-k competitivo |
| Parámetros | Matriz densa global | Bloques por módulo (GroupGRUCell) |
| Actualización | Todo el estado en cada paso | Solo módulos activos |
| Comunicación | Densa e interconectada | Dispersa (standard / GWT / DVNC) |
| Olvido catastrófico | Alta vulnerabilidad | Mitigado por congelamiento de inactivos |
| Generalización out-of-distribution | Pobre | Robusta por especialización |

---

## Arquitectura

```
x_t ──► InputAttentionRIM ──► scores [B, K]
        (Q desde h_{t-1,i}       │
         K/V desde x_t)          ▼
                            Selector top-k (STE / Gumbel)
                                 │
                    ┌────────────▼────────────┐
                    │    GroupGRUCell          │  ← FAST params
                    │    (einsum vectorizado)  │
                    └────────────┬────────────┘
                                 │
                    h_t = M ⊙ h_new + (1-M) ⊙ h_{t-1}   (Hadamard)
                                 │
                    ┌────────────▼────────────┐
                    │  Comunicación (modo):    │  ← SLOW params
                    │  • standard: MH residual │
                    │  • gwt:  Global Workspace│
                    │  • dvnc: VQ codebook     │
                    └────────────┬────────────┘
                                 │
                    output_t = LayerNorm(h + OutputProj(h))
```

### Sub-módulos

| Clase | Descripción | Referencia |
|-------|------------|------------|
| `GroupGRUCell` | GRU K grupos vectorizado (`einsum`), O(1) kernels CUDA | §"Cuello de Botella Iterativo" |
| `_InputAttentionRIM` | Atencion de entrada paper-exact: Q←h, K/V←x | Ecuaciones (1)-(5) |
| `_MultiHeadCommResidual` | MH-Att con residual: h=Att(h̃)+h̃ | Ecuación (10) |
| `_GlobalWorkspace` | GWT: competencia→buffer→broadcast | §"GWT" |
| `_DVNCCodebook` | DVNC: cuantización VQ-VAE + STE | §"DVNC" |
| `RIMsState` | Dataclass de estado completo | — |
| `RecurrentIndependentMechanisms` | Módulo principal | — |

---

## Instalación

```bash
# Dentro del entorno MSC
pip install torch>=2.0  # único requisito externo

# Copiar al paquete de consciencia
cp rims.py src/consciousness/layers/rims.py
```

El módulo asume la existencia de `.base.ConsciousnessLayerBase` y `.config.LayerConfig` del stack MSC. Para uso standalone, hereda de `nn.Module` directamente.

---

## Uso rápido

```python
from consciousness.layers.rims import RecurrentIndependentMechanisms

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
print(state.to_dict())
# {'num_active': 3.0, 'activation_rate': 0.5, 'attention_entropy': ...,
#  'comm_norm': ..., 'vq_loss': 0.0}
```

---

## Configuración completa

```python
rims = RecurrentIndependentMechanisms(
    input_size   = 128,
    hidden_size  = 384,
    num_rims     = 6,
    num_active   = 3,

    # Comunicación: 'standard' | 'gwt' | 'dvnc'
    comm_mode    = 'gwt',
    ws_slots     = 2,          # slots del workspace (solo gwt)
    num_codes    = 128,        # vocab del codebook (solo dvnc)

    # Routing: 'ste' (Straight-Through) | 'gumbel'
    routing      = 'gumbel',
    gumbel_temp  = 1.0,        # temperatura inicial (aprendible)

    num_heads    = 4,
    dropout      = 0.1,
)
```

### Paso con estado previo (autoregresivo)

```python
hidden = rims.reset_hidden(batch_size=32, device=device)
outputs = []
for t in range(seq_len):
    out_t, state = rims(x[:, t], hidden=state.hidden_states)
    outputs.append(out_t)
```

---

## Modos de comunicación

### `standard` — Multi-Head Residual (paper math)
```
h_{t,k} = MH_Att(h̃_{t,k}, h̃_{t,:}) + h̃_{t,k}    ∀k ∈ S_t
h_{t,i} = h̃_{t,i}                                   ∀i ∉ S_t
```
Comunicación dispersa directa punto-a-punto entre módulos activos.

### `gwt` — Global Workspace Theory
```
write:     ws_t = Att_write(ws, hidden[active])
broadcast: h_t  = h + Att_read(h, ws_t)          (todos los módulos)
```
Cuello de botella centralizado. Fuerza compresión máxima: solo lo esencial pasa al workspace. Recomendado para tareas de razonamiento y zero-shot.

### `dvnc` — Discrete-Valued Neural Communication
```
z_q  = codebook[argmin_c ||z - c||²]
z_st = z + (z_q - z).detach()          (Straight-Through)
```
Mensaje discreto antes de comunicar. Previene ruido de punto flotante entre módulos. Fuerza un "lenguaje interno" simbólico. Exporta `vq_loss` para el optimizador.

```python
# El VQ loss debe sumarse a la loss principal
out, state = rims(x)
loss = task_loss + 0.1 * state.vq_loss
```

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

---

## API de referencia

### `RecurrentIndependentMechanisms`

```python
forward(x, hidden=None) -> (output, RIMsState)
```
- `x`: `[B, input_size]` o `[B, T, input_size]`
- `hidden`: `[B, num_rims, rim_size]` o `None`
- Retorna `output` del mismo rango temporal que `x`

```python
reset_hidden(batch_size, device) -> Tensor  # [B, K, rim_size]
fast_params()  -> List[Parameter]
slow_params()  -> List[Parameter]
get_statistics() -> Dict[str, Any]
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

    def to_dict(self) -> Dict[str, Any]: ...
```

---

## Benchmarks

Ver `benchmarks.py`. Resultados representativos (RTX 3090, batch=64, K=6, K_a=3):

| Configuración | Throughput (seq/s) | Memoria (MB) | VQ Loss |
|--------------|-------------------|--------------|---------|
| v4.1 (loop) | ~1,200 | 890 | — |
| v5.0 standard | ~4,800 | 340 | — |
| v5.0 gwt | ~4,100 | 360 | — |
| v5.0 dvnc | ~3,600 | 380 | ~0.12 |

El `GroupGRUCell` vectorizado consigue **~4x speedup** sobre el loop de `GRUCell` individual.

---

## Tests

```bash
python -m pytest tests_rims.py -v
```

Cobertura: 22 tests — validación matemática, sparsidad, gradientes, modos de comunicación, routing, fast/slow params, manejo de secuencias y entradas puntuales.

---

## Integración con MSC

```python
# En EntityBrainV4 / GlobalWorkspaceEA1V2
from consciousness.layers.rims import RecurrentIndependentMechanisms

self.rims = RecurrentIndependentMechanisms(
    input_size  = self.perception_dim,
    hidden_size = self.hidden_dim,
    num_rims    = 6,
    num_active  = 3,
    comm_mode   = 'gwt',    # coherente con GWT pillar del workspace
    routing     = 'gumbel',
)

# En el forward del workspace
rims_out, rims_state = self.rims(perception_embedding, hidden=prev_hidden)
self.metrics['rims_activation'] = rims_state.activation_rate
```

---

## Changelog

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

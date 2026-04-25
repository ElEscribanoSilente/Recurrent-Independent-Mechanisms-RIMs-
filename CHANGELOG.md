# CHANGELOG — Recurrent Independent Mechanisms (RIMs)

## [5.1.1] — 2026-04-25

### Resumen

Nueve correcciones aplicadas sobre v5.1.0 tras revisión estática completa: dos bugs funcionales que propagaban gradientes incorrectamente (VQ loss invertido y entropía con grafo roto), tres fixes de robustez en inicialización, una corrección del orden de operaciones en el inactivity decay, eliminación de un sync GPU↔CPU costoso en cada forward, y dos mejoras menores de tipado y configurabilidad.

**Breaking changes (limitados)**: Checkpoints DVNC entrenados con v5.1.0 requieren re-entrenamiento o ajuste fino del codebook por la inversión del VQ loss. Modelos con `comm_mode='standard'` o `'gwt'` y `inactivity_decay=0` son numéricamente equivalentes.

---

### 🐛 BUGFIX — VQ loss invertido (`_DVNCCodebook`)

**Severidad: CRÍTICA (DVNC)**

| Aspecto | v5.1.0 | v5.1.1 |
|---------|--------|--------|
| `codebook_loss` | `(z_q.detach() - z)²` → propaga al **encoder** | `(z_q - z.detach())²` → propaga al **codebook** |
| `commitment_loss` | `(z_q - z.detach())²` → propaga al **codebook** | `(z - z_q.detach())²` → propaga al **encoder** |
| Resultado | Roles intercambiados | Posición canónica (van den Oord 2017, eq. 3) |

**Problema**: Los `.detach()` estaban en posición invertida respecto a la formulación canónica VQ-VAE. El término que debía entrenar el codebook entrenaba al encoder y viceversa. En la práctica, esto degrada la convergencia del codebook y debilita el commitment del encoder hacia los códigos discretos.

---

### 🐛 BUGFIX — `activation_entropy` rompía el grafo (`_DVNCCodebook`)

**Severidad: ALTA (DVNC)**

**Problema**: En `_step()` se calculaba la entropía con `.item()`, convirtiendo a `float` Python y descartando el gradiente hacia `attention_weights`. Luego dentro del codebook se reconstruía via `torch.tensor(...)` en cada paso, alocando memoria sin recuperar el grafo.

**Fix**:
- `_DVNCCodebook.forward()` ahora acepta `activation_entropy: Optional[Any]` (tensor o float).
- Si es **tensor**: la modulación de `beta_eff` mantiene el gradiente diferenciable.
- Si es **float**: se calcula sigmoid en Python sin allocar tensor por paso.
- `_step()` ahora pasa **tensor** (gradiente vivo).

---

### 🔧 MEJORA — Inicialización per-slice de `W_q` (`_InputAttentionRIM`)

| Aspecto | v5.1.0 | v5.1.1 |
|---------|--------|--------|
| Init | `xavier_uniform_(W_q.view(K, d_key, rim_size))` | `for k: xavier_uniform_(W_q.data[k])` |
| `fan_in/fan_out` | No estándar para tensor 3D | Cálculo correcto en 2D |

**Problema**: `xavier_uniform_` sobre un tensor 3D usa `_calculate_fan_in_and_fan_out` con interpretación de "receptive field" que no aplica aquí. Además, `view(...)` sobre un tensor que ya tiene esa forma es redundante y confuso.

**Fix**: Loop explícito por slice 2D — cada `W_q[k]` se inicializa con xavier estándar.

---

### 🔧 MEJORA — `_init_weights` respeta submódulos auto-inicializados

**Problema**: `RecurrentIndependentMechanisms._init_weights()` recorría `self.modules()` y reinicializaba todos los `nn.Linear` con xavier, sobrescribiendo la inicialización específica de `_InputAttentionRIM` y de la última capa zero-init de `_GlobalWorkspace.ws_generator`.

**Fix**: Submódulos especializados marcan `self._initialized = True` en `__init__`. El padre detecta este flag y los excluye del recorrido. Aplicado a `_InputAttentionRIM`, `_GlobalWorkspace`, `_DVNCCodebook`.

---

### 🔧 MEJORA — Gating real en `_GlobalWorkspace._generate_workspace`

| Aspecto | v5.1.0 | v5.1.1 |
|---------|--------|--------|
| `ws_generator[-1]` init | xavier (lejos de zero) | `zeros_(weight)` + `zeros_(bias)` |
| Comportamiento al inicio | Output ≠ 0, fallback no domina | Output ≈ 0, fallback domina |
| Docstring vs realidad | Inconsistente | Consistente |

**Problema**: El docstring afirmaba "el MLP se inicializa cerca de zero, así al inicio ws ≈ 0 y el fallback domina via la suma residual". Esto era falso — xavier no inicializa cerca de zero. El gating "implícito" no existía.

**Fix**: Inicialización explícita a zero de la última capa del `ws_generator`. Ahora la afirmación del docstring es verdadera y el modelo arranca usando solo el fallback estático, aprendiendo gradualmente a usar el path dinámico.

---

### 🔧 MEJORA — Fingerprint NCO opt-in (sync GPU↔CPU eliminado)

| Aspecto | v5.1.0 | v5.1.1 |
|---------|--------|--------|
| Cómputo | Cada `forward()` | Solo si `compute_fingerprint=True` |
| Sync GPU→CPU | Forzado en cada paso | Solo bajo demanda |
| Default | Activado | **Desactivado** |

**Problema**: `_compute_fingerprint` incluye `.cpu()`, forzando una barrera de sincronización en cada forward. En training a alta velocidad esto introduce latencia significativa que no es necesaria salvo para auditoría puntual del NCO.

**Fix**:
- Nuevo kwarg `compute_fingerprint: bool = False` en `__init__`.
- `_compute_fingerprint()` documentado como "sync caro, llamar bajo demanda".
- `RIMsState.fingerprint` queda como `""` cuando está desactivado.
- `RIMsState.to_dict()` omite la clave `'fingerprint'` cuando está vacía.

---

### 🐛 BUGFIX — Inactivity decay aplicado en orden incorrecto (`_step`)

**Severidad: MEDIA**

| Aspecto | v5.1.0 | v5.1.1 |
|---------|--------|--------|
| Orden | LayerNorm → Decay | Decay → Mezcla → LayerNorm |
| Sobre qué opera | Output post-norm de todos | Estado preservado `h_{t-1}` de inactivos |
| Consistencia con docstring | No | Sí |

**Problema**: En v5.1.0 el decay se aplicaba **después** del LayerNorm sobre el resultado mezclado, contradiciendo el docstring que sugería operar sobre el estado stale preservado de módulos inactivos. El LayerNorm normalizaba primero y luego el decay encogía, produciendo dinámicas distintas a las documentadas.

**Fix**: Reordenado:
1. Calcular `h_decayed = decay(h_{t-1}, active, inactivity_steps)` — solo afecta a inactivos.
2. Calcular `h_all = GroupGRU(x_attended, h_{t-1})` — sobre el hidden original (no contaminar dinámica de activos).
3. Mezcla Hadamard: `h_t = M·h_all + (1-M)·h_decayed`.
4. LayerNorm al final.

---

### 🔧 MEJORA — `inactivity_cap` parametrizable

| Aspecto | v5.1.0 | v5.1.1 |
|---------|--------|--------|
| `cap` | Hardcoded `100.0` | `inactivity_cap: float = 100.0` |
| Validación | — | `> 0` |

Magic number eliminado. Ahora configurable por instancia, con validación en `__init__`.

---

### 🔧 MEJORA — Tensores escalares con shape `[]` en lugar de `[1]`

**Problema**: `torch.zeros(1)` produce un tensor de shape `[1]`, no escalar. Llamar `.item()` funciona pero levanta deprecation warning en versiones recientes de PyTorch.

**Fix**: `torch.zeros(())` en `vq_loss` y `total_vq` → escalar canónico shape `[]`.

---

### 🆕 API — `extra_repr` para diagnóstico

`print(model)` ahora muestra hiperparámetros relevantes inline:

```
RecurrentIndependentMechanisms(
  input_size=128, hidden_size=384, num_rims=6, num_active=3,
  rim_size=64, comm_mode='dvnc', routing='gumbel',
  inactivity_decay=0.001, inactivity_cap=100.0,
  compute_fingerprint=False
  ...
)
```

---

### 🆕 API — Validaciones extendidas en `__init__`

| Validación | v5.1.0 | v5.1.1 |
|---|---|---|
| `num_active ∈ [1, num_rims]` | ✓ | ✓ |
| `hidden_size % num_rims == 0` | ✓ | ✓ |
| `rim_size % num_heads == 0` | ✓ | ✓ |
| `inactivity_cap > 0` | — | ✓ |
| `inactivity_decay ∈ [0, 1)` | — | ✓ |

---

### ⚠️ Breaking Changes

1. **VQ loss DVNC**: La dirección de optimización del codebook cambió. Modelos pre-entrenados con `comm_mode='dvnc'` en v5.1.0 NO son compatibles sin re-entrenamiento o fine-tuning del codebook.
2. **Inactivity decay**: El orden de aplicación cambió. Modelos con `inactivity_decay > 0` producirán trayectorias numéricamente distintas a v5.1.0 (el comportamiento nuevo es el documentado correctamente).
3. **Default de `compute_fingerprint`**: Pasó de implícitamente activado (siempre) a `False`. Si tu pipeline NCO depende del fingerprint, debes pasar `compute_fingerprint=True` explícitamente.

### Compatibilidad numérica garantizada

Modelos con la siguiente configuración son numéricamente equivalentes a v5.1.0:
- `comm_mode='standard'` o `'gwt'`
- `inactivity_decay=0` (decay desactivado)
- `compute_fingerprint=False`

### Migración v5.1.0 → v5.1.1

```python
# Cargar checkpoint v5.1.0
state_dict = torch.load('rims_v51.pt')
model = RecurrentIndependentMechanisms(
    ...,
    comm_mode='dvnc',
    compute_fingerprint=False,  # explicito (default cambio)
)
model.load_state_dict(state_dict)

# Si comm_mode='dvnc': re-entrenar codebook unos pasos
# (recomendado: congelar encoder y entrenar solo codebook por ~1 epoch)
for p in model.parameters(): p.requires_grad = False
for p in model.comm_layer.codebook.parameters(): p.requires_grad = True
# ... loop de fine-tuning ...
```

---

### Conteo de cambios

| Categoría | Cantidad |
|-----------|----------|
| Bugfix crítico (gradiente) | 2 |
| Bugfix (orden de operaciones) | 1 |
| Mejora de inicialización | 3 |
| Mejora de performance | 1 |
| Configurabilidad / API | 2 |
| **Total** | **9** |

---

## [5.1.0] — 2026-03-22

### Resumen

Nueve mejoras aplicadas sobre v5.0.0: una corrección funcional crítica en la máscara de comunicación, tres mejoras de estabilidad numérica, dos mejoras arquitectónicas (inactivity decay + workspace dinámico), una mejora de fidelidad al paper (W_q per-módulo), una integración con TREC-BAC (fingerprint NCO), y una validación de meta-learning (cobertura fast/slow).

**Breaking changes**: `reset_hidden()` ahora retorna `Tuple[Tensor, Tensor]`. `forward()` acepta nuevo kwarg `inactivity_steps`. `RIMsState` tiene dos campos nuevos.

---

### 🐛 BUGFIX — Máscara de comunicación (`_MultiHeadCommResidual`)

**Severidad: CRÍTICA**

| Aspecto | v5.0 | v5.1 |
|---------|------|------|
| Máscara | Columnas (sources) enmascaradas | Solo filas (queries) enmascaradas |
| Efecto  | Módulos inactivos bloqueados como fuente | Todos los módulos son fuente |

**Problema**: v5.0 enmascaraba columnas con `active_mask`, bloqueando keys/values de módulos inactivos. El paper especifica que *todos* los módulos son fuente y solo los activos emiten queries. Un módulo inactivo con estado relevante acumulado quedaba invisible para los activos.

**Fix**: La máscara se aplica a filas (queries de inactivos → atención zero) y el softmax se computa sobre todas las columnas sin restricción. Se elimina el path `masked_fill(-inf) → softmax → nan_to_num` y se reemplaza por `softmax → multiplicar por row_mask`.

---

### 🔧 MEJORA — W_q per-módulo en `_InputAttentionRIM`

| Aspecto | v5.0 | v5.1 |
|---------|------|------|
| `W_q`   | `nn.Linear` compartido | `nn.Parameter(K, d_key, rim_size)` + einsum |
| Fidelidad | Aproximada | Exacta (paper eq. 3) |

**Antes**: Un único `nn.Linear(rim_size, d_key)` se aplicaba idénticamente a todos los módulos. **Ahora**: Cada módulo k tiene su propia matriz `W_q^{(k)}` implementada como tensor 3D con einsum `'bkr,kdr->bkd'`, consistente con el patrón ya usado en `GroupGRUCell`.

---

### 🔧 MEJORA — Temperatura Gumbel via softplus

| Aspecto | v5.0 | v5.1 |
|---------|------|------|
| Parametrización | `log_gumbel_temp` + `exp().clamp(0.1, 10.0)` | `_raw_gumbel_temp` + `softplus() + 0.1` |
| Gradiente en bordes | Discontinuo (clamp corta) | Continuo en todo el rango |
| Rango efectivo | [0.1, 10.0] hard | [0.1, +∞) soft |

La `@property gumbel_temp` encapsula la transformación. El parámetro interno se inicializa via inverse softplus para que el valor inicial coincida con el `gumbel_temp` solicitado.

---

### 🔧 MEJORA — Manejo robusto de softmax (`_MultiHeadCommResidual`, `_GlobalWorkspace`)

**Antes**: `F.softmax(attn, dim=-1).nan_to_num(0.0)` — si todas las columnas son `-inf`, softmax produce NaN que se reemplaza silenciosamente. El gradiente queda indefinido.

**Después**:
- `_MultiHeadCommResidual`: softmax se computa sin masking previo, luego se multiplica por `row_mask.float()`. Filas inactivas → atención zero con gradiente limpio.
- `_GlobalWorkspace`: se detectan filas sin módulos activos *antes* del softmax (`any_active`), y se aplica zero post-softmax solo donde es necesario. El path `-inf → NaN` se elimina.

---

### 🆕 FEATURE — Inactivity Decay

**Parámetro**: `inactivity_decay: float = 0.001`

Módulos que permanecen inactivos durante muchos pasos consecutivos sufren un decay exponencial suave en su estado oculto:

```
decay_ratio = min(steps_inactivo, 100) / 100
h_inactive *= (1 - lambda_decay * decay_ratio)
```

- **Cap**: 100 pasos máximo de decay (evita colapso a zero).
- **Reset**: Al activarse, el contador vuelve a 0 y el decay se detiene.
- **Tracking**: `inactivity_steps` se mantiene como tensor `[B, K]` en `RIMsState` y se propaga entre timesteps.

**Motivación**: En secuencias largas (>50 pasos), módulos permanentemente inactivos mantienen estado stale que puede contaminar la comunicación cuando finalmente se reactivan.

---

### 🆕 FEATURE — Workspace Dinámico (`_GlobalWorkspace`)

| Aspecto | v5.0 | v5.1 |
|---------|------|------|
| Buffer inicial | `nn.Parameter` estático | MLP condicionado al contexto + fallback |
| Adaptabilidad | Fija | Dinámica por batch/timestep |

El buffer del workspace se genera condicionado al promedio de los estados activos:

```
context = mean(h_activos)  →  MLP  →  ws_slots vectores
ws = ws_generado + ws_fallback_estático
```

La suma residual con el fallback estático garantiza estabilidad durante la inicialización (el MLP se inicializa cerca de zero, así el fallback domina al principio).

---

### 🆕 FEATURE — Commitment Adaptativo (`_DVNCCodebook`)

| Aspecto | v5.0 | v5.1 |
|---------|------|------|
| `beta` | Fijo (`commitment`) | `beta_base * sigmoid(entropy / scale)` |
| Bajo estrés | Codebook rígido | Codebook relajado |

**Nuevo parámetro**: `entropy_scale: float = 1.0`

Cuando la entropía de activación es baja (pocos módulos activos = sistema bajo estrés), el commitment loss baja automáticamente, dando más libertad al codebook para representar mensajes fuera del vocabulario aprendido.

**Integración**: `_step()` ahora pasa `activation_entropy` al `_DVNCCodebook` cuando `comm_mode='dvnc'`.

---

### 🆕 FEATURE — NCO Fingerprint

**Función**: `_compute_fingerprint(hidden, precision=4) -> str`

Computa un hash SHA-256 truncado (16 hex chars = 64 bits) del estado oculto cuantizado. El NCO puede comparar fingerprints entre checkpoints sin almacenar tensores completos:

```python
if state_a.fingerprint != state_b.fingerprint:
    nco.flag_ontological_divergence(state_a, state_b)
```

**Campo**: `RIMsState.fingerprint: str` — se computa automáticamente al final de `forward()`.

---

### 🔧 MEJORA — Validación fast/slow exhaustiva

**Nuevo método**: `_validate_param_groups()`

Se ejecuta automáticamente en `__init__()`. Verifica que:
1. `fast_params()` ∩ `slow_params()` = ∅ (sin duplicados)
2. `fast_params()` ∪ `slow_params()` = `self.parameters()` (sin omisiones)

Lanza `RuntimeError` si se viola cualquier condición. Previene bugs silenciosos al agregar módulos nuevos.

---

### 📊 Métricas nuevas

| Métrica | Tipo | Descripción |
|---------|------|-------------|
| `max_inactivity` | int | Máximo pasos consecutivos inactivo en el batch |
| `mean_inactivity` | float | Promedio de inactividad acumulada |
| `fingerprint` | str | Hash del estado (en `RIMsState.to_dict()`) |

---

### ⚠️ Breaking Changes

1. **`reset_hidden()`** retorna `Tuple[Tensor, Tensor]` en lugar de `Tensor`. El segundo elemento es `inactivity_steps`.
2. **`forward()`** acepta kwarg opcional `inactivity_steps: Optional[Tensor]`.
3. **`RIMsState`** tiene dos campos nuevos: `inactivity_steps: Tensor` y `fingerprint: str`.
4. **`_DVNCCodebook.forward()`** acepta kwarg opcional `activation_entropy: Optional[float]`.
5. **`log_gumbel_temp`** renombrado a `_raw_gumbel_temp` (afecta checkpoint loading).

### Migración de checkpoints v5.0 → v5.1

```python
state_dict = torch.load('rims_v50.pt')
# Renombrar parametro de temperatura
state_dict['_raw_gumbel_temp'] = state_dict.pop('log_gumbel_temp')
# Inicializar nuevos parametros del workspace generator (si comm_mode='gwt')
# Se cargarán con valores por defecto al usar strict=False
model.load_state_dict(state_dict, strict=False)
```

---

### Conteo de cambios

| Categoría | Cantidad |
|-----------|----------|
| Bugfix crítico | 1 |
| Mejora de fidelidad | 1 |
| Estabilidad numérica | 3 |
| Feature nueva | 4 |
| Validación/safety | 1 |
| **Total** | **10** |

---

*Author:Escribano Silente (MSC Framework)*
*Solicitado por: Esraderey (Observer MSC)*

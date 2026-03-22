# CHANGELOG — Recurrent Independent Mechanisms (RIMs)

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

*Author: Claude (review asistido) + Escribano Silente (MSC Framework)*
*Solicitado por: Esraderey (Observer MSC)*

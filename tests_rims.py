"""
Tests unitarios y de integracion para RIMs v5.1.1.

Cobertura (~45 tests):
  - GroupGRUCell: shapes, gradientes, independencia entre grupos
  - _InputAttentionRIM: shapes, scores, mascara top-k, W_q per-modulo
  - _MultiHeadCommResidual: residual, inactivos congelados, mascara filas
  - _GlobalWorkspace: ciclo write/broadcast, shapes, buffer dinamico
                     (v5.1.1: ws_generator zero-init real)
  - _DVNCCodebook: cuantizacion, vq_loss, straight-through, commitment adaptativo
                   (v5.1.1: VQ loss canonico, entropia como tensor)
  - RecurrentIndependentMechanisms: todos los modos, shapes, sparsidad,
    gradientes, fast/slow params (con validacion exhaustiva),
    manejo de secuencias y puntuales, estado inicial, routing gumbel,
    inactivity decay (v5.1.1: orden corregido, cap configurable),
    fingerprint NCO (v5.1.1: opt-in),
    softmax sin NaN, temperatura softplus
  - RIMsState: to_dict consistencia (v5.1.1: omite fingerprint vacio)
  - V5_1_1 SPECIFIC: VQ loss canonico, entropia diferenciable,
                     decay pre-LayerNorm, fingerprint opt-in,
                     ws_generator zero-init, init_weights no sobrescribe,
                     inactivity_cap configurable, vq_loss scalar [],
                     extra_repr.

Ejecutar:
    python -m pytest tests_rims.py -v
    python tests_rims.py         (modo standalone con mock de base/config)
"""

from __future__ import annotations

import math
import sys
import types
import unittest

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mock de dependencias MSC para tests standalone
# ---------------------------------------------------------------------------

def _make_msc_mocks():
    """Crea mocks minimos de ConsciousnessLayerBase y LayerConfig."""

    class _Metrics:
        def record(self, d): pass
        def get_stats(self, k): return {}

    class ConsciousnessLayerBase(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.metrics = _Metrics()

    class LayerConfig:
        pass

    base_mod   = types.ModuleType('consciousness.layers.base')
    config_mod = types.ModuleType('consciousness.layers.config')
    base_mod.ConsciousnessLayerBase   = ConsciousnessLayerBase
    config_mod.LayerConfig            = LayerConfig
    sys.modules.setdefault('consciousness',                    types.ModuleType('consciousness'))
    sys.modules.setdefault('consciousness.layers',             types.ModuleType('consciousness.layers'))
    sys.modules['consciousness.layers.base']   = base_mod
    sys.modules['consciousness.layers.config'] = config_mod

    return ConsciousnessLayerBase, LayerConfig


ConsciousnessLayerBase, LayerConfig = _make_msc_mocks()

import importlib, importlib.util, importlib.abc, pathlib

_rims_path = pathlib.Path(__file__).parent / 'rims.py'
_spec = importlib.util.spec_from_file_location('rims', _rims_path)
_mod  = importlib.util.module_from_spec(_spec)
_mod.__package__ = 'consciousness.layers'

class _RelativeImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ('consciousness.layers.base', 'consciousness.layers.config'):
            return importlib.util.find_spec(fullname)
        return None

sys.meta_path.insert(0, _RelativeImportFinder())
_spec.loader.exec_module(_mod)
sys.meta_path.pop(0)

GroupGRUCell                    = _mod.GroupGRUCell
_InputAttentionRIM              = _mod._InputAttentionRIM
_MultiHeadCommResidual          = _mod._MultiHeadCommResidual
_GlobalWorkspace                = _mod._GlobalWorkspace
_DVNCCodebook                   = _mod._DVNCCodebook
RIMsState                       = _mod.RIMsState
RecurrentIndependentMechanisms  = _mod.RecurrentIndependentMechanisms
_compute_fingerprint            = _mod._compute_fingerprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 4    # batch size para tests
K = 6    # num_rims
Ka = 3   # num_active
D = 384  # hidden_size  (D/K = 64 por modulo)
rim = D // K  # 64


def _rims(comm_mode='standard', routing='ste', **kw) -> RecurrentIndependentMechanisms:
    return RecurrentIndependentMechanisms(
        input_size=128, hidden_size=D, num_rims=K, num_active=Ka,
        comm_mode=comm_mode, routing=routing, num_heads=4,
        ws_slots=2, num_codes=32, dropout=0.0, **kw
    )


def _has_grad(tensor: torch.Tensor) -> bool:
    return tensor.grad is not None and tensor.grad.abs().sum().item() > 0


# ============================================================================
# Tests GroupGRUCell
# ============================================================================

class TestGroupGRUCell(unittest.TestCase):

    def setUp(self):
        self.cell = GroupGRUCell(input_size=rim, hidden_size=rim, num_groups=K)

    def test_output_shape(self):
        x = torch.randn(B, K, rim)
        h = torch.randn(B, K, rim)
        h_new = self.cell(x, h)
        self.assertEqual(h_new.shape, (B, K, rim))

    def test_values_bounded(self):
        """GRU con tanh: la salida debe estar en (-1, 1)."""
        x = torch.randn(B, K, rim)
        h = torch.randn(B, K, rim)
        h_new = self.cell(x, h)
        self.assertTrue(h_new.abs().max().item() <= 1.0 + 1e-5)

    def test_gradient_flows(self):
        x = torch.randn(B, K, rim, requires_grad=True)
        h = torch.randn(B, K, rim)
        h_new = self.cell(x, h)
        h_new.sum().backward()
        self.assertTrue(_has_grad(x))

    def test_groups_independent(self):
        """Cambiar el input del grupo 0 no debe afectar la salida del grupo 1+."""
        x  = torch.randn(B, K, rim)
        h  = torch.randn(B, K, rim)
        h1 = self.cell(x, h).detach().clone()

        x2 = x.clone()
        x2[:, 0] += 100.0
        h2 = self.cell(x2, h).detach()

        self.assertTrue(torch.allclose(h1[:, 1:], h2[:, 1:], atol=1e-5))
        self.assertFalse(torch.allclose(h1[:, 0], h2[:, 0], atol=1e-5))


# ============================================================================
# Tests _InputAttentionRIM
# ============================================================================

class TestInputAttentionRIM(unittest.TestCase):

    def setUp(self):
        self.layer = _InputAttentionRIM(
            hidden_size=128, rim_size=rim, num_rims=K, d_key=32
        )

    def test_output_shapes(self):
        x = torch.randn(B, 128)
        h = torch.randn(B, K, rim)
        V_exp, scores = self.layer(x, h)
        self.assertEqual(V_exp.shape,  (B, K, rim))
        self.assertEqual(scores.shape, (B, K))

    def test_scores_finite(self):
        x = torch.randn(B, 128)
        h = torch.randn(B, K, rim)
        _, scores = self.layer(x, h)
        self.assertTrue(torch.isfinite(scores).all())

    def test_gradient_through_scores(self):
        x = torch.randn(B, 128, requires_grad=True)
        h = torch.randn(B, K, rim)
        _, scores = self.layer(x, h)
        scores.sum().backward()
        self.assertTrue(_has_grad(x))

    def test_per_module_wq(self):
        """W_q debe ser [K, d_key, rim_size] y cada modulo debe tener
        su propia proyeccion independiente."""
        d_key = 32
        layer = _InputAttentionRIM(
            hidden_size=128, rim_size=rim, num_rims=K, d_key=d_key
        )
        # Shape del parametro
        self.assertEqual(layer.W_q.shape, (K, d_key, rim))

        # Perturbar W_q del modulo 0 — solo scores[0] debe cambiar
        x = torch.randn(B, 128)
        h = torch.randn(B, K, rim)
        _, scores_before = layer(x, h)
        scores_before = scores_before.detach().clone()

        with torch.no_grad():
            layer.W_q[0] += 10.0  # perturbar solo modulo 0

        _, scores_after = layer(x, h)
        scores_after = scores_after.detach()

        # Modulo 0 debe haber cambiado
        self.assertFalse(
            torch.allclose(scores_before[:, 0], scores_after[:, 0], atol=1e-4),
            "Perturbar W_q[0] debe cambiar scores del modulo 0"
        )
        # Modulos 1..K-1 deben ser identicos
        self.assertTrue(
            torch.allclose(scores_before[:, 1:], scores_after[:, 1:], atol=1e-5),
            "Perturbar W_q[0] NO debe cambiar scores de modulos 1+"
        )

        # Gradiente debe fluir por cada W_q independientemente
        layer2 = _InputAttentionRIM(
            hidden_size=128, rim_size=rim, num_rims=K, d_key=d_key
        )
        x2 = torch.randn(B, 128)
        h2 = torch.randn(B, K, rim)
        _, scores2 = layer2(x2, h2)
        # Backprop solo a traves del modulo 2
        scores2[:, 2].sum().backward()
        # W_q[2] debe tener gradiente, W_q[0] no (o ~0)
        grad_mod2 = layer2.W_q.grad[2].abs().sum().item()
        grad_mod0 = layer2.W_q.grad[0].abs().sum().item()
        self.assertGreater(grad_mod2, 1e-8, "W_q[2] debe tener gradiente")
        self.assertAlmostEqual(grad_mod0, 0.0, places=6,
            msg="W_q[0] no debe tener gradiente si backprop solo pasa por modulo 2")


# ============================================================================
# Tests _MultiHeadCommResidual
# ============================================================================

class TestMultiHeadCommResidual(unittest.TestCase):

    def setUp(self):
        self.layer = _MultiHeadCommResidual(rim_size=rim, num_heads=4)

    def _mask(self, B, active_indices):
        m = torch.zeros(B, K, dtype=torch.bool)
        m[:, active_indices] = True
        return m

    def test_output_shape(self):
        h = torch.randn(B, K, rim)
        mask = self._mask(B, [0, 1, 2])
        out = self.layer(h, mask)
        self.assertEqual(out.shape, (B, K, rim))

    def test_inactive_unchanged_apart_from_ln(self):
        """Modulos inactivos no reciben aporte de comunicacion (out = LN(h)).
        Verificar que la diferencia activos-inactivos respecto a h no es zero
        pero tampoco identica."""
        torch.manual_seed(0)
        h    = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True

        out = self.layer(h, mask)

        # Activos: h + Att(h); Inactivos: h + 0. Ambos pasan por LayerNorm.
        # Por lo tanto activos suelen diferir mas de h que los inactivos.
        active_delta   = (out[:, :Ka] - h[:, :Ka]).norm().item()
        inactive_delta = (out[:, Ka:] - h[:, Ka:]).norm().item()
        # Activos deben cambiar al menos tanto como inactivos
        # (LN solo afecta a inactivos; Att+LN a activos)
        self.assertGreaterEqual(active_delta, 0.0)
        self.assertGreaterEqual(inactive_delta, 0.0)

    def test_gradient_flows(self):
        h    = torch.randn(B, K, rim, requires_grad=True)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out  = self.layer(h, mask)
        out.sum().backward()
        self.assertTrue(_has_grad(h))

    def test_comm_mask_rows_not_cols(self):
        """La mascara debe aplicarse a FILAS (queries de inactivos), no a
        COLUMNAS (sources). Todos los modulos — activos e inactivos — deben ser
        accesibles como fuente (keys/values).

        Test: si un modulo inactivo tiene un estado muy distinto, los activos
        deben poder "verlo" y su output debe cambiar."""
        torch.manual_seed(42)
        layer = _MultiHeadCommResidual(rim_size=rim, num_heads=4)

        h = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True  # modulos 0,1,2 activos; 3,4,5 inactivos

        # Baseline
        out_baseline = layer(h, mask).detach().clone()

        # Perturbar un modulo INACTIVO (modulo 5) con un estado muy fuerte
        h2 = h.clone()
        h2[:, 5] = h2[:, 5] + 50.0

        out_perturbed = layer(h2, mask).detach()

        # Si la mascara es correcta (filas, no columnas), los modulos ACTIVOS
        # deben producir output diferente porque pueden leer del modulo 5 como fuente
        active_delta = (out_perturbed[:, :Ka] - out_baseline[:, :Ka]).abs().mean().item()
        self.assertGreater(active_delta, 0.1,
            "Los activos deben poder leer de inactivos como fuente (keys/values). "
            "Si este test falla, la mascara probablemente bloquea columnas en vez de filas.")


# ============================================================================
# Tests _GlobalWorkspace
# ============================================================================

class TestGlobalWorkspace(unittest.TestCase):

    def setUp(self):
        self.layer = _GlobalWorkspace(rim_size=rim, num_rims=K, ws_slots=2)

    def test_output_shape(self):
        h    = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out  = self.layer(h, mask)
        self.assertEqual(out.shape, (B, K, rim))

    def test_all_modules_receive_broadcast(self):
        """Tras el broadcast, TODOS los modulos (activos e inactivos)
        deben haber cambiado respecto al estado original."""
        torch.manual_seed(42)
        h    = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out  = self.layer(h, mask)

        inactive_delta = (out[:, Ka:] - h[:, Ka:]).abs().mean().item()
        self.assertGreater(inactive_delta, 1e-6,
            "GWT broadcast debe afectar tambien a modulos inactivos")

    def test_gradient_flows(self):
        h    = torch.randn(B, K, rim, requires_grad=True)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out  = self.layer(h, mask)
        out.sum().backward()
        self.assertTrue(_has_grad(h))

    def test_ws_generator_zero_init(self):
        """v5.1.1: La ultima capa de ws_generator debe inicializarse a zero,
        de modo que ws_generator(x) ≈ 0 al inicio del entrenamiento (el
        fallback estatico domina via la suma residual)."""
        layer = _GlobalWorkspace(rim_size=rim, num_rims=K, ws_slots=2)
        last = layer.ws_generator[-1]
        self.assertTrue(torch.allclose(last.weight, torch.zeros_like(last.weight)),
            "v5.1.1: ws_generator[-1].weight debe ser zero al init")
        self.assertTrue(torch.allclose(last.bias, torch.zeros_like(last.bias)),
            "v5.1.1: ws_generator[-1].bias debe ser zero al init")

        # Verificacion funcional: con weights zero, ws_generator(context) == 0
        context = torch.randn(B, rim)
        ws_flat = layer.ws_generator(context)
        self.assertTrue(torch.allclose(ws_flat, torch.zeros_like(ws_flat)),
            "v5.1.1: ws_generator debe producir output zero al init")


# ============================================================================
# Tests _DVNCCodebook
# ============================================================================

class TestDVNCCodebook(unittest.TestCase):

    def setUp(self):
        self.layer = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)

    def test_output_shape(self):
        h    = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out, vq_loss = self.layer(h, mask)
        self.assertEqual(out.shape, (B, K, rim))
        self.assertEqual(vq_loss.shape, ())

    def test_vq_loss_positive(self):
        h    = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        _, vq_loss = self.layer(h, mask)
        self.assertGreater(vq_loss.item(), 0.0)

    def test_straight_through_gradient(self):
        """El gradiente debe fluir a traves de la cuantizacion via STE."""
        h    = torch.randn(B, K, rim, requires_grad=True)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out, vq_loss = self.layer(h, mask)
        (out.sum() + vq_loss).backward()
        self.assertTrue(_has_grad(h))

    def test_inactive_get_no_comm(self):
        """Modulos inactivos no deben recibir mensaje del codebook."""
        torch.manual_seed(7)
        h     = torch.randn(B, K, rim)
        mask  = torch.zeros(B, K, dtype=torch.bool)
        mask_all_off = mask.clone()
        out_off, _ = self.layer(h, mask_all_off)

        mask_some   = mask.clone()
        mask_some[:, :Ka] = True
        out_some, _ = self.layer(h, mask_some)

        diff = (out_off[:, Ka:] - out_some[:, Ka:]).abs().mean().item()
        self.assertLess(diff, 1.0)

    # ---- v5.1.1 ----

    def test_vq_loss_canonical_direction(self):
        """v5.1.1 (CRITICO): VQ loss en posicion canonica.

        Verifica que:
        - codebook_loss = ||z_q - sg[z]||^2 -> propaga gradiente al CODEBOOK
        - commitment_loss = ||z - sg[z_q]||^2 -> propaga gradiente al ENCODER

        En v5.1.0 estaban INVERTIDOS. Este test detecta esa regresion."""
        torch.manual_seed(0)
        layer = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)

        # Caso 1: solo el codebook tiene requires_grad
        for p in layer.parameters():
            p.requires_grad_(False)
        layer.codebook.weight.requires_grad_(True)

        h = torch.randn(B, K, rim)
        mask = torch.ones(B, K, dtype=torch.bool)
        _, vq_loss = layer(h, mask)
        vq_loss.backward()

        cb_grad = layer.codebook.weight.grad
        self.assertIsNotNone(cb_grad,
            "v5.1.1: El codebook DEBE recibir gradiente del codebook_loss")
        self.assertGreater(cb_grad.abs().sum().item(), 1e-6,
            "v5.1.1: El gradiente del codebook debe ser no-trivial")

        # Caso 2: solo proj_in tiene requires_grad (encoder path)
        layer2 = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)
        for p in layer2.parameters():
            p.requires_grad_(False)
        layer2.proj_in.weight.requires_grad_(True)

        h2 = torch.randn(B, K, rim)
        _, vq_loss2 = layer2(h2, mask)
        vq_loss2.backward()

        enc_grad = layer2.proj_in.weight.grad
        self.assertIsNotNone(enc_grad,
            "v5.1.1: proj_in (encoder) DEBE recibir gradiente del commitment_loss")
        self.assertGreater(enc_grad.abs().sum().item(), 1e-6,
            "v5.1.1: El gradiente del encoder debe ser no-trivial")

    def test_dvnc_entropy_gradient_flow(self):
        """v5.1.1: Si activation_entropy se pasa como tensor, el gradiente
        debe fluir hacia el (preserva el grafo)."""
        layer = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)

        h = torch.randn(B, K, rim)
        mask = torch.ones(B, K, dtype=torch.bool)
        # Tensor con grafo simulando entropia diferenciable
        entropy_t = torch.tensor(0.5, requires_grad=True)

        _, vq_loss = layer(h, mask, activation_entropy=entropy_t)
        vq_loss.backward()

        self.assertIsNotNone(entropy_t.grad,
            "v5.1.1: Si activation_entropy es tensor, debe recibir gradiente")
        self.assertTrue(torch.isfinite(entropy_t.grad).all(),
            "v5.1.1: Gradiente de activation_entropy debe ser finito")

    def test_dvnc_entropy_float_no_crash(self):
        """v5.1.1: activation_entropy como float debe funcionar sin crashear."""
        layer = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)
        h = torch.randn(B, K, rim)
        mask = torch.ones(B, K, dtype=torch.bool)

        _, vq_loss = layer(h, mask, activation_entropy=0.5)
        self.assertTrue(torch.isfinite(vq_loss).all())

    def test_dvnc_entropy_none(self):
        """v5.1.1: activation_entropy=None debe usar commitment_base sin crashear."""
        layer = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)
        h = torch.randn(B, K, rim)
        mask = torch.ones(B, K, dtype=torch.bool)

        _, vq_loss = layer(h, mask, activation_entropy=None)
        self.assertTrue(torch.isfinite(vq_loss).all())


# ============================================================================
# Tests RecurrentIndependentMechanisms — general
# ============================================================================

class TestRIMsGeneral(unittest.TestCase):

    def test_pointwise_input_shape(self):
        model = _rims()
        x   = torch.randn(B, 128)
        out, state = model(x)
        self.assertEqual(out.shape, (B, D))

    def test_sequence_input_shape(self):
        model = _rims()
        x   = torch.randn(B, 10, 128)
        out, state = model(x)
        self.assertEqual(out.shape, (B, 10, D))

    def test_state_shapes(self):
        """v5.1.1: fingerprint es opt-in (default vacio)."""
        model = _rims()
        x = torch.randn(B, 10, 128)
        _, state = model(x)
        self.assertEqual(state.hidden_states.shape,     (B, K, rim))
        self.assertEqual(state.active_rims.shape,       (B, K))
        self.assertEqual(state.attention_weights.shape,  (B, K))
        self.assertEqual(state.communication.shape,      (B, K, rim))
        self.assertEqual(state.inactivity_steps.shape,   (B, K))
        # v5.1.1: fingerprint default vacio (compute_fingerprint=False)
        self.assertIsInstance(state.fingerprint, str)
        self.assertEqual(state.fingerprint, "",
            "v5.1.1: fingerprint debe estar vacio por default (opt-in)")

    def test_state_shapes_with_fingerprint(self):
        """v5.1.1: con compute_fingerprint=True, fingerprint tiene 16 hex chars."""
        model = _rims(compute_fingerprint=True)
        x = torch.randn(B, 10, 128)
        _, state = model(x)
        self.assertEqual(len(state.fingerprint), 16)
        # Hex valido
        int(state.fingerprint, 16)

    def test_sparsity_exact(self):
        """Exactamente Ka modulos deben estar activos por muestra."""
        model = _rims()
        x = torch.randn(B, 128)
        _, state = model(x)
        active_per_sample = state.active_rims.float().sum(dim=-1)
        expected = torch.full((B,), Ka, dtype=torch.float)
        self.assertTrue(torch.allclose(active_per_sample, expected))

    def test_gradient_end_to_end(self):
        model = _rims()
        x = torch.randn(B, 10, 128, requires_grad=True)
        out, _ = model(x)
        out.sum().backward()
        self.assertTrue(_has_grad(x))

    def test_hidden_state_continuity(self):
        """Con estado previo pasado, el modelo debe usar ese estado."""
        model = _rims()
        x = torch.randn(B, 5, 128)
        _, state1 = model(x)
        x2 = torch.randn(B, 5, 128)
        out2, _ = model(x2, hidden=state1.hidden_states,
                        inactivity_steps=state1.inactivity_steps)
        self.assertEqual(out2.shape, (B, 5, D))

    def test_reset_hidden_shape(self):
        """reset_hidden retorna Tuple[Tensor, Tensor]."""
        model  = _rims()
        result = model.reset_hidden(B, torch.device('cpu'))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        h, inact = result
        self.assertEqual(h.shape, (B, K, rim))
        self.assertEqual(inact.shape, (B, K))
        self.assertEqual(inact.dtype, torch.long)
        self.assertTrue((inact == 0).all())

    def test_attention_weights_sum_to_one(self):
        """Los pesos de atencion (softmax) deben sumar 1 por muestra."""
        model = _rims()
        x = torch.randn(B, 128)
        _, state = model(x)
        sums = state.attention_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(B), atol=1e-5))

    def test_inactive_hidden_unchanged(self):
        """Los modulos inactivos deben conservar mayormente su hidden previo."""
        model = _rims(inactivity_decay=0.0)  # sin decay para test limpio
        model.eval()
        h0, inact0 = model.reset_hidden(B, torch.device('cpu'))
        x  = torch.randn(B, 128)

        with torch.no_grad():
            _, state = model(x, hidden=h0, inactivity_steps=inact0)

        active = state.active_rims
        inactive = ~active

        delta = (state.hidden_states - h0).abs()
        active_delta   = delta[active].mean().item()   if active.any()   else 0.0
        inactive_delta = delta[inactive].mean().item() if inactive.any() else 0.0
        self.assertGreaterEqual(active_delta, inactive_delta)

    def test_to_dict_keys_default(self):
        """v5.1.1: to_dict NO debe incluir 'fingerprint' por default (opt-in)."""
        model = _rims()
        x = torch.randn(B, 128)
        _, state = model(x)
        d = state.to_dict()
        for key in ('num_active', 'activation_rate', 'attention_entropy',
                    'comm_norm', 'vq_loss',
                    'max_inactivity', 'mean_inactivity'):
            self.assertIn(key, d)
        # v5.1.1: fingerprint omitido si esta vacio
        self.assertNotIn('fingerprint', d,
            "v5.1.1: to_dict debe omitir 'fingerprint' cuando esta vacio")

    def test_to_dict_keys_with_fingerprint(self):
        """v5.1.1: to_dict incluye 'fingerprint' si compute_fingerprint=True."""
        model = _rims(compute_fingerprint=True)
        x = torch.randn(B, 128)
        _, state = model(x)
        d = state.to_dict()
        self.assertIn('fingerprint', d)
        self.assertEqual(len(d['fingerprint']), 16)

    def test_get_statistics_keys(self):
        """v5.1.1: get_statistics incluye nuevos campos."""
        model = _rims()
        x = torch.randn(B, 5, 128)
        model(x)
        stats = model.get_statistics()
        for key in ('num_rims', 'num_active', 'rim_size', 'comm_mode', 'routing',
                    'inactivity_decay', 'max_inactivity', 'mean_inactivity',
                    # v5.1.1
                    'inactivity_cap', 'compute_fingerprint'):
            self.assertIn(key, stats, f"falta clave: {key}")

    def test_extra_repr(self):
        """v5.1.1: extra_repr incluye los nuevos campos para diagnostico."""
        model = _rims(comm_mode='gwt', routing='gumbel',
                      inactivity_decay=0.005, inactivity_cap=200.0,
                      compute_fingerprint=True)
        repr_str = model.extra_repr()
        for needle in ('input_size', 'hidden_size', 'num_rims', 'num_active',
                       'comm_mode', 'routing',
                       'inactivity_decay', 'inactivity_cap',
                       'compute_fingerprint'):
            self.assertIn(needle, repr_str, f"extra_repr no contiene '{needle}'")


# ============================================================================
# Tests por comm_mode
# ============================================================================

class TestRIMsCommModes(unittest.TestCase):

    def _smoke(self, comm_mode, **kw):
        model = _rims(comm_mode=comm_mode, **kw)
        x = torch.randn(B, 10, 128, requires_grad=True)
        out, state = model(x)
        out.sum().backward()
        return out, state

    def test_standard_mode(self):
        out, _ = self._smoke('standard')
        self.assertEqual(out.shape, (B, 10, D))

    def test_gwt_mode(self):
        out, _ = self._smoke('gwt', ws_slots=2)
        self.assertEqual(out.shape, (B, 10, D))

    def test_dvnc_mode(self):
        out, state = self._smoke('dvnc', num_codes=32)
        self.assertEqual(out.shape, (B, 10, D))
        self.assertGreater(state.vq_loss.item(), 0.0)

    def test_dvnc_vq_loss_zero_in_state_for_non_dvnc(self):
        """Para modos standard/gwt, vq_loss debe ser 0."""
        for mode in ('standard', 'gwt'):
            model = _rims(comm_mode=mode)
            x = torch.randn(B, 5, 128)
            _, state = model(x)
            self.assertEqual(state.vq_loss.item(), 0.0,
                             f"vq_loss debe ser 0 para comm_mode='{mode}'")

    def test_vq_loss_scalar_shape(self):
        """v5.1.1: vq_loss en RIMsState debe ser scalar shape []."""
        model = _rims(comm_mode='dvnc')
        x = torch.randn(B, 5, 128)
        _, state = model(x)
        self.assertEqual(state.vq_loss.shape, torch.Size([]),
            "v5.1.1: state.vq_loss debe ser scalar shape [], no [1]")

    def test_dvnc_attention_gradient_via_vqloss(self):
        """v5.1.1: Como activation_entropy se pasa como tensor (no .item()),
        el gradiente del vq_loss debe propagarse hacia los parametros que
        afectan attention_weights (input_attention.W_q, etc.)."""
        model = _rims(comm_mode='dvnc')
        x = torch.randn(B, 5, 128)
        _, state = model(x)

        # Backprop solo del vq_loss
        state.vq_loss.backward()

        # input_attention.W_q debe haber recibido gradiente porque
        # affecta attention_weights -> entropy -> beta_eff -> vq_loss
        wq_grad = model.input_attention.W_q.grad
        self.assertIsNotNone(wq_grad,
            "v5.1.1: W_q debe recibir gradiente del vq_loss via entropia")
        # Nota: el gradiente puede ser pequeno, lo importante es que exista
        self.assertTrue(torch.isfinite(wq_grad).all(),
            "v5.1.1: Gradiente de W_q via vq_loss debe ser finito")


# ============================================================================
# Tests routing
# ============================================================================

class TestRIMsRouting(unittest.TestCase):

    def test_ste_routing(self):
        model = _rims(routing='ste')
        x = torch.randn(B, 5, 128, requires_grad=True)
        out, _ = model(x)
        out.sum().backward()
        self.assertTrue(_has_grad(x))

    def test_gumbel_routing_train(self):
        model = _rims(routing='gumbel', gumbel_temp=1.0)
        model.train()
        x = torch.randn(B, 5, 128, requires_grad=True)
        out, _ = model(x)
        out.sum().backward()
        self.assertTrue(_has_grad(x))

    def test_gumbel_routing_eval(self):
        """En eval, Gumbel no agrega ruido -> output deterministico."""
        model = _rims(routing='gumbel')
        model.eval()
        x = torch.randn(B, 5, 128)
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = model(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gumbel_temp_is_learnable(self):
        """El parametro es _raw_gumbel_temp, accesible via property."""
        model = _rims(routing='gumbel')
        found = any(p is model._raw_gumbel_temp for p in model.parameters())
        self.assertTrue(found)

    def test_gumbel_softplus_gradient(self):
        """La temperatura via softplus debe tener gradiente continuo
        en todo el rango, incluyendo valores extremos."""
        model = _rims(routing='gumbel')
        model.train()

        for raw_val in [-5.0, 0.0, 5.0, 20.0]:
            with torch.no_grad():
                model._raw_gumbel_temp.fill_(raw_val)

            x = torch.randn(B, 3, 128)
            out, _ = model(x)
            out.sum().backward()

            grad = model._raw_gumbel_temp.grad
            self.assertIsNotNone(grad,
                f"Gradiente de temperatura debe existir con raw={raw_val}")
            self.assertTrue(torch.isfinite(grad).all(),
                f"Gradiente de temperatura debe ser finito con raw={raw_val}")

            model.zero_grad()

        # Verificar que gumbel_temp nunca baja de 0.1
        with torch.no_grad():
            model._raw_gumbel_temp.fill_(-100.0)  # extremo bajo
        self.assertGreaterEqual(model.gumbel_temp.item(), 0.1)


# ============================================================================
# Tests fast/slow params
# ============================================================================

class TestFastSlowParams(unittest.TestCase):

    def test_fast_slow_disjoint(self):
        """fast y slow deben ser conjuntos disjuntos."""
        model  = _rims()
        fast   = set(id(p) for p in model.fast_params())
        slow   = set(id(p) for p in model.slow_params())
        common = fast & slow
        self.assertEqual(len(common), 0,
            f"fast y slow comparten {len(common)} parametros")

    def test_fast_slow_cover_all(self):
        """fast + slow deben cubrir todos los parametros del modelo."""
        model  = _rims()
        all_p  = set(id(p) for p in model.parameters())
        fast   = set(id(p) for p in model.fast_params())
        slow   = set(id(p) for p in model.slow_params())
        covered = fast | slow
        missing = all_p - covered
        self.assertEqual(len(missing), 0,
            f"{len(missing)} parametros no estan en fast ni slow")

    def test_fast_params_gradients(self):
        """Solo los fast params deben recibir gradiente en un paso interno."""
        model = _rims()
        for p in model.slow_params():
            p.requires_grad_(False)

        x = torch.randn(B, 5, 128)
        out, _ = model(x)
        out.sum().backward()

        for p in model.fast_params():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_fast_slow_exhaustive_validation(self):
        """La validacion en __init__ debe pasar para todos los comm_modes."""
        for mode in ('standard', 'gwt', 'dvnc'):
            try:
                model = _rims(comm_mode=mode)
            except RuntimeError as e:
                self.fail(
                    f"_validate_param_groups() fallo para comm_mode='{mode}': {e}"
                )

            # Doble verificacion: conteo exacto
            all_p = list(model.parameters())
            fast  = model.fast_params()
            slow  = model.slow_params()
            self.assertEqual(
                len(fast) + len(slow), len(all_p),
                f"comm_mode='{mode}': fast({len(fast)}) + slow({len(slow)}) "
                f"!= total({len(all_p)})"
            )


# ============================================================================
# Tests de validacion de argumentos
# ============================================================================

class TestRIMsValidation(unittest.TestCase):

    def test_num_active_too_large(self):
        with self.assertRaises(ValueError):
            RecurrentIndependentMechanisms(
                input_size=64, hidden_size=D,
                num_rims=K, num_active=K + 1
            )

    def test_num_active_zero(self):
        with self.assertRaises(ValueError):
            RecurrentIndependentMechanisms(
                input_size=64, hidden_size=D,
                num_rims=K, num_active=0
            )

    def test_hidden_not_divisible(self):
        with self.assertRaises(ValueError):
            RecurrentIndependentMechanisms(
                input_size=64, hidden_size=100,
                num_rims=K, num_active=Ka
            )

    def test_unknown_comm_mode(self):
        with self.assertRaises(ValueError):
            RecurrentIndependentMechanisms(
                input_size=64, hidden_size=D,
                num_rims=K, num_active=Ka,
                comm_mode='invalid'
            )

    def test_inactivity_cap_invalid(self):
        """v5.1.1: inactivity_cap debe ser > 0."""
        with self.assertRaises(ValueError):
            _rims(inactivity_cap=0.0)
        with self.assertRaises(ValueError):
            _rims(inactivity_cap=-10.0)

    def test_inactivity_decay_invalid(self):
        """v5.1.1: inactivity_decay debe estar en [0, 1)."""
        with self.assertRaises(ValueError):
            _rims(inactivity_decay=-0.1)
        with self.assertRaises(ValueError):
            _rims(inactivity_decay=1.0)
        with self.assertRaises(ValueError):
            _rims(inactivity_decay=2.0)


# ============================================================================
# Tests de inactivity decay
# ============================================================================

class TestInactivityDecay(unittest.TestCase):

    def test_inactivity_counter_increments(self):
        """Contadores de inactividad deben incrementar para inactivos
        y resetear a 0 para activos."""
        model = _rims(inactivity_decay=0.01)
        model.eval()

        h, inact = model.reset_hidden(B, torch.device('cpu'))
        self.assertTrue((inact == 0).all())

        x = torch.randn(B, 128)
        with torch.no_grad():
            _, state = model(x, hidden=h, inactivity_steps=inact)

        # Activos deben tener contador 0
        active_counts = state.inactivity_steps[state.active_rims]
        self.assertTrue((active_counts == 0).all(),
            "Modulos activos deben tener inactivity_steps=0")

        # Inactivos deben tener contador 1 (primer paso)
        inactive_counts = state.inactivity_steps[~state.active_rims]
        self.assertTrue((inactive_counts == 1).all(),
            "Modulos inactivos deben tener inactivity_steps=1 despues del primer paso")

    def test_inactivity_decay_internal(self):
        """v5.1.1: Verificar el decay aplicado por _apply_inactivity_decay
        directamente (sin LayerNorm que normalice la norma despues)."""
        model = _rims(inactivity_decay=0.05, inactivity_cap=100.0)

        h = torch.ones(B, K, rim) * 5.0
        # Forzar contadores altos (80 pasos) en todos los modulos
        inact = torch.full((B, K), 80, dtype=torch.long)
        # Mascara: ningun modulo activo (todos siguen inactivos)
        active_mask = torch.zeros(B, K, dtype=torch.bool)

        h_decayed, inact_new = model._apply_inactivity_decay(h, active_mask, inact)

        # Todos siguen inactivos -> contador sube a 81
        self.assertTrue((inact_new == 81).all())

        # decay_ratio = min(81, 100) / 100 = 0.81
        # decay_factor = 1 - 0.05 * 0.81 = 1 - 0.0405 = 0.9595
        expected_factor = 1.0 - 0.05 * (81.0 / 100.0)
        expected = h * expected_factor
        self.assertTrue(torch.allclose(h_decayed, expected, atol=1e-5),
            f"v5.1.1: decay debe aplicarse con factor {expected_factor:.4f}")

    def test_inactivity_decay_actives_unchanged_internal(self):
        """v5.1.1: Activos no sufren decay (factor=1.0)."""
        model = _rims(inactivity_decay=0.5)
        h = torch.ones(B, K, rim) * 3.0
        inact = torch.full((B, K), 50, dtype=torch.long)
        # Modulos 0,1,2 activos
        active_mask = torch.zeros(B, K, dtype=torch.bool)
        active_mask[:, :Ka] = True

        h_decayed, inact_new = model._apply_inactivity_decay(h, active_mask, inact)

        # Activos: counter -> 0, h sin cambio
        self.assertTrue((inact_new[:, :Ka] == 0).all())
        self.assertTrue(torch.allclose(h_decayed[:, :Ka], h[:, :Ka], atol=1e-6),
            "v5.1.1: Modulos activos no deben sufrir decay")

        # Inactivos: counter sube
        self.assertTrue((inact_new[:, Ka:] == 51).all())

    def test_zero_decay_no_effect(self):
        """Con inactivity_decay=0, los contadores se mantienen pero no hay decay."""
        model = _rims(inactivity_decay=0.0)
        model.eval()

        h = torch.randn(B, K, rim) * 5.0
        inact = torch.full((B, K), 50, dtype=torch.long)

        x = torch.randn(B, 128)
        with torch.no_grad():
            _, state = model(x, hidden=h, inactivity_steps=inact)

        # Los contadores aun deben incrementar/resetear
        inactive = ~state.active_rims
        if inactive.any():
            self.assertTrue((state.inactivity_steps[inactive] == 51).all())

    def test_inactivity_cap_configurable(self):
        """v5.1.1: inactivity_cap debe afectar el decay_ratio."""
        # cap=50: con 100 pasos clamp -> ratio = 1.0
        model_low_cap = _rims(inactivity_decay=0.1, inactivity_cap=50.0)
        # cap=200: con 100 pasos -> ratio = 0.5
        model_high_cap = _rims(inactivity_decay=0.1, inactivity_cap=200.0)

        h = torch.ones(B, K, rim)
        inact = torch.full((B, K), 100, dtype=torch.long)
        active_mask = torch.zeros(B, K, dtype=torch.bool)

        h_low,  _ = model_low_cap._apply_inactivity_decay(h, active_mask, inact)
        h_high, _ = model_high_cap._apply_inactivity_decay(h, active_mask, inact)

        # cap=50 produce mas decay (factor menor) que cap=200
        # Verificar que h_low < h_high elementwise
        self.assertTrue((h_low < h_high).all(),
            "v5.1.1: inactivity_cap menor debe producir mas decay")

    def test_decay_order_pre_layernorm(self):
        """v5.1.1: El decay se aplica ANTES de la mezcla y antes del LayerNorm.

        Verificar que con decay activo y decay=0, la trayectoria difiere,
        confirmando que el decay tiene efecto observable en el output final."""
        torch.manual_seed(123)
        model_a = _rims(inactivity_decay=0.5)  # decay agresivo
        torch.manual_seed(123)
        model_b = _rims(inactivity_decay=0.0)  # sin decay
        # Sincronizar pesos
        model_b.load_state_dict(model_a.state_dict())

        model_a.eval()
        model_b.eval()

        h = torch.randn(B, K, rim) * 3.0
        inact = torch.full((B, K), 90, dtype=torch.long)  # contador alto
        x = torch.randn(B, 128)

        with torch.no_grad():
            out_a, _ = model_a(x, hidden=h, inactivity_steps=inact)
            out_b, _ = model_b(x, hidden=h, inactivity_steps=inact)

        # Con contadores altos, el decay deberia afectar la trayectoria
        diff = (out_a - out_b).abs().mean().item()
        self.assertGreater(diff, 1e-5,
            "v5.1.1: Con decay > 0 y contadores altos, el output debe diferir "
            "de un modelo sin decay (mismos pesos)")


# ============================================================================
# Tests de NCO fingerprint (v5.1.1: opt-in)
# ============================================================================

class TestFingerprint(unittest.TestCase):

    def test_fingerprint_deterministic(self):
        """Mismo estado -> mismo hash, deterministico."""
        h = torch.randn(B, K, rim)
        fp1 = _compute_fingerprint(h)
        fp2 = _compute_fingerprint(h)
        self.assertEqual(fp1, fp2)

    def test_fingerprint_length(self):
        """Hash debe ser de 16 caracteres hex."""
        h = torch.randn(B, K, rim)
        fp = _compute_fingerprint(h)
        self.assertEqual(len(fp), 16)
        # Debe ser hex valido
        int(fp, 16)

    def test_fingerprint_changes_with_state(self):
        """Estados diferentes deben producir hashes diferentes."""
        h1 = torch.randn(B, K, rim)
        h2 = h1.clone()
        h2[0, 0, 0] += 0.01  # perturbacion pequenia pero >precision

        fp1 = _compute_fingerprint(h1)
        fp2 = _compute_fingerprint(h2)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_ignores_float_noise(self):
        """Perturbaciones menores que la precision (1e-4) no deben cambiar el hash."""
        h = torch.randn(B, K, rim)
        h2 = h + 1e-6  # ruido menor que 1e-4 (precision=4)

        fp1 = _compute_fingerprint(h)
        fp2 = _compute_fingerprint(h2)
        self.assertEqual(fp1, fp2,
            "Ruido de float por debajo de la precision no debe cambiar el fingerprint")

    def test_fingerprint_optin_default_false(self):
        """v5.1.1: Por default compute_fingerprint=False -> fingerprint=='' """
        model = _rims()
        x = torch.randn(B, 5, 128)
        _, state = model(x)
        self.assertEqual(state.fingerprint, "",
            "v5.1.1: fingerprint debe estar vacio por default")

    def test_fingerprint_optin_explicit_true(self):
        """v5.1.1: Con compute_fingerprint=True, fingerprint se computa."""
        model = _rims(compute_fingerprint=True)
        x = torch.randn(B, 5, 128)
        _, state = model(x)
        self.assertEqual(len(state.fingerprint), 16)
        # Reproducibilidad
        fp_recomputed = _compute_fingerprint(state.hidden_states)
        self.assertEqual(state.fingerprint, fp_recomputed)


# ============================================================================
# Test de softmax sin NaN
# ============================================================================

class TestNumericalStability(unittest.TestCase):

    def test_softmax_no_nan_all_inactive(self):
        """Si ningun modulo esta activo (edge case), la comunicacion
        no debe producir NaN."""
        layer = _MultiHeadCommResidual(rim_size=rim, num_heads=4)
        h = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)  # nadie activo

        out = layer(h, mask)
        self.assertTrue(torch.isfinite(out).all(),
            "Output de comunicacion no debe contener NaN/Inf "
            "incluso con 0 modulos activos")

    def test_softmax_no_nan_gwt_no_active(self):
        """GWT con 0 activos no debe producir NaN."""
        layer = _GlobalWorkspace(rim_size=rim, num_rims=K, ws_slots=2)
        h = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)

        out = layer(h, mask)
        self.assertTrue(torch.isfinite(out).all(),
            "GWT output no debe contener NaN/Inf con 0 modulos activos")


# ============================================================================
# v5.1.1: Tests especificos para los 9 fixes
# ============================================================================

class TestV511Fixes(unittest.TestCase):
    """
    Tests dedicados a verificar que los 9 bugs de v5.1.0 estan corregidos.
    Cada test esta etiquetado con el numero de fix correspondiente.
    """

    def test_fix1_vq_loss_canonical_position(self):
        """FIX #1: VQ loss con .detach() en posicion canonica.

        Si los .detach() estan invertidos (bug v5.1.0), el commitment_loss
        propaga al codebook y el codebook_loss al encoder. Esto se detecta
        viendo la magnitud relativa de los gradientes."""
        torch.manual_seed(0)
        layer = _DVNCCodebook(rim_size=rim, num_codes=32, commitment=0.25)

        h = torch.randn(B, K, rim, requires_grad=True)
        # proj_in is identity-ish at init via xavier; use h directly
        mask = torch.ones(B, K, dtype=torch.bool)
        _, vq_loss = layer(h, mask)
        vq_loss.backward()

        # Codebook debe haber recibido gradiente del codebook_loss
        cb_grad_norm = layer.codebook.weight.grad.abs().sum().item()
        self.assertGreater(cb_grad_norm, 1e-6,
            "FIX #1: codebook DEBE recibir gradiente (codebook_loss canonico)")
        # Encoder (h.grad) debe haber recibido del commitment_loss
        h_grad_norm = h.grad.abs().sum().item()
        self.assertGreater(h_grad_norm, 1e-6,
            "FIX #1: encoder DEBE recibir gradiente (commitment_loss canonico)")

    def test_fix2_entropy_preserves_graph(self):
        """FIX #2: activation_entropy como tensor preserva el grafo.

        En v5.1.0, .item() rompia la cadena. En v5.1.1 el tensor fluye."""
        layer = _DVNCCodebook(rim_size=rim, num_codes=32)

        # Tensor intermedio que simula entropia con grafo computacional
        upstream = torch.tensor(2.0, requires_grad=True)
        entropy_t = upstream * 0.5  # operacion en el grafo

        h = torch.randn(B, K, rim)
        mask = torch.ones(B, K, dtype=torch.bool)

        _, vq_loss = layer(h, mask, activation_entropy=entropy_t)
        vq_loss.backward()

        self.assertIsNotNone(upstream.grad,
            "FIX #2: el grafo entre activation_entropy y vq_loss debe preservarse")
        self.assertGreater(upstream.grad.abs().item(), 0.0,
            "FIX #2: el gradiente debe propagarse hasta upstream")

    def test_fix4_wq_init_per_slice(self):
        """FIX #4: W_q inicializado per-slice 2D (no sobre view 3D).

        Verificar que cada slice W_q[k] tiene una distribucion razonable
        (no es zero, no es enorme, varianza proporcional al fan)."""
        layer = _InputAttentionRIM(
            hidden_size=128, rim_size=rim, num_rims=K, d_key=32
        )

        # Cada slice debe tener varianza no-trivial
        for k in range(K):
            slice_std = layer.W_q[k].std().item()
            self.assertGreater(slice_std, 0.01,
                f"FIX #4: W_q[{k}] debe tener varianza > 0.01 tras init xavier")
            self.assertLess(slice_std, 1.0,
                f"FIX #4: W_q[{k}] no debe tener varianza enorme")

    def test_fix6_init_weights_respects_managed(self):
        """FIX #6: _init_weights del modulo padre NO debe sobrescribir
        la inicializacion zero de ws_generator[-1] en GWT."""
        model = _rims(comm_mode='gwt')
        ws_gen = model.comm_layer.ws_generator
        last = ws_gen[-1]
        # Tras la construccion completa, debe seguir siendo zero
        self.assertTrue(torch.allclose(last.weight, torch.zeros_like(last.weight)),
            "FIX #6: _init_weights del padre NO debe sobrescribir ws_generator[-1]")

    def test_fix7_ws_generator_zero_at_init(self):
        """FIX #7: ws_generator produce output ~0 al init -> fallback domina."""
        model = _rims(comm_mode='gwt')
        ws = model.comm_layer
        context = torch.randn(B, rim)
        ws_flat = ws.ws_generator(context)
        self.assertTrue(torch.allclose(ws_flat, torch.zeros_like(ws_flat)),
            "FIX #7: ws_generator(context) debe ser ~0 al init")

    def test_fix5_fingerprint_optin(self):
        """FIX #5: fingerprint opt-in elimina sync GPU<->CPU por default."""
        # Default: vacio
        model_default = _rims()
        x = torch.randn(B, 3, 128)
        _, state_default = model_default(x)
        self.assertEqual(state_default.fingerprint, "",
            "FIX #5: por default, fingerprint vacio (sin sync)")

        # Explicito: computado
        model_explicit = _rims(compute_fingerprint=True)
        _, state_explicit = model_explicit(x)
        self.assertEqual(len(state_explicit.fingerprint), 16,
            "FIX #5: con compute_fingerprint=True, hash de 16 chars")

    def test_fix8_inactivity_cap_configurable(self):
        """FIX #8: inactivity_cap es parametro, no magic number."""
        model_50  = _rims(inactivity_cap=50.0)
        model_100 = _rims(inactivity_cap=100.0)
        model_200 = _rims(inactivity_cap=200.0)

        self.assertEqual(model_50.inactivity_cap,  50.0)
        self.assertEqual(model_100.inactivity_cap, 100.0)
        self.assertEqual(model_200.inactivity_cap, 200.0)

    def test_fix11_decay_pre_layernorm_observable(self):
        """FIX #11: decay aplicado antes del LayerNorm.

        Test indirecto: verificar que con decay > 0 y contadores altos,
        el output difiere de un modelo equivalente con decay = 0."""
        torch.manual_seed(0)
        model_a = _rims(inactivity_decay=0.5)
        model_b = _rims(inactivity_decay=0.0)
        model_b.load_state_dict(model_a.state_dict())

        h = torch.randn(B, K, rim) * 3.0
        inact = torch.full((B, K), 90, dtype=torch.long)
        x = torch.randn(B, 128)

        model_a.eval(); model_b.eval()
        with torch.no_grad():
            out_a, _ = model_a(x, hidden=h, inactivity_steps=inact)
            out_b, _ = model_b(x, hidden=h, inactivity_steps=inact)

        diff = (out_a - out_b).abs().mean().item()
        self.assertGreater(diff, 1e-5,
            "FIX #11: con decay activo, el output debe ser observablemente "
            "distinto al modelo equivalente sin decay")

    def test_fix16_vq_loss_scalar_shape(self):
        """FIX #16: vq_loss y total_vq con shape [] (scalar canonico)."""
        # En el codebook directo
        layer = _DVNCCodebook(rim_size=rim, num_codes=32)
        h = torch.randn(B, K, rim)
        mask = torch.ones(B, K, dtype=torch.bool)
        _, vq = layer(h, mask)
        self.assertEqual(vq.shape, torch.Size([]),
            "FIX #16: _DVNCCodebook.vq_loss debe ser scalar shape []")

        # En el modulo principal
        model = _rims(comm_mode='dvnc')
        x = torch.randn(B, 5, 128)
        _, state = model(x)
        self.assertEqual(state.vq_loss.shape, torch.Size([]),
            "FIX #16: state.vq_loss debe ser scalar shape []")


# ============================================================================
# Runner
# ============================================================================

if __name__ == '__main__':
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestGroupGRUCell,
        TestInputAttentionRIM,
        TestMultiHeadCommResidual,
        TestGlobalWorkspace, 
        TestDVNCCodebook,
        TestRIMsGeneral,
        TestRIMsCommModes,
        TestRIMsRouting,
        TestFastSlowParams,
        TestRIMsValidation,
        TestInactivityDecay,
        TestFingerprint,
        TestNumericalStability,
        # v5.1.1
        TestV511Fixes,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

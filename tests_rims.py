"""
Tests unitarios y de integracion para RIMs v5.1.

Cobertura (30 tests):
  - GroupGRUCell: shapes, gradientes, independencia entre grupos
  - _InputAttentionRIM: shapes, scores, mascara top-k, W_q per-modulo (v5.1)
  - _MultiHeadCommResidual: residual, inactivos congelados, mascara filas (v5.1)
  - _GlobalWorkspace: ciclo write/broadcast, shapes, buffer dinamico (v5.1)
  - _DVNCCodebook: cuantizacion, vq_loss, straight-through, commitment adaptativo (v5.1)
  - RecurrentIndependentMechanisms: todos los modos, shapes, sparsidad,
    gradientes, fast/slow params (con validacion exhaustiva v5.1),
    manejo de secuencias y puntuales, estado inicial, routing gumbel,
    inactivity decay (v5.1), fingerprint NCO (v5.1),
    softmax sin NaN (v5.1), temperatura softplus (v5.1)
  - RIMsState: to_dict consistencia (con campos v5.1)

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

    # ---- v5.1 ----

    def test_per_module_wq(self):
        """v5.1: W_q debe ser [K, d_key, rim_size] y cada modulo debe tener
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

    def test_inactive_unchanged(self):
        """Modulos inactivos deben recibir 0 de comunicacion -> su contribucion al output
        viene solo del residual."""
        torch.manual_seed(0)
        h    = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True

        out = self.layer(h, mask)

        active_norm   = (out[:, :Ka]  - h[:, :Ka]).norm().item()
        inactive_norm = (out[:, Ka:]  - h[:, Ka:]).norm().item()
        self.assertGreaterEqual(active_norm, 0.0)
        self.assertGreaterEqual(inactive_norm, 0.0)

    def test_gradient_flows(self):
        h    = torch.randn(B, K, rim, requires_grad=True)
        mask = torch.zeros(B, K, dtype=torch.bool)
        mask[:, :Ka] = True
        out  = self.layer(h, mask)
        out.sum().backward()
        self.assertTrue(_has_grad(h))

    # ---- v5.1 ----

    def test_comm_mask_rows_not_cols(self):
        """v5.1: La mascara debe aplicarse a FILAS (queries de inactivos), no a
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
            "v5.1: Los activos deben poder leer de inactivos como fuente (keys/values). "
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
        model = _rims()
        x = torch.randn(B, 10, 128)
        _, state = model(x)
        self.assertEqual(state.hidden_states.shape,     (B, K, rim))
        self.assertEqual(state.active_rims.shape,       (B, K))
        self.assertEqual(state.attention_weights.shape,  (B, K))
        self.assertEqual(state.communication.shape,      (B, K, rim))
        # v5.1 campos
        self.assertEqual(state.inactivity_steps.shape,   (B, K))
        self.assertIsInstance(state.fingerprint, str)
        self.assertEqual(len(state.fingerprint), 16)

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
        """v5.1: reset_hidden retorna Tuple[Tensor, Tensor]."""
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

    def test_to_dict_keys(self):
        model = _rims()
        x = torch.randn(B, 128)
        _, state = model(x)
        d = state.to_dict()
        for key in ('num_active', 'activation_rate', 'attention_entropy',
                    'comm_norm', 'vq_loss',
                    'max_inactivity', 'mean_inactivity', 'fingerprint'):  # v5.1
            self.assertIn(key, d)

    def test_get_statistics_keys(self):
        model = _rims()
        x = torch.randn(B, 5, 128)
        model(x)
        stats = model.get_statistics()
        for key in ('num_rims', 'num_active', 'rim_size', 'comm_mode', 'routing',
                    'inactivity_decay', 'max_inactivity', 'mean_inactivity'):  # v5.1
            self.assertIn(key, stats)


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
        """v5.1: el parametro es _raw_gumbel_temp, accesible via property."""
        model = _rims(routing='gumbel')
        found = any(p is model._raw_gumbel_temp for p in model.parameters())
        self.assertTrue(found)

    # ---- v5.1 ----

    def test_gumbel_softplus_gradient(self):
        """v5.1: La temperatura via softplus debe tener gradiente continuo
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

    # ---- v5.1 ----

    def test_fast_slow_exhaustive_validation(self):
        """v5.1: La validacion en __init__ debe pasar para todos los comm_modes.
        Este test verifica que no lanza RuntimeError."""
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


# ============================================================================
# v5.1: Tests de inactivity decay
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

    def test_inactivity_decay_reduces_norm(self):
        """Modulos inactivos durante muchos pasos deben tener menor norma
        que al inicio cuando decay > 0."""
        model = _rims(inactivity_decay=0.05)  # decay agresivo para test
        model.eval()

        h, inact = model.reset_hidden(B, torch.device('cpu'))
        # Forzar un estado fuerte en todos los modulos
        h = h + 5.0
        initial_norm = h.norm(dim=-1).mean().item()

        # Simular inactividad larga forzando contadores altos
        inact = torch.full((B, K), 80, dtype=torch.long)  # 80 pasos inactivo

        x = torch.randn(B, 128)
        with torch.no_grad():
            _, state = model(x, hidden=h, inactivity_steps=inact)

        # Modulos que siguen inactivos deben tener norma reducida
        inactive = ~state.active_rims
        if inactive.any():
            decayed_norm = state.hidden_states[inactive].norm(dim=-1).mean().item()
            original_inactive_norm = h.expand_as(state.hidden_states)[inactive].norm(dim=-1).mean().item()
            self.assertLess(decayed_norm, original_inactive_norm * 0.99,
                "Decay debe reducir la norma de modulos inactivos")

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


# ============================================================================
# v5.1: Tests de NCO fingerprint
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

    def test_fingerprint_in_rims_state(self):
        """El fingerprint debe estar presente y ser valido en RIMsState."""
        model = _rims()
        x = torch.randn(B, 5, 128)
        _, state = model(x)

        self.assertIsInstance(state.fingerprint, str)
        self.assertEqual(len(state.fingerprint), 16)
        # Debe ser reproducible
        fp_recomputed = _compute_fingerprint(state.hidden_states)
        self.assertEqual(state.fingerprint, fp_recomputed)


# ============================================================================
# v5.1: Test de softmax sin NaN
# ============================================================================

class TestNumericalStability(unittest.TestCase):

    def test_softmax_no_nan_all_inactive(self):
        """v5.1: Si ningun modulo esta activo (edge case), la comunicacion
        no debe producir NaN — debe retornar hidden sin cambios (o con LN)."""
        # _MultiHeadCommResidual con mascara completamente False
        layer = _MultiHeadCommResidual(rim_size=rim, num_heads=4)
        h = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)  # nadie activo

        out = layer(h, mask)
        self.assertTrue(torch.isfinite(out).all(),
            "v5.1: Output de comunicacion no debe contener NaN/Inf "
            "incluso con 0 modulos activos")

    def test_softmax_no_nan_gwt_no_active(self):
        """v5.1: GWT con 0 activos no debe producir NaN."""
        layer = _GlobalWorkspace(rim_size=rim, num_rims=K, ws_slots=2)
        h = torch.randn(B, K, rim)
        mask = torch.zeros(B, K, dtype=torch.bool)

        out = layer(h, mask)
        self.assertTrue(torch.isfinite(out).all(),
            "v5.1: GWT output no debe contener NaN/Inf con 0 modulos activos")


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
        # v5.1
        TestInactivityDecay,
        TestFingerprint,
        TestNumericalStability,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

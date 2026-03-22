"""
Benchmarks para RIMs v5.1.

Mide:
  1. GroupGRUCell vs loop de GRUCell — speedup del vectorizado
  2. Throughput por comm_mode (standard / gwt / dvnc)
  3. Throughput por routing (ste / gumbel)
  4. Escalado en num_rims (K = 4, 6, 8, 12)
  5. Uso de memoria peak por configuracion
  6. Overhead del ciclo de secuencia (T = 1, 10, 50, 100)
  7. [v5.1] Overhead de inactivity decay (lambda = 0, 0.001, 0.01, 0.1)
  8. [v5.1] Overhead de NCO fingerprint
  9. [v5.1] v5.0 vs v5.1 comm_mode comparison (mascara corregida, dynamic GWT, adaptive DVNC)

Ejecutar:
    python benchmarks_rims.py
    python benchmarks_rims.py --device cuda   (si hay GPU disponible)
    python benchmarks_rims.py --quick         (iteraciones reducidas)

Salida: tabla ASCII + archivo benchmarks_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
import math
from typing import Callable, Dict, List

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mock MSC para standalone
# ---------------------------------------------------------------------------

def _mock_msc():
    class _Metrics:
        def record(self, d): pass
        def get_stats(self, k): return {}

    class ConsciousnessLayerBase(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.metrics = _Metrics()

    class LayerConfig:
        pass

    import importlib.util, pathlib
    base_mod   = types.ModuleType('consciousness.layers.base')
    config_mod = types.ModuleType('consciousness.layers.config')
    base_mod.ConsciousnessLayerBase   = ConsciousnessLayerBase
    config_mod.LayerConfig            = LayerConfig
    sys.modules.setdefault('consciousness',            types.ModuleType('consciousness'))
    sys.modules.setdefault('consciousness.layers',     types.ModuleType('consciousness.layers'))
    sys.modules['consciousness.layers.base']   = base_mod
    sys.modules['consciousness.layers.config'] = config_mod

    spec = importlib.util.spec_from_file_location(
        'rims', pathlib.Path(__file__).parent / 'rims.py'
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'consciousness.layers'
    spec.loader.exec_module(mod)
    return mod


_rims_mod = _mock_msc()
GroupGRUCell                   = _rims_mod.GroupGRUCell
RecurrentIndependentMechanisms = _rims_mod.RecurrentIndependentMechanisms
_compute_fingerprint           = _rims_mod._compute_fingerprint

# ---------------------------------------------------------------------------
# Utilidades de medicion
# ---------------------------------------------------------------------------

WARMUP    = 3
ITERS_STD = 30
ITERS_MEM = 5

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure_throughput(
    fn: Callable,
    iters: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """
    Ejecuta fn() iters veces y retorna throughput (samples/s) y latencia media (ms).
    """
    for _ in range(WARMUP):
        fn()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    elapsed = time.perf_counter() - t0

    mean_ms = elapsed / iters * 1000
    tput    = batch_size * iters / elapsed
    return {'latency_ms': mean_ms, 'throughput_seq_s': tput}


def measure_memory(fn: Callable, device: torch.device) -> float:
    """Retorna memoria peak en MB (solo CUDA)."""
    if device.type != 'cuda':
        return float('nan')
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(ITERS_MEM):
        fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_latency_us(fn: Callable, iters: int, device: torch.device) -> float:
    """Mide latencia en microsegundos (para overhead de operaciones ligeras)."""
    for _ in range(WARMUP):
        fn()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1_000_000  # microsegundos


# ---------------------------------------------------------------------------
# Benchmark 1: GroupGRUCell vs loop de GRUCell
# ---------------------------------------------------------------------------

def bench_grouped_vs_loop(device: torch.device, iters: int, B=64) -> Dict:
    K, rim = 6, 64
    x = torch.randn(B, K, rim, device=device)
    h = torch.randn(B, K, rim, device=device)

    # GroupGRUCell vectorizado
    grouped = GroupGRUCell(rim, rim, K).to(device)

    def fn_grouped():
        return grouped(x, h)

    # Loop naive
    cells = nn.ModuleList([nn.GRUCell(rim, rim) for _ in range(K)]).to(device)

    def fn_loop():
        parts = []
        for i in range(K):
            parts.append(cells[i](x[:, i], h[:, i]))
        return torch.stack(parts, dim=1)

    r_grouped = measure_throughput(fn_grouped, iters, device, B)
    r_loop    = measure_throughput(fn_loop,    iters, device, B)

    speedup = r_loop['latency_ms'] / r_grouped['latency_ms']
    return {
        'GroupGRUCell_ms': r_grouped['latency_ms'],
        'LoopGRUCell_ms':  r_loop['latency_ms'],
        'speedup_x':       speedup,
    }


# ---------------------------------------------------------------------------
# Benchmark 2: comm_mode throughput
# ---------------------------------------------------------------------------

def bench_comm_modes(device: torch.device, iters: int, B=64, T=20) -> Dict:
    results = {}
    for mode in ('standard', 'gwt', 'dvnc'):
        model = RecurrentIndependentMechanisms(
            input_size=128, hidden_size=384, num_rims=6, num_active=3,
            comm_mode=mode, dropout=0.0, num_codes=32, ws_slots=2,
        ).to(device).eval()

        x = torch.randn(B, T, 128, device=device)

        def fn(m=model, _x=x):
            with torch.no_grad():
                return m(_x)

        r = measure_throughput(fn, iters, device, B)
        r['memory_MB'] = measure_memory(fn, device)
        r['params'] = sum(p.numel() for p in model.parameters())
        results[mode] = r
    return results


# ---------------------------------------------------------------------------
# Benchmark 3: routing throughput
# ---------------------------------------------------------------------------

def bench_routing(device: torch.device, iters: int, B=64, T=20) -> Dict:
    results = {}
    for routing in ('ste', 'gumbel'):
        model = RecurrentIndependentMechanisms(
            input_size=128, hidden_size=384, num_rims=6, num_active=3,
            routing=routing, dropout=0.0,
        ).to(device).eval()

        x = torch.randn(B, T, 128, device=device)

        def fn(m=model, _x=x):
            with torch.no_grad():
                return m(_x)

        results[routing] = measure_throughput(fn, iters, device, B)
    return results


# ---------------------------------------------------------------------------
# Benchmark 4: escalado en num_rims
# ---------------------------------------------------------------------------

def bench_num_rims(device: torch.device, iters: int, B=64, T=10) -> Dict:
    results = {}
    for K in (4, 6, 8, 12):
        hidden_size = K * 64
        Ka = max(1, K // 2)
        try:
            model = RecurrentIndependentMechanisms(
                input_size=128, hidden_size=hidden_size,
                num_rims=K, num_active=Ka, dropout=0.0,
            ).to(device).eval()
            x = torch.randn(B, T, 128, device=device)

            def fn(m=model, _x=x):
                with torch.no_grad():
                    return m(_x)

            r = measure_throughput(fn, iters, device, B)
            r['params'] = sum(p.numel() for p in model.parameters())
            results[f'K={K}'] = r
        except Exception as e:
            results[f'K={K}'] = {'error': str(e)}
    return results


# ---------------------------------------------------------------------------
# Benchmark 5: escalado en seq_len
# ---------------------------------------------------------------------------

def bench_seq_len(device: torch.device, iters: int, B=32) -> Dict:
    model = RecurrentIndependentMechanisms(
        input_size=128, hidden_size=384, num_rims=6, num_active=3, dropout=0.0,
    ).to(device).eval()

    results = {}
    for T in (1, 10, 50, 100):
        x = torch.randn(B, T, 128, device=device)

        def fn(_x=x):
            with torch.no_grad():
                return model(_x)

        r = measure_throughput(fn, iters, device, B)
        r['tokens_per_s'] = B * T * iters / (r['latency_ms'] * iters / 1000)
        results[f'T={T}'] = r
    return results


# ---------------------------------------------------------------------------
# Benchmark 6 [v5.1]: overhead de inactivity decay
# ---------------------------------------------------------------------------

def bench_inactivity_decay(device: torch.device, iters: int, B=64, T=20) -> Dict:
    """
    Compara throughput con distintos valores de lambda decay.
    lambda=0 efectivamente desactiva el decay (baseline).
    """
    results = {}
    for lam in (0.0, 0.001, 0.01, 0.1):
        model = RecurrentIndependentMechanisms(
            input_size=128, hidden_size=384, num_rims=6, num_active=3,
            dropout=0.0, inactivity_decay=lam,
        ).to(device).eval()

        x = torch.randn(B, T, 128, device=device)

        def fn(m=model, _x=x):
            with torch.no_grad():
                return m(_x)

        r = measure_throughput(fn, iters, device, B)
        results[f'lambda={lam}'] = r
    return results


# ---------------------------------------------------------------------------
# Benchmark 7 [v5.1]: overhead de NCO fingerprint
# ---------------------------------------------------------------------------

def bench_fingerprint(device: torch.device, iters: int) -> Dict:
    """
    Mide el overhead de _compute_fingerprint en aislamiento.
    Compara distintos tamaños de estado oculto.
    """
    results = {}
    for B, K, D in [(1, 6, 64), (32, 6, 64), (64, 6, 64), (64, 12, 128)]:
        hidden = torch.randn(B, K, D, device=device)
        label = f'B={B},K={K},D={D}'

        def fn(_h=hidden):
            return _compute_fingerprint(_h)

        lat_us = measure_latency_us(fn, iters * 3, device)
        results[label] = {
            'latency_us': lat_us,
            'tensor_size_KB': hidden.nelement() * 4 / 1024,
        }
    return results


# ---------------------------------------------------------------------------
# Benchmark 8 [v5.1]: autoregressive loop con estado propagado
# ---------------------------------------------------------------------------

def bench_autoregressive(device: torch.device, iters: int, B=32, T=50) -> Dict:
    """
    Mide el overhead real de propagar inactivity_steps y generar fingerprint
    en un loop autoregresivo tipico (paso a paso).
    Compara: (a) batch completo T pasos, (b) loop paso a paso con propagacion.
    """
    model = RecurrentIndependentMechanisms(
        input_size=128, hidden_size=384, num_rims=6, num_active=3,
        dropout=0.0, inactivity_decay=0.001,
    ).to(device).eval()

    x = torch.randn(B, T, 128, device=device)

    # (a) Batch completo
    def fn_batch():
        with torch.no_grad():
            return model(x)

    # (b) Loop autoregresivo con propagacion
    def fn_loop():
        with torch.no_grad():
            hidden, inact = model.reset_hidden(B, device)
            for t in range(T):
                out_t, state = model(x[:, t], hidden=hidden, inactivity_steps=inact)
                hidden = state.hidden_states
                inact  = state.inactivity_steps
            return out_t, state

    r_batch = measure_throughput(fn_batch, iters, device, B)
    r_loop  = measure_throughput(fn_loop,  iters, device, B)

    overhead_pct = (r_loop['latency_ms'] / r_batch['latency_ms'] - 1.0) * 100

    return {
        'batch_T_steps':      r_batch,
        'loop_step_by_step':  r_loop,
        'loop_overhead_pct':  overhead_pct,
    }


# ---------------------------------------------------------------------------
# Benchmark 9 [v5.1]: comm_mode parameter count comparison
# ---------------------------------------------------------------------------

def bench_param_count(device: torch.device) -> Dict:
    """
    Cuenta parametros por grupo (fast/slow) y totales para cada comm_mode.
    Muestra el overhead de v5.1 features (ws_generator, per-module W_q, etc).
    """
    results = {}
    for mode in ('standard', 'gwt', 'dvnc'):
        model = RecurrentIndependentMechanisms(
            input_size=128, hidden_size=384, num_rims=6, num_active=3,
            comm_mode=mode, dropout=0.0, num_codes=32, ws_slots=2,
        ).to(device)

        total  = sum(p.numel() for p in model.parameters())
        n_fast = sum(p.numel() for p in model.fast_params())
        n_slow = sum(p.numel() for p in model.slow_params())

        results[mode] = {
            'total_params':  total,
            'fast_params':   n_fast,
            'slow_params':   n_slow,
            'fast_pct':      n_fast / total * 100,
            'slow_pct':      n_slow / total * 100,
            'coverage_ok':   n_fast + n_slow == total,
        }
    return results


# ---------------------------------------------------------------------------
# Formateo de tabla ASCII
# ---------------------------------------------------------------------------

def _table(title: str, data: Dict, cols: List[str], fmt: str = '.2f') -> str:
    lines = [f'\n{"="*70}', f'  {title}', '='*70]
    col_w = max(max(len(c) for c in cols), 14)
    header = '  '.join(f'{c:<{col_w}}' for c in cols)
    lines.append(header)
    lines.append('-' * len(header))
    for row_name, row_vals in data.items():
        if isinstance(row_vals, dict) and 'error' not in row_vals:
            vals = []
            for c in cols[1:]:
                v = row_vals.get(c, float('nan'))
                if isinstance(v, bool):
                    vals.append(str(v).rjust(col_w))
                elif isinstance(v, float):
                    vals.append(f'{v:{fmt}}'.rjust(col_w))
                elif isinstance(v, int):
                    vals.append(f'{v:,}'.rjust(col_w))
                else:
                    vals.append(str(v).rjust(col_w))
            lines.append(f'{row_name:<{col_w}}  ' + '  '.join(vals))
        else:
            err = row_vals.get('error', '?') if isinstance(row_vals, dict) else '?'
            lines.append(f'{row_name:<{col_w}}  ERROR: {err}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(device: torch.device, quick: bool = False):
    iters = 10 if quick else ITERS_STD
    n_benchmarks = 9
    print(f'\nRIMs v5.1 — Benchmarks  |  device={device}  |  iters={iters}')
    print('='*70)

    results = {}

    # 1. GroupGRUCell vs loop
    print(f'\n[1/{n_benchmarks}] GroupGRUCell vectorizado vs loop de GRUCell...')
    r1 = bench_grouped_vs_loop(device, iters)
    results['grouped_vs_loop'] = r1
    print(f"  GroupGRUCell : {r1['GroupGRUCell_ms']:.3f} ms")
    print(f"  Loop GRUCell : {r1['LoopGRUCell_ms']:.3f} ms")
    print(f"  Speedup      : {r1['speedup_x']:.2f}x")

    # 2. comm_mode
    print(f'\n[2/{n_benchmarks}] Throughput por comm_mode (B=64, T=20)...')
    r2 = bench_comm_modes(device, iters)
    results['comm_modes'] = r2
    print(_table(
        'Comm Mode Throughput',
        r2,
        ['mode', 'latency_ms', 'throughput_seq_s', 'memory_MB', 'params'],
    ))

    # 3. routing
    print(f'\n[3/{n_benchmarks}] Throughput por routing...')
    r3 = bench_routing(device, iters)
    results['routing'] = r3
    print(_table(
        'Routing Throughput',
        r3,
        ['routing', 'latency_ms', 'throughput_seq_s'],
    ))

    # 4. escalado num_rims
    print(f'\n[4/{n_benchmarks}] Escalado en num_rims (B=64, T=10)...')
    r4 = bench_num_rims(device, iters)
    results['num_rims_scaling'] = r4
    print(_table(
        'Num RIMs Scaling',
        r4,
        ['config', 'latency_ms', 'throughput_seq_s', 'params'],
    ))

    # 5. seq_len
    print(f'\n[5/{n_benchmarks}] Escalado en seq_len (B=32)...')
    r5 = bench_seq_len(device, iters)
    results['seq_len_scaling'] = r5
    print(_table(
        'Seq Len Scaling',
        r5,
        ['config', 'latency_ms', 'throughput_seq_s', 'tokens_per_s'],
    ))

    # 6. [v5.1] inactivity decay overhead
    print(f'\n[6/{n_benchmarks}] [v5.1] Overhead de inactivity decay...')
    r6 = bench_inactivity_decay(device, iters)
    results['inactivity_decay'] = r6
    print(_table(
        'Inactivity Decay Overhead',
        r6,
        ['lambda', 'latency_ms', 'throughput_seq_s'],
    ))

    # 7. [v5.1] fingerprint overhead
    print(f'\n[7/{n_benchmarks}] [v5.1] Overhead de NCO fingerprint...')
    r7 = bench_fingerprint(device, iters)
    results['fingerprint'] = r7
    print(_table(
        'NCO Fingerprint Overhead',
        r7,
        ['config', 'latency_us', 'tensor_size_KB'],
    ))

    # 8. [v5.1] autoregressive loop
    print(f'\n[8/{n_benchmarks}] [v5.1] Autoregressive loop (B=32, T=50)...')
    r8 = bench_autoregressive(device, iters)
    results['autoregressive'] = r8
    print(f"  Batch (T pasos)     : {r8['batch_T_steps']['latency_ms']:.3f} ms")
    print(f"  Loop (paso a paso)  : {r8['loop_step_by_step']['latency_ms']:.3f} ms")
    print(f"  Overhead del loop   : {r8['loop_overhead_pct']:.1f}%")

    # 9. [v5.1] parameter count breakdown
    print(f'\n[9/{n_benchmarks}] [v5.1] Parameter count por comm_mode...')
    r9 = bench_param_count(device)
    results['param_count'] = r9
    print(_table(
        'Parameter Count Breakdown',
        r9,
        ['mode', 'total_params', 'fast_params', 'slow_params', 'fast_pct', 'slow_pct', 'coverage_ok'],
    ))

    # Summary
    print(f'\n{"="*70}')
    print('  RESUMEN v5.1')
    print('='*70)
    if 'lambda=0.0' in r6 and 'lambda=0.001' in r6:
        decay_overhead = (
            r6['lambda=0.001']['latency_ms'] / r6['lambda=0.0']['latency_ms'] - 1.0
        ) * 100
        print(f"  Decay overhead (lambda=0.001 vs 0):  {decay_overhead:+.1f}%")
    if 'B=64,K=6,D=64' in r7:
        print(f"  Fingerprint latency (B=64,K=6):      {r7['B=64,K=6,D=64']['latency_us']:.0f} µs")
    print(f"  Autoregressive loop overhead:        {r8['loop_overhead_pct']:.1f}%")
    for mode, vals in r9.items():
        print(f"  {mode:>10} params: {vals['total_params']:,}  "
              f"(fast {vals['fast_pct']:.0f}% / slow {vals['slow_pct']:.0f}%)  "
              f"coverage={'OK' if vals['coverage_ok'] else 'FAIL'}")

    # Guardar JSON
    out_path = 'benchmarks_results.json'
    with open(out_path, 'w') as f:
        def _clean(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            return obj
        json.dump(_clean(results), f, indent=2)
    print(f'\nResultados guardados en {out_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RIMs v5.1 Benchmarks')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--quick',  action='store_true', help='Menos iteraciones')
    args = parser.parse_args()

    dev = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA no disponible, usando CPU')
        dev = torch.device('cpu')

    run_all(dev, quick=args.quick)

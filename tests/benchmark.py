"""
MSC Serial v2.2 — Benchmark completo
=====================================
Compara MSC Serial vs pickle vs torch.save vs json en múltiples escenarios.

Ejecutar: python msc_serial/benchmark.py
"""
import sys
import os
import io
import time
import struct
import pickle
import json
import zlib
import dataclasses
from pathlib import Path

# Importar MSC Serial desde el mismo directorio
sys.path.insert(0, os.path.dirname(__file__))
import msc_serial as msc

import numpy as np
import torch


# ─────────────────────── Helpers ────────────────────────────

def fmt_bytes(n):
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    else:
        return f"{n/(1024*1024):.2f} MB"


def fmt_time(ms):
    if ms < 0.001:
        return f"{ms*1000:.2f} us"
    elif ms < 1:
        return f"{ms:.3f} ms"
    else:
        return f"{ms:.1f} ms"


def bench_roundtrip(name, encode_fn, decode_fn, obj, rounds=200):
    """Mide encode + decode, retorna dict con tiempos y tamaños."""
    # Warmup
    for _ in range(min(5, rounds)):
        data = encode_fn(obj)
        decode_fn(data)

    # Encode
    t0 = time.perf_counter()
    for _ in range(rounds):
        data = encode_fn(obj)
    encode_ms = (time.perf_counter() - t0) / rounds * 1000

    # Decode
    t0 = time.perf_counter()
    for _ in range(rounds):
        decode_fn(data)
    decode_ms = (time.perf_counter() - t0) / rounds * 1000

    size = len(data) if isinstance(data, (bytes, bytearray)) else len(data.getvalue()) if hasattr(data, 'getvalue') else 0

    return {
        'name': name,
        'encode_ms': encode_ms,
        'decode_ms': decode_ms,
        'total_ms': encode_ms + decode_ms,
        'size': size,
    }


# ─────────────── Encode/decode wrappers ─────────────────────

def msc_encode(obj):
    return msc.dumps(obj)

def msc_decode(data):
    return msc.loads(data, strict=False)

def msc_encode_crc(obj):
    return msc.dumps(obj, with_crc=True)

def msc_decode_crc(data):
    return msc.loads(data, strict=False)

def msc_encode_compressed(obj):
    buf = io.BytesIO()
    msc.dump_compressed(obj, buf)
    return buf.getvalue()

def msc_decode_compressed(data):
    return msc.load_compressed(io.BytesIO(data), strict=False)

def pickle_encode(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_decode(data):
    return pickle.loads(data)

def torch_save_encode(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()

def torch_save_decode(data):
    return torch.load(io.BytesIO(data), weights_only=False)

def json_encode(obj):
    return json.dumps(obj).encode('utf-8')

def json_decode(data):
    return json.loads(data)

def numpy_encode(obj):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f'arr_{i}': v for i, v in enumerate(obj.values()) if isinstance(v, np.ndarray)})
    return buf.getvalue()


# ─────────────────────── Test payloads ──────────────────────

def make_payloads():
    """Genera payloads de prueba de diferentes tamaños y tipos."""

    @msc.register
    @dataclasses.dataclass
    class ModelCheckpoint:
        version: str = "5.2"
        epoch: int = 0
        loss: float = 0.0
        config: dict = dataclasses.field(default_factory=dict)

    payloads = {}

    # 1. Primitivos mixtos (simula config)
    payloads['config_small'] = {
        'name': 'MSC Runtime',
        'state_size': 256,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'enabled': True,
        'layers': [64, 128, 256],
        'device': 'cuda',
    }

    # 2. Config grande (simula RuntimeConfig completo)
    payloads['config_large'] = {f'param_{i}': float(i) * 0.01 for i in range(200)}
    payloads['config_large'].update({
        'name': 'ConsciousnessRuntime',
        'modules': ['global_workspace', 'attention_schema', 'iit', 'strange_loop'],
        'nested': {'brain': {'type': 'entity_v5', 'backbone': 'mamba'}},
    })

    # 3. Tensor pequeño (bias)
    payloads['tensor_small'] = torch.randn(128)

    # 4. Tensor medio (capa FC)
    payloads['tensor_medium'] = torch.randn(256, 256)

    # 5. Tensor grande (embedding)
    payloads['tensor_large'] = torch.randn(1024, 512)

    # 6. Dict de tensores (simula state_dict parcial)
    payloads['state_dict_mini'] = {
        'fc1.weight': torch.randn(128, 64),
        'fc1.bias': torch.randn(128),
        'fc2.weight': torch.randn(64, 128),
        'fc2.bias': torch.randn(64),
    }

    # 7. State dict mediano
    payloads['state_dict_medium'] = {}
    for i in range(10):
        payloads['state_dict_medium'][f'layer_{i}.weight'] = torch.randn(256, 256)
        payloads['state_dict_medium'][f'layer_{i}.bias'] = torch.randn(256)

    # 8. Numpy arrays
    payloads['numpy_arrays'] = {
        'states': np.random.randn(1000, 64).astype(np.float32),
        'actions': np.random.randint(0, 10, 1000).astype(np.int64),
        'rewards': np.random.randn(1000).astype(np.float32),
    }

    # 9. Lista de experiencias (simula replay buffer)
    payloads['replay_buffer'] = [
        {'state': list(range(64)), 'action': i % 10, 'reward': float(i) * 0.01, 'done': i % 50 == 0}
        for i in range(500)
    ]

    # 10. Strings pesados (simula logs)
    payloads['text_heavy'] = {
        'logs': [f"[cycle {i}] phi={np.random.rand():.4f} level=CONSCIOUS action=3" for i in range(1000)],
        'metadata': {'experiment': 'consciousness_v4', 'cycles': 3000},
    }

    # 11. Dataclass registrado
    payloads['dataclass'] = ModelCheckpoint(
        version="5.2", epoch=100, loss=0.0342,
        config={'state_size': 256, 'lr': 0.001, 'layers': [64, 128]}
    )

    # 12. Mixto: tensores + numpy + primitivos (simula checkpoint completo)
    payloads['full_checkpoint'] = {
        'version': '5.2',
        'epoch': 100,
        'loss': 0.0342,
        'weights': {
            'fc1.weight': torch.randn(128, 64),
            'fc1.bias': torch.randn(128),
        },
        'optimizer_state': {
            'step': 10000,
            'lr': 0.0003,
            'betas': (0.9, 0.999),
        },
        'metrics': {
            'phi_history': np.random.rand(100).astype(np.float32).tolist(),
            'loss_history': np.random.rand(100).astype(np.float32).tolist(),
        },
    }

    return payloads


# ──────────────────── Correctness checks ────────────────────

def _deep_equal(a, b):
    """Comparacion profunda que maneja tensores y numpy arrays."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def verify_correctness(payloads):
    """Verifica que MSC roundtrip preserva datos correctamente."""
    print("\n  ── Verificacion de correctness ──\n")
    ok_count = 0
    fail_count = 0

    for name, obj in payloads.items():
        try:
            encoded = msc.dumps(obj)
            decoded = msc.loads(encoded, strict=False)

            # Verificar por tipo
            if isinstance(obj, torch.Tensor):
                if isinstance(decoded, torch.Tensor) and torch.equal(obj, decoded):
                    ok_count += 1
                else:
                    print(f"  FAIL {name}: tensor mismatch")
                    fail_count += 1
            elif isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
                # Dict con tensores: verificar keys y tensor equality
                keys_ok = set(decoded.keys()) == set(obj.keys())
                tensors_ok = all(
                    torch.equal(decoded[k], v) if isinstance(v, torch.Tensor)
                    else (np.array_equal(decoded[k], v) if isinstance(v, np.ndarray) else decoded[k] == v)
                    for k, v in obj.items()
                )
                if keys_ok and tensors_ok:
                    ok_count += 1
                else:
                    print(f"  FAIL {name}: dict+tensor mismatch")
                    fail_count += 1
            elif isinstance(obj, dict) and any(isinstance(v, np.ndarray) for v in obj.values()):
                all_ok = True
                for k, v in obj.items():
                    if isinstance(v, np.ndarray):
                        if not np.array_equal(decoded[k], v):
                            all_ok = False
                    elif decoded.get(k) != v:
                        all_ok = False
                if all_ok:
                    ok_count += 1
                else:
                    print(f"  FAIL {name}: dict+numpy mismatch")
                    fail_count += 1
            else:
                # Comparacion generica recursiva
                if _deep_equal(obj, decoded):
                    ok_count += 1
                else:
                    print(f"  FAIL {name}: value mismatch")
                    fail_count += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            fail_count += 1

    total = ok_count + fail_count
    print(f"  Correctness: {ok_count}/{total} OK")
    if fail_count:
        print(f"  *** {fail_count} FALLOS ***")
    return fail_count == 0


# ────────────────────── Main benchmark ──────────────────────

def run_benchmarks():
    payloads = make_payloads()

    print("=" * 78)
    print(f"  MSC Serial v{msc.__version__} — Benchmark Suite")
    print(f"  Python {sys.version.split()[0]} | PyTorch {torch.__version__} | NumPy {np.__version__}")
    print("=" * 78)

    # Correctness primero
    if not verify_correctness(payloads):
        print("\n  *** ABORTANDO: fallos de correctness ***")
        sys.exit(1)

    # ── Benchmark por payload ──
    print("\n" + "=" * 78)
    print("  BENCHMARK: MSC Serial vs Pickle vs torch.save")
    print("=" * 78)

    # Header
    print(f"\n  {'Payload':<25} {'Method':<18} {'Encode':>9} {'Decode':>9} {'Total':>9} {'Size':>10}")
    print(f"  {'─'*25} {'─'*18} {'─'*9} {'─'*9} {'─'*9} {'─'*10}")

    for payload_name, obj in payloads.items():
        results = []

        # MSC Serial
        try:
            r = bench_roundtrip('MSC', msc_encode, msc_decode, obj)
            results.append(r)
        except Exception as e:
            results.append({'name': 'MSC', 'encode_ms': -1, 'decode_ms': -1, 'total_ms': -1, 'size': 0, 'error': str(e)})

        # MSC + CRC
        try:
            r = bench_roundtrip('MSC+CRC', msc_encode_crc, msc_decode_crc, obj)
            results.append(r)
        except Exception:
            pass

        # MSC Compressed
        try:
            r = bench_roundtrip('MSC+zlib', msc_encode_compressed, msc_decode_compressed, obj, rounds=100)
            results.append(r)
        except Exception:
            pass

        # Pickle
        try:
            r = bench_roundtrip('pickle', pickle_encode, pickle_decode, obj)
            results.append(r)
        except Exception as e:
            results.append({'name': 'pickle', 'encode_ms': -1, 'decode_ms': -1, 'total_ms': -1, 'size': 0, 'error': str(e)})

        # torch.save (solo para tensores/dicts con tensores)
        has_tensors = isinstance(obj, torch.Tensor) or (
            isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values())
        )
        if has_tensors:
            try:
                r = bench_roundtrip('torch.save', torch_save_encode, torch_save_decode, obj, rounds=50)
                results.append(r)
            except Exception:
                pass

        # JSON (solo para serializable)
        is_json_safe = not isinstance(obj, (torch.Tensor,)) and not (
            isinstance(obj, dict) and any(isinstance(v, (torch.Tensor, np.ndarray)) for v in obj.values())
        ) and not dataclasses.is_dataclass(obj)
        if is_json_safe:
            try:
                r = bench_roundtrip('json', json_encode, json_decode, obj, rounds=200)
                results.append(r)
            except Exception:
                pass

        # Imprimir grupo
        print()
        for i, r in enumerate(results):
            prefix = payload_name if i == 0 else ''
            if r.get('error'):
                print(f"  {prefix:<25} {r['name']:<18} {'ERROR':>9}")
            else:
                print(f"  {prefix:<25} {r['name']:<18} "
                      f"{fmt_time(r['encode_ms']):>9} "
                      f"{fmt_time(r['decode_ms']):>9} "
                      f"{fmt_time(r['total_ms']):>9} "
                      f"{fmt_bytes(r['size']):>10}")

    # ── Benchmark de escalabilidad ──
    print("\n" + "=" * 78)
    print("  ESCALABILIDAD: Tensores de tamaño creciente")
    print("=" * 78)
    print(f"\n  {'Shape':<20} {'MSC enc':>9} {'MSC dec':>9} {'pkl enc':>9} {'pkl dec':>9} {'MSC size':>10} {'pkl size':>10} {'ratio':>6}")
    print(f"  {'─'*20} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*10} {'─'*6}")

    shapes = [(10, 10), (100, 100), (256, 256), (512, 512), (1024, 1024)]
    for shape in shapes:
        t = torch.randn(*shape)
        rounds = 200 if shape[0] <= 256 else 50

        r_msc = bench_roundtrip('msc', msc_encode, msc_decode, t, rounds=rounds)
        r_pkl = bench_roundtrip('pkl', pickle_encode, pickle_decode, t, rounds=rounds)

        ratio = r_msc['size'] / r_pkl['size'] if r_pkl['size'] > 0 else 0
        print(f"  {str(shape):<20} "
              f"{fmt_time(r_msc['encode_ms']):>9} {fmt_time(r_msc['decode_ms']):>9} "
              f"{fmt_time(r_pkl['encode_ms']):>9} {fmt_time(r_pkl['decode_ms']):>9} "
              f"{fmt_bytes(r_msc['size']):>10} {fmt_bytes(r_pkl['size']):>10} "
              f"{ratio:>5.2f}x")

    # ── Overhead analysis ──
    print("\n" + "=" * 78)
    print("  OVERHEAD: bytes de metadata por tipo")
    print("=" * 78)

    overhead_tests = [
        ("None", None),
        ("bool", True),
        ("int 42", 42),
        ("int 2^64", 2**64),
        ("float", 3.14),
        ("str 'hello'", "hello"),
        ("bytes(100)", bytes(100)),
        ("list [1,2,3]", [1, 2, 3]),
        ("dict {a:1}", {"a": 1}),
        ("tensor(1)", torch.tensor(1.0)),
        ("tensor(10)", torch.randn(10)),
        ("tensor(100,100)", torch.randn(100, 100)),
        ("ndarray(100,100)", np.random.randn(100, 100)),
    ]

    print(f"\n  {'Tipo':<25} {'MSC':>10} {'pickle':>10} {'delta':>10} {'overhead%':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, obj in overhead_tests:
        msc_size = len(msc.dumps(obj))
        pkl_size = len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        delta = msc_size - pkl_size
        pct = (delta / pkl_size * 100) if pkl_size > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {fmt_bytes(msc_size):>10} {fmt_bytes(pkl_size):>10} "
              f"{sign}{delta:>9} {pct:>+9.1f}%")

    # ── Resumen ──
    print("\n" + "=" * 78)
    print("  RESUMEN")
    print("=" * 78)

    # Test rapido con el payload mas representativo
    obj = payloads['state_dict_mini']
    r_msc = bench_roundtrip('MSC', msc_encode, msc_decode, obj, rounds=500)
    r_pkl = bench_roundtrip('pickle', pickle_encode, pickle_decode, obj, rounds=500)
    r_tch = bench_roundtrip('torch', torch_save_encode, torch_save_decode, obj, rounds=100)

    speed_vs_pkl = r_pkl['total_ms'] / r_msc['total_ms'] if r_msc['total_ms'] > 0 else 0
    speed_vs_tch = r_tch['total_ms'] / r_msc['total_ms'] if r_msc['total_ms'] > 0 else 0
    size_vs_pkl = r_msc['size'] / r_pkl['size'] if r_pkl['size'] > 0 else 0
    size_vs_tch = r_msc['size'] / r_tch['size'] if r_tch['size'] > 0 else 0

    print(f"""
  Payload: state_dict_mini (4 tensores, ~57K params)

  MSC Serial:  {fmt_time(r_msc['total_ms'])} roundtrip, {fmt_bytes(r_msc['size'])}
  pickle:      {fmt_time(r_pkl['total_ms'])} roundtrip, {fmt_bytes(r_pkl['size'])}
  torch.save:  {fmt_time(r_tch['total_ms'])} roundtrip, {fmt_bytes(r_tch['size'])}

  MSC vs pickle:     {speed_vs_pkl:.2f}x velocidad, {size_vs_pkl:.2f}x tamaño
  MSC vs torch.save: {speed_vs_tch:.2f}x velocidad, {size_vs_tch:.2f}x tamaño

  Seguridad: MSC no ejecuta codigo arbitrario al deserializar.
             pickle y torch.save(weights_only=False) SI lo hacen.
""")
    print("=" * 78)


if __name__ == '__main__':
    run_benchmarks()

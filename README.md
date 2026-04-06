# MSCS — Safe Serialization for Python

**v2.3.0** | [Changelog](CHANGELOG.md) | [PyPI](https://pypi.org/project/mscs/)

> **Status: Beta** — API is stable but the format may evolve. Not yet battle-tested in large-scale production.

A secure, fast, binary serialization library. Drop-in replacement for `pickle` that **does not execute arbitrary code** during deserialization of unregistered classes.

Built for AI/ML workflows — native support for **NumPy arrays** and **PyTorch tensors** with zero-copy performance.

## Why not pickle?

```python
# pickle: arbitrary code execution on load
data = pickle.loads(untrusted_bytes)  # can run os.system("rm -rf /")

# mscs: only reconstructs explicitly registered classes
data = mscs.loads(untrusted_bytes)    # MSCSecurityError if class not registered
```

## Comparison with Alternatives

| Feature | mscs | pickle | safetensors | torch.save |
|---------|------|--------|-------------|------------|
| No arbitrary code execution | Partial* | No | Yes | No |
| HMAC authentication | Yes | No | No | No |
| Custom class support | Yes (registry) | Yes | No | Yes |
| NumPy arrays | Yes | Yes | Yes | Yes |
| PyTorch tensors | Yes | Yes | Yes | Yes |
| Circular references | Yes | Yes | No | Yes |
| Zero dependencies | Yes | Yes | Yes (Rust) | No |
| Compression built-in | Yes (zlib) | No | No | No |

\* **mscs executes `__setstate__`** on registered classes. See [Security Model](#security-model) for details.

**When to use safetensors instead:** If you only need to serialize tensors and arrays (model weights, embeddings), [safetensors](https://github.com/huggingface/safetensors) is the industry standard — it's written in Rust, truly zero-code-execution, and widely adopted. Use mscs when you need to serialize **mixed Python objects** (configs, custom classes, nested structures) alongside tensors.

## Install

```bash
pip install mscs              # core (no dependencies)
pip install mscs[numpy]       # + numpy support
pip install mscs[torch]       # + numpy + PyTorch tensor support
pip install mscs[all]         # everything
```

## Quick Start

```python
import mscs

# Primitives, collections, nested structures — just works
data = {"model": "v5.2", "lr": 0.001, "layers": [64, 128, 256]}
encoded = mscs.dumps(data)
decoded = mscs.loads(encoded)

# NumPy arrays
import numpy as np
arr = np.random.randn(100, 100).astype(np.float32)
encoded = mscs.dumps(arr)

# PyTorch tensors — no .numpy() conversion needed
import torch
weights = torch.randn(256, 256)
encoded = mscs.dumps(weights)  # safe, no pickle involved

# Full model checkpoints
checkpoint = {
    "epoch": 100,
    "model_state": {k: v for k, v in model.state_dict().items()},
    "optimizer_lr": 0.0003,
}
mscs.dump(checkpoint, open("checkpoint.mscs", "wb"))
restored = mscs.load(open("checkpoint.mscs", "rb"))
```

## Custom Classes

```python
import mscs
from dataclasses import dataclass

@mscs.register
@dataclass
class Config:
    state_size: int = 256
    lr: float = 0.001

config = Config(512, 0.0003)
data = mscs.dumps(config)
restored = mscs.loads(data)  # Config(state_size=512, lr=0.0003)

# Unregistered classes raise MSCSecurityError in strict mode
mscs.loads(data_with_unknown_class)  # MSCSecurityError

# Or get a dict fallback in non-strict mode
mscs.loads(data_with_unknown_class, strict=False)  # {'__class__': '...', '__state__': {...}}
```

### Backward Compatibility with Renamed Classes

```python
mscs.register_alias("my_module.OldConfig", Config)
```

### Register All Classes in a Module

```python
import my_models
mscs.register_module(my_models)
```

## Compression & Integrity

```python
# zlib compression
with open("data.mscs.z", "wb") as f:
    mscs.dump_compressed(large_obj, f)

with open("data.mscs.z", "rb") as f:
    obj = mscs.load_compressed(f)

# CRC32 integrity check (detects accidental corruption, NOT tamper-proof)
data = mscs.dumps(obj, with_crc=True)
mscs.loads(data)  # verifies CRC, raises MSCDecodeError if corrupted

# HMAC-SHA256 authentication (cryptographic, tamper-proof)
key = b'your-secret-key-here'
data = mscs.dumps(obj, hmac_key=key)
mscs.loads(data, hmac_key=key)          # verifies HMAC, raises MSCSecurityError if tampered
mscs.loads(data)                         # MSCSecurityError: no key provided for signed payload
mscs.loads(unsigned_data, hmac_key=key)  # MSCSecurityError: anti-downgrade protection
```

## API Reference

### Core

| Function | Description |
|----------|------------|
| `dumps(obj, *, with_crc=False, hmac_key=None) -> bytes` | Serialize to bytes |
| `loads(data, *, strict=True, hmac_key=None) -> Any` | Deserialize from bytes |
| `dump(obj, file, *, with_crc=False, hmac_key=None)` | Serialize to file (binary mode) |
| `load(file, *, strict=True, hmac_key=None) -> Any` | Deserialize from file |
| `dump_compressed(obj, file, level=6)` | Serialize with zlib compression |
| `load_compressed(file) -> Any` | Deserialize compressed data |

### Registry

| Function | Description |
|----------|------------|
| `register(cls) -> cls` | Register class as safe (also works as decorator) |
| `register_alias(old_path, cls)` | Map old class path to new class |
| `register_module(module) -> list` | Register all classes in a module |

### Utilities

| Function | Description |
|----------|------------|
| `inspect(data) -> dict` | Get metadata without deserializing |
| `benchmark(obj, rounds=100) -> dict` | Measure encode/decode performance |
| `copy(obj) -> obj` | Deep copy via serialization round-trip |

## Supported Types

| Type | Notes |
|------|-------|
| `None`, `bool`, `int`, `float`, `complex` | Ints up to 8192 bytes (~19,700 digits) |
| `str`, `bytes`, `bytearray` | UTF-8, ref-tracked |
| `list`, `tuple`, `dict`, `set`, `frozenset` | Circular refs supported |
| `datetime`, `date`, `time`, `timedelta` | ISO 8601 |
| `Decimal`, `UUID`, `Path` | Lossless |
| `Enum` | Must be registered |
| `numpy.ndarray` | dtype whitelist enforced |
| `torch.Tensor` | Auto CPU transfer, preserves requires_grad |
| `dataclass`, `__slots__`, `__dict__` objects | Must be registered |

## Performance

Benchmarked on a single machine (results may vary by hardware and payload):

**Payload: `state_dict` with 4 tensors (~57K parameters, dominated by contiguous float32 buffers):**

| Method | Roundtrip | Size |
|--------|-----------|------|
| **mscs** | **~0.1 ms** | **~65 KB** |
| pickle | ~0.6 ms | ~68 KB |
| torch.save | ~0.4 ms | ~67 KB |

mscs is fast for tensor-heavy payloads because it writes raw buffers with minimal framing overhead. **For small, nested Python structures (dicts, strings, configs), the speedup is smaller.** Always benchmark with your actual data.

Run `python tests/benchmark.py` to reproduce on your machine.

## Security Model

mscs provides a **defense-in-depth** approach, but it is **not a sandbox**. Understand the boundaries:

### What mscs prevents

1. **No dynamic imports**: Class names in the binary stream are only used as registry lookup keys — never passed to `importlib`
2. **Explicit registry**: Custom classes must be registered before deserialization; unregistered classes raise `MSCSecurityError`
3. **NumPy dtype whitelist**: Blocks `object`, `void`, and structured dtypes that could execute code
4. **Configurable limits**: `MAX_DEPTH=256`, `MAX_SIZE=512MB`, `MAX_COLLECTION=10M`, `MAX_INT_BYTES=8192`
5. **Anti zip-bomb**: `load_compressed` validates both compressed and decompressed sizes with bounded reads
6. **Path null byte rejection**: Paths containing null bytes are rejected
7. **CRC32 corruption detection**: Optional checksum to detect accidental data corruption (not cryptographic — an attacker can forge CRC32)
8. **HMAC-SHA256 authentication**: Optional cryptographic signature to detect intentional tampering. Anti-downgrade protection prevents stripping the HMAC flag.
9. **Trailing bytes rejection**: Payloads with unexpected bytes after the serialized object are rejected
10. **Integer size limit**: Ints larger than `MAX_INT_BYTES` (8192 bytes, ~19,700 digits) are rejected to prevent CPU exhaustion attacks

### What mscs does NOT prevent

1. **`__setstate__` execution**: If you register a class that implements `__setstate__`, that method **will execute** during deserialization. Only register classes you trust.
2. **Path traversal**: Deserialized `Path` objects may contain `../` sequences. The consumer must validate paths before using them for file I/O.
3. **Malicious registered classes**: The security boundary is the registry. If you register a class with dangerous behavior in `__init__`, `__setstate__`, or property setters, mscs cannot protect you.

**Rule of thumb**: mscs is safe for deserializing untrusted *data* as long as your registry only contains trusted *classes*.

## Binary Format

```
┌──────────┬─────────┬───────┬──────────┬────────────────┬──────────────┬──────────────┐
│ Magic(4) │ Ver.(1) │ Fl(1) │ Tag(1)   │ Payload(var)   │ CRC32(4)?    │ HMAC(32)?    │
│ "MSCS"   │ 0x02    │ bits  │ type tag │ type-dependent  │ if flag 0x01 │ if flag 0x02 │
└──────────┴─────────┴───────┴──────────┴────────────────┴──────────────┴──────────────┘
```

**Header** (6 bytes fixed):
- Bytes 0-3: Magic `MSCS` (0x4D534353)
- Byte 4: Format version (currently `0x02`)
- Byte 5: Flags (bit 0 = CRC32 appended, bit 1 = HMAC-SHA256 appended)

CRC32 and HMAC are mutually exclusive (HMAC is strictly superior).

**Payload**: Recursive type-length-value encoding. Each value starts with a 1-byte type tag:

| Tag | Type | Payload format |
|-----|------|----------------|
| 0x00 | None | (empty) |
| 0x01 | bool | 1 byte (0x00/0x01) |
| 0x02 | int | `<H>` byte count + signed little-endian bytes |
| 0x03 | float | `<d>` IEEE 754 double |
| 0x04 | str | `<I>` byte count + UTF-8 |
| 0x05 | bytes | `<I>` byte count + raw |
| 0x06 | list | `<I>` item count + items |
| 0x07 | tuple | `<I>` item count + items |
| 0x08 | dict | `<I>` pair count + key/value pairs |
| 0x09 | set | `<I>` item count + items (sorted) |
| 0x0A | ndarray | str(meta) + `<I>` data size + raw buffer |
| 0x0B | object | str(class_path) + encoded(state) |
| 0x0C | complex | `<dd>` real, imag |
| 0x0D | frozenset | `<I>` item count + items (sorted) |
| 0x0E | datetime | `<H>` str len + ISO 8601 string |
| 0x0F | date | `<HBB>` year, month, day |
| 0x10 | time | `<H>` str len + ISO 8601 string |
| 0x11 | timedelta (legacy) | `<iiI>` days, seconds, microseconds |
| 0x12 | Decimal | `<H>` str len + decimal string |
| 0x13 | Enum | str(class_path) + encoded(value) |
| 0x14 | bytearray | `<I>` byte count + raw |
| 0x15 | ref | `<I>` reference ID |
| 0x16 | UUID | 16 bytes raw |
| 0x17 | Path | `<I>` str len + UTF-8 path string |
| 0x18 | Tensor | str(meta) + `<I>` data size + raw buffer |
| 0x19 | timedelta2 | `<iiI>` days, seconds, microseconds |

**ndarray meta**: `"{dtype}|{shape}"` where shape is `"dim0xdim1x..."` (e.g., `"float32|100x100"`).

**Tensor meta**: `"{dtype}|{shape}|{requires_grad}"` (e.g., `"float32|256x256|0"`).

**Reference tracking**: Mutable containers (list, dict, set, etc.), strings, bytes, and arrays are assigned incrementing IDs. Tag 0x15 refers back to a previously seen object by ID, enabling circular reference support.

## License

MIT

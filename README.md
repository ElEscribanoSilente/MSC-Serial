# MSCS — Safe Serialization for Python

A secure, fast, binary serialization library. Drop-in replacement for `pickle` that **never executes arbitrary code** during deserialization.

Built for AI/ML workflows — native support for **NumPy arrays** and **PyTorch tensors** with zero-copy performance.

## Why not pickle?

```python
# pickle: arbitrary code execution on load
data = pickle.loads(untrusted_bytes)  # can run os.system("rm -rf /")

# mscs: only reconstructs explicitly registered classes
data = mscs.loads(untrusted_bytes)    # MSCSecurityError if class not registered
```

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
encoded = mscs.dumps(arr)  # 39 KB (vs 39.5 KB pickle)

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

# CRC32 integrity check
data = mscs.dumps(obj, with_crc=True)
mscs.loads(data)  # verifies CRC, raises MSCDecodeError if corrupted
```

## API Reference

### Core

| Function | Description |
|----------|------------|
| `dumps(obj, *, with_crc=False) -> bytes` | Serialize to bytes |
| `loads(data, *, strict=True) -> Any` | Deserialize from bytes |
| `dump(obj, file, **kwargs)` | Serialize to file (binary mode) |
| `load(file, **kwargs) -> Any` | Deserialize from file |
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
| `None`, `bool`, `int`, `float`, `complex` | Arbitrary precision ints |
| `str`, `bytes`, `bytearray` | UTF-8, ref-tracked |
| `list`, `tuple`, `dict`, `set`, `frozenset` | Circular refs supported |
| `datetime`, `date`, `time`, `timedelta` | ISO 8601 |
| `Decimal`, `UUID`, `Path` | Lossless |
| `Enum` | Must be registered |
| `numpy.ndarray` | dtype whitelist enforced |
| `torch.Tensor` | Auto CPU transfer, preserves requires_grad |
| `dataclass`, `__slots__`, `__dict__` objects | Must be registered |

## Performance

Benchmarked on a state_dict with 4 tensors (~57K parameters):

| Method | Roundtrip | Size | Safe |
|--------|-----------|------|------|
| **mscs** | **0.098 ms** | **65 KB** | **Yes** |
| pickle | 0.580 ms | 68 KB | No (RCE) |
| torch.save | 0.437 ms | 67 KB | No (RCE) |

**5.9x faster than pickle, 4.1x faster than torch.save.**

## Security Model

1. **No code execution**: Deserialization only reconstructs data, never runs arbitrary code
2. **Explicit registry**: Custom classes must be registered before deserialization
3. **No dynamic imports**: Class names in the binary stream are only used as registry keys
4. **NumPy dtype whitelist**: Blocks `object`, `void`, and structured dtypes
5. **Configurable limits**: `MAX_DEPTH=256`, `MAX_SIZE=512MB`, `MAX_COLLECTION=10M`
6. **Anti zip-bomb**: `load_compressed` validates both compressed and decompressed sizes
7. **CRC32 integrity**: Optional checksum to detect corruption

## Binary Format

```
[MSCS][version:1][flags:1][type_tag:1][...payload...]
```

## License

MIT

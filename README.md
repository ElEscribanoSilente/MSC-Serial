# MSC Serial (mscs)

Binary serialization for Python with explicit class registration, optional
HMAC-SHA256 authentication, and NumPy/PyTorch support.

Version **2.6.0** · Python **3.9+** · MIT license

MSCS supports primitive values, containers, shared references, many cyclic
graphs, registered classes and dataclasses. It uses its own wire format;
it cannot read pickle files or implement every pickle protocol feature.

## Install

```bash
python -m pip install mscs
python -m pip install "mscs[numpy]"  # NumPy >=1.20
python -m pip install "mscs[torch]"  # NumPy >=1.20 and PyTorch >=2.8
```

The base package has no mandatory runtime dependencies. Optional dependency
versions must also support your Python version and platform.

## Quick start

```python
import mscs

data = {"name": "example", "values": [1, 2, 3]}
blob = mscs.dumps(data)
assert mscs.loads(blob) == data

with open("example.msc", "wb") as file:
    mscs.dump(data, file)
with open("example.msc", "rb") as file:
    restored = mscs.load(file)
```

Register only classes whose reconstruction code you trust:

```python
from dataclasses import dataclass
import mscs

@mscs.register
@dataclass
class Point:
    x: float
    y: float

point = mscs.loads(mscs.dumps(Point(1.0, 2.0)))
assert isinstance(point, Point)
```

Classes are identified by module and qualified name. Keep those names stable
across producer and consumer; use `register_alias(old_name, cls)` for migrations.
Enums also require registration to restore their class identity.

## Authentication and limits

```python
import secrets
import mscs

key = secrets.token_bytes(32)  # Example: securely share/persist your real key.
signed = mscs.dumps({"value": 42}, hmac_key=key)
value = mscs.loads(
    signed,
    hmac_key=key,
    max_size=8 * 1024 * 1024,
    max_depth=64,
    max_hash_work=100_000,
)
```

Providing `hmac_key` on load requires a valid signature and rejects unsigned
v2 and legacy v1 input. Authentication is checked before object reconstruction.
For compressed files, bounded decompression precedes authentication of the
inner message. HMAC does not encrypt data. `with_crc=True` detects accidental
corruption; CRC is not authentication and cannot be combined with `hmac_key`.

| Load option | Default | Meaning |
| --- | --- | --- |
| `strict` | `True` | Reject unregistered classes; `False` returns fallback data for them. |
| `max_size` | 512 MiB | Maximum encoded message size, including framing. |
| `max_depth` | 256 | Maximum nesting depth; also available on encoding. |
| `max_hash_work` | 1,000,000 | Cumulative expanded builtin hash work for keys, set items and Enum values. |

Use per-call options; rebinding exported `MAX_*` constants does not change the
decoder defaults. Python object overhead can make memory usage substantially
larger than `max_size`. Hash work limits do not bound collision costs, equality
comparisons or registered class methods. For hostile workloads, also impose
process memory and execution-time limits appropriate to your application.

The registry is a trust boundary, not a sandbox. Reconstruction may execute
registered `__new__`, `__setstate__`, attribute descriptors, hash/equality
methods and Enum `_missing_` hooks. `strict=False` does not disable registered
class hooks. MSCS does not import arbitrary classes named by a payload, but
registering a class explicitly authorizes its reconstruction behavior.

## Arrays, tensors and object state

```python
import numpy as np
import mscs

array = np.arange(12, dtype=np.float32).reshape(3, 4)
assert np.array_equal(mscs.loads(mscs.dumps(array)), array)
```

Numeric and boolean NumPy dtypes are supported; object, structured, void and
string dtypes are rejected. Arrays are reconstructed into copied buffers.
Dense PyTorch tensors are serialized through CPU storage, including `bfloat16`
and resolved conjugate/negative views. Restored tensors are on CPU; device
placement, storage views and autograd graphs are not preserved. The
`requires_grad` flag is retained where the dtype supports it.

Dataclass fields, inherited/private slots and unset slots are handled explicitly.
Shared references and supported cycles preserve identity. Cycles that require
passing unresolved tuple state into an opaque `__setstate__` fail with
`MSCDecodeError`; use a mutable container to break that reconstruction dependency.

## API

| Function | Purpose |
| --- | --- |
| `dumps(obj, **options)` / `loads(blob, **options)` | Encode/decode bytes. |
| `dump(obj, file, **options)` / `load(file, **options)` | Use an open binary file. |
| `dump_compressed(obj, file, level=6, **options)` / `load_compressed(file, **options)` | Use a bounded zlib container around an MSCS message. |
| `register(cls)` / `register_alias(name, cls)` | Allow a class or its former wire name. |
| `register_module(module)` | Register classes from a trusted module; review what it exposes. |
| `copy(obj)` | Round-trip a trusted object with non-strict fallback behavior. |
| `inspect(blob)` | Read basic framing metadata; does not authenticate or validate the entire message. |
| `benchmark(obj, rounds=100)` | Run the included local serialization benchmark. |

Encoding options are `with_crc`, `hmac_key` and `max_depth`. Decoding options
are `strict`, `hmac_key`, `max_size`, `max_depth` and `max_hash_work`.
Catch `MSCError` for library errors, or its subclasses `MSCEncodeError`,
`MSCDecodeError` and `MSCSecurityError` for more specific handling.

## Upgrading to 2.6

- Readers continue to accept supported v1/v2 messages, subject to authentication
  and resource limits. Wire v2 remains the writer's version.
- New field-and-slot dataclass states and `bfloat16` metadata require 2.6 readers.
  Upgrade consumers before writing those forms. Field-only frozen slotted
  dataclasses keep their existing list state representation.
- Primitive-backed Enums now retain their registered type. Older messages
  that stored them as scalars cannot recover the original Enum class.
- Invalid Enum values produce bounded builtin diagnostics. Stored members and
  registered `_missing_` hooks are used for resolution; custom Enum metaclass
  `__call__` is not a deserialization hook.
- The new hash work budget may reject large legitimate hashed graphs; choose
  an explicit budget after measuring your workload. PyTorch extras now require 2.8+.

See the [changelog](https://github.com/ElEscribanoSilente/MSC-Serial/blob/main/CHANGELOG.md)
for the complete release history.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python tests/security_audit.py
```

Install `.[all,test]` to exercise the optional array and tensor tests. The CI
configuration targets Python 3.9–3.14 on Linux and Windows, with separate base
and optional dependency jobs. The suite includes mutation, authentication,
resource-limit and object graph regressions. Passing tests does not establish
absence of other defects.

[Release instructions](https://github.com/ElEscribanoSilente/MSC-Serial/blob/main/RELEASING.md)
· [Security policy](https://github.com/ElEscribanoSilente/MSC-Serial/blob/main/SECURITY.md)
· [License](https://github.com/ElEscribanoSilente/MSC-Serial/blob/main/LICENSE)

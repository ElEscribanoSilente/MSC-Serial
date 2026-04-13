# Changelog

All notable changes to MSCS are documented here.

## [2.4.0] — 2026-04-12

### Added
- **Native `collections.deque` support** — tag `0x1A`. Preserves `maxlen` and supports circular references. Format: `<i>` maxlen (-1 if None) + `<I>` item count + items.

### Fixed
- **`__getstate__`/`__setstate__` ignored on dataclasses** — dataclasses that defined `__getstate__` were serialized by walking their fields directly (via `dataclasses.fields()`), which caused `MSCEncodeError` if any field contained unsupported types (e.g., `deque`). The encoder and decoder now check for `__getstate__`/`__setstate__` **before** checking `is_dataclass`. Priority order: `__getstate__`/`__setstate__` > dataclass fields > `__slots__` > `__dict__`.

### Backward Compatibility
- Wire format is fully backward compatible. Payloads from v2.3, v2.2, v2.1, v2.0, and v1.0 load without changes.
- The `deque` tag (`0x1A`) is new — payloads containing `deque` cannot be read by v2.3 or earlier (forward compatibility is not guaranteed).

---

## [2.3.0] — 2026-04-06

### Added
- **HMAC-SHA256 authentication** — `hmac_key=` parameter in `dumps()`/`loads()`/`dump()`/`load()` for cryptographic payload signing. Uses `hmac.compare_digest()` for timing-safe verification.
- **Anti-downgrade protection** — providing `hmac_key` for an unsigned payload raises `MSCSecurityError`, preventing silent HMAC stripping attacks.
- **Trailing bytes validation** — `loads()` now rejects payloads with unexpected bytes after the serialized object.
- **`MAX_INT_BYTES = 8192`** — integers larger than 8192 bytes (~19,700 decimal digits) are rejected on encode and decode, preventing CPU exhaustion via crafted payloads.
- **Path null byte rejection** — deserialized `Path` objects containing null bytes raise `MSCSecurityError`.
- **Thread-safe registry** — `register()`, `register_alias()`, and `register_module()` now use `threading.Lock`.
- **pytest test suite** — 151 unit tests covering roundtrip, security, edge cases, backward compat, threading, file I/O, HMAC, trailing bytes, and int limits.
- **Hypothesis fuzzing** — 18 property-based tests (3,500+ examples) covering roundtrip, adversarial binary payloads, HMAC tampering, and resource limits.
- **`test` optional dependency** — `pip install mscs[test]` installs `pytest` and `hypothesis`.

### Fixed
- **Tuple/frozenset ref desync** — decoder now reserves ref slots before decoding children, matching encoder order. Previously `('', '')` and similar tuples with shared refs would fail roundtrip.
- **OBJ ref desync** — decoder now reserves ref slot for objects before decoding their state, fixing failures with nested registered classes (e.g., dataclass containing another dataclass).
- **`id()` reuse in OBJ encoder** — temporary state dicts extracted from dataclasses/`__slots__` objects are now pinned to prevent CPython from reusing their `id()`, which caused false ref hits and silent data corruption in nested objects.
- **Top-level optional imports** — `numpy` and `torch` are now imported once at module load, not inside every `_encode()`/`_decode()` call.
- **`load_compressed` bounded read** — now reads at most `MAX_COMPRESSED + 1` bytes instead of unbounded `file.read()`.

### Changed
- `dumps()` signature: added `hmac_key` parameter (backward compatible, defaults to `None`).
- `loads()` signature: added `hmac_key` parameter (backward compatible, defaults to `None`).
- `dump()` and `load()` now use explicit keyword arguments instead of `**kwargs`.
- Flags byte (offset 5): bit 1 now indicates HMAC-SHA256 (32 bytes appended after payload). CRC (bit 0) and HMAC (bit 1) are mutually exclusive.
- `register()` and `register_module()` docstrings now warn that `__setstate__` of registered classes **will execute** during deserialization.

### Backward Compatibility
- Wire format is fully backward compatible. Payloads from v2.2, v2.1, v2.0, and v1.0 load without changes.
- Existing code calling `dumps()`/`loads()` without `hmac_key` works identically.
- `MAX_INT_BYTES` may reject integers that v2.2 accepted (> 8192 bytes / ~19,700 digits). This is intentional for security.

---

## [2.2.1] — 2025-xx-xx

### Fixed
- PyPI project URLs now point to the correct GitHub repository.

---

## [2.2.0] — 2025-xx-xx

### Added
- Native `torch.Tensor` support (tag `0x18`) — serializes dtype, shape, and `requires_grad` without manual numpy conversion.
- `register_alias(old_path, cls)` for backward compatibility with renamed/moved classes.

### Fixed
- `timedelta` now uses dedicated tag `_TIMEDELTA2` (`0x19`), eliminating the ambiguous heuristic between v2.0 and v2.1 formats.
- `_encode_str` validates length against `MAX_STRING`.
- `load_compressed` protected against zip bombs (validates both compressed and decompressed sizes).

### Backward Compatibility
- Retrocompatible with v2.1, v2.0, and v1.0 payloads.

---

## [2.1.0]

### Added
- Native `UUID` support.
- Native `pathlib.Path` support.
- `register_module()` for bulk class registration.
- `copy()` — deep copy via serialization round-trip.
- `inspect()` now shows root tag name.
- Decode error breadcrumbs (path context).

### Fixed
- `timedelta` now encodes days/seconds/microseconds separately (v2.0 lost precision via `total_seconds()` float).
- NumPy dtype validation against whitelist of safe types.

---

## [2.0.0]

### Added
- Class registry (replaces dynamic `importlib`).
- Types: `complex`, `frozenset`, `datetime`/`date`/`time`/`timedelta`, `Decimal`, `Enum`, `bytearray`.
- Circular reference detection and handling.
- Depth and size limits.
- Typed exceptions (`MSCEncodeError`, `MSCDecodeError`, `MSCSecurityError`).
- `benchmark()` utility.
- Optional CRC32 integrity check.

---

## [1.0.0]

- Initial release. Basic serialization of primitives, collections, numpy arrays, and registered objects.

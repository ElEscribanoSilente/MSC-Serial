# Changelog

All notable changes to MSCS are documented here.

## [2.5.0] — 2026-07-14

Minor release: a batch of high/medium data-integrity, DoS, and container-integrity fixes, plus new backward-compatible public API (per-call `max_size` / `max_depth` limits). Wire format unchanged.

### Fixed
- **Silent loss of `__slots__` attributes (High / data integrity)** — the encoder's slot extraction missed three shapes, all round-tripping without error but returning incomplete instances: (1) `__slots__ = "name"` declared as a plain string was iterated character by character, serializing an empty state; (2) hybrid classes (a `__slots__` base with a `__dict__` subclass) took the `__dict__` branch and dropped every inherited slot — and the decoder wrote slot values into `obj.__dict__`, where the slot descriptor shadows them; (3) private slots (`__name`) were looked up unmangled, missing the real `_Class__name` attribute. Slot names are now collected once per class via `_collect_slot_names()`: full MRO walk using each class's own `__slots__` (str/iterable/dict forms), CPython name-mangling applied, `__dict__`/`__weakref__` entries excluded; the encoder merges slots with instance `__dict__` for hybrids (effective values win over shadowed dict keys), and the decoder routes each state key to its store — slot names through their descriptor (`setattr`), everything else directly into the instance `__dict__`. Routing per store matters: blanket `setattr` would let a spurious-but-legal `"__dict__"` key inside an instance dict *replace the object's entire `__dict__`* (silent attribute clobbering and cross-object identity aliasing — caught by independent blind review of this fix). On slots-only classes a non-slot key fails closed (`AttributeError` → `MSCDecodeError`). Unset slots stay unset after round-trip. Anchors: `TestSlotsRoundtrip` (13 tests).
- **Silent corruption of circular references through tuples and registered objects (High / data integrity)** — the decoder reserved a `None` placeholder for tuples, frozensets, and objects before decoding their children, and `_REF` returned that placeholder as-is. Any cycle passing through a tuple, any self- or mutually-referencing registered object, and any cycle through a `strict=False` fallback dict decoded silently into structures with `None` holes — no exception, semantically wrong data. Objects are now created and published to the ref table **before** their state is decoded (identity first, state second — pickle semantics), and refs into a tuple still under construction resolve through deferred fix-ups that patch the insertion site when the tuple materializes, cascading through nested pending tuples. Anchors: `test_tuple_cycle`, `test_nested_tuple_cycle`, `test_tuple_cycle_multiple_pending_slots`, `test_registered_object_self_reference`, `test_registered_objects_mutual_reference`, `test_object_in_tuple_cycle`, `test_slots_object_in_tuple_cycle`, `test_frozen_dataclass_in_tuple_cycle`, `test_unregistered_fallback_self_reference`, `test_unregistered_fallback_tuple_cycle`, `test_deque_tuple_cycle`, `test_dict_value_tuple_cycle`.

### Security
- **`load_compressed` now validates container integrity as strictly as `loads()` validates the payload (Medium / integrity, untrusted files)** — the compression layer was lenient: it never verified the zlib stream ended cleanly (`decompressor.eof`), never rejected bytes after the stream (`decompressor.unused_data`), never checked the decompressed length against the declared `orig_size`, a truncated 4-byte header leaked a raw `struct.error`, and a corrupt (non-truncated) stream leaked a raw `zlib.error`. So a truncated stream (missing part of the ADLER32 checksum), a forged `orig_size`, trailing garbage, and two concatenated containers were all accepted as valid — parser-differential fodder — while bit-rot raised the wrong exception type. It now fails closed on each with `MSCDecodeError`: incomplete/truncated header, `eof=False`, non-empty `unused_data`, `len(out) != orig_size`, and any `zlib.error` are all wrapped. Blind review also caught a false-reject the earlier compressed-read bound introduced: incompressible data (random/encrypted/already-compressed) compresses to slightly *more* than its raw size, so the compressed read is now bounded by `zlib.compressBound(max_size)` rather than `max_size`, and a legitimate `dump_compressed`→`load_compressed` with a tightly-tuned `max_size` no longer fails. Anchors: `TestCompressedIntegrity` (16 tests).
- **`MAX_SIZE` now bounds the total payload, not just individual fields (High / memory DoS)** — `loads()` never checked the total input length: per-field caps (`MAX_STRING` 100 MB, `MAX_COLLECTION` 10 M, `MAX_SIZE` per array) all held while their *sum* was unbounded, so a payload of many individually-valid 100 MB byte strings could force allocation far beyond `MAX_SIZE`, and `load()` materialized the whole file via `file.read()` before any check. `loads()` now rejects `len(data) > max_size` up front (before version dispatch, so v1 and v2 alike) — since every decoded string/collection lives inside the input, bounding the input bounds the cumulative total. `load()` reads at most `max_size + 1` bytes and rejects a larger file without materializing it. `load_compressed()` bounds **both** the decompressed output and the compressed read to `max_size` (the latter capped by `MAX_COMPRESSED`), so lowering `max_size` actually lowers peak memory on the compressed side too (caught by blind review — the compressed read was still fixed at `MAX_COMPRESSED`). The limit applies to the whole encoded blob (data + framing): a single field whose raw size reaches `MAX_SIZE` crosses the default by the framing overhead, so `copy()` bounds its own trusted round-trip by the exact blob length rather than the default. Peak memory is still a *multiple* of `max_size` (Python object overhead is ~6× for tiny primitives), so `max_size` should be set conservatively for untrusted input — see below. Anchors: `TestTotalSizeLimit` (14 tests).
- **Resource limits are now genuinely configurable — per call, not by attribute rebinding** — `dumps`/`dump` accept `max_depth`; `loads`/`load`/`load_compressed` accept `max_size` and `max_depth`, each defaulting to the module constant. Previously the only "knobs" were the module constants, and rebinding the re-exported `mscs.MAX_DEPTH` / `mscs.MAX_SIZE` silently did nothing (the encoder/decoder read `_core`'s own globals, not the package-level name), so `mscs.MAX_DEPTH = 1` was a no-op. The per-call parameters are the supported, thread-safe mechanism; the module constants remain read-only defaults. Anchors: `test_max_depth_configurable_per_call_decode`, `test_max_depth_configurable_per_call_encode`, `test_module_constant_rebind_has_no_effect`.

- **Forged forward-ref payloads now fail closed** — hand-crafted payloads placing a `_REF` to a slot still under construction in unpatchable positions (dict key, set/frozenset member, self-referential root tuple, Enum value) previously decoded into silently corrupted structures containing `None`. They now raise `MSCDecodeError`, and `loads()` additionally verifies — in both the v1 and v2 branches — that no unresolved placeholder survives decoding. None of these shapes can be produced by the encoder (they would require hashing a cycle). Anchors: `TestForgedPendingRefs` (7 tests).

- **User code never observes the internal sentinel** — Python's attribute protocol (`__setattr__` overrides, data descriptors/properties, custom `__setstate__`) executes user code at assignment time. A pending value is therefore *never assigned*: the deferred fix-up performs the first and only assignment with the real value. `__setstate__` code that *reads* a not-yet-patched attribute of a sibling object gets a plain `AttributeError` (wrapped in `MSCDecodeError`) instead of a sentinel. Found by independent blind review of this fix. Anchors: `test_custom_setattr_never_sees_pending`, `test_property_setter_never_sees_pending`, `test_setstate_copying_nested_pending_attr_fails_closed`, `test_setstate_holding_nested_object_resolves_after_load`.

### Changed
- One legitimate-but-unresolvable shape now fails closed instead of corrupting: a class with a custom `__setstate__` whose state contains a reference to a tuple still under construction (a cycle through a tuple into an opaque `__setstate__`) raises `MSCDecodeError` with guidance, where it previously received `None` silently. Break the cycle through a mutable container, or drop `__setstate__`. Anchor: `test_setstate_object_in_tuple_cycle_fails_closed`.

### Performance
- The per-item pending checks cost ~11% on a container-dense synthetic microbenchmark (2000-key dict of small lists/tuples). On buffer-dominated payloads (tensors, arrays, long strings) the overhead is proportional to container count, not data size, and is noise.

### Backward Compatibility
- Wire format unchanged; the encoder is untouched. All valid v2.x/v1.0 payloads still load. Graphs with cycles through tuples/objects that previously decoded **corrupted** now decode **correctly** from the same bytes.
- New keyword-only parameters (`max_size`, `max_depth`) are additive with backward-compatible defaults. One behavior change: `loads()`/`load()` now reject a blob larger than `max_size` (default 512 MB). A payload that legitimately serializes above 512 MB — rare — must be loaded with an explicit higher `max_size`.

### Tests
- Suite grows from 215 to 284 (66 new regression anchors + 2 behavior controls); zero regressions.

---

## [2.4.1] — 2026-07-06

### Security
- **HMAC authentication bypass via v1 downgrade (Critical)** — `loads()` dispatched v1 payloads (`MSCS\x01`) before any HMAC verification, silently ignoring `hmac_key` and forcing `strict=False`. An attacker could forge an unauthenticated v1 payload and bypass HMAC authentication entirely, and trigger `__setstate__` of registered classes with attacker data. `loads()` now rejects a v1 payload when `hmac_key` is provided (fail-closed), closing the downgrade from a signed v2 payload to an unsigned v1 one. Anchors: `test_hmac_v1_downgrade_rejected`, `test_hmac_v1_object_downgrade_rejected`.
- **Zip-bomb memory-exhaustion DoS in `load_compressed` (Medium)** — `zlib.decompress()` materialized the entire decompressed output before the `MAX_SIZE` check (`bufsize` is a hint, not a cap), so a small crafted payload could exhaust memory despite the guard. Decompression is now incremental and bounded to `MAX_SIZE + 1`, aborting as soon as the limit is crossed; peak memory stays near `MAX_SIZE` regardless of the bomb's true size. Anchor: `test_zip_bomb_bounded_memory`.
- **ENUM tag type confusion (Medium)** — the decoder called `cls(value)` for any registered class referenced by an `ENUM` tag without checking it was an `Enum`, invoking arbitrary constructors with attacker-controlled arguments (beyond the documented `__setstate__` boundary). It now requires `issubclass(cls, Enum)` before instantiating. Anchor: `test_enum_tag_rejects_non_enum_class`.

### Fixed
- **Frozen dataclass round-trip** — a `@dataclass(frozen=True)` encoded correctly but failed to decode: the decoder assigned fields with `setattr()`, which frozen instances forbid (`FrozenInstanceError` → `MSCDecodeError`). The dataclass branch now uses `object.__setattr__()`, matching how dataclass-generated `__init__` populates frozen instances. Anchor: `test_frozen_dataclass_roundtrip`.

### Changed
- v1 payloads now honor the caller's `strict` argument instead of forcing `strict=False`. Under the default `strict=True`, a v1 payload referencing an unregistered class now raises `MSCSecurityError` instead of returning a fallback dict.

### Backward Compatibility
- Wire format unchanged; all valid v2.4/v2.3/v2.2/v2.1/v2.0/v1.0 payloads still round-trip.
- Two behavior changes affect **v1** payloads only: loading a v1 payload with `hmac_key` set is now rejected (it could never be authenticated), and unregistered objects in a v1 payload under the default `strict=True` now raise instead of returning a fallback dict. Load such legacy payloads without `hmac_key` (and with `strict=False` if the fallback dict is required).

### Tests
- Suite grows from 207 to 215 (8 new regression anchors and controls); zero regressions.

---

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

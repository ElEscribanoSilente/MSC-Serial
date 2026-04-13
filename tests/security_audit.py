"""
MSC Serial v2.2 — Adversarial Security Audit
=============================================
Tests crafted payloads, edge cases, and attack vectors.
"""
import sys
import os
import struct
import io
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import mscs
from mscs._core import (
    MAGIC, VERSION, _NONE, _BOOL, _INT, _FLOAT, _STR, _BYTES,
    _LIST, _TUPLE, _DICT, _SET, _NDARRAY, _OBJ, _COMPLEX,
    _FROZENSET, _DATETIME, _DATE, _TIME, _TIMEDELTA, _DECIMAL,
    _ENUM, _BYTEARRAY, _REF, _UUID, _PATH, _TENSOR, _TIMEDELTA2,
    _DEQUE, MAX_DEPTH, MAX_SIZE, MAX_COLLECTION, MAX_STRING, _Encoder,
    _registry, _is_safe_dtype,
)

results = []

def test_reject(name, fn):
    """Expects MSCDecodeError, MSCSecurityError, or MSCEncodeError."""
    try:
        fn()
        results.append(('FAIL', name, 'No exception raised — should have been rejected'))
    except (mscs.MSCDecodeError, mscs.MSCSecurityError, mscs.MSCEncodeError) as e:
        results.append(('OK', name, type(e).__name__))
    except Exception as e:
        results.append(('BUG', name, f'{type(e).__name__}: {e}'))

def test_ok(name, fn):
    """Expects success."""
    try:
        fn()
        results.append(('OK', name, 'passed'))
    except Exception as e:
        results.append(('BUG', name, f'{type(e).__name__}: {e}'))

def test_note(name, fn, note):
    """Run fn, always OK, but record a note (informational)."""
    try:
        result = fn()
        results.append(('NOTE', name, f'{note}: {result!r}'))
    except Exception as e:
        results.append(('NOTE', name, f'{note}: {type(e).__name__}: {e}'))

HEADER = MAGIC + VERSION + b'\x00'


def build_meta_payload(tag_byte, meta_str, data_bytes):
    """Build a payload with tag + _encode_str(meta) + length + data."""
    buf = io.BytesIO()
    buf.write(HEADER)
    buf.write(tag_byte)
    # _encode_str format: _STR tag + <I length + raw
    raw_meta = meta_str.encode('utf-8')
    buf.write(_STR)
    buf.write(struct.pack('<I', len(raw_meta)))
    buf.write(raw_meta)
    # data
    buf.write(struct.pack('<I', len(data_bytes)))
    buf.write(data_bytes)
    return buf.getvalue()


def main():
    print("=" * 70)
    print(f"  MSC Serial v{mscs.__version__} — Adversarial Security Audit")
    print("=" * 70)

    # ═══════════════ 1. MALFORMED PAYLOADS ═══════════════
    print("\n  -- 1. Malformed payloads --")

    test_reject("empty bytes", lambda: mscs.loads(b''))
    test_reject("too short (3 bytes)", lambda: mscs.loads(b'MSC'))
    test_reject("wrong magic", lambda: mscs.loads(b'EVIL\x02\x00\x00'))
    test_reject("future version 0xFF", lambda: mscs.loads(b'MSCS\xFF\x00\x00'))
    test_reject("unknown tag 0xFF", lambda: mscs.loads(HEADER + b'\xFF'))
    test_reject("truncated int (says 4B, has 0)", lambda: mscs.loads(HEADER + _INT + b'\x04\x00'))
    test_reject("truncated str (says 100B, has 0)", lambda: mscs.loads(HEADER + _STR + struct.pack('<I', 100)))
    test_reject("truncated list (says 5, has 0)", lambda: mscs.loads(HEADER + _LIST + struct.pack('<I', 5)))
    test_reject("truncated dict (says 1kv, has 0)", lambda: mscs.loads(HEADER + _DICT + struct.pack('<I', 1)))
    test_reject("truncated ndarray (no meta)", lambda: mscs.loads(HEADER + _NDARRAY))
    test_reject("truncated tensor (no meta)", lambda: mscs.loads(HEADER + _TENSOR))
    test_reject("just header, no payload", lambda: mscs.loads(HEADER))
    test_reject("header + null byte", lambda: mscs.loads(HEADER + b'\x00\x00'))  # None + extra garbage? Actually this decodes None fine

    # ═══════════════ 2. RESOURCE EXHAUSTION (DoS) ═══════════════
    print("\n  -- 2. Resource exhaustion --")

    test_reject("list count > MAX_COLLECTION", lambda: mscs.loads(HEADER + _LIST + struct.pack('<I', MAX_COLLECTION + 1)))
    test_reject("dict count > MAX_COLLECTION", lambda: mscs.loads(HEADER + _DICT + struct.pack('<I', MAX_COLLECTION + 1)))
    test_reject("set count > MAX_COLLECTION", lambda: mscs.loads(HEADER + _SET + struct.pack('<I', MAX_COLLECTION + 1)))
    test_reject("str len > MAX_STRING", lambda: mscs.loads(HEADER + _STR + struct.pack('<I', MAX_STRING + 1)))
    test_reject("bytes len > MAX_STRING", lambda: mscs.loads(HEADER + _BYTES + struct.pack('<I', MAX_STRING + 1)))
    test_reject("bytearray len > MAX_STRING", lambda: mscs.loads(HEADER + _BYTEARRAY + struct.pack('<I', MAX_STRING + 1)))

    # Depth bomb: 300 nested lists
    depth_bomb = HEADER
    for _ in range(300):
        depth_bomb += _LIST + struct.pack('<I', 1)
    depth_bomb += _NONE
    test_reject("depth bomb (300 nested lists)", lambda: mscs.loads(depth_bomb))

    # INT with max <H> length = 65535, but no data
    test_reject("int n_bytes=65535, no data", lambda: mscs.loads(HEADER + _INT + struct.pack('<H', 65535)))

    # ═══════════════ 3. REFERENCE MANIPULATION ═══════════════
    print("\n  -- 3. Reference attacks --")

    test_reject("forward ref id=999", lambda: mscs.loads(HEADER + _REF + struct.pack('<I', 999)))
    test_reject("forward ref id=0 (empty refs)", lambda: mscs.loads(HEADER + _REF + struct.pack('<I', 0)))
    test_reject("ref id=0xFFFFFFFF", lambda: mscs.loads(HEADER + _REF + struct.pack('<I', 0xFFFFFFFF)))

    # Self-referencing: list that contains a REF to itself
    # This is a valid pattern (circular ref), test it works
    def test_circular():
        circ = [1, 2]
        circ.append(circ)
        data = mscs.dumps(circ)
        dec = mscs.loads(data, strict=False)
        assert dec[2] is dec, "Circular ref not preserved"
    test_ok("circular ref roundtrip", test_circular)

    # ═══════════════ 4. TYPE CONFUSION / DTYPE ATTACKS ═══════════════
    print("\n  -- 4. Type confusion --")

    test_reject("ndarray dtype=object", lambda: mscs.loads(build_meta_payload(_NDARRAY, 'object|10', b'\x00' * 80), strict=False))
    test_reject("ndarray dtype=void", lambda: mscs.loads(build_meta_payload(_NDARRAY, 'void|10', b'\x00' * 80), strict=False))
    test_reject("ndarray dtype=O", lambda: mscs.loads(build_meta_payload(_NDARRAY, 'O|10', b'\x00' * 80), strict=False))
    test_reject("tensor dtype=object", lambda: mscs.loads(build_meta_payload(_TENSOR, 'object|10|0', b'\x00' * 80), strict=False))

    # Shape/data mismatch
    test_reject("ndarray shape 100x100 but only 16B data",
                lambda: mscs.loads(build_meta_payload(_NDARRAY, 'float32|100x100', b'\x00' * 16), strict=False))

    # Valid dtype pass
    test_ok("dtype float32 accepted", lambda: assert_true(_is_safe_dtype('float32')))
    test_ok("dtype <f8 accepted", lambda: assert_true(_is_safe_dtype('<f8')))
    test_ok("dtype int64 accepted", lambda: assert_true(_is_safe_dtype('int64')))
    test_ok("dtype object blocked", lambda: assert_true(not _is_safe_dtype('object')))
    test_ok("dtype void blocked", lambda: assert_true(not _is_safe_dtype('void')))
    test_ok("dtype V8 blocked", lambda: assert_true(not _is_safe_dtype('V8')))

    # ═══════════════ 5. OBJECT DESERIALIZATION ═══════════════
    print("\n  -- 5. Object deserialization --")

    # Build an OBJ payload for unregistered class
    obj_payload = io.BytesIO()
    obj_payload.write(HEADER + _OBJ)
    cls_s = '__main__.MaliciousClass'.encode('utf-8')
    obj_payload.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
    obj_payload.write(_DICT + struct.pack('<I', 0))  # empty state
    obj_bytes = obj_payload.getvalue()

    test_reject("unregistered class strict=True", lambda: mscs.loads(obj_bytes, strict=True))

    def test_unregistered_fallback():
        result = mscs.loads(obj_bytes, strict=False)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result['__class__'] == '__main__.MaliciousClass'
    test_ok("unregistered class strict=False -> dict fallback", test_unregistered_fallback)

    # Enum with unregistered class
    enum_payload = io.BytesIO()
    enum_payload.write(HEADER + _ENUM)
    enum_s = 'evil.module.BadEnum'.encode('utf-8')
    enum_payload.write(_STR + struct.pack('<I', len(enum_s)) + enum_s)
    enum_payload.write(_INT + struct.pack('<H', 1) + b'\x01')
    test_reject("unregistered enum strict=True", lambda: mscs.loads(enum_payload.getvalue(), strict=True))

    # __setstate__ execution (registered class) — security boundary
    import dataclasses

    @mscs.register
    @dataclasses.dataclass
    class SafeClass:
        value: int = 0

    setstate_called = []

    @mscs.register
    class HasSetstate:
        def __init__(self):
            self.data = None
        def __getstate__(self):
            return {'data': self.data}
        def __setstate__(self, state):
            setstate_called.append(True)
            self.data = state.get('data')

    def test_setstate_exec():
        obj = HasSetstate()
        obj.data = 42
        data = mscs.dumps(obj)
        setstate_called.clear()
        mscs.loads(data)
        assert len(setstate_called) == 1, "__setstate__ not called"
    test_ok("__setstate__ executes for registered class", test_setstate_exec)

    # ═══════════════ 6. PATH SECURITY ═══════════════
    print("\n  -- 6. Path deserialization --")

    import pathlib

    def test_path_traversal():
        payload = HEADER + _PATH
        p = '../../../../etc/passwd'.encode('utf-8')
        payload += struct.pack('<I', len(p)) + p
        result = mscs.loads(payload, strict=False)
        return result
    test_note("path traversal string", test_path_traversal,
              "FINDING: Path objects with traversal ARE deserialized — consumer must validate")

    def test_path_null():
        payload = HEADER + _PATH
        p = '/tmp/evil\x00hidden'.encode('utf-8')
        payload += struct.pack('<I', len(p)) + p
        result = mscs.loads(payload, strict=False)
        return result
    test_note("path with null byte", test_path_null,
              "INFO: null bytes in paths")

    # ═══════════════ 7. CRC INTEGRITY ═══════════════
    print("\n  -- 7. CRC integrity --")

    data_crc = mscs.dumps({'secret': 42}, with_crc=True)
    corrupted = bytearray(data_crc)
    corrupted[10] ^= 0xFF
    test_reject("corrupted byte with CRC", lambda: mscs.loads(bytes(corrupted)))

    # CRC truncated
    test_reject("truncated CRC (missing 2 bytes)", lambda: mscs.loads(data_crc[:-2]))

    # Without CRC flag — corruption not detected (by design)
    data_no_crc = mscs.dumps({'secret': 42}, with_crc=False)
    corrupted2 = bytearray(data_no_crc)
    if len(corrupted2) > 10:
        corrupted2[10] ^= 0xFF
    test_note("corrupted byte WITHOUT CRC", lambda: mscs.loads(bytes(corrupted2), strict=False),
              "BY DESIGN: no CRC = no corruption detection")

    # ═══════════════ 8. COMPRESSION ATTACKS ═══════════════
    print("\n  -- 8. Compression attacks --")

    test_reject("compressed: fake orig_size > MAX_SIZE",
                lambda: mscs.load_compressed(io.BytesIO(struct.pack('<I', MAX_SIZE + 1) + zlib.compress(b'x'))))

    # Actual decompression bomb: small compressed data that expands hugely
    # We can't actually create a 512MB+ bomb in a test, but we test the limit
    test_reject("compressed: orig_size too large",
                lambda: mscs.load_compressed(io.BytesIO(struct.pack('<I', 600_000_000) + zlib.compress(b'x'))))

    # ═══════════════ 9. INTEGER EDGE CASES ═══════════════
    print("\n  -- 9. Integer edge cases --")

    test_ok("int 0 roundtrip", lambda: assert_eq(mscs.loads(mscs.dumps(0), strict=False), 0))
    test_ok("int -1 roundtrip", lambda: assert_eq(mscs.loads(mscs.dumps(-1), strict=False), -1))
    test_ok("int 2^1000 roundtrip", lambda: assert_eq(mscs.loads(mscs.dumps(2**1000), strict=False), 2**1000))
    test_ok("int -(2^500) roundtrip", lambda: assert_eq(mscs.loads(mscs.dumps(-(2**500)), strict=False), -(2**500)))
    test_ok("int MAX_INT64 roundtrip", lambda: assert_eq(mscs.loads(mscs.dumps(2**63 - 1), strict=False), 2**63 - 1))

    # ═══════════════ 10. DATETIME EDGE CASES ═══════════════
    print("\n  -- 10. Datetime edge cases --")

    dt_bad = HEADER + _DATETIME
    dt_str = 'not-a-date'.encode('utf-8')
    dt_bad += struct.pack('<H', len(dt_str)) + dt_str
    test_reject("malformed datetime string", lambda: mscs.loads(dt_bad, strict=False))

    date_bad = HEADER + _DATE + struct.pack('<HBB', 2025, 13, 32)
    test_reject("invalid date month=13 day=32", lambda: mscs.loads(date_bad, strict=False))

    time_bad = HEADER + _TIME
    t_str = 'garbage'.encode('utf-8')
    time_bad += struct.pack('<H', len(t_str)) + t_str
    test_reject("malformed time string", lambda: mscs.loads(time_bad, strict=False))

    # ═══════════════ 11. DECIMAL EDGE CASES ═══════════════
    print("\n  -- 11. Decimal edge cases --")

    dec_bad = HEADER + _DECIMAL
    dec_str = 'not_a_number'.encode('utf-8')
    dec_bad += struct.pack('<H', len(dec_str)) + dec_str
    test_reject("malformed Decimal string", lambda: mscs.loads(dec_bad, strict=False))

    # ═══════════════ 12. REGISTRY ISOLATION ═══════════════
    print("\n  -- 12. Registry isolation --")

    def test_registry_no_pollution():
        count_before = len(_registry)
        try:
            mscs.loads(obj_bytes, strict=False)
        except Exception:
            pass
        assert len(_registry) == count_before, f"Registry grew! {count_before} -> {len(_registry)}"
    test_ok("registry not polluted by loads()", test_registry_no_pollution)

    # ═══════════════ 13. V1 BACKWARD COMPAT ═══════════════
    print("\n  -- 13. Backward compatibility --")

    # v1 payload: MSCS\x01 + INT(42)
    v1_data = b'MSCS\x01' + _INT + struct.pack('<H', 1) + b'\x2a'
    def test_v1():
        result = mscs.loads(v1_data, strict=False)
        assert result == 42
    test_ok("v1.0 payload loads (strict forced False)", test_v1)

    # ═══════════════ 14. CONCURRENT REGISTRY ACCESS ═══════════════
    print("\n  -- 14. Thread safety --")

    import threading

    def test_thread_registry():
        errors = []
        @mscs.register
        @dataclasses.dataclass
        class ThreadTest:
            n: int = 0

        def worker(i):
            try:
                obj = ThreadTest(n=i)
                data = mscs.dumps(obj)
                dec = mscs.loads(data)
                if dec.n != i:
                    errors.append(f"Worker {i}: expected n={i}, got n={dec.n}")
            except Exception as e:
                errors.append(f"Worker {i}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError(f"{len(errors)} thread errors: {errors[:3]}")
    test_ok("concurrent register+dumps+loads (20 threads)", test_thread_registry)

    # ═══════════════ 15. TIMEDELTA AMBIGUITY ═══════════════
    print("\n  -- 15. Timedelta v2.0/v2.1/v2.2 --")

    from datetime import timedelta

    # v2.2 uses _TIMEDELTA2 — unambiguous
    def test_td_v22():
        td = timedelta(days=5, seconds=3661, microseconds=123456)
        data = mscs.dumps(td)
        assert data[6:7] == _TIMEDELTA2, f"Expected TIMEDELTA2 tag, got {data[6]:02x}"
        dec = mscs.loads(data, strict=False)
        assert dec == td
    test_ok("timedelta v2.2 uses TIMEDELTA2 tag", test_td_v22)

    # Legacy TIMEDELTA tag should still decode
    legacy_td = HEADER + _TIMEDELTA + struct.pack('<iiI', 5, 3661, 123456)
    def test_td_legacy():
        dec = mscs.loads(legacy_td, strict=False)
        assert dec == timedelta(days=5, seconds=3661, microseconds=123456)
    test_ok("legacy TIMEDELTA tag still decodes", test_td_legacy)

    # ═══════════════ 16. DEQUE SECURITY (v2.4.0) ═══════════════
    print("\n  -- 16. Deque adversarial payloads --")

    def craft_deque(maxlen_raw, count, items=b''):
        p = HEADER + _DEQUE
        p += struct.pack('<i', maxlen_raw)
        p += struct.pack('<I', count)
        p += items
        return p

    test_reject("deque maxlen=-2", lambda: mscs.loads(craft_deque(-2, 0), strict=False))
    test_reject("deque maxlen=-1000", lambda: mscs.loads(craft_deque(-1000, 0), strict=False))
    test_reject("deque maxlen=INT32_MIN", lambda: mscs.loads(craft_deque(-(2**31), 0), strict=False))
    test_reject("deque maxlen=0, count=100 (CPU waste)", lambda: mscs.loads(craft_deque(0, 100, _NONE * 100), strict=False))
    test_reject("deque maxlen=2, count=100 (CPU waste)", lambda: mscs.loads(craft_deque(2, 100, _NONE * 100), strict=False))
    test_reject("deque count > MAX_COLLECTION", lambda: mscs.loads(craft_deque(-1, MAX_COLLECTION + 1), strict=False))
    test_reject("deque truncated (no maxlen)", lambda: mscs.loads(HEADER + _DEQUE + b'\x00\x00', strict=False))
    test_reject("deque truncated (no count)", lambda: mscs.loads(HEADER + _DEQUE + struct.pack('<i', -1), strict=False))
    test_reject("deque truncated items", lambda: mscs.loads(craft_deque(-1, 5), strict=False))

    def test_deque_valid():
        from collections import deque
        d = deque([1, 2, 3], maxlen=5)
        data = mscs.dumps(d)
        r = mscs.loads(data, strict=False)
        assert list(r) == [1, 2, 3] and r.maxlen == 5
    test_ok("deque valid roundtrip (maxlen=5, 3 items)", test_deque_valid)

    # ═══════════════ 17. DTYPE S/U/V BYPASS (v2.4.0 fix) ═══════════════
    print("\n  -- 17. Dtype S/U/V shorthand bypass --")

    test_ok("dtype U8 blocked (unicode)", lambda: assert_true(not _is_safe_dtype('U8')))
    test_ok("dtype U16 blocked", lambda: assert_true(not _is_safe_dtype('U16')))
    test_ok("dtype S8 blocked (string)", lambda: assert_true(not _is_safe_dtype('S8')))
    test_ok("dtype S16 blocked", lambda: assert_true(not _is_safe_dtype('S16')))
    test_ok("dtype V8 blocked (void)", lambda: assert_true(not _is_safe_dtype('V8')))
    test_ok("dtype <U8 blocked (with byteorder)", lambda: assert_true(not _is_safe_dtype('<U8')))
    test_ok("dtype >S16 blocked", lambda: assert_true(not _is_safe_dtype('>S16')))
    test_ok("dtype v8 blocked (void lowercase)", lambda: assert_true(not _is_safe_dtype('v8')))
    test_ok("dtype u8 safe (uint64, not unicode)", lambda: assert_true(_is_safe_dtype('u8')))
    test_ok("dtype u2 safe (uint16)", lambda: assert_true(_is_safe_dtype('u2')))
    test_ok("dtype uint8 safe (full name)", lambda: assert_true(_is_safe_dtype('uint8')))

    # ═══════════════ RESULTS ═══════════════
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    bugs = [r for r in results if r[0] == 'BUG']
    fails = [r for r in results if r[0] == 'FAIL']
    oks = [r for r in results if r[0] == 'OK']
    notes = [r for r in results if r[0] == 'NOTE']

    for status, name, detail in results:
        icons = {'OK': ' OK ', 'FAIL': 'FAIL', 'BUG': '*BUG', 'NOTE': 'NOTE'}
        print(f"  [{icons[status]}] {name:<50} {detail}")

    print(f"\n  Total: {len(results)} | OK: {len(oks)} | Notes: {len(notes)} | Fails: {len(fails)} | Bugs: {len(bugs)}")

    if bugs:
        print(f"\n  *** {len(bugs)} BUGS FOUND ***")
        for _, name, detail in bugs:
            print(f"    - {name}: {detail}")
        sys.exit(2)
    if fails:
        print(f"\n  *** {len(fails)} MISSING REJECTIONS ***")
        for _, name, detail in fails:
            print(f"    - {name}: {detail}")
        sys.exit(1)

    if notes:
        print(f"\n  Security notes ({len(notes)}):")
        for _, name, detail in notes:
            print(f"    - {name}: {detail}")

    print(f"\n  All {len(oks)} security checks passed.")
    print("=" * 70)


def assert_true(v):
    assert v, f"Expected True, got {v!r}"

def assert_eq(a, b):
    assert a == b, f"Expected {b!r}, got {a!r}"


if __name__ == '__main__':
    main()

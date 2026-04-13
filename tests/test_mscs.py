"""
MSCS — pytest test suite
=========================
Covers roundtrip correctness, security boundaries, edge cases, and limits.

Run: pytest tests/test_mscs.py -v
"""
import sys
import os
import struct
import io
import math
import zlib
import threading
import dataclasses
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from uuid import UUID
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import mscs
from mscs._core import (
    MAGIC, VERSION, MAX_DEPTH, MAX_SIZE, MAX_COLLECTION, MAX_STRING,
    MAX_COMPRESSED, _NONE, _BOOL, _INT, _FLOAT, _STR, _BYTES,
    _LIST, _TUPLE, _DICT, _SET, _NDARRAY, _OBJ, _COMPLEX,
    _FROZENSET, _DATETIME, _DATE, _TIME, _TIMEDELTA, _DECIMAL,
    _ENUM, _BYTEARRAY, _REF, _UUID, _PATH, _TENSOR, _TIMEDELTA2, _DEQUE,
    _is_safe_dtype, _registry,
)

HEADER = MAGIC + VERSION + b'\x00'

# Optional deps
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════
# ROUNDTRIP TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPrimitiveRoundtrip:
    def test_none(self):
        assert mscs.loads(mscs.dumps(None), strict=False) is None

    @pytest.mark.parametrize("val", [True, False])
    def test_bool(self, val):
        assert mscs.loads(mscs.dumps(val), strict=False) is val

    @pytest.mark.parametrize("val", [0, 1, -1, 42, -42, 2**63 - 1, -(2**63), 2**1000, -(2**500)])
    def test_int(self, val):
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    @pytest.mark.parametrize("val", [0.0, 1.5, -3.14, 1e308, -1e308, float('inf'), float('-inf')])
    def test_float(self, val):
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_float_nan(self):
        result = mscs.loads(mscs.dumps(float('nan')), strict=False)
        assert math.isnan(result)

    @pytest.mark.parametrize("val", [0+0j, 1+2j, -3.14+2.71j, complex(float('inf'), float('-inf'))])
    def test_complex(self, val):
        assert mscs.loads(mscs.dumps(val), strict=False) == val


class TestStringBytesRoundtrip:
    @pytest.mark.parametrize("val", ["", "hello", "emoji \U0001f389", "a" * 10000, "\x00\x01\x02"])
    def test_str(self, val):
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    @pytest.mark.parametrize("val", [b"", b"hello", bytes(range(256)), b"\x00" * 1000])
    def test_bytes(self, val):
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_bytearray(self):
        val = bytearray(b"hello world")
        result = mscs.loads(mscs.dumps(val), strict=False)
        assert result == val
        assert isinstance(result, bytearray)


class TestCollectionRoundtrip:
    def test_list(self):
        val = [1, "two", 3.0, None, True, [4, 5]]
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_empty_list(self):
        assert mscs.loads(mscs.dumps([]), strict=False) == []

    def test_tuple(self):
        val = (1, "two", (3, 4))
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_empty_tuple(self):
        assert mscs.loads(mscs.dumps(()), strict=False) == ()

    def test_dict(self):
        val = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_empty_dict(self):
        assert mscs.loads(mscs.dumps({}), strict=False) == {}

    def test_set(self):
        val = {1, 2, 3, "four"}
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_empty_set(self):
        assert mscs.loads(mscs.dumps(set()), strict=False) == set()

    def test_frozenset(self):
        val = frozenset([1, 2, 3])
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_nested_collections(self):
        val = {"list": [1, (2, 3)], "set": {4, 5}, "tuple": ({"a": 1},)}
        assert mscs.loads(mscs.dumps(val), strict=False) == val


class TestDatetimeRoundtrip:
    def test_datetime(self):
        val = datetime(2025, 6, 15, 10, 30, 45, 123456)
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_date(self):
        val = date(2025, 12, 31)
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_time(self):
        val = time(23, 59, 59, 999999)
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_timedelta(self):
        val = timedelta(days=5, seconds=3661, microseconds=123456)
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_timedelta_negative(self):
        val = timedelta(days=-3, seconds=100)
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_timedelta_zero(self):
        val = timedelta(0)
        assert mscs.loads(mscs.dumps(val), strict=False) == val


class TestSpecialTypesRoundtrip:
    def test_decimal(self):
        for s in ["0", "3.14159265358979323846", "-1e-100", "Infinity", "-Infinity"]:
            val = Decimal(s)
            assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_uuid(self):
        val = UUID("12345678-1234-5678-1234-567812345678")
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_path(self):
        val = Path("/tmp/test/file.txt")
        assert mscs.loads(mscs.dumps(val), strict=False) == val


class TestCircularReferences:
    def test_circular_list(self):
        lst = [1, 2]
        lst.append(lst)
        data = mscs.dumps(lst)
        result = mscs.loads(data, strict=False)
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] is result

    def test_circular_dict(self):
        d = {"a": 1}
        d["self"] = d
        data = mscs.dumps(d)
        result = mscs.loads(data, strict=False)
        assert result["a"] == 1
        assert result["self"] is result

    def test_shared_reference(self):
        shared = [1, 2, 3]
        val = {"a": shared, "b": shared}
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert result["a"] is result["b"]


# ═══════════════════════════════════════════════════════════════════
# NUMPY TESTS
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestNumpyRoundtrip:
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "bool"])
    def test_array_dtypes(self, dtype):
        arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = mscs.loads(mscs.dumps(arr), strict=False)
        assert np.array_equal(arr, result)
        assert arr.dtype == result.dtype

    def test_2d_array(self):
        arr = np.random.randn(10, 20).astype(np.float32)
        result = mscs.loads(mscs.dumps(arr), strict=False)
        assert np.array_equal(arr, result)
        assert arr.shape == result.shape

    def test_scalar_array(self):
        arr = np.array(3.14, dtype=np.float32)
        result = mscs.loads(mscs.dumps(arr), strict=False)
        assert np.array_equal(arr, result)

    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        result = mscs.loads(mscs.dumps(arr), strict=False)
        assert np.array_equal(arr, result)

    def test_large_array(self):
        arr = np.random.randn(100, 100).astype(np.float32)
        result = mscs.loads(mscs.dumps(arr), strict=False)
        assert np.array_equal(arr, result)


# ═══════════════════════════════════════════════════════════════════
# TORCH TESTS
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTorchRoundtrip:
    def test_tensor_1d(self):
        t = torch.randn(128)
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert torch.equal(t, result)

    def test_tensor_2d(self):
        t = torch.randn(64, 32)
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert torch.equal(t, result)

    def test_tensor_requires_grad(self):
        t = torch.randn(10, requires_grad=True)
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert torch.equal(t.detach(), result.detach())
        assert result.requires_grad is True

    def test_tensor_no_grad(self):
        t = torch.randn(10, requires_grad=False)
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert result.requires_grad is False

    def test_state_dict(self):
        sd = {
            "fc1.weight": torch.randn(128, 64),
            "fc1.bias": torch.randn(128),
        }
        result = mscs.loads(mscs.dumps(sd), strict=False)
        for k in sd:
            assert torch.equal(sd[k], result[k])

    def test_scalar_tensor(self):
        t = torch.tensor(3.14)
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert torch.equal(t, result)


# ═══════════════════════════════════════════════════════════════════
# CUSTOM CLASS TESTS
# ═══════════════════════════════════════════════════════════════════

class TestCustomClasses:
    def test_dataclass_roundtrip(self):
        @mscs.register
        @dataclasses.dataclass
        class Point:
            x: float = 0.0
            y: float = 0.0

        p = Point(1.5, 2.5)
        result = mscs.loads(mscs.dumps(p))
        assert result.x == 1.5
        assert result.y == 2.5

    def test_slots_roundtrip(self):
        @mscs.register
        class SlottedObj:
            __slots__ = ('a', 'b')
            def __init__(self, a=0, b=0):
                self.a = a
                self.b = b

        obj = SlottedObj(10, 20)
        result = mscs.loads(mscs.dumps(obj))
        assert result.a == 10
        assert result.b == 20

    def test_getstate_setstate(self):
        @mscs.register
        class CustomState:
            def __init__(self):
                self.data = None
            def __getstate__(self):
                return {"data": self.data}
            def __setstate__(self, state):
                self.data = state["data"]

        obj = CustomState()
        obj.data = {"key": "value"}
        result = mscs.loads(mscs.dumps(obj))
        assert result.data == {"key": "value"}

    def test_unregistered_strict_raises(self):
        payload = io.BytesIO()
        payload.write(HEADER + _OBJ)
        cls_s = b'fake.module.UnknownClass'
        payload.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
        payload.write(_DICT + struct.pack('<I', 0))

        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(payload.getvalue(), strict=True)

    def test_unregistered_non_strict_fallback(self):
        payload = io.BytesIO()
        payload.write(HEADER + _OBJ)
        cls_s = b'fake.module.UnknownClass'
        payload.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
        payload.write(_DICT + struct.pack('<I', 0))

        result = mscs.loads(payload.getvalue(), strict=False)
        assert isinstance(result, dict)
        assert result["__class__"] == "fake.module.UnknownClass"

    def test_register_alias(self):
        @mscs.register
        @dataclasses.dataclass
        class NewName:
            value: int = 0

        mscs.register_alias("old.module.OldName", NewName)

        payload = io.BytesIO()
        payload.write(HEADER + _OBJ)
        cls_s = b'old.module.OldName'
        payload.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
        state = mscs.dumps({"value": 42})[6:]  # strip header
        payload.write(state)

        result = mscs.loads(payload.getvalue())
        assert isinstance(result, NewName)
        assert result.value == 42


# ═══════════════════════════════════════════════════════════════════
# SECURITY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestMalformedPayloads:
    def test_empty_bytes(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(b'')

    def test_too_short(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(b'MSC')

    def test_wrong_magic(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(b'EVIL\x02\x00\x00')

    def test_future_version(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(b'MSCS\xFF\x00\x00')

    def test_unknown_tag(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + b'\xFF')

    def test_truncated_int(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _INT + b'\x04\x00')

    def test_truncated_str(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _STR + struct.pack('<I', 100))

    def test_truncated_list(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _LIST + struct.pack('<I', 5))

    def test_truncated_dict(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _DICT + struct.pack('<I', 1))

    def test_just_header(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER)


class TestResourceExhaustion:
    def test_list_exceeds_max_collection(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _LIST + struct.pack('<I', MAX_COLLECTION + 1))

    def test_dict_exceeds_max_collection(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _DICT + struct.pack('<I', MAX_COLLECTION + 1))

    def test_set_exceeds_max_collection(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _SET + struct.pack('<I', MAX_COLLECTION + 1))

    def test_str_exceeds_max_string(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _STR + struct.pack('<I', MAX_STRING + 1))

    def test_bytes_exceeds_max_string(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _BYTES + struct.pack('<I', MAX_STRING + 1))

    def test_depth_bomb(self):
        payload = HEADER
        for _ in range(300):
            payload += _LIST + struct.pack('<I', 1)
        payload += _NONE
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)


class TestReferenceAttacks:
    def test_forward_ref(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _REF + struct.pack('<I', 999))

    def test_ref_empty(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _REF + struct.pack('<I', 0))

    def test_ref_max_id(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(HEADER + _REF + struct.pack('<I', 0xFFFFFFFF))


class TestDtypeSecurity:
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "bool", "<f4", ">i8"])
    def test_safe_dtypes_accepted(self, dtype):
        assert _is_safe_dtype(dtype) is True

    @pytest.mark.parametrize("dtype", ["object", "O", "void", "V", "V8"])
    def test_dangerous_dtypes_blocked(self, dtype):
        assert _is_safe_dtype(dtype) is False

    @pytest.mark.parametrize("dtype", [
        "U8", "U16", "U32",        # unicode string shorthand (UPPERCASE)
        "S8", "S16", "S32",        # byte string shorthand (UPPERCASE)
        "V8", "V16",               # void shorthand (UPPERCASE)
        "<U8", ">S16", "|V32",     # with byteorder prefix
        "v8", "v16",               # lowercase void shorthand
    ])
    def test_string_unicode_void_shorthand_blocked(self, dtype):
        """SEC-03: S<n>/U<n>/V<n> shorthand dtypes must be blocked."""
        assert _is_safe_dtype(dtype) is False

    @pytest.mark.parametrize("dtype", ["u1", "u2", "u4", "u8"])
    def test_unsigned_int_shorthand_accepted(self, dtype):
        """Ensure lowercase u<n> (uint) is NOT blocked by S/U/V filter."""
        assert _is_safe_dtype(dtype) is True


class TestPathSecurity:
    def test_null_byte_rejected(self):
        payload = HEADER + _PATH
        p = '/tmp/evil\x00hidden'.encode('utf-8')
        payload += struct.pack('<I', len(p)) + p
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(payload, strict=False)

    def test_traversal_deserializes(self):
        """Path traversal strings are deserialized — consumer must validate."""
        val = Path("../../../../etc/passwd")
        result = mscs.loads(mscs.dumps(val), strict=False)
        assert str(result) == str(val)


class TestEnumSecurity:
    def test_unregistered_enum_strict(self):
        payload = io.BytesIO()
        payload.write(HEADER + _ENUM)
        enum_s = b'evil.module.BadEnum'
        payload.write(_STR + struct.pack('<I', len(enum_s)) + enum_s)
        payload.write(_INT + struct.pack('<H', 1) + b'\x01')
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(payload.getvalue(), strict=True)


# ═══════════════════════════════════════════════════════════════════
# CRC INTEGRITY
# ═══════════════════════════════════════════════════════════════════

class TestCRC:
    def test_crc_valid(self):
        data = mscs.dumps({"key": 42}, with_crc=True)
        assert mscs.loads(data) == {"key": 42}

    def test_crc_corrupted(self):
        data = mscs.dumps({"key": 42}, with_crc=True)
        corrupted = bytearray(data)
        corrupted[10] ^= 0xFF
        with pytest.raises(mscs.MSCDecodeError, match="CRC32"):
            mscs.loads(bytes(corrupted))

    def test_crc_truncated(self):
        data = mscs.dumps({"key": 42}, with_crc=True)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(data[:-2])


# ═══════════════════════════════════════════════════════════════════
# COMPRESSION
# ═══════════════════════════════════════════════════════════════════

class TestCompression:
    def test_roundtrip(self):
        val = {"data": list(range(1000))}
        buf = io.BytesIO()
        mscs.dump_compressed(val, buf)
        buf.seek(0)
        result = mscs.load_compressed(buf, strict=False)
        assert result == val

    def test_orig_size_exceeds_limit(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(
                io.BytesIO(struct.pack('<I', MAX_SIZE + 1) + zlib.compress(b'x'))
            )

    def test_compressed_size_limit(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(
                io.BytesIO(struct.pack('<I', 600_000_000) + zlib.compress(b'x'))
            )


# ═══════════════════════════════════════════════════════════════════
# DATETIME EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestDatetimeEdgeCases:
    def test_malformed_datetime(self):
        payload = HEADER + _DATETIME
        s = b'not-a-date'
        payload += struct.pack('<H', len(s)) + s
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_invalid_date(self):
        payload = HEADER + _DATE + struct.pack('<HBB', 2025, 13, 32)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_malformed_time(self):
        payload = HEADER + _TIME
        s = b'garbage'
        payload += struct.pack('<H', len(s)) + s
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_malformed_decimal(self):
        payload = HEADER + _DECIMAL
        s = b'not_a_number'
        payload += struct.pack('<H', len(s)) + s
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_v1_payload(self):
        v1_data = b'MSCS\x01' + _INT + struct.pack('<H', 1) + b'\x2a'
        result = mscs.loads(v1_data, strict=False)
        assert result == 42

    def test_timedelta_v22_uses_new_tag(self):
        td = timedelta(days=5, seconds=3661, microseconds=123456)
        data = mscs.dumps(td)
        assert data[6:7] == _TIMEDELTA2

    def test_legacy_timedelta_tag_decodes(self):
        payload = HEADER + _TIMEDELTA + struct.pack('<iiI', 5, 3661, 123456)
        result = mscs.loads(payload, strict=False)
        assert result == timedelta(days=5, seconds=3661, microseconds=123456)


# ═══════════════════════════════════════════════════════════════════
# REGISTRY ISOLATION
# ═══════════════════════════════════════════════════════════════════

class TestRegistryIsolation:
    def test_loads_does_not_pollute_registry(self):
        payload = io.BytesIO()
        payload.write(HEADER + _OBJ)
        cls_s = b'ghost.module.GhostClass'
        payload.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
        payload.write(_DICT + struct.pack('<I', 0))

        count_before = len(_registry)
        mscs.loads(payload.getvalue(), strict=False)
        assert len(_registry) == count_before


# ═══════════════════════════════════════════════════════════════════
# THREAD SAFETY
# ═══════════════════════════════════════════════════════════════════

class TestThreadSafety:
    def test_concurrent_register_and_roundtrip(self):
        errors = []

        @mscs.register
        @dataclasses.dataclass
        class ThreadTestObj:
            n: int = 0

        def worker(i):
            try:
                obj = ThreadTestObj(n=i)
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

        assert not errors, f"Thread errors: {errors}"


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

class TestInspect:
    def test_inspect_dict(self):
        data = mscs.dumps({"a": 1})
        info = mscs.inspect(data)
        assert info["valid"] is True
        assert info["version"] == 2
        assert info["root_type"] == "dict"

    def test_inspect_invalid(self):
        info = mscs.inspect(b'NOPE')
        assert info["valid"] is False

    def test_inspect_crc_flag(self):
        data = mscs.dumps(42, with_crc=True)
        info = mscs.inspect(data)
        assert info["has_crc"] is True


class TestCopy:
    def test_copy_dict(self):
        val = {"a": [1, 2, 3], "b": "hello"}
        result = mscs.copy(val)
        assert result == val
        assert result is not val
        assert result["a"] is not val["a"]


class TestBenchmarkUtil:
    def test_benchmark_returns_dict(self):
        result = mscs.benchmark([1, 2, 3], rounds=10)
        assert "encode_ms" in result
        assert "decode_ms" in result
        assert "raw_bytes" in result
        assert "compressed_bytes" in result
        assert result["rounds"] == 10


# ═══════════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════════

class TestFileIO:
    def test_dump_load(self, tmp_path):
        val = {"model": "test", "params": [1, 2, 3]}
        filepath = tmp_path / "test.mscs"
        with open(filepath, "wb") as f:
            mscs.dump(val, f)
        with open(filepath, "rb") as f:
            result = mscs.load(f, strict=False)
        assert result == val

    def test_dump_load_compressed(self, tmp_path):
        val = {"data": list(range(1000))}
        filepath = tmp_path / "test.mscs.z"
        with open(filepath, "wb") as f:
            mscs.dump_compressed(val, f)
        with open(filepath, "rb") as f:
            result = mscs.load_compressed(f, strict=False)
        assert result == val

    def test_dump_load_with_hmac(self, tmp_path):
        key = b'secret-key-for-test'
        val = {"secure": True, "data": [1, 2, 3]}
        filepath = tmp_path / "test.mscs"
        with open(filepath, "wb") as f:
            mscs.dump(val, f, hmac_key=key)
        with open(filepath, "rb") as f:
            result = mscs.load(f, strict=False, hmac_key=key)
        assert result == val


# ═══════════════════════════════════════════════════════════════════
# HMAC AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════

class TestHMAC:
    KEY = b'test-hmac-key-256bit-long-enough!'

    def test_hmac_roundtrip(self):
        val = {"secret": 42, "nested": [1, 2, 3]}
        data = mscs.dumps(val, hmac_key=self.KEY)
        result = mscs.loads(data, strict=False, hmac_key=self.KEY)
        assert result == val

    def test_hmac_flag_set(self):
        data = mscs.dumps(42, hmac_key=self.KEY)
        assert data[5] & 0x02  # flag bit 1

    def test_hmac_rejects_tampered_payload(self):
        data = mscs.dumps(42, hmac_key=self.KEY)
        tampered = bytearray(data)
        tampered[7] ^= 0xFF
        with pytest.raises(mscs.MSCSecurityError, match="HMAC"):
            mscs.loads(bytes(tampered), strict=False, hmac_key=self.KEY)

    def test_hmac_rejects_wrong_key(self):
        data = mscs.dumps(42, hmac_key=self.KEY)
        with pytest.raises(mscs.MSCSecurityError, match="HMAC"):
            mscs.loads(data, strict=False, hmac_key=b'wrong-key-different')

    def test_hmac_rejects_missing_key(self):
        data = mscs.dumps(42, hmac_key=self.KEY)
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(data, strict=False)

    def test_hmac_downgrade_attack(self):
        """Providing key for unsigned payload = rejected."""
        data = mscs.dumps(42)  # no hmac
        with pytest.raises(mscs.MSCSecurityError, match="downgrade"):
            mscs.loads(data, strict=False, hmac_key=self.KEY)

    def test_hmac_and_crc_mutually_exclusive(self):
        with pytest.raises(mscs.MSCEncodeError):
            mscs.dumps(42, with_crc=True, hmac_key=self.KEY)

    def test_hmac_truncated(self):
        data = mscs.dumps(42, hmac_key=self.KEY)
        with pytest.raises((mscs.MSCDecodeError, mscs.MSCSecurityError)):
            mscs.loads(data[:-10], strict=False, hmac_key=self.KEY)


# ═══════════════════════════════════════════════════════════════════
# MAX_INT_BYTES LIMIT
# ═══════════════════════════════════════════════════════════════════

class TestMaxIntBytes:
    def test_normal_large_int_accepted(self):
        """2^1000 is ~126 bytes, well under 8192 limit."""
        val = 2 ** 1000
        assert mscs.loads(mscs.dumps(val), strict=False) == val

    def test_huge_int_encode_rejected(self):
        """Int exceeding MAX_INT_BYTES should be rejected on encode."""
        from mscs._core import MAX_INT_BYTES
        huge = 2 ** (MAX_INT_BYTES * 8 + 8)
        with pytest.raises(mscs.MSCEncodeError, match="Entero demasiado grande"):
            mscs.dumps(huge)

    def test_huge_int_decode_rejected(self):
        """Crafted payload with oversized int should be rejected on decode."""
        from mscs._core import MAX_INT_BYTES
        n_bytes = MAX_INT_BYTES + 1
        payload = HEADER + _INT + struct.pack('<H', n_bytes) + b'\x01' * n_bytes
        with pytest.raises(mscs.MSCDecodeError, match="Entero demasiado grande"):
            mscs.loads(payload, strict=False)

    def test_max_boundary_accepted(self):
        """Int at exactly MAX_INT_BYTES should work."""
        from mscs._core import MAX_INT_BYTES
        val = 2 ** (MAX_INT_BYTES * 8 - 9)  # fits in MAX_INT_BYTES
        data = mscs.dumps(val)
        assert mscs.loads(data, strict=False) == val


# ═══════════════════════════════════════════════════════════════════
# TRAILING BYTES VALIDATION
# ═══════════════════════════════════════════════════════════════════

class TestTrailingBytes:
    def test_trailing_bytes_rejected(self):
        data = mscs.dumps(42)
        tampered = data + b'\x00\x01\x02'
        with pytest.raises(mscs.MSCDecodeError, match="Trailing bytes"):
            mscs.loads(tampered, strict=False)

    def test_clean_payload_accepted(self):
        data = mscs.dumps({"a": [1, 2, 3]})
        assert mscs.loads(data, strict=False) == {"a": [1, 2, 3]}

    def test_trailing_bytes_with_crc(self):
        """CRC payload + extra bytes: CRC mismatch or trailing detected."""
        data = mscs.dumps(42, with_crc=True)
        tampered = data + b'\xFF'
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(tampered, strict=False)


# ═══════════════════════════════════════════════════════════════════
# BUG 1 — __getstate__/__setstate__ ON DATACLASS
# ═══════════════════════════════════════════════════════════════════

class TestDataclassGetstate:
    def test_dataclass_with_getstate_roundtrip(self):
        """Dataclass with __getstate__/__setstate__ must use them over field walking."""
        from collections import deque

        @mscs.register
        @dataclasses.dataclass
        class FooDC:
            x: int = 1
            q: object = dataclasses.field(default_factory=lambda: deque([1, 2, 3]))

            def __getstate__(self):
                return {'x': self.x, 'q': list(self.q)}

            def __setstate__(self, s):
                object.__setattr__(self, 'x', s['x'])
                object.__setattr__(self, 'q', deque(s['q']))

        f = FooDC()
        data = mscs.dumps(f)
        result = mscs.loads(data)
        assert result.x == 1
        assert list(result.q) == [1, 2, 3]
        assert isinstance(result.q, deque)

    def test_dataclass_without_getstate_still_works(self):
        """Normal dataclass (no __getstate__) should still use field walking."""
        @mscs.register
        @dataclasses.dataclass
        class BarDC:
            a: int = 10
            b: str = "hello"

        obj = BarDC(42, "world")
        data = mscs.dumps(obj)
        result = mscs.loads(data)
        assert result.a == 42
        assert result.b == "world"

    def test_dataclass_getstate_transforms_state(self):
        """__getstate__ that transforms data must be respected."""
        @mscs.register
        @dataclasses.dataclass
        class TransformDC:
            values: list = dataclasses.field(default_factory=lambda: [1, 2, 3])

            def __getstate__(self):
                return {'values': [v * 10 for v in self.values]}

            def __setstate__(self, s):
                object.__setattr__(self, 'values', [v // 10 for v in s['values']])

        obj = TransformDC([5, 6, 7])
        data = mscs.dumps(obj)
        result = mscs.loads(data)
        assert result.values == [5, 6, 7]


# ═══════════════════════════════════════════════════════════════════
# BUG 2 — DEQUE NATIVE SUPPORT
# ═══════════════════════════════════════════════════════════════════

class TestDequeRoundtrip:
    def test_deque_basic(self):
        from collections import deque
        val = deque([1, 2, 3])
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert isinstance(result, deque)
        assert list(result) == [1, 2, 3]
        assert result.maxlen is None

    def test_deque_with_maxlen(self):
        from collections import deque
        val = deque([1, 2, 3], maxlen=5)
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert isinstance(result, deque)
        assert list(result) == [1, 2, 3]
        assert result.maxlen == 5

    def test_deque_empty(self):
        from collections import deque
        val = deque()
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert isinstance(result, deque)
        assert len(result) == 0
        assert result.maxlen is None

    def test_deque_empty_with_maxlen(self):
        from collections import deque
        val = deque(maxlen=10)
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert isinstance(result, deque)
        assert len(result) == 0
        assert result.maxlen == 10

    def test_deque_nested(self):
        from collections import deque
        val = {"history": deque([1, 2, 3], maxlen=100), "data": [4, 5]}
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert isinstance(result["history"], deque)
        assert list(result["history"]) == [1, 2, 3]
        assert result["history"].maxlen == 100
        assert result["data"] == [4, 5]

    def test_deque_mixed_types(self):
        from collections import deque
        val = deque(["hello", 42, 3.14, None, True])
        data = mscs.dumps(val)
        result = mscs.loads(data, strict=False)
        assert list(result) == ["hello", 42, 3.14, None, True]

    def test_deque_circular_ref(self):
        from collections import deque
        d = deque([1, 2])
        d.append(d)
        data = mscs.dumps(d)
        result = mscs.loads(data, strict=False)
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] is result


# ═══════════════════════════════════════════════════════════════════
# SECURITY: DEQUE ADVERSARIAL PAYLOADS
# ═══════════════════════════════════════════════════════════════════

class TestDequeSecurity:
    def _craft_deque_payload(self, maxlen_raw, count, items_data=b''):
        """Build a raw deque payload with arbitrary maxlen and count."""
        payload = HEADER + _DEQUE
        payload += struct.pack('<i', maxlen_raw)
        payload += struct.pack('<I', count)
        payload += items_data
        return payload

    def test_negative_maxlen_rejected(self):
        """SEC-01: maxlen < -1 must be rejected."""
        payload = self._craft_deque_payload(-2, 0)
        with pytest.raises(mscs.MSCDecodeError, match="maxlen"):
            mscs.loads(payload, strict=False)

    def test_very_negative_maxlen_rejected(self):
        """SEC-01: maxlen = -1000 must be rejected."""
        payload = self._craft_deque_payload(-1000, 0)
        with pytest.raises(mscs.MSCDecodeError, match="maxlen"):
            mscs.loads(payload, strict=False)

    def test_min_int32_maxlen_rejected(self):
        """SEC-01: maxlen = INT32_MIN must be rejected."""
        payload = self._craft_deque_payload(-(2**31), 0)
        with pytest.raises(mscs.MSCDecodeError, match="maxlen"):
            mscs.loads(payload, strict=False)

    def test_maxlen0_with_items_rejected(self):
        """SEC-02: maxlen=0 with count>0 must be rejected (CPU waste DoS)."""
        # Craft: maxlen=0, count=100, 100 None items
        items = _NONE * 100
        payload = self._craft_deque_payload(0, 100, items)
        with pytest.raises(mscs.MSCDecodeError, match="excede maxlen"):
            mscs.loads(payload, strict=False)

    def test_maxlen_less_than_count_rejected(self):
        """SEC-02: count > maxlen must be rejected (CPU waste DoS)."""
        # maxlen=2, count=100 — would decode 100 items keeping only 2
        items = _NONE * 100
        payload = self._craft_deque_payload(2, 100, items)
        with pytest.raises(mscs.MSCDecodeError, match="excede maxlen"):
            mscs.loads(payload, strict=False)

    def test_maxlen_equals_count_accepted(self):
        """maxlen == count is valid and must work."""
        from collections import deque
        d = deque([1, 2, 3], maxlen=3)
        data = mscs.dumps(d)
        result = mscs.loads(data, strict=False)
        assert list(result) == [1, 2, 3]
        assert result.maxlen == 3

    def test_maxlen_none_unlimited_accepted(self):
        """maxlen=-1 (None/unlimited) with any count must work."""
        from collections import deque
        d = deque(range(100))
        data = mscs.dumps(d)
        result = mscs.loads(data, strict=False)
        assert len(result) == 100
        assert result.maxlen is None

    def test_deque_count_exceeds_max_collection(self):
        """Deque count > MAX_COLLECTION must be rejected."""
        from mscs._core import MAX_COLLECTION
        payload = self._craft_deque_payload(-1, MAX_COLLECTION + 1)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_deque_truncated_maxlen(self):
        """Truncated deque (missing maxlen bytes) must error."""
        payload = HEADER + _DEQUE + b'\x00\x00'  # only 2 bytes, need 4
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_deque_truncated_count(self):
        """Truncated deque (has maxlen but no count) must error."""
        payload = HEADER + _DEQUE + struct.pack('<i', -1)  # maxlen only
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_deque_truncated_items(self):
        """Deque that claims 5 items but has none must error."""
        payload = self._craft_deque_payload(-1, 5)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

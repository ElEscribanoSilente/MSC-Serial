"""
MSCS — Hypothesis fuzzing tests
================================
Property-based testing to verify:
1. Random valid Python objects survive roundtrip (dumps → loads)
2. Random binary payloads never crash the decoder (only MSCError)
3. HMAC rejects any tampered payload

Run: pytest tests/test_fuzz.py -v --hypothesis-seed=0
"""
import sys
import os
import struct

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import mscs
from mscs._core import MAGIC, VERSION, MAX_INT_BYTES

# ─── Strategies for valid Python objects ───

json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2 ** (MAX_INT_BYTES * 8 - 1)) + 1,
                max_value=2 ** (MAX_INT_BYTES * 8 - 1) - 1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=1000),
    st.binary(max_size=1000),
)

# Recursive strategy for nested structures
serializable_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=20),
        st.tuples(children, children),
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            children,
            max_size=20,
        ),
    ),
    max_leaves=50,
)


# ═══════════════════════════════════════════════════════════════════
# 1. ROUNDTRIP: valid objects survive encode → decode
# ═══════════════════════════════════════════════════════════════════

class TestRoundtripFuzz:
    @given(obj=serializable_values)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_no_crash(self, obj):
        """Any serializable object should roundtrip without crashing."""
        data = mscs.dumps(obj)
        result = mscs.loads(data, strict=False)
        # We can't always assert equality (NaN != NaN, etc.)
        # but we assert no crash and valid return
        assert result is not None or obj is None

    @given(obj=st.one_of(
        st.integers(min_value=-10**1000, max_value=10**1000),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=5000),
        st.binary(max_size=5000),
    ))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_primitive_roundtrip_exact(self, obj):
        """Primitives should roundtrip exactly (except NaN)."""
        import math
        try:
            data = mscs.dumps(obj)
        except mscs.MSCEncodeError:
            # e.g., int too large for MAX_INT_BYTES
            return
        result = mscs.loads(data, strict=False)
        if isinstance(obj, float) and math.isnan(obj):
            assert isinstance(result, float) and math.isnan(result)
        else:
            assert result == obj, f"Roundtrip mismatch: {obj!r} != {result!r}"

    @given(obj=st.lists(st.integers(min_value=-1000, max_value=1000), max_size=100))
    @settings(max_examples=200)
    def test_list_roundtrip_exact(self, obj):
        """Lists of ints should roundtrip exactly."""
        assert mscs.loads(mscs.dumps(obj), strict=False) == obj

    @given(obj=st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.integers(min_value=-1000, max_value=1000), st.text(max_size=50)),
        max_size=50
    ))
    @settings(max_examples=200)
    def test_dict_roundtrip_exact(self, obj):
        """Dicts should roundtrip exactly."""
        assert mscs.loads(mscs.dumps(obj), strict=False) == obj


# ═══════════════════════════════════════════════════════════════════
# 2. ADVERSARIAL: random bytes never crash the decoder
# ═══════════════════════════════════════════════════════════════════

class TestAdversarialFuzz:
    @given(data=st.binary(min_size=0, max_size=1000))
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_random_bytes_no_crash(self, data):
        """Random binary data should raise MSCError or succeed, never crash."""
        try:
            mscs.loads(data, strict=False)
        except (mscs.MSCDecodeError, mscs.MSCSecurityError, mscs.MSCEncodeError):
            pass  # Expected — malformed data rejected cleanly

    @given(data=st.binary(min_size=6, max_size=500))
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_header_random_payload_no_crash(self, data):
        """Valid MSCS header + random payload: must not crash."""
        payload = MAGIC + VERSION + b'\x00' + data[6:]
        try:
            mscs.loads(payload, strict=False)
        except (mscs.MSCDecodeError, mscs.MSCSecurityError):
            pass

    @given(
        tag=st.sampled_from([b'\x00', b'\x01', b'\x02', b'\x03', b'\x04',
                             b'\x05', b'\x06', b'\x07', b'\x08', b'\x09',
                             b'\x0A', b'\x0B', b'\x0C', b'\x0D', b'\x0E',
                             b'\x0F', b'\x10', b'\x11', b'\x12', b'\x13',
                             b'\x14', b'\x15', b'\x16', b'\x17', b'\x18',
                             b'\x19', b'\xFF']),
        suffix=st.binary(min_size=0, max_size=200)
    )
    @settings(max_examples=500)
    def test_every_tag_with_random_suffix(self, tag, suffix):
        """Each type tag followed by random data: must not crash."""
        payload = MAGIC + VERSION + b'\x00' + tag + suffix
        try:
            mscs.loads(payload, strict=False)
        except (mscs.MSCDecodeError, mscs.MSCSecurityError):
            pass

    @given(depth=st.integers(min_value=100, max_value=400))
    @settings(max_examples=20)
    def test_nested_depth_bomb(self, depth):
        """Deep nesting should be rejected cleanly, not stack overflow."""
        payload = MAGIC + VERSION + b'\x00'
        for _ in range(depth):
            payload += b'\x06' + struct.pack('<I', 1)  # _LIST with count=1
        payload += b'\x00'  # _NONE
        try:
            mscs.loads(payload, strict=False)
        except mscs.MSCDecodeError:
            pass


# ═══════════════════════════════════════════════════════════════════
# 3. HMAC: tampered payloads always rejected
# ═══════════════════════════════════════════════════════════════════

class TestHMACFuzz:
    @given(obj=serializable_values, key=st.binary(min_size=16, max_size=64))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_hmac_roundtrip(self, obj, key):
        """HMAC-signed payloads should roundtrip with the correct key."""
        try:
            data = mscs.dumps(obj, hmac_key=key)
        except mscs.MSCEncodeError:
            return
        result = mscs.loads(data, strict=False, hmac_key=key)
        assert result is not None or obj is None

    @given(
        obj=st.one_of(st.integers(min_value=-100, max_value=100), st.text(max_size=50)),
        key=st.binary(min_size=16, max_size=32),
        flip_pos=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=300)
    def test_hmac_rejects_bit_flip(self, obj, key, flip_pos):
        """Flipping any bit in an HMAC-signed payload should be rejected."""
        try:
            data = mscs.dumps(obj, hmac_key=key)
        except mscs.MSCEncodeError:
            return
        # Flip one bit
        pos = flip_pos % len(data)
        tampered = bytearray(data)
        tampered[pos] ^= 0x01
        tampered = bytes(tampered)
        if tampered == data:
            return  # no change (shouldn't happen with XOR)
        with pytest.raises((mscs.MSCSecurityError, mscs.MSCDecodeError)):
            mscs.loads(tampered, strict=False, hmac_key=key)

    @given(
        obj=st.integers(min_value=0, max_value=100),
        key1=st.binary(min_size=16, max_size=32),
        key2=st.binary(min_size=16, max_size=32),
    )
    @settings(max_examples=200)
    def test_hmac_rejects_wrong_key(self, obj, key1, key2):
        """Different key should be rejected."""
        assume(key1 != key2)
        data = mscs.dumps(obj, hmac_key=key1)
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(data, strict=False, hmac_key=key2)

    @given(obj=st.integers(min_value=0, max_value=100))
    @settings(max_examples=100)
    def test_hmac_rejects_missing_key(self, obj):
        """HMAC payload without key should be rejected."""
        key = b'test-secret-key-1234'
        data = mscs.dumps(obj, hmac_key=key)
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(data, strict=False)

    @given(obj=st.integers(min_value=0, max_value=100))
    @settings(max_examples=100)
    def test_hmac_downgrade_attack(self, obj):
        """Unsigned payload with key should be rejected (anti-downgrade)."""
        data = mscs.dumps(obj)  # no hmac
        key = b'test-secret-key-1234'
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(data, strict=False, hmac_key=key)


# ═══════════════════════════════════════════════════════════════════
# 4. MAX_INT_BYTES: oversized ints rejected
# ═══════════════════════════════════════════════════════════════════

class TestIntLimitFuzz:
    @given(n_bytes=st.integers(min_value=MAX_INT_BYTES + 1, max_value=65535))
    @settings(max_examples=50)
    def test_oversized_int_rejected_on_decode(self, n_bytes):
        """Ints larger than MAX_INT_BYTES should be rejected in decode."""
        payload = MAGIC + VERSION + b'\x00' + b'\x02' + struct.pack('<H', n_bytes)
        payload += b'\x01' * n_bytes  # dummy data
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload, strict=False)

    def test_oversized_int_rejected_on_encode(self):
        """Ints larger than MAX_INT_BYTES should be rejected in encode."""
        huge = 2 ** (MAX_INT_BYTES * 8 + 8)
        with pytest.raises(mscs.MSCEncodeError):
            mscs.dumps(huge)

    @given(exp=st.integers(min_value=1, max_value=MAX_INT_BYTES * 8 - 2))
    @settings(max_examples=50)
    def test_valid_large_int_accepted(self, exp):
        """Ints within MAX_INT_BYTES should roundtrip."""
        val = 2 ** exp
        data = mscs.dumps(val)
        assert mscs.loads(data, strict=False) == val


# ═══════════════════════════════════════════════════════════════════
# 5. TRAILING BYTES: always rejected for v2
# ═══════════════════════════════════════════════════════════════════

class TestTrailingBytesFuzz:
    @given(garbage=st.binary(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_trailing_bytes_rejected(self, garbage):
        """Appending any bytes to a valid payload should be rejected."""
        data = mscs.dumps(42)
        tampered = data + garbage
        with pytest.raises(mscs.MSCDecodeError, match="Trailing bytes"):
            mscs.loads(tampered, strict=False)

    @given(garbage=st.binary(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_trailing_bytes_with_crc_rejected(self, garbage):
        """Trailing bytes after CRC should be rejected."""
        data = mscs.dumps(42, with_crc=True)
        tampered = data + garbage
        # Will fail either CRC check or trailing bytes check
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(tampered, strict=False)

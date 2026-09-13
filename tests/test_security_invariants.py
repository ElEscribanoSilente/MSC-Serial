"""Authentication, dtype and object reconstruction security invariants.

Collected by pytest as part of the public regression suite.
"""
import builtins
import io
import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mscs
from mscs import _core


@pytest.fixture
def registry():
    previous = dict(_core._registry)
    yield
    _core._registry.clear()
    _core._registry.update(previous)


def pack_compressed(data):
    return struct.pack('<I', len(data)) + zlib.compress(data)


@pytest.mark.parametrize('strict', [True, False])
def test_every_signed_bit_is_authenticated_before_object_hooks(registry, strict):
    calls = []

    @mscs.register
    class Record:
        def __new__(cls):
            calls.append('new')
            return super().__new__(cls)

        def __setstate__(self, state):
            calls.append('setstate')
            self.__dict__.update(state)

    obj = Record()
    obj.value = 7
    key = b'local-audit-key-with-no-external-use'
    signed = mscs.dumps(obj, hmac_key=key)
    for offset in range(len(signed)):
        for bit in range(8):
            calls.clear()
            altered = bytearray(signed)
            altered[offset] ^= 1 << bit
            with pytest.raises(mscs.MSCError):
                mscs.loads(bytes(altered), hmac_key=key, strict=strict)
            assert calls == [], (offset, bit, calls)
    calls.clear()
    restored = mscs.loads(signed, hmac_key=key, strict=strict)
    assert restored.value == 7
    assert calls == ['new', 'setstate']


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('transport', ['loads', 'load', 'load_compressed'])
def test_authentication_rejects_all_unsigned_versions(registry, strict, transport):
    calls = []

    @mscs.register
    class Record:
        def __setstate__(self, state):
            calls.append('setstate')

    obj = Record()
    obj.value = 1
    unsigned = mscs.dumps(obj)
    signed = mscs.dumps(obj, hmac_key=b'local-key')
    variants = [
        unsigned,
        unsigned[:4] + b'\x01' + unsigned[6:],
        mscs.dumps(obj, with_crc=True),
        signed[:-32],
        signed[:-1],
        mscs.dumps(obj, hmac_key=b'another-key'),
    ]
    for data in variants:
        calls.clear()
        kwargs = dict(hmac_key=b'local-key', strict=strict)
        with pytest.raises(mscs.MSCError):
            if transport == 'loads':
                mscs.loads(data, **kwargs)
            elif transport == 'load':
                mscs.load(io.BytesIO(data), **kwargs)
            else:
                mscs.load_compressed(io.BytesIO(pack_compressed(data)), **kwargs)
        assert calls == []


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('version', [1, 2])
def test_unregistered_class_never_imported_or_instantiated(registry, monkeypatch, strict, version):
    calls = []

    class Unregistered:
        def __new__(cls):
            calls.append('new')
            return super().__new__(cls)

        def __setstate__(self, state):
            calls.append('setstate')

    Unregistered.__module__ = 'mscs_audit_nonexistent_module'
    obj = Unregistered()
    obj.value = 9
    data = mscs.dumps(obj)
    if version == 1:
        data = data[:4] + b'\x01' + data[6:]
    calls.clear()
    original_import = builtins.__import__

    def observe_import(name, *args, **kwargs):
        if name.startswith('mscs_audit_nonexistent_module'):
            calls.append('import')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', observe_import)
    if strict:
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(data, strict=True)
    else:
        result = mscs.loads(data, strict=False)
        assert result['__state__'] == {'value': 9}
    assert calls == []


@pytest.mark.parametrize('key', [b'', b'a', bytes(range(32))])
def test_signed_numpy_and_tensor_checkpoint(key):
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    tensor.requires_grad_(True)
    original = {'weights': tensor, 'same': tensor, 'array': np.arange(6, dtype='>i4')}
    restored = mscs.loads(mscs.dumps(original, hmac_key=key), hmac_key=key)
    assert torch.equal(restored['weights'], tensor)
    assert restored['weights'].requires_grad
    assert restored['weights'] is restored['same']
    assert np.array_equal(original['array'], restored['array'])
    assert restored['array'].dtype == original['array'].dtype


@pytest.mark.parametrize('dtype_name', [
    'bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64', 'complex64', 'complex128',
])
def test_supported_tensor_dtypes(dtype_name):
    torch = pytest.importorskip('torch')
    dtype = getattr(torch, dtype_name)
    original = torch.tensor([[0, 1], [1, 0]], dtype=dtype).t()
    restored = mscs.loads(mscs.dumps(original))
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert torch.equal(restored, original)


def test_bfloat16_checkpoint_roundtrip():
    torch = pytest.importorskip('torch')
    original = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    restored = mscs.loads(mscs.dumps(original))
    assert restored.dtype == original.dtype
    assert torch.equal(restored, original)


def test_conjugate_tensor_roundtrip():
    torch = pytest.importorskip('torch')
    original = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64).conj()
    restored = mscs.loads(mscs.dumps(original))
    assert restored.dtype == original.dtype
    assert torch.equal(restored, original)

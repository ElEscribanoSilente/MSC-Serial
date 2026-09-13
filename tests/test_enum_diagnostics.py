"""Bounded Enum error handling, including legacy wire and custom resolution."""
from enum import Enum, Flag, IntEnum, IntFlag
import enum
import io
from pathlib import Path
import struct
import sys
import tracemalloc
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mscs
from mscs import _core as core


@pytest.fixture(autouse=True)
def registry():
    previous = dict(core._registry)
    yield
    core._registry.clear()
    core._registry.update(previous)


def enum_wire(cls, value, version=2):
    buf = io.BytesIO(b'MSCS\x02\x00' + core._ENUM)
    buf.seek(0, 2)
    enc = core._Encoder(buf)
    enc._encode_str(core._class_key(cls))
    enc.encode(value)
    raw = buf.getvalue()
    return raw if version == 2 else raw[:4] + b'\x01' + raw[6:]


def shared_value():
    value = 'x' * 128
    for _ in range(12):
        value = (value, value)
    return value


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('transport', ['loads', 'load', 'load_compressed'])
def test_invalid_enum_diagnostic_does_not_expand_shared_value(version, strict, transport):
    @mscs.register
    class Known(Enum):
        A = 1
    raw = enum_wire(Known, shared_value(), version)
    options = dict(strict=strict, max_size=len(raw), max_hash_work=10_000)
    packed = struct.pack('<I', len(raw)) + zlib.compress(raw)
    tracemalloc.start()
    try:
        with pytest.raises(mscs.MSCDecodeError) as caught:
            if transport == 'loads':
                mscs.loads(raw, **options)
            elif transport == 'load':
                mscs.load(io.BytesIO(raw), **options)
            else:
                mscs.load_compressed(io.BytesIO(packed), **options)
        assert len(str(caught.value)) < 256
        assert caught.value.__context__ is None
        assert tracemalloc.get_traced_memory()[1] < 128 * 1024
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize('base', [Enum, IntEnum, Flag, IntFlag])
def test_invalid_value_for_enum_and_flags_has_small_error(base):
    @mscs.register
    class Known(base):
        A = 1
    with pytest.raises(mscs.MSCDecodeError) as caught:
        mscs.loads(enum_wire(Known, shared_value()))
    assert len(str(caught.value)) < 256


@pytest.mark.parametrize('outcome', ['member', 'none', 'invalid', 'raises'])
def test_missing_hook_is_called_once_without_builtin_diagnostic(outcome):
    calls = []
    @mscs.register
    class Known(Enum):
        A = 1
        @classmethod
        def _missing_(cls, value):
            calls.append(type(value))
            if outcome == 'member':
                return cls.A
            if outcome == 'none':
                return None
            if outcome == 'invalid':
                return value
            raise ValueError('hook rejected value')
    raw = enum_wire(Known, shared_value())
    if outcome == 'member':
        assert mscs.loads(raw) is Known.A
    else:
        with pytest.raises(mscs.MSCDecodeError) as caught:
            mscs.loads(raw)
        assert len(str(caught.value)) < 256
    assert calls == [tuple]


def test_valid_shared_tuple_and_unhashable_members_retain_identity():
    @mscs.register
    class Known(Enum):
        SHARED = shared_value()
        LIST = [1, 2]
        DICT = {'x': 1}
        FROZENSET = frozenset([1, 2])
    for member in Known:
        assert mscs.loads(mscs.dumps(member)) is member


@pytest.mark.skipif(sys.version_info < (3, 13), reason='value aliases require Python 3.13')
def test_value_aliases_include_unhashable_aliases():
    @mscs.register
    class Known(Enum):
        A = 1
    Known.A._add_value_alias_(2)
    Known.A._add_value_alias_([3])
    for value in [2, [3]]:
        assert mscs.loads(enum_wire(Known, value)) is Known.A


@pytest.mark.skipif(sys.version_info < (3, 11), reason='FlagBoundary requires Python 3.11')
@pytest.mark.parametrize('boundary', ['STRICT', 'CONFORM', 'EJECT', 'KEEP'])
def test_flag_boundary_semantics(boundary):
    cls = Flag('Known', {'A': 1, 'B': 2}, boundary=getattr(enum, boundary))
    mscs.register(cls)
    raw = enum_wire(cls, 5)
    if boundary == 'STRICT':
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(raw)
    else:
        result = mscs.loads(raw)
        assert result == cls(5) and type(result) is type(cls(5))


def test_custom_enum_metaclass_call_is_not_a_deserialization_hook():
    class Meta(type(Enum)):
        def __call__(cls, value):
            raise AssertionError('member reconstruction must not invoke the metaclass constructor')
    @mscs.register
    class Known(Enum, metaclass=Meta):
        A = 1
    assert mscs.loads(mscs.dumps(Known.A)) is Known.A

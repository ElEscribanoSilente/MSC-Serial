"""Regression tests for the September 2026 P1/P2 repairs."""
import dataclasses
from enum import Enum, IntEnum, IntFlag
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
def restore_registry():
    previous = dict(core._registry)
    yield
    core._registry.clear()
    core._registry.update(previous)


def shared_tuple(depth):
    value = (0, 0)
    for _ in range(depth - 1):
        value = (value, value)
    return value


def hashed_blob(depth, tag=core._DICT, version=2):
    # Build bytes without hashing or rendering the expanded key in the test.
    def encode_tuple(level, ref):
        if level == 1:
            return core._TUPLE + struct.pack('<I', 2) + b'\x02\x01\x00\x00' * 2
        return (core._TUPLE + struct.pack('<I', 2) + encode_tuple(level - 1, ref + 1)
                + core._REF + struct.pack('<I', ref + 1))
    header = b'MSCS' + bytes([version]) + (b'\x00' if version == 2 else b'')
    return header + tag + struct.pack('<I', 1) + encode_tuple(depth, 1) + (core._NONE if tag == core._DICT else b'')


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('tag', [core._DICT, core._SET, core._FROZENSET])
def test_hash_budget_covers_all_containers_and_versions(version, tag):
    with pytest.raises(mscs.MSCDecodeError, match='hash'):
        mscs.loads(hashed_blob(8, tag, version), max_hash_work=64)


@pytest.mark.skipif(not hasattr(core, 'MAX_HASH_WORK'), reason='safe baseline: guard not implemented yet')
@pytest.mark.parametrize('tag', [core._DICT, core._SET, core._FROZENSET])
def test_default_budget_rejects_exponential_graph_without_expansion(tag):
    data = hashed_blob(60, tag)
    tracemalloc.start()
    try:
        with pytest.raises(mscs.MSCDecodeError, match='hash'):
            mscs.loads(data, max_size=len(data))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024


def test_shared_key_keeps_identity_and_small_memory_footprint():
    data = hashed_blob(14)
    tracemalloc.start()
    try:
        result = mscs.loads(data)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    key = next(iter(result))
    assert key[0] is key[1]
    assert peak < 100_000


def test_large_integer_dict_key_never_needs_decimal_repr():
    key = 10 ** 5000
    assert mscs.loads(mscs.dumps({key: 'ok'})) == {key: 'ok'}


def test_dict_breadcrumb_does_not_call_registered_repr():
    @mscs.register
    class Key:
        def __repr__(self):
            raise AssertionError('repr must not run while decoding a key')

    result = mscs.loads(mscs.dumps({Key(): 1}))
    assert list(result.values()) == [1]


@pytest.mark.parametrize('transport', ['load', 'load_compressed'])
def test_hash_budget_is_forwarded_by_file_apis(transport):
    data = hashed_blob(8)
    if transport == 'load_compressed':
        data = struct.pack('<I', len(data)) + zlib.compress(data)
    with pytest.raises(mscs.MSCDecodeError, match='hash'):
        getattr(mscs, transport)(io.BytesIO(data), max_hash_work=64)


def test_hash_budget_is_cumulative_and_per_call():
    payload = mscs.dumps({1: 2, 3: 4})
    with pytest.raises(mscs.MSCDecodeError, match='hash'):
        mscs.loads(payload, max_hash_work=1)
    assert mscs.loads(payload, max_hash_work=2) == {1: 2, 3: 4}
    assert mscs.loads(mscs.dumps([1, 2]), max_hash_work=0) == [1, 2]


@pytest.mark.skipif(not hasattr(core, 'MAX_HASH_WORK'), reason='safe baseline: guard not implemented yet')
def test_reference_chain_depth_is_checked_before_native_hashing():
    # Lexical depth stays small while each subsequent tuple references the last.
    header = b'MSCS\x02\x00'
    items = [core._TUPLE + struct.pack('<I', 0)]
    for ref in range(1, 80):
        items.append(core._TUPLE + struct.pack('<I', 1) + core._REF + struct.pack('<I', ref))
    items.append(core._SET + struct.pack('<I', 1) + core._REF + struct.pack('<I', 80))
    data = header + core._LIST + struct.pack('<I', len(items)) + b''.join(items)
    with pytest.raises(mscs.MSCDecodeError, match='hash'):
        mscs.loads(data, max_depth=32)


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('base', [IntEnum, IntFlag, Enum])
def test_registered_enums_keep_type_and_identity(base, strict):
    @mscs.register
    class Member(base):
        A = 1
    assert mscs.loads(mscs.dumps(Member.A), strict=strict) is Member.A
    assert mscs.copy([Member.A])[0] is Member.A


def test_string_and_float_enums_keep_type():
    @mscs.register
    class Text(str, Enum):
        A = 'a'
    @mscs.register
    class Number(float, Enum):
        A = 1.5
    for value in [Text.A, Number.A]:
        assert mscs.loads(mscs.dumps(value)) is value


def test_unregistered_primitive_enum_obeys_strict():
    class Member(IntEnum):
        A = 1
    data = mscs.dumps(Member.A)
    with pytest.raises(mscs.MSCSecurityError):
        mscs.loads(data)
    assert mscs.loads(data, strict=False)['__value__'] == 1


def test_enum_alias_and_non_enum_registry_guard():
    class Member(Enum):
        A = 1
    mscs.register_alias(core._class_key(Member), Member)
    assert mscs.loads(mscs.dumps(Member.A), strict=False) is Member.A
    class NotEnum:
        def __init__(self, value):
            raise AssertionError('non-Enum constructor must not execute')
    data = mscs.dumps(Member.A)
    mscs.register_alias(core._class_key(Member), NotEnum)
    for strict in [True, False]:
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(data, strict=strict)


@pytest.mark.skipif(not hasattr(core, 'MAX_HASH_WORK'), reason='safe baseline: guard not implemented yet')
def test_enum_value_hashing_has_the_same_budget():
    @mscs.register
    class Member(Enum):
        A = (0, 0)
    buf = io.BytesIO(b'MSCS\x02\x00')
    buf.seek(6)
    enc = core._Encoder(buf)
    buf.write(core._ENUM)
    enc._encode_str(core._class_key(Member))
    enc.encode(shared_tuple(25))
    with pytest.raises(mscs.MSCDecodeError, match='hash'):
        mscs.loads(buf.getvalue())


@pytest.mark.parametrize('frozen', [False, True])
def test_dataclass_preserves_inherited_private_and_unset_slots(frozen):
    class Base:
        __slots__ = ('x', '__private', 'unset', '__weakref__')
    @mscs.register
    @dataclasses.dataclass(frozen=frozen)
    class Child(Base):
        y: int
    obj = Child(1)
    object.__setattr__(obj, 'x', 2)
    object.__setattr__(obj, '_Base__private', obj)
    result = mscs.loads(mscs.dumps(obj))
    assert result.x == 2
    assert result._Base__private is result
    assert not hasattr(result, 'unset')


def test_dataclass_inherited_slot_in_tuple_cycle():
    class Base:
        __slots__ = 'parent'
    @mscs.register
    @dataclasses.dataclass
    class Child(Base):
        value: int
    obj = Child(2)
    root = (obj,)
    obj.parent = root
    result = mscs.loads(mscs.dumps(root))
    assert result[0].parent is result


@pytest.mark.skipif(sys.version_info < (3, 10), reason='dataclass slots added in 3.10')
def test_frozen_slots_dataclass_keeps_inherited_slot_and_legacy_list_state():
    class Base:
        __slots__ = ('x',)
    @mscs.register
    @dataclasses.dataclass(frozen=True, slots=True)
    class Child(Base):
        y: int
    obj = Child(1)
    object.__setattr__(obj, 'x', 2)
    result = mscs.loads(mscs.dumps(obj))
    assert (result.x, result.y) == (2, 1)
    # 2.5.1 used the generated __getstate__ list for this class.
    buf = io.BytesIO(b'MSCS\x02\x00'); buf.seek(6)
    enc = core._Encoder(buf); enc.ref_counter = 1
    buf.write(core._OBJ); enc._encode_str(core._class_key(Child)); enc.encode([7])
    legacy = mscs.loads(buf.getvalue())
    assert legacy.y == 7 and not hasattr(legacy, 'x')


@pytest.mark.parametrize('shape', [(), (0,), (2, 3)])
def test_bfloat16_preserves_raw_bits_and_grad(shape):
    torch = pytest.importorskip('torch')
    count = 1
    for n in shape:
        count *= n
    bits = torch.tensor([0x3f80, -32768, 0x7fc1, 0x7f80, -128, 0][:count], dtype=torch.int16)
    original = bits.reshape(shape).view(torch.bfloat16).requires_grad_(True)
    result = mscs.loads(mscs.dumps([original, original]))
    assert result[0] is result[1]
    assert result[0].requires_grad
    assert result[0].dtype == torch.bfloat16
    assert torch.equal(result[0].view(torch.int16), original.view(torch.int16))


def test_conjugate_view_preserves_shared_identity_and_grad():
    torch = pytest.importorskip('torch')
    original = torch.tensor([1 + 2j], requires_grad=True).conj()
    result = mscs.loads(mscs.dumps([original, original]))
    assert result[0] is result[1]
    assert result[0].requires_grad and torch.equal(result[0], original)


def test_torch_extras_exclude_known_incompatible_abi_pair():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    from packaging.requirements import Requirement
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    for extra in ['torch', 'all']:
        reqs = {r.name: r for r in map(Requirement, config['project']['optional-dependencies'][extra])}
        assert not ('2.0.0' in reqs['torch'].specifier and '2.0.2' in reqs['numpy'].specifier)
        assert '2.8.0' in reqs['torch'].specifier and '2.0.2' in reqs['numpy'].specifier


@pytest.mark.parametrize('budget', [-1, 1.5, True, '100'])
def test_invalid_hash_budget_is_rejected(budget):
    with pytest.raises(ValueError, match='max_hash_work'):
        mscs.loads(mscs.dumps(None), max_hash_work=budget)


def test_flag_combination_and_unhashable_enum_value():
    @mscs.register
    class Flags(IntFlag):
        A = 1
        B = 2
    @mscs.register
    class Lists(Enum):
        A = [1, 2]
    assert mscs.copy(Flags.A | Flags.B) is Flags.A | Flags.B
    assert mscs.copy(Lists.A) is Lists.A


@pytest.mark.skipif(sys.version_info < (3, 10), reason='dataclass slots added in 3.10')
def test_frozen_slots_custom_hooks_keep_priority():
    calls = []
    @mscs.register
    @dataclasses.dataclass(frozen=True, slots=True)
    class Record:
        value: int
    # Python 3.10 replaces hooks during slots=True decoration. Install the
    # custom pair afterwards so every supported Python tests the live hooks.
    def getstate(self):
        calls.append('get')
        return self.value + 10
    def setstate(self, state):
        calls.append('set')
        object.__setattr__(self, 'value', state - 10)
    Record.__getstate__ = getstate
    Record.__setstate__ = setstate
    assert mscs.loads(mscs.dumps(Record(7))).value == 7
    assert calls == ['get', 'set']


@pytest.mark.skipif(sys.version_info < (3, 10), reason='dataclass slots added in 3.10')
def test_frozen_slots_without_extra_state_keeps_legacy_list_wire():
    @dataclasses.dataclass(frozen=True, slots=True)
    class Record:
        value: int
    result = mscs.loads(mscs.dumps(Record(3)), strict=False)
    assert result['__state__'] == [3]


@pytest.mark.parametrize('key', ['__dict__', 'undeclared', 10 ** 5000], ids=['dict', 'unknown', 'large-int'])
def test_dataclass_state_filter_stays_closed_and_diagnostic_is_bounded(key):
    @mscs.register
    @dataclasses.dataclass
    class Record:
        value: int
    buf = io.BytesIO(b'MSCS\x02\x00')
    buf.seek(6)
    enc = core._Encoder(buf)
    enc.ref_counter = 1
    buf.write(core._OBJ)
    enc._encode_str(core._class_key(Record))
    enc.encode({key: 1})
    with pytest.raises(mscs.MSCDecodeError, match='field') as error:
        mscs.loads(buf.getvalue())
    assert len(str(error.value)) < 300


def test_bfloat16_noncontiguous_signed_and_explicit_little_endian():
    torch = pytest.importorskip('torch')
    original = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
    data = mscs.dumps(original, hmac_key=b'regression-key')
    assert torch.equal(mscs.loads(data, hmac_key=b'regression-key'), original)
    one = torch.tensor([1], dtype=torch.bfloat16)
    assert mscs.dumps(one).endswith(b'\x02\x00\x00\x00\x80\x3f')


@pytest.mark.parametrize('tag,raw', [(core._TENSOR, b'\x80'), (core._TENSOR, b'\x80\x3f' * 2),
                                   (core._NDARRAY, b'\x80\x3f')])
def test_bfloat16_malformed_payload_and_ndarray_tag_are_rejected(tag, raw):
    pytest.importorskip('torch')
    buf = io.BytesIO(b'MSCS\x02\x00')
    buf.seek(6)
    buf.write(tag)
    core._Encoder(buf)._encode_str('bfloat16|1|0' if tag == core._TENSOR else 'bfloat16|1')
    buf.write(struct.pack('<I', len(raw)) + raw)
    with pytest.raises((mscs.MSCDecodeError, mscs.MSCSecurityError)):
        mscs.loads(buf.getvalue())

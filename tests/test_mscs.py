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
    _is_safe_dtype, _registry, _class_key,
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

    def test_shared_tuple_reference(self):
        x = (1, 2)
        val = [x, x]
        result = mscs.loads(mscs.dumps(val), strict=False)
        assert result[0] is result[1]
        assert result[0] == (1, 2)

    def test_tuple_cycle(self):
        # Ciclo que atraviesa una tupla: root -> lista -> root
        child = []
        root = (child,)
        child.append(root)
        result = mscs.loads(mscs.dumps(root), strict=False)
        assert isinstance(result, tuple)
        assert result[0][0] is result

    def test_nested_tuple_cycle(self):
        # La tupla interior queda pendiente hasta que la exterior se resuelve
        inner_list = []
        root = (inner_list,)
        inner_list.append((root, 42))
        result = mscs.loads(mscs.dumps(root), strict=False)
        assert result[0][0][1] == 42
        assert result[0][0][0] is result

    def test_tuple_cycle_multiple_pending_slots(self):
        # Una tupla con DOS slots pendientes hacia el mismo ancestro
        holder = []
        root = (holder,)
        holder.append((root, root))
        result = mscs.loads(mscs.dumps(root), strict=False)
        inner = result[0][0]
        assert inner[0] is result
        assert inner[1] is result

    def test_dict_value_tuple_cycle(self):
        d = {}
        t = (d,)
        d["t"] = t
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert result[0]["t"] is result

    def test_deque_tuple_cycle(self):
        import collections
        dq = collections.deque()
        t = (dq,)
        dq.append(t)
        result = mscs.loads(mscs.dumps(t), strict=False)
        assert result[0][0] is result

    def test_registered_object_self_reference(self):
        @mscs.register
        class SelfRefNode:
            pass

        n = SelfRefNode()
        n.me = n
        result = mscs.loads(mscs.dumps(n))
        assert isinstance(result, SelfRefNode)
        assert result.me is result

    def test_registered_objects_mutual_reference(self):
        @mscs.register
        class MutualA:
            pass

        @mscs.register
        class MutualB:
            pass

        a = MutualA()
        b = MutualB()
        a.other = b
        b.other = a
        result = mscs.loads(mscs.dumps(a))
        assert isinstance(result.other, MutualB)
        assert result.other.other is result

    def test_object_in_tuple_cycle(self):
        @mscs.register
        class TupleCycleNode:
            pass

        n = TupleCycleNode()
        root = (n,)
        n.me = root
        result = mscs.loads(mscs.dumps(root))
        assert result[0].me is result

    def test_slots_object_in_tuple_cycle(self):
        @mscs.register
        class SlotsCycleNode:
            __slots__ = ("me",)

        n = SlotsCycleNode()
        t = (n,)
        n.me = t
        result = mscs.loads(mscs.dumps(t))
        assert result[0].me is result

    def test_frozen_dataclass_in_tuple_cycle(self):
        @mscs.register
        @dataclasses.dataclass(frozen=True)
        class FrozenCycleNode:
            x: object = None

        n = FrozenCycleNode()
        t = (n,)
        object.__setattr__(n, "x", t)
        result = mscs.loads(mscs.dumps(t))
        assert result[0].x is result

    def test_setstate_object_direct_cycle(self):
        # __setstate__ custom con auto-referencia directa (sin tupla):
        # la instancia existe antes de decodificar el estado.
        @mscs.register
        class StatefulNode:
            def __getstate__(self):
                return {"me": self.me}

            def __setstate__(self, state):
                self.me = state["me"]

        n = StatefulNode()
        n.me = n
        result = mscs.loads(mscs.dumps(n))
        assert result.me is result

    def test_setstate_object_in_tuple_cycle_fails_closed(self):
        # Un __setstate__ custom recibiría el sentinela de una tupla aún en
        # construcción: irresoluble sin corrupción -> error explícito.
        @mscs.register
        class OpaqueStateNode:
            def __getstate__(self):
                return {"t": self.t}

            def __setstate__(self, state):
                self.t = state["t"]

        n = OpaqueStateNode()
        t = (n,)
        n.t = t
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(mscs.dumps(t))

    def test_unregistered_fallback_self_reference(self):
        class UnregisteredCycleNode:
            pass

        n = UnregisteredCycleNode()
        n.me = n
        result = mscs.loads(mscs.dumps(n), strict=False)
        assert result["__state__"]["me"] is result

    def test_unregistered_fallback_tuple_cycle(self):
        class UnregisteredTupleNode:
            pass

        n = UnregisteredTupleNode()
        root = (n,)
        n.me = root
        result = mscs.loads(mscs.dumps(root), strict=False)
        assert result[0]["__state__"]["me"] is result

    def test_custom_setattr_never_sees_pending(self):
        # El __setattr__ de usuario debe ver UNA asignación con el valor
        # real, nunca el sentinela interno de una tupla en construcción.
        seen = []

        @mscs.register
        class AuditedSlotsNode:
            __slots__ = ("me",)

            def __setattr__(self, k, v):
                seen.append(type(v).__name__)
                object.__setattr__(self, k, v)

        n = AuditedSlotsNode()
        t = (n,)
        object.__setattr__(n, "me", t)
        result = mscs.loads(mscs.dumps(t))
        assert result[0].me is result
        assert seen == ["tuple"], f"__setattr__ vio: {seen}"

    def test_property_setter_never_sees_pending(self):
        # Un data descriptor (property) invocado vía object.__setattr__ en
        # la rama dataclass tampoco debe ver el sentinela.
        seen = []

        @mscs.register
        @dataclasses.dataclass
        class ShadowedFieldNode:
            me: object = None

        def _get(self):
            return self.__dict__.get("me")

        def _set(self, v):
            seen.append(type(v).__name__)
            self.__dict__["me"] = v

        ShadowedFieldNode.me = property(_get, _set)

        d = ShadowedFieldNode.__new__(ShadowedFieldNode)
        t = (d,)
        d.__dict__["me"] = t
        result = mscs.loads(mscs.dumps(t))
        assert result[0].me is result
        assert seen == ["tuple"], f"property.__set__ vio: {seen}"

    def test_setstate_copying_nested_pending_attr_fails_closed(self):
        # __setstate__ que LEE un atributo aún no parcheado de un objeto
        # anidado (en ciclo vía tupla) debe fallar visible, no recibir ni
        # copiar el sentinela interno.
        @mscs.register
        class InnerBack:
            pass

        @mscs.register
        class CopierNode:
            def __getstate__(self):
                return {"peer": self.peer}

            def __setstate__(self, state):
                self.peer = state["peer"]
                self.snapshot = self.peer.back  # lee en pleno decode

        b = InnerBack()
        a = CopierNode()
        a.peer = b
        root = (a,)
        b.back = root
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(mscs.dumps(root))

    def test_setstate_holding_nested_object_resolves_after_load(self):
        # Variante resoluble del caso anterior: __setstate__ solo guarda la
        # referencia sin leer el atributo pendiente — el fix-up lo parchea
        # y el grafo final queda correcto.
        @mscs.register
        class InnerBack2:
            pass

        @mscs.register
        class HolderNode:
            def __getstate__(self):
                return {"peer": self.peer}

            def __setstate__(self, state):
                self.peer = state["peer"]

        b = InnerBack2()
        a = HolderNode()
        a.peer = b
        root = (a,)
        b.back = root
        result = mscs.loads(mscs.dumps(root))
        assert result[0].peer.back is result


class TestForgedPendingRefs:
    """Payloads forjados con _REF hacia slots aún en construcción.

    Ninguno es producible por el encoder (exigirían hashear un ciclo o
    resolver una tupla contra sí misma): deben fallar cerrado, nunca
    entregar un placeholder o corromper silenciosamente.
    """

    def test_self_referential_root_tuple(self):
        # TUPLE(n=1) cuyo único item es _REF a su propio slot: deadlock
        payload = HEADER + _TUPLE + struct.pack('<I', 1) + _REF + struct.pack('<I', 0)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)

    def test_pending_ref_as_dict_key(self):
        # TUPLE(n=1)[ DICT(n=1){ _REF 0: None } ]
        payload = (
            HEADER + _TUPLE + struct.pack('<I', 1)
            + _DICT + struct.pack('<I', 1)
            + _REF + struct.pack('<I', 0)
            + _NONE
        )
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)

    def test_pending_ref_in_set(self):
        # TUPLE(n=1)[ SET(n=1){ _REF 0 } ]
        payload = (
            HEADER + _TUPLE + struct.pack('<I', 1)
            + _SET + struct.pack('<I', 1)
            + _REF + struct.pack('<I', 0)
        )
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)

    def test_pending_ref_in_frozenset(self):
        # TUPLE(n=1)[ FROZENSET(n=1){ _REF 0 } ]
        payload = (
            HEADER + _TUPLE + struct.pack('<I', 1)
            + _FROZENSET + struct.pack('<I', 1)
            + _REF + struct.pack('<I', 0)
        )
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)

    def test_self_referential_frozenset(self):
        # FROZENSET(n=1){ _REF 0 } directo, sin tupla exterior
        payload = HEADER + _FROZENSET + struct.pack('<I', 1) + _REF + struct.pack('<I', 0)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)

    @pytest.mark.parametrize("strict", [True, False])
    def test_pending_ref_as_enum_value(self, strict):
        # TUPLE(n=1)[ ENUM(path, _REF 0) ]: en strict=False el sentinela
        # escaparia dentro del dict {'__enum__', '__value__'} sin fix-up
        path_bytes = b"forged.module.NoSuchEnum"
        payload = (
            HEADER + _TUPLE + struct.pack('<I', 1)
            + _ENUM + _STR + struct.pack('<I', len(path_bytes)) + path_bytes
            + _REF + struct.pack('<I', 0)
        )
        with pytest.raises((mscs.MSCDecodeError, mscs.MSCSecurityError)):
            mscs.loads(payload, strict=strict)


class TestSlotsRoundtrip:
    """Extracción/restauración de __slots__: string, herencia, híbridas,
    name mangling, __dict__/__weakref__ declarados en slots."""

    def test_slots_declared_as_string(self):
        @mscs.register
        class OneSlotStr:
            __slots__ = "value"

            def __init__(self):
                self.value = 7

        d = mscs.loads(mscs.dumps(OneSlotStr()))
        assert d.value == 7

    def test_inherited_slot_in_hybrid_class(self):
        class SlotBaseH:
            __slots__ = ("x",)

        @mscs.register
        class HybridH(SlotBaseH):
            pass

        obj = HybridH()
        obj.x = 1
        obj.y = 2
        d = mscs.loads(mscs.dumps(obj))
        assert d.x == 1
        assert d.y == 2

    def test_slot_with_name_mangling(self):
        @mscs.register
        class SecretSlot:
            __slots__ = ("__token",)

            def __init__(self):
                self.__token = "abc"

            def get_token(self):
                return self.__token

        d = mscs.loads(mscs.dumps(SecretSlot()))
        assert d.get_token() == "abc"

    def test_slot_mangling_class_with_leading_underscore(self):
        @mscs.register
        class _PrivSlot:
            __slots__ = ("__k",)

            def __init__(self):
                self.__k = 5

            def get_k(self):
                return self.__k

        d = mscs.loads(mscs.dumps(_PrivSlot()))
        assert d.get_k() == 5

    def test_slots_declaring_dict_and_weakref(self):
        @mscs.register
        class DictInSlots:
            __slots__ = ("x", "__dict__", "__weakref__")

        w = DictInSlots()
        w.x = 10
        w.dynamic = 20
        d = mscs.loads(mscs.dumps(w))
        assert d.x == 10
        assert d.dynamic == 20

    def test_unset_slot_stays_unset(self):
        @mscs.register
        class PartialSlotsRT:
            __slots__ = ("a", "b")

        p = PartialSlotsRT()
        p.a = 1
        d = mscs.loads(mscs.dumps(p))
        assert d.a == 1
        assert not hasattr(d, "b")

    def test_multilevel_slot_inheritance(self):
        class SlotL1:
            __slots__ = ("a",)

        class SlotL2(SlotL1):
            __slots__ = ("b",)

        @mscs.register
        class SlotL3(SlotL2):
            __slots__ = ("c",)

        obj = SlotL3()
        obj.a, obj.b, obj.c = 1, 2, 3
        d = mscs.loads(mscs.dumps(obj))
        assert (d.a, d.b, d.c) == (1, 2, 3)

    def test_hybrid_class_in_tuple_cycle(self):
        # Interacción con la resolución de ciclos: el slot heredado se
        # restaura vía setattr y el fix-up parchea el hueco de la tupla.
        class SlotBaseCycle:
            __slots__ = ("ref",)

        @mscs.register
        class HybridCycleNode(SlotBaseCycle):
            pass

        h = HybridCycleNode()
        t = (h,)
        h.ref = t
        h.tag = "d"
        d = mscs.loads(mscs.dumps(t))
        assert d[0].ref is d
        assert d[0].tag == "d"

    def test_slots_dict_value_shadowed_by_slot(self):
        # En una híbrida, una clave espuria de __dict__ homónima de un slot
        # es inaccesible (el descriptor gana): se serializa el valor efectivo.
        class SlotBaseShadow:
            __slots__ = ("x",)

        @mscs.register
        class HybridShadow(SlotBaseShadow):
            pass

        obj = HybridShadow()
        obj.x = 1
        obj.__dict__["x"] = 99  # inaccesible vía atributo
        assert obj.x == 1
        d = mscs.loads(mscs.dumps(obj))
        assert d.x == 1

    def test_hybrid_spurious_dict_key_is_faithful(self):
        # Una clave literal "__dict__" dentro del __dict__ de instancia es
        # legal: debe restaurarse como clave, jamás pasar por setattr (que
        # reemplazaría el dict del objeto entero).
        class SlotBaseSpur:
            __slots__ = ("x",)

        @mscs.register
        class HybridSpur(SlotBaseSpur):
            pass

        obj = HybridSpur()
        obj.x = 1
        obj.__dict__["__dict__"] = "weird but legal"
        obj.__dict__["__weakref__"] = "also legal"
        d = mscs.loads(mscs.dumps(obj))
        assert d.x == 1
        assert d.__dict__["__dict__"] == "weird but legal"
        assert d.__dict__["__weakref__"] == "also legal"

    def test_hybrid_spurious_dict_key_no_clobbering(self):
        # El valor dict de una clave espuria "__dict__" no debe pisar ni
        # inyectar atributos reales.
        class SlotBaseClob:
            __slots__ = ("x",)

        @mscs.register
        class HybridClob(SlotBaseClob):
            pass

        obj = HybridClob()
        obj.x = 1
        obj.y = 2
        obj.__dict__["__dict__"] = {"evil": "payload", "y": "clobbered"}
        d = mscs.loads(mscs.dumps(obj))
        assert d.y == 2
        assert not hasattr(d, "evil")
        assert d.__dict__["__dict__"] == {"evil": "payload", "y": "clobbered"}

    def test_hybrid_spurious_dict_key_no_aliasing(self):
        # El __dict__ del objeto decodificado nunca debe SER otro objeto
        # del grafo (aliasing de identidad vía setattr('__dict__', ...)).
        class SlotBaseAlias:
            __slots__ = ("x",)

        @mscs.register
        class HybridAlias(SlotBaseAlias):
            pass

        obj = HybridAlias()
        obj.x = 1
        shared = {"k": "v"}
        obj.__dict__["__dict__"] = shared
        h2, shared2 = mscs.loads(mscs.dumps([obj, shared]))
        assert h2.__dict__ is not shared2
        assert h2.__dict__["__dict__"] is shared2  # la ref compartida, como valor

    def test_forged_dict_key_on_pure_slots_fails_closed(self):
        # Payload forjado: clase slots-pura registrada con una clave
        # "__dict__" en el estado -> setattr -> AttributeError -> error.
        @mscs.register
        class PureSlotsForged:
            __slots__ = ("a",)

        path = _class_key(PureSlotsForged).encode()
        key = b"__dict__"
        payload = (
            HEADER + _OBJ
            + _STR + struct.pack('<I', len(path)) + path
            + _DICT + struct.pack('<I', 1)
            + _STR + struct.pack('<I', len(key)) + key
            + _NONE
        )
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(payload)


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

    def test_frozen_dataclass_roundtrip(self):
        """Regression: frozen dataclasses encoded fine but the decoder used
        setattr, which frozen instances forbid, so decode raised
        MSCDecodeError. The dataclass branch now uses object.__setattr__."""
        @mscs.register
        @dataclasses.dataclass(frozen=True)
        class FrozenPoint:
            x: int = 0
            y: int = 0

        p = FrozenPoint(3, 4)
        result = mscs.loads(mscs.dumps(p))
        assert isinstance(result, FrozenPoint)
        assert result == FrozenPoint(3, 4)

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


class TestTotalSizeLimit:
    """MAX_SIZE acota el blob total, no solo campos individuales; y los
    límites relevantes son configurables por llamada (per-call)."""

    def test_loads_rejects_blob_over_max_size(self):
        # Un blob real por encima de un max_size chico se rechaza ANTES de
        # decodificar (tope total, no por campo).
        blob = mscs.dumps([b"x" * 2000])
        assert len(blob) > 1000
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob, max_size=1000)

    def test_loads_accepts_blob_at_max_size_boundary(self):
        blob = mscs.dumps([1, 2, 3])
        # Exactamente len(blob) debe pasar; len(blob)-1 debe fallar.
        assert mscs.loads(blob, max_size=len(blob), strict=False) == [1, 2, 3]
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob, max_size=len(blob) - 1)

    def test_cumulative_strings_bounded_by_max_size(self):
        # El ataque del hallazgo: muchos bytes-strings individualmente
        # válidos (< MAX_STRING) cuya SUMA excede el tope. Con un max_size
        # acotado, el blob entero se rechaza.
        blob = mscs.dumps([b"a" * 100_000 for _ in range(5)])  # ~500 KB
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob, max_size=200_000)
        # Sin acotar (default grande) sigue funcionando.
        out = mscs.loads(blob, strict=False)
        assert sum(len(x) for x in out) == 500_000

    def test_load_bounds_file_read(self):
        # load() no debe materializar un archivo mayor que max_size.
        blob = mscs.dumps([b"y" * 5000])
        buf = io.BytesIO(blob)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load(buf, max_size=1000)

    def test_load_roundtrip_within_limit(self):
        blob = mscs.dumps({"a": 1, "b": [2, 3]})
        buf = io.BytesIO(blob)
        assert mscs.load(buf, strict=False, max_size=10_000) == {"a": 1, "b": [2, 3]}

    def test_max_depth_configurable_per_call_decode(self):
        # El knob real: max_depth per-call SÍ afecta el decode.
        nested = [[[[[1]]]]]
        blob = mscs.dumps(nested)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob, max_depth=2)
        assert mscs.loads(blob, max_depth=50, strict=False) == nested

    def test_max_depth_configurable_per_call_encode(self):
        with pytest.raises(mscs.MSCEncodeError):
            mscs.dumps([[[[1]]]], max_depth=2)

    def test_module_constant_rebind_has_no_effect(self):
        # Documentar el footgun: reasignar mscs.MAX_DEPTH NO cambia el
        # comportamiento; el knob soportado es el parámetro per-call.
        original = mscs.MAX_DEPTH
        try:
            mscs.MAX_DEPTH = 1
            # Sigue decodificando anidamiento profundo pese al "1".
            assert mscs.loads(mscs.dumps([[[1]]]), strict=False) == [[[1]]]
        finally:
            mscs.MAX_DEPTH = original

    def test_load_compressed_respects_max_size(self):
        import zlib as _zlib
        obj = [b"z" * 100_000]
        raw = mscs.dumps(obj)
        buf = io.BytesIO()
        buf.write(struct.pack('<I', len(raw)))
        buf.write(_zlib.compress(raw))
        buf.seek(0)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(buf, max_size=1000)

    def test_v1_blob_over_max_size_rejected(self):
        # El tope total aplica también a la rama v1 (antes del dispatch).
        v1_blob = MAGIC + b'\x01' + _LIST + struct.pack('<I', 0)
        # Un blob v1 legítimo pequeño pasa...
        assert mscs.loads(v1_blob, strict=False) == []
        # ...pero uno mayor que max_size se rechaza igual que v2.
        big_v1 = MAGIC + b'\x01' + _BYTES + struct.pack('<I', 4000) + b'q' * 4000
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(big_v1, strict=False, max_size=1000)

    def test_load_compressed_read_bounded_by_max_size(self):
        # La lectura del blob COMPRIMIDO se acota por max_size, no solo por
        # MAX_COMPRESSED: bajar max_size reduce el pico de memoria del lado
        # comprimido. Un SpyFile registra los tamaños pedidos a read().
        import zlib as _zlib

        class _SpyFile:
            def __init__(self, data):
                self._buf = io.BytesIO(data)
                self.reads = []

            def read(self, n=-1):
                self.reads.append(n)
                return self._buf.read(n)

        raw = mscs.dumps([b"z" * 2000])
        blob = struct.pack('<I', len(raw)) + _zlib.compress(raw)
        spy = _SpyFile(blob)
        mscs.load_compressed(spy, max_size=50_000, strict=False)
        # Primera lectura: 4 bytes (header orig_size). Segunda: el blob
        # comprimido, acotado a compressBound(max_size)+1 (holgura de zlib
        # sobre datos incompresibles), muy por debajo de MAX_COMPRESSED.
        expected = mscs._core._zlib_compress_bound(50_000) + 1
        assert spy.reads[0] == 4
        assert spy.reads[1] == expected, spy.reads
        assert spy.reads[1] < mscs.MAX_COMPRESSED

    def test_load_compressed_forged_small_header_large_body(self):
        # orig_size forjado pequeño (pasa) pero cuerpo comprimido grande:
        # el tope de lectura comprimida por max_size lo rechaza.
        import zlib as _zlib
        raw = mscs.dumps(list(range(2000)))
        body = _zlib.compress(raw)
        assert len(body) > 1000
        forged = struct.pack('<I', 10) + body  # miente: dice 10 bytes
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(io.BytesIO(forged), strict=False, max_size=1000)

    def test_total_limit_is_on_blob_including_framing(self, monkeypatch):
        # MAX_SIZE acota el blob TOTAL (datos + framing). Un único campo cuyo
        # tamaño crudo llega al límite cruza el default por el framing.
        monkeypatch.setattr(mscs._core, "MAX_SIZE", 10_000)
        payload = b"x" * 10_000  # blob = 6 header + 1 tag + 4 len + 10_000
        blob = mscs.dumps(payload)
        assert len(blob) > 10_000
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob)  # default resuelve a MAX_SIZE=10_000

    def test_copy_not_limited_by_default_max_size(self, monkeypatch):
        # copy() confía en su propio output: no debe romperse aunque el blob
        # cruce el MAX_SIZE por defecto (round-trip de objeto propio).
        monkeypatch.setattr(mscs._core, "MAX_SIZE", 10_000)
        payload = b"x" * 10_000
        # loads con default rechaza el blob (datos+framing > MAX_SIZE)...
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(mscs.dumps(payload))
        # ...pero copy() lo acota a sus propios bytes y funciona.
        assert mscs.copy(payload) == payload


class TestCompressedIntegrity:
    """load_compressed valida el contenedor comprimido con la misma
    estrictez que loads() aplica al payload: header completo, stream zlib
    íntegro, sin bytes finales, orig_size veraz."""

    @staticmethod
    def _framed(orig_size, body):
        return io.BytesIO(struct.pack('<I', orig_size) + body)

    def test_roundtrip_baseline(self):
        obj = [1, 2, 3, "hello", {"k": [4, 5]}]
        buf = io.BytesIO()
        mscs.dump_compressed(obj, buf)
        buf.seek(0)
        assert mscs.load_compressed(buf, strict=False) == obj

    def test_rejects_fake_orig_size(self):
        import zlib as _zlib
        raw = mscs.dumps([1, 2, 3])
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(self._framed(len(raw) + 999, _zlib.compress(raw)),
                                 strict=False)

    def test_rejects_truncated_zlib_stream(self):
        # Falta parte del ADLER32 final: el decoder antes devolvía datos
        # parciales sin error (eof=False no se comprobaba).
        import zlib as _zlib
        raw = mscs.dumps([1, 2, 3, "hello"])
        good = _zlib.compress(raw)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(self._framed(len(raw), good[:-2]), strict=False)

    def test_rejects_trailing_garbage_after_stream(self):
        import zlib as _zlib
        raw = mscs.dumps([1, 2, 3])
        good = _zlib.compress(raw)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(self._framed(len(raw), good + b"garbage"),
                                 strict=False)

    def test_rejects_concatenated_containers(self):
        # Dos streams zlib concatenados: el segundo queda en unused_data.
        import zlib as _zlib
        raw = mscs.dumps([1, 2, 3])
        blob = struct.pack('<I', len(raw)) + _zlib.compress(raw) + _zlib.compress(raw)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(io.BytesIO(blob), strict=False)

    def test_truncated_header_2_bytes_clean_error(self):
        # Antes lanzaba struct.error crudo; ahora MSCDecodeError.
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(io.BytesIO(b'\x01\x02'), strict=False)

    def test_empty_header_clean_error(self):
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(io.BytesIO(b''), strict=False)

    def test_bomb_with_lying_header_still_bounded(self):
        # Control: el endurecimiento no debe romper el corte anti zip-bomb.
        import zlib as _zlib
        bomb = _zlib.compress(b'\x00' * 2_000_000)
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(self._framed(10, bomb), strict=False,
                                 max_size=100_000)

    def test_compressed_roundtrip_with_hmac(self):
        # El endurecimiento no interfiere con kwargs (hmac_key) hacia loads().
        key = b"k" * 32
        obj = {"secret": [1, 2, 3]}
        buf = io.BytesIO()
        mscs.dump_compressed(obj, buf, hmac_key=key)
        buf.seek(0)
        assert mscs.load_compressed(buf, strict=False, hmac_key=key) == obj

    @pytest.mark.parametrize("corrupt", ["garbage", "bitflip", "adler"])
    def test_corrupt_stream_wrapped_not_raw_zlib_error(self, corrupt):
        # Un stream corrupto (no truncado) debe dar MSCDecodeError, nunca
        # un zlib.error crudo que el caller no atrapa con except MSCError.
        import zlib as _zlib
        raw = mscs.dumps([1, 2, 3, "hello"])
        good = _zlib.compress(raw)
        if corrupt == "garbage":
            body = b'\xDE\xAD\xBE\xEF' * 20
        elif corrupt == "bitflip":
            body = bytearray(good); body[len(body) // 2] ^= 0xFF; body = bytes(body)
        else:  # adler
            body = bytearray(good); body[-1] ^= 0xFF; body = bytes(body)
        blob = struct.pack('<I', len(raw)) + body
        with pytest.raises(mscs.MSCDecodeError):
            mscs.load_compressed(io.BytesIO(blob), strict=False)

    @pytest.mark.parametrize("n,level", [(50_000, 0), (100_000, 6), (200_000, 9)])
    def test_incompressible_roundtrip_with_tight_max_size(self, n, level):
        # Datos incompresibles comprimen a MÁS que su tamaño crudo; con
        # max_size ajustado al tamaño real, el bound de lectura comprimida
        # (compressBound) no debe rechazar un dump_compressed legítimo.
        import os as _os, zlib as _zlib
        obj = _os.urandom(n)
        raw = mscs.dumps(obj)
        blob = struct.pack('<I', len(raw)) + _zlib.compress(raw, level)
        assert len(blob) - 4 > len(raw)  # confirma que comprimido > crudo
        out = mscs.load_compressed(io.BytesIO(blob), strict=False, max_size=len(raw))
        assert out == obj

    def test_empty_object_compressed_roundtrip(self):
        # Casos vacíos/pequeños no deben tropezar con los bounds ajustados.
        for obj in (None, [], {}, ""):
            buf = io.BytesIO()
            mscs.dump_compressed(obj, buf)
            buf.seek(0)
            assert mscs.load_compressed(buf, strict=False) == obj


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

    def test_enum_tag_rejects_non_enum_class(self):
        """Regression: an ENUM tag pointing at a registered NON-enum class
        must be rejected, not invoke cls(value). Before the fix the decoder
        called cls(attacker_value) on any registered class (type confusion)."""
        init_seen = []

        @mscs.register
        class NotAnEnum:
            def __init__(self, v):
                init_seen.append(v)

        payload = io.BytesIO()
        payload.write(HEADER + _ENUM)
        cp = _class_key(NotAnEnum).encode()
        payload.write(_STR + struct.pack('<I', len(cp)) + cp)
        payload.write(_INT + struct.pack('<H', 1) + b'\x2a')  # value = 42
        with pytest.raises(mscs.MSCSecurityError, match="no-Enum"):
            mscs.loads(payload.getvalue(), strict=True)
        assert init_seen == [], "constructor must NOT run on ENUM type confusion"

    def test_valid_enum_still_roundtrips(self):
        """Control: a genuine registered Enum must still decode."""
        import enum

        @mscs.register
        class Color(enum.Enum):
            RED = 1
            GREEN = 2

        data = mscs.dumps(Color.GREEN)
        assert mscs.loads(data, strict=True) is Color.GREEN


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

    def test_zip_bomb_bounded_memory(self, monkeypatch):
        """Regression: a payload decompressing far beyond MAX_SIZE must be
        rejected WITHOUT materializing the whole bomb. MAX_SIZE is shrunk so
        the test is cheap; the boundedness property is scale-invariant.
        Before the fix, zlib.decompress() allocated the full output first,
        so peak memory tracked the bomb size, not MAX_SIZE."""
        import tracemalloc
        from mscs import _core

        monkeypatch.setattr(_core, 'MAX_SIZE', 1 << 20)     # 1 MB cap
        compressed = zlib.compress(b'\x00' * (32 << 20), 9)  # 32 MB decompressed
        blob = struct.pack('<I', 8) + compressed             # tiny orig_size hint

        tracemalloc.start()
        try:
            with pytest.raises(mscs.MSCDecodeError):
                mscs.load_compressed(io.BytesIO(blob))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 4 * (1 << 20), f"peak {peak:,}B implies full decompression"

    def test_compressed_roundtrip_larger(self):
        """Control: a legitimate compressed payload still roundtrips."""
        val = {"blob": b'\x00' * 100_000, "n": list(range(500))}
        buf = io.BytesIO()
        mscs.dump_compressed(val, buf)
        buf.seek(0)
        assert mscs.load_compressed(buf, strict=False) == val


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

    def test_v1_respects_strict_true(self):
        """Regression: v1 must honor the caller's strict flag instead of
        forcing strict=False. An unregistered class under strict=True
        (the default) must raise, not silently return a fallback dict."""
        forged = io.BytesIO()
        forged.write(b'MSCS\x01' + _OBJ)
        cls_s = b'ghost.module.GhostClass'
        forged.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
        forged.write(_DICT + struct.pack('<I', 0))
        payload = forged.getvalue()
        with pytest.raises(mscs.MSCSecurityError):
            mscs.loads(payload, strict=True)
        # strict=False still yields the fallback dict (backward compat).
        result = mscs.loads(payload, strict=False)
        assert result == {'__class__': 'ghost.module.GhostClass', '__state__': {}}

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

    def test_hmac_v1_downgrade_rejected(self):
        """Regression: a v1 payload cannot carry HMAC, so supplying a key
        must be rejected as a downgrade — NOT decoded ignoring the key.

        Before the fix, loads() dispatched v1 before any HMAC logic and
        returned the attacker's forged payload while silently ignoring
        hmac_key, fully bypassing authentication."""
        forged_v1 = b'MSCS\x01' + _INT + struct.pack('<H', 1) + b'\x2a'
        with pytest.raises(mscs.MSCSecurityError, match="downgrade"):
            mscs.loads(forged_v1, hmac_key=self.KEY)

    def test_hmac_v1_object_downgrade_rejected(self):
        """A v1 OBJ payload for a registered class must not run __setstate__
        when a key is supplied: the missing HMAC is rejected first."""
        forged = io.BytesIO()
        forged.write(b'MSCS\x01' + _OBJ)
        cls_s = b'attacker.module.Anything'
        forged.write(_STR + struct.pack('<I', len(cls_s)) + cls_s)
        forged.write(_DICT + struct.pack('<I', 0))
        with pytest.raises(mscs.MSCSecurityError, match="downgrade"):
            mscs.loads(forged.getvalue(), hmac_key=self.KEY)

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


# ═══════════════════════════════════════════════════════════════════
# PERITAJE 2026-07-23 — anclas de regresión
# ═══════════════════════════════════════════════════════════════════

def _forge_obj(cls_path: str, state: dict) -> bytes:
    """Forja un blob v2 con un único OBJ(cls_path, state). El `state` es lo que
    un atacante controla: un dict serializado con el encoder real (sin refs
    internas, así que la numeración de refs del OBJ es irrelevante aquí)."""
    buf = io.BytesIO()
    buf.write(MAGIC + VERSION + b'\x00')
    enc = mscs._core._Encoder(buf)
    enc.buf.write(_OBJ)
    enc._encode_str(cls_path)
    enc.encode(state)
    return buf.getvalue()


class TestDataclassKeyFiltering:
    """Ancla del hallazgo ALTA: la rama dataclass del decoder no filtraba las
    claves del state contra dataclasses.fields(cls), permitiendo clobbering de
    __dict__ e invocación de setters de @property con datos del atacante.
    Gemelo del fix de slots (ver TestSlotsRoundtrip)."""

    def test_dataclass_roundtrip_still_works(self):
        @mscs.register
        @dataclasses.dataclass
        class DCOk:
            x: int = 0
            y: int = 0

        result = mscs.loads(mscs.dumps(DCOk(1, 2)))
        assert result.x == 1 and result.y == 2

    def test_forged_dict_key_on_dataclass_fails_closed(self):
        # Clave espuria '__dict__' -> reemplazo del dict de instancia entero.
        @mscs.register
        @dataclasses.dataclass
        class DCClob:
            x: int = 0
            y: int = 0

        blob = _forge_obj(_class_key(DCClob),
                          {'__dict__': {'is_admin': True, 'injected': 'x'}})
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob)

    def test_forged_property_key_on_dataclass_fails_closed(self):
        # Clave que coincide con una @property (no field) -> invocaría su setter
        # fuera del modelo documentado ("solo __setstate__ se ejecuta").
        side = []

        @mscs.register
        @dataclasses.dataclass
        class DCProp:
            _v: bool = False

            @property
            def admin(self):
                return self._v

            @admin.setter
            def admin(self, value):
                side.append(value)
                object.__setattr__(self, '_v', value)

        blob = _forge_obj(_class_key(DCProp), {'admin': True})
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob)
        assert side == []  # el setter jamás se invocó

    def test_forged_nonfield_key_on_dataclass_fails_closed(self):
        # Cualquier clave que no sea un field declarado se rechaza (fail-closed).
        @mscs.register
        @dataclasses.dataclass
        class DCExtra:
            x: int = 0

        blob = _forge_obj(_class_key(DCExtra), {'x': 1, 'surprise': 999})
        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(blob)

    def test_frozen_dataclass_roundtrip_still_works(self):
        # El fix no debe romper el round-trip de frozen (object.__setattr__).
        @mscs.register
        @dataclasses.dataclass(frozen=True)
        class DCFrozen:
            a: int = 0
            b: str = ""

        result = mscs.loads(mscs.dumps(DCFrozen(7, "ok")))
        assert result.a == 7 and result.b == "ok"


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestNdarrayShapeLimit:
    """Ancla del hallazgo ALTA: el conteo de dimensiones del shape no pasaba por
    el cap MAX_COLLECTION/MAX_NDARRAY_DIMS -> DoS de amplificación de memoria
    (el .split('x') materializaba millones de substrings antes de que numpy
    rechazara). El cap corta O(1) sin materializar."""

    def _forge_ndarray(self, meta: str, raw: bytes = b'\x00') -> bytes:
        buf = io.BytesIO()
        buf.write(MAGIC + VERSION + b'\x00')
        enc = mscs._core._Encoder(buf)
        enc.buf.write(_NDARRAY)
        enc._encode_str(meta)
        enc.buf.write(struct.pack('<I', len(raw)))
        enc.buf.write(raw)
        return buf.getvalue()

    def test_excessive_shape_dimensions_rejected(self):
        # 100 dimensiones (> el cap): rechazo por el guard propio, con su
        # mensaje distintivo — NO por el error de numpy (que llegaría después
        # de materializar el split).
        meta = "uint8|" + "x".join(["1"] * 100)
        blob = self._forge_ndarray(meta)
        with pytest.raises(mscs.MSCDecodeError, match="dimensiones"):
            mscs.loads(blob)

    def test_huge_shape_dimensions_rejected(self):
        # 5M dimensiones: antes materializaba ~250MB de substrings; ahora se
        # corta antes del split. Solo verifica el rechazo (sin medir memoria).
        meta = "uint8|" + "x".join(["1"] * 5_000_000)
        blob = self._forge_ndarray(meta)
        with pytest.raises(mscs.MSCDecodeError, match="dimensiones"):
            mscs.loads(blob)

    def test_huge_single_dimension_token_rejected(self):
        # Gemelo del cap de dimensiones: UN solo token gigante (cero 'x' ->
        # ndim=1, pasa el cap de conteo) fuerza int() O(n^2) sobre el token —
        # DoS de CPU. La longitud del token se acota antes de int(), sin
        # depender de sys.int_max_str_digits (mutable global, ausente en
        # Python <3.11). Desactivamos esa mitigación del intérprete para
        # probar el cap de mscs, no el de CPython.
        old = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(0)
            meta = "uint8|" + "9" * 10000
            blob = self._forge_ndarray(meta)
            with pytest.raises(mscs.MSCDecodeError, match="d[íi]gitos"):
                mscs.loads(blob)
        finally:
            sys.set_int_max_str_digits(old)

    def test_normal_ndarray_still_works(self):
        arr = np.arange(24, dtype='float32').reshape(2, 3, 4)
        result = mscs.loads(mscs.dumps(arr))
        assert result.shape == (2, 3, 4)
        assert np.array_equal(result, arr)

    def test_moderate_multidim_ndarray_still_works(self):
        # Un array con varias dimensiones (bajo el cap) hace round-trip.
        arr = np.zeros((1,) * 16, dtype='uint8')
        result = mscs.loads(mscs.dumps(arr))
        assert result.shape == (1,) * 16


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTensorShapeLimit:
    """Gemelo de TestNdarrayShapeLimit para el tag _TENSOR."""

    def _forge_tensor(self, meta: str, raw: bytes = b'\x00') -> bytes:
        buf = io.BytesIO()
        buf.write(MAGIC + VERSION + b'\x00')
        enc = mscs._core._Encoder(buf)
        enc.buf.write(_TENSOR)
        enc._encode_str(meta)
        enc.buf.write(struct.pack('<I', len(raw)))
        enc.buf.write(raw)
        return buf.getvalue()

    def test_excessive_tensor_shape_dimensions_rejected(self):
        meta = "float32|" + "x".join(["1"] * 100)
        blob = self._forge_tensor(meta)
        with pytest.raises(mscs.MSCDecodeError, match="dimensiones"):
            mscs.loads(blob)

    def test_normal_tensor_still_works(self):
        t = torch.zeros(2, 3, 4)
        result = mscs.loads(mscs.dumps(t))
        assert tuple(result.shape) == (2, 3, 4)


class TestTrailingBytesV1:
    """Ancla del hallazgo MEDIA: la rama v1 de loads() no validaba trailing
    bytes (la v2 sí). Gemelo de TestTrailingBytes para el formato v1."""

    def _v1_int(self, value: int = 42) -> bytes:
        return b'MSCS\x01' + _INT + struct.pack('<H', 1) + bytes([value])

    def test_v1_trailing_bytes_rejected(self):
        with pytest.raises(mscs.MSCDecodeError, match="[Tt]railing"):
            mscs.loads(self._v1_int() + b'\xde\xad', strict=False)

    def test_v1_smuggling_rejected(self):
        # Dos mensajes v1 concatenados: el segundo no debe pasar como basura.
        with pytest.raises(mscs.MSCDecodeError, match="[Tt]railing"):
            mscs.loads(self._v1_int(1) + self._v1_int(2), strict=False)

    def test_v1_clean_still_works(self):
        assert mscs.loads(self._v1_int(42), strict=False) == 42


class TestControlParity:
    """Meta-patrón del proyecto: un control aplicado a una rama pero no a su
    gemela. Estos tests fijan que los controles clave se aplican en TODAS las
    ramas paralelas, no solo en la que disparó el bug histórico."""

    def test_trailing_bytes_rejected_in_v1_and_v2(self):
        # v2
        with pytest.raises(mscs.MSCDecodeError, match="[Tt]railing"):
            mscs.loads(mscs.dumps(42) + b'\x00', strict=False)
        # v1
        v1 = b'MSCS\x01' + _INT + struct.pack('<H', 1) + b'\x2a'
        with pytest.raises(mscs.MSCDecodeError, match="[Tt]railing"):
            mscs.loads(v1 + b'\x00', strict=False)

    def test_spurious_dict_key_rejected_in_dataclass_and_slots(self):
        # slots puros (rama ya endurecida en 2026-07-13)
        @mscs.register
        class ParitySlots:
            __slots__ = ('a',)

        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(_forge_obj(_class_key(ParitySlots), {'__dict__': {'evil': 1}}))

        # dataclass (rama gemela endurecida en 2026-07-23)
        @mscs.register
        @dataclasses.dataclass
        class ParityDC:
            a: int = 0

        with pytest.raises(mscs.MSCDecodeError):
            mscs.loads(_forge_obj(_class_key(ParityDC), {'__dict__': {'evil': 1}}))

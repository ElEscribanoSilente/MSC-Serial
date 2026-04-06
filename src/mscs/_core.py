"""
MSC Serial v2.2
===============
Reemplazo personal y seguro de pickle.

Soporta: dict, list, tuple, set, frozenset, str, int, float, complex,
         bool, None, bytes, bytearray, datetime, date, time, timedelta,
         Decimal, UUID, Path, Enum, numpy arrays, torch.Tensor,
         dataclasses, objetos con __slots__, objetos custom registrados,
         referencias circulares.

API compatible con pickle:
  msc.dump(obj, file)
  msc.load(file)
  msc.dumps(obj) -> bytes
  msc.loads(data) -> obj
  msc.dump_compressed(obj, file)
  msc.load_compressed(file)

Extras:
  msc.register(cls)           # registrar clase segura para deserialización
  msc.register_alias(old, c)  # alias para clases renombradas (backward compat)
  msc.register_module(mod)    # registrar todas las clases de un módulo
  msc.inspect(data) -> dict   # metadata sin deserializar
  msc.benchmark(obj) -> dict  # medir rendimiento
  msc.copy(obj) -> obj        # deep copy via round-trip

Seguridad:
  - No ejecuta código arbitrario al deserializar
  - Solo reconstruye objetos de clases explícitamente registradas
  - Límites de profundidad y tamaño configurables
  - Formato auditable con magic bytes + versión
  - Sin importlib dinámico en deserialización
  - Validación de numpy dtypes contra whitelist
  - Protección anti zip-bomb en load_compressed
  - NOTA: la seguridad del registry depende de que solo se registren
    clases confiables. __setstate__ de clases registradas SE EJECUTA.
  - NOTA: ref tracking usa id(obj); como el encoder mantiene refs a
    todos los objetos serializados, los IDs no se reutilizan durante
    una sola llamada a encode().

Changelog v2.3.0:
  - ADD: HMAC-SHA256 autenticación criptográfica (hmac_key= en dumps/loads)
  - ADD: Protección anti-downgrade (payload sin HMAC + clave = rechazado)
  - ADD: Validación de trailing bytes (basura al final del payload = error)
  - ADD: MAX_INT_BYTES=8192 — previene CPU exhaustion con ints enormes
  - ADD: Rechazo de null bytes en Path (MSCSecurityError)
  - ADD: Thread-safe registry con threading.Lock
  - ADD: Test suite con pytest (169 tests) + fuzzing con Hypothesis
  - FIX: Refs de tuple/frozenset desincronizadas encoder↔decoder
  - FIX: id() reuse de dicts temporales en OBJ anidados (dataclasses)
  - FIX: OBJ decoder no reservaba ref slot antes de decodear hijos
  - FIX: Imports de numpy/torch movidos a top-level (elimina try/except
         repetido en cada llamada a encode/decode)
  - FIX: load_compressed usa read con límite (no lee archivo completo)
  - Retrocompatible con payloads v2.2, v2.1, v2.0 y v1.0

Changelog v2.2:
  - FIX: timedelta usa tag dedicado _TIMEDELTA2 (0x19) — elimina la
         ambiguedad heuristica entre formatos v2.0 y v2.1
  - FIX: _encode_str ahora valida longitud contra MAX_STRING
  - FIX: load_compressed protegido contra zip bombs (valida tamaño
         comprimido Y descomprimido)
  - ADD: soporte nativo torch.Tensor (tag 0x18) — serializa dtype,
         shape, requires_grad sin conversión manual a numpy
  - ADD: register_alias(old_path, cls) para backward-compat con
         checkpoints de clases renombradas/movidas
  - Retrocompatible con payloads v2.1, v2.0 y v1.0

Changelog v2.1:
  - FIX: timedelta ahora codifica days/seconds/microseconds por separado
         (v2.0 perdía precisión al usar total_seconds() como float)
  - FIX: validación de numpy dtype contra whitelist de tipos seguros
  - ADD: soporte UUID nativo
  - ADD: soporte pathlib.Path nativo
  - ADD: register_module() para registro masivo de clases
  - ADD: copy() — deep copy vía serialización round-trip
  - ADD: inspect() ahora muestra nombre del tag raíz
  - ADD: contexto de ruta en errores de decode (breadcrumbs)
  - Retrocompatible con payloads v2.0 y v1.0

Changelog v2.0:
  - Registry de clases seguras (elimina importlib dinámico)
  - Soporte: complex, frozenset, datetime/date/time/timedelta, Decimal, Enum
  - Detección y manejo de referencias circulares
  - Límites de profundidad y tamaño máximo
  - Streaming encode/decode para objetos grandes
  - Mejor manejo de errores con excepciones tipadas
  - Soporte bytearray nativo
  - Benchmark integrado
  - Validación de integridad con CRC32 opcional
"""

import struct
import io
import zlib
import hmac
import hashlib
import threading
import inspect as _inspect_mod
import dataclasses
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID
from typing import Any, Type, Dict, Optional, Set, List

# ─────────────── OPTIONAL DEPENDENCIES (top-level) ───────────────

try:
    import numpy as _np
except ImportError:
    _np = None

try:
    import torch as _torch
except ImportError:
    _torch = None

__version__ = "2.3.0"
__all__ = [
    "dump", "load", "dumps", "loads",
    "dump_compressed", "load_compressed",
    "register", "register_alias", "register_module",
    "inspect", "benchmark", "copy",
    "MSCError", "MSCEncodeError", "MSCDecodeError", "MSCSecurityError",
    "MAX_INT_BYTES",
]

# ─────────────────────── EXCEPTIONS ───────────────────────────────

class MSCError(Exception):
    """Base para errores de MSC Serial."""

class MSCEncodeError(MSCError):
    """Error durante serialización."""

class MSCDecodeError(MSCError):
    """Error durante deserialización."""

class MSCSecurityError(MSCError):
    """Intento de deserializar clase no registrada."""

# ─────────────────────── TYPE TAGS ────────────────────────────────

_NONE       = b'\x00'
_BOOL       = b'\x01'
_INT        = b'\x02'
_FLOAT      = b'\x03'
_STR        = b'\x04'
_BYTES      = b'\x05'
_LIST       = b'\x06'
_TUPLE      = b'\x07'
_DICT       = b'\x08'
_SET        = b'\x09'
_NDARRAY    = b'\x0A'
_OBJ        = b'\x0B'
_COMPLEX    = b'\x0C'
_FROZENSET  = b'\x0D'
_DATETIME   = b'\x0E'
_DATE       = b'\x0F'
_TIME       = b'\x10'
_TIMEDELTA  = b'\x11'
_DECIMAL    = b'\x12'
_ENUM       = b'\x13'
_BYTEARRAY  = b'\x14'
_REF        = b'\x15'
_UUID       = b'\x16'
_PATH       = b'\x17'
_TENSOR     = b'\x18'
_TIMEDELTA2 = b'\x19'  # v2.2: timedelta sin ambiguedad

_TAG_NAMES: Dict[int, str] = {
    0x00: 'None',    0x01: 'bool',      0x02: 'int',       0x03: 'float',
    0x04: 'str',     0x05: 'bytes',     0x06: 'list',      0x07: 'tuple',
    0x08: 'dict',    0x09: 'set',       0x0A: 'ndarray',   0x0B: 'object',
    0x0C: 'complex', 0x0D: 'frozenset', 0x0E: 'datetime',  0x0F: 'date',
    0x10: 'time',    0x11: 'timedelta', 0x12: 'Decimal',   0x13: 'Enum',
    0x14: 'bytearray', 0x15: 'ref',    0x16: 'UUID',      0x17: 'Path',
    0x18: 'tensor',  0x19: 'timedelta2',
}

MAGIC   = b'MSCS'
VERSION = b'\x02'  # formato binario sigue siendo v2; cambios son aditivos

# ─────────────────────── LIMITS ───────────────────────────────────

MAX_DEPTH       = 256
MAX_SIZE        = 512 * 1024 * 1024  # 512 MB
MAX_COMPRESSED  = 512 * 1024 * 1024  # 512 MB (compressed input limit, anti zip-bomb)
MAX_COLLECTION  = 10_000_000
MAX_STRING      = 100 * 1024 * 1024  # 100 MB
MAX_INT_BYTES   = 8192              # ~19,700 dígitos decimales

_HMAC_DIGEST_SIZE = 32  # SHA-256

# ─────────────────── NUMPY DTYPE WHITELIST ────────────────────────

_SAFE_NUMPY_DTYPES: Set[str] = {
    # Enteros
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    # Flotantes
    'float16', 'float32', 'float64', 'float128',
    # Complejos
    'complex64', 'complex128', 'complex256',
    # Bool y bytes
    'bool', 'bool_',
    # Strings fijos
    # (aceptamos S<n> y U<n> por regex abajo)
}


import re as _re
_RE_DTYPE_SHORT = _re.compile(r'[fiubcUSV]\d+')
_RE_DTYPE_LONG  = _re.compile(r'(int|uint|float|complex|bool)\d*_?')


def _is_safe_dtype(dtype_str: str) -> bool:
    """Valida que un dtype string sea seguro (no structured/object/void)."""
    clean = dtype_str.strip()
    # Rechazar explícitamente tipos peligrosos (case-insensitive)
    if clean.lower() in ('object', 'o', 'void', 'v'):
        return False
    clean = clean.lower()
    # Tipos simples directos
    if clean in _SAFE_NUMPY_DTYPES:
        return True
    # Con prefijo de byteorder: <f4, >i8, =f8, |b1, etc.
    if len(clean) > 1 and clean[0] in '<>=|!':
        clean = clean[1:]
    # Numpy shorthand: f4, f8, i4, i8, u2, b1, c8, c16, etc.
    if _RE_DTYPE_SHORT.fullmatch(clean):
        # Rechazar V (void) — ya cubierto arriba
        if clean[0] == 'V':
            return False
        return True
    # Nombre completo con bitsize: float32, int64, etc.
    if _RE_DTYPE_LONG.fullmatch(clean):
        return True
    return False


# ─────────────────────── REGISTRY ─────────────────────────────────

_registry: Dict[str, Type] = {}
_registry_lock = threading.Lock()


def _class_key(cls: Type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register(cls: Type) -> Type:
    """
    Registra una clase como segura para deserialización.
    Puede usarse como decorador:

        @msc.register
        @dataclass
        class MiObjeto:
            x: float
            y: float

    NOTA: __setstate__ de clases registradas SE EJECUTA durante
    deserialización. Solo registra clases confiables.
    """
    key = _class_key(cls)
    with _registry_lock:
        _registry[key] = cls
    return cls


def register_module(module) -> List[Type]:
    """
    Registra todas las clases definidas en un módulo.
    Retorna lista de clases registradas.

        import my_models
        msc.register_module(my_models)

    NOTA: __setstate__ de clases registradas SE EJECUTA durante
    deserialización. Solo registra módulos confiables.
    """
    registered = []
    for name, obj in _inspect_mod.getmembers(module, _inspect_mod.isclass):
        # Solo clases definidas EN el módulo (no importadas de stdlib, etc.)
        if obj.__module__ == module.__name__:
            register(obj)
            registered.append(obj)
    return registered


def register_alias(alias: str, cls: Type) -> None:
    """
    Registra un alias para una clase (backward-compat con checkpoints viejos).

        # La clase se renombro de OldName a NewName
        msc.register_alias("my_module.OldName", NewName)
    """
    with _registry_lock:
        _registry[alias] = cls


def _is_registered(class_path: str) -> bool:
    return class_path in _registry


def _get_registered(class_path: str) -> Type:
    if class_path not in _registry:
        raise MSCSecurityError(
            f"Clase no registrada: {class_path!r}. "
            f"Usa msc.register({class_path.rsplit('.', 1)[-1]}) antes de deserializar."
        )
    return _registry[class_path]


# ──────────────────────── ENCODER ─────────────────────────────────

class _Encoder:
    __slots__ = ('buf', 'depth', 'refs', 'ref_counter', 'use_refs', '_pinned')

    def __init__(self, buf: io.BytesIO, *, use_refs: bool = True):
        self.buf = buf
        self.depth = 0
        self.refs: Dict[int, int] = {}   # id(obj) -> ref_id
        self.ref_counter = 0
        self.use_refs = use_refs
        self._pinned: list = []  # prevent GC of temporary objects (id reuse)

    def encode(self, obj: Any):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MSCEncodeError(
                f"Profundidad máxima excedida ({MAX_DEPTH}). "
                f"¿Referencia circular no detectada?"
            )
        try:
            self._encode(obj)
        finally:
            self.depth -= 1

    def _assign_ref(self, obj: Any) -> bool:
        """Retorna True si el objeto ya fue serializado (escribe REF)."""
        if not self.use_refs:
            return False
        oid = id(obj)
        if oid in self.refs:
            self.buf.write(_REF)
            self.buf.write(struct.pack('<I', self.refs[oid]))
            return True
        self.refs[oid] = self.ref_counter
        self.ref_counter += 1
        return False

    def _write_length(self, n: int, max_val: int = MAX_COLLECTION, label: str = "colección"):
        if n > max_val:
            raise MSCEncodeError(f"Tamaño de {label} excede límite: {n:,} > {max_val:,}")
        self.buf.write(struct.pack('<I', n))

    def _encode(self, obj: Any):
        buf = self.buf

        # ── Singletons y primitivos inmutables (sin ref tracking) ──

        if obj is None:
            buf.write(_NONE)
            return

        if isinstance(obj, bool):  # antes de int
            buf.write(_BOOL)
            buf.write(b'\x01' if obj else b'\x00')
            return

        if isinstance(obj, int):
            buf.write(_INT)
            if obj == 0:
                buf.write(struct.pack('<H', 1))
                buf.write(b'\x00')
            else:
                n_bytes = (obj.bit_length() + 8) // 8
                if n_bytes > MAX_INT_BYTES:
                    raise MSCEncodeError(
                        f"Entero demasiado grande: {n_bytes:,} bytes "
                        f"(límite: {MAX_INT_BYTES:,})"
                    )
                raw = obj.to_bytes(n_bytes, 'little', signed=True)
                buf.write(struct.pack('<H', len(raw)))
                buf.write(raw)
            return

        if isinstance(obj, float):
            buf.write(_FLOAT)
            buf.write(struct.pack('<d', obj))
            return

        if isinstance(obj, complex):
            buf.write(_COMPLEX)
            buf.write(struct.pack('<dd', obj.real, obj.imag))
            return

        # ── Strings y bytes (ref tracking para grandes) ──

        if isinstance(obj, str):
            if self._assign_ref(obj):
                return
            buf.write(_STR)
            raw = obj.encode('utf-8')
            self._write_length(len(raw), MAX_STRING, "string")
            buf.write(raw)
            return

        if isinstance(obj, bytearray):
            if self._assign_ref(obj):
                return
            buf.write(_BYTEARRAY)
            self._write_length(len(obj), MAX_STRING, "bytearray")
            buf.write(bytes(obj))
            return

        if isinstance(obj, bytes):
            if self._assign_ref(obj):
                return
            buf.write(_BYTES)
            self._write_length(len(obj), MAX_STRING, "bytes")
            buf.write(obj)
            return

        # ── UUID ──

        if isinstance(obj, UUID):
            buf.write(_UUID)
            buf.write(obj.bytes)  # siempre 16 bytes
            return

        # ── Path ──

        if isinstance(obj, Path):
            buf.write(_PATH)
            raw = str(obj).encode('utf-8')
            self._write_length(len(raw), MAX_STRING, "path")
            buf.write(raw)
            return

        # ── Tipos temporales ──

        if isinstance(obj, datetime):
            buf.write(_DATETIME)
            ts = obj.isoformat()
            raw = ts.encode('utf-8')
            buf.write(struct.pack('<H', len(raw)))
            buf.write(raw)
            return

        if isinstance(obj, date):
            buf.write(_DATE)
            buf.write(struct.pack('<HBB', obj.year, obj.month, obj.day))
            return

        if isinstance(obj, time):
            buf.write(_TIME)
            ts = obj.isoformat()
            raw = ts.encode('utf-8')
            buf.write(struct.pack('<H', len(raw)))
            buf.write(raw)
            return

        if isinstance(obj, timedelta):
            # v2.2: tag dedicado sin ambiguedad con v2.0
            buf.write(_TIMEDELTA2)
            buf.write(struct.pack('<iiI', obj.days, obj.seconds, obj.microseconds))
            return

        if isinstance(obj, Decimal):
            buf.write(_DECIMAL)
            raw = str(obj).encode('utf-8')
            buf.write(struct.pack('<H', len(raw)))
            buf.write(raw)
            return

        # ── Enum ──

        if isinstance(obj, Enum):
            buf.write(_ENUM)
            cls_path = _class_key(type(obj))
            self._encode_str(cls_path)
            self.encode(obj.value)
            return

        # ── Colecciones (con ref tracking) ──

        if isinstance(obj, list):
            if self._assign_ref(obj):
                return
            buf.write(_LIST)
            self._write_length(len(obj))
            for item in obj:
                self.encode(item)
            return

        if isinstance(obj, tuple):
            if self._assign_ref(obj):
                return
            buf.write(_TUPLE)
            self._write_length(len(obj))
            for item in obj:
                self.encode(item)
            return

        if isinstance(obj, frozenset):
            if self._assign_ref(obj):
                return
            buf.write(_FROZENSET)
            items = sorted(obj, key=repr)
            self._write_length(len(items))
            for item in items:
                self.encode(item)
            return

        if isinstance(obj, set):
            if self._assign_ref(obj):
                return
            buf.write(_SET)
            items = sorted(obj, key=repr)
            self._write_length(len(items))
            for item in items:
                self.encode(item)
            return

        if isinstance(obj, dict):
            if self._assign_ref(obj):
                return
            buf.write(_DICT)
            self._write_length(len(obj))
            for k, v in obj.items():
                self.encode(k)
                self.encode(v)
            return

        # ── Numpy ──

        if _np is not None and isinstance(obj, _np.ndarray):
            if self._assign_ref(obj):
                return
            dtype_s = str(obj.dtype)
            if not _is_safe_dtype(dtype_s):
                raise MSCEncodeError(
                    f"numpy dtype no permitido: {dtype_s!r}. "
                    f"Solo se permiten dtypes numéricos simples."
                )
            buf.write(_NDARRAY)
            shape_s = 'x'.join(map(str, obj.shape)) if obj.shape else ''
            meta = f"{dtype_s}|{shape_s}"
            self._encode_str(meta)
            raw = obj.tobytes()
            self._write_length(len(raw), MAX_SIZE, "ndarray data")
            buf.write(raw)
            return

        # ── PyTorch Tensor ──

        if _torch is not None and isinstance(obj, _torch.Tensor):
            if self._assign_ref(obj):
                return
            t = obj.detach().cpu().contiguous()
            arr = t.numpy()
            dtype_s = str(arr.dtype)
            if not _is_safe_dtype(dtype_s):
                raise MSCEncodeError(
                    f"torch dtype no permitido: {obj.dtype} (numpy: {dtype_s!r})"
                )
            buf.write(_TENSOR)
            shape_s = 'x'.join(map(str, arr.shape)) if arr.shape else ''
            requires_grad = '1' if obj.requires_grad else '0'
            meta = f"{dtype_s}|{shape_s}|{requires_grad}"
            self._encode_str(meta)
            raw = arr.tobytes()
            self._write_length(len(raw), MAX_SIZE, "tensor data")
            buf.write(raw)
            return

        # ── Objeto registrado ──

        if self._assign_ref(obj):
            return

        buf.write(_OBJ)
        cls_path = _class_key(type(obj))
        self._encode_str(cls_path)

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            state = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        elif hasattr(obj, '__slots__') and not hasattr(obj, '__dict__'):
            state = {}
            for cls in type(obj).__mro__:
                for s in getattr(cls, '__slots__', ()):
                    if hasattr(obj, s) and s not in state:
                        state[s] = getattr(obj, s)
        elif '__getstate__' in type(obj).__dict__ or any(
            '__getstate__' in c.__dict__ for c in type(obj).__mro__[:-1]
            if c is not object
        ):
            state = obj.__getstate__()
        elif hasattr(obj, '__dict__'):
            state = obj.__dict__
        else:
            raise MSCEncodeError(f"No se puede serializar: {type(obj)!r}")

        # Pin temporary state dicts to prevent id() reuse. CPython may
        # reuse the id of a temporary dict after it goes out of scope,
        # causing false ref hits on subsequent OBJ state dicts.
        self._pinned.append(state)
        self.encode(state)

    def _encode_str(self, s: str):
        """Encode string directamente sin ref tracking (para metadata interna)."""
        self.buf.write(_STR)
        raw = s.encode('utf-8')
        if len(raw) > MAX_STRING:
            raise MSCEncodeError(f"Metadata string excede limite: {len(raw):,} > {MAX_STRING:,}")
        self.buf.write(struct.pack('<I', len(raw)))
        self.buf.write(raw)


# ──────────────────────── DECODER ─────────────────────────────────

class _Decoder:
    __slots__ = ('buf', 'depth', 'refs', 'strict', 'path')

    def __init__(self, buf: io.BytesIO, *, strict: bool = True):
        self.buf = buf
        self.depth = 0
        self.refs: Dict[int, Any] = {}
        self.strict = strict
        self.path: List[str] = []  # breadcrumbs para errores

    def decode(self) -> Any:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MSCDecodeError(
                f"Profundidad máxima excedida ({MAX_DEPTH}) en {self._path_str()}"
            )
        try:
            return self._decode()
        except (MSCDecodeError, MSCSecurityError):
            raise
        except Exception as e:
            raise MSCDecodeError(
                f"Error en {self._path_str()}: {e}"
            ) from e
        finally:
            self.depth -= 1

    def _path_str(self) -> str:
        return ' → '.join(self.path) if self.path else '<root>'

    def _read(self, n: int) -> bytes:
        data = self.buf.read(n)
        if len(data) < n:
            raise MSCDecodeError(
                f"Fin inesperado en {self._path_str()}: "
                f"esperaba {n} bytes, obtuvo {len(data)}"
            )
        return data

    def _read_length(self, max_val: int = MAX_COLLECTION) -> int:
        n = struct.unpack('<I', self._read(4))[0]
        if n > max_val:
            raise MSCDecodeError(
                f"Tamaño excede límite en {self._path_str()}: {n:,} > {max_val:,}"
            )
        return n

    def _store_ref(self, obj: Any) -> Any:
        self.refs[len(self.refs)] = obj
        return obj

    def _decode(self) -> Any:
        tag = self._read(1)

        if tag == _NONE:
            return None

        if tag == _BOOL:
            return self._read(1) == b'\x01'

        if tag == _INT:
            n = struct.unpack('<H', self._read(2))[0]
            if n > MAX_INT_BYTES:
                raise MSCDecodeError(
                    f"Entero demasiado grande: {n:,} bytes "
                    f"(límite: {MAX_INT_BYTES:,}) en {self._path_str()}"
                )
            return int.from_bytes(self._read(n), 'little', signed=True)

        if tag == _FLOAT:
            return struct.unpack('<d', self._read(8))[0]

        if tag == _COMPLEX:
            r, i = struct.unpack('<dd', self._read(16))
            return complex(r, i)

        if tag == _REF:
            ref_id = struct.unpack('<I', self._read(4))[0]
            if ref_id not in self.refs:
                raise MSCDecodeError(
                    f"Referencia inválida: {ref_id} en {self._path_str()}"
                )
            return self.refs[ref_id]

        if tag == _STR:
            n = self._read_length(MAX_STRING)
            s = self._read(n).decode('utf-8')
            return self._store_ref(s)

        if tag == _BYTES:
            n = self._read_length(MAX_STRING)
            b = self._read(n)
            return self._store_ref(b)

        if tag == _BYTEARRAY:
            n = self._read_length(MAX_STRING)
            ba = bytearray(self._read(n))
            return self._store_ref(ba)

        if tag == _UUID:
            raw = self._read(16)
            return UUID(bytes=raw)

        if tag == _PATH:
            n = self._read_length(MAX_STRING)
            s = self._read(n).decode('utf-8')
            if '\x00' in s:
                raise MSCSecurityError(
                    f"Path contiene null bytes — posible ataque de inyección"
                )
            return Path(s)

        if tag == _DATETIME:
            n = struct.unpack('<H', self._read(2))[0]
            s = self._read(n).decode('utf-8')
            return datetime.fromisoformat(s)

        if tag == _DATE:
            y, m, d = struct.unpack('<HBB', self._read(4))
            return date(y, m, d)

        if tag == _TIME:
            n = struct.unpack('<H', self._read(2))[0]
            s = self._read(n).decode('utf-8')
            return time.fromisoformat(s)

        if tag == _TIMEDELTA2:
            # v2.2: tag dedicado, sin ambiguedad
            days, secs, us = struct.unpack('<iiI', self._read(12))
            return timedelta(days=days, seconds=secs, microseconds=us)

        if tag == _TIMEDELTA:
            # Legacy: payloads v2.0/v2.1 usaban el mismo tag para 2 formatos.
            # Heuristica: v2.1 = (days:i4, seconds:i4, microseconds:U4)
            #             v2.0 = (days:i4, total_seconds:f8)
            raw12 = self._read(12)
            days_21, secs_21, us_21 = struct.unpack('<iiI', raw12)
            if 0 <= secs_21 < 86400 and us_21 < 1_000_000:
                return timedelta(days=days_21, seconds=secs_21, microseconds=us_21)
            # Fallback v2.0
            _days_20, total_20 = struct.unpack('<id', raw12)
            return timedelta(seconds=total_20)

        if tag == _DECIMAL:
            n = struct.unpack('<H', self._read(2))[0]
            s = self._read(n).decode('utf-8')
            return Decimal(s)

        if tag == _ENUM:
            class_path = self._decode_str()
            self.path.append(f'Enum({class_path})')
            value = self.decode()
            self.path.pop()
            if self.strict:
                cls = _get_registered(class_path)
                return cls(value)
            else:
                return {'__enum__': class_path, '__value__': value}

        if tag == _LIST:
            n = self._read_length()
            result = []
            self._store_ref(result)
            for i in range(n):
                self.path.append(f'[{i}]')
                result.append(self.decode())
                self.path.pop()
            return result

        if tag == _TUPLE:
            n = self._read_length()
            # Reserve ref slot BEFORE decoding children (matches encoder order)
            ref_id = len(self.refs)
            self.refs[ref_id] = None  # placeholder
            items = []
            for i in range(n):
                self.path.append(f'({i})')
                items.append(self.decode())
                self.path.pop()
            t = tuple(items)
            self.refs[ref_id] = t  # fill placeholder
            return t

        if tag == _FROZENSET:
            n = self._read_length()
            # Reserve ref slot BEFORE decoding children (matches encoder order)
            ref_id = len(self.refs)
            self.refs[ref_id] = None  # placeholder
            items = frozenset(self.decode() for _ in range(n))
            self.refs[ref_id] = items  # fill placeholder
            return items

        if tag == _SET:
            n = self._read_length()
            result = set()
            self._store_ref(result)
            result.update(self.decode() for _ in range(n))
            return result

        if tag == _DICT:
            n = self._read_length()
            result = {}
            self._store_ref(result)
            for _ in range(n):
                k = self.decode()
                self.path.append(f'.{k!r}' if isinstance(k, str) else f'[{k!r}]')
                v = self.decode()
                self.path.pop()
                result[k] = v
            return result

        if tag == _NDARRAY:
            if _np is None:
                raise MSCDecodeError("numpy requerido para deserializar arrays")
            meta = self._decode_str()
            dtype_str, shape_str = meta.split('|')
            if not _is_safe_dtype(dtype_str):
                raise MSCSecurityError(
                    f"numpy dtype no permitido en deserialización: {dtype_str!r}"
                )
            shape = tuple(int(x) for x in shape_str.split('x')) if shape_str else ()
            n = self._read_length(MAX_SIZE)
            raw = self._read(n)
            arr = _np.frombuffer(raw, dtype=_np.dtype(dtype_str)).copy().reshape(shape)
            return self._store_ref(arr)

        if tag == _TENSOR:
            if _torch is None or _np is None:
                raise MSCDecodeError("torch y numpy requeridos para deserializar tensores")
            meta = self._decode_str()
            parts = meta.split('|')
            dtype_str, shape_str = parts[0], parts[1]
            requires_grad = parts[2] == '1' if len(parts) > 2 else False
            if not _is_safe_dtype(dtype_str):
                raise MSCSecurityError(
                    f"tensor dtype no permitido: {dtype_str!r}"
                )
            shape = tuple(int(x) for x in shape_str.split('x')) if shape_str else ()
            n = self._read_length(MAX_SIZE)
            raw = self._read(n)
            arr = _np.frombuffer(raw, dtype=_np.dtype(dtype_str)).copy().reshape(shape)
            t = _torch.from_numpy(arr)
            if requires_grad:
                t = t.requires_grad_(True)
            return self._store_ref(t)

        if tag == _OBJ:
            # Reserve ref slot BEFORE decoding children (matches encoder order)
            ref_id = len(self.refs)
            self.refs[ref_id] = None  # placeholder

            class_path = self._decode_str()
            self.path.append(class_path.rsplit('.', 1)[-1])
            state = self.decode()
            self.path.pop()

            if self.strict:
                cls = _get_registered(class_path)
            elif _is_registered(class_path):
                cls = _registry[class_path]
            else:
                fallback = {'__class__': class_path, '__state__': state}
                self.refs[ref_id] = fallback
                return fallback

            obj = cls.__new__(cls)
            if dataclasses.is_dataclass(cls):
                for k, v in state.items():
                    setattr(obj, k, v)
            elif hasattr(obj, '__slots__') and not hasattr(obj, '__dict__'):
                for k, v in state.items():
                    setattr(obj, k, v)
            elif '__setstate__' in type(obj).__dict__ or any(
                '__setstate__' in c.__dict__ for c in type(obj).__mro__[:-1]
                if c is not object
            ):
                obj.__setstate__(state)
            elif hasattr(obj, '__dict__'):
                obj.__dict__.update(state)
            else:
                for k, v in state.items():
                    setattr(obj, k, v)
            self.refs[ref_id] = obj
            return obj

        raise MSCDecodeError(
            f"Tag desconocido: {tag!r} en {self._path_str()}"
        )

    def _decode_str(self) -> str:
        """Decode string sin afectar ref counter (para metadata interna)."""
        tag = self._read(1)
        if tag != _STR:
            raise MSCDecodeError(
                f"Esperaba STR tag, obtuvo {tag!r} en {self._path_str()}"
            )
        n = self._read_length(MAX_STRING)
        return self._read(n).decode('utf-8')


# ──────────────────────── PUBLIC API ──────────────────────────────

def dumps(obj: Any, *, with_crc: bool = False,
          hmac_key: Optional[bytes] = None) -> bytes:
    """
    Serializa obj a bytes.

    with_crc: añade CRC32 para detectar corrupción accidental.
    hmac_key: si se proporciona, añade HMAC-SHA256 para autenticación
              criptográfica. Verificado en loads() con la misma clave.
              Mutuamente exclusivo con with_crc (HMAC es estrictamente
              superior).
    """
    if with_crc and hmac_key is not None:
        raise MSCEncodeError(
            "with_crc y hmac_key son mutuamente exclusivos. "
            "HMAC ya incluye protección de integridad."
        )
    buf = io.BytesIO()
    buf.write(MAGIC + VERSION)
    flags = 0x00
    if with_crc:
        flags |= 0x01
    if hmac_key is not None:
        flags |= 0x02
    buf.write(struct.pack('B', flags))
    enc = _Encoder(buf)
    enc.encode(obj)
    data = buf.getvalue()
    if with_crc:
        crc = zlib.crc32(data) & 0xFFFFFFFF
        data += struct.pack('<I', crc)
    if hmac_key is not None:
        mac = hmac.new(hmac_key, data, hashlib.sha256).digest()
        data += mac
    return data


def loads(data: bytes, *, strict: bool = True,
          hmac_key: Optional[bytes] = None) -> Any:
    """
    Deserializa bytes a objeto.

    strict=True: solo reconstruye clases registradas (lanza MSCSecurityError).
    strict=False: clases no registradas retornan dict fallback.
    hmac_key: si se proporciona, verifica HMAC-SHA256 antes de deserializar.
              Lanza MSCSecurityError si la firma no coincide.
    """
    if len(data) < 6:
        raise MSCDecodeError("Datos demasiado cortos para ser MSC Serial")
    if data[:4] != MAGIC:
        raise MSCDecodeError(f"Magic bytes inválidos: {data[:4]!r}")
    ver = data[4:5]

    if ver == b'\x01':
        # Retrocompatibilidad con v1.0 (sin flags, sin trailing validation)
        buf = io.BytesIO(data)
        buf.seek(5)
        dec = _Decoder(buf, strict=False)
        return dec.decode()

    if ver != VERSION:
        raise MSCDecodeError(f"Versión no soportada: {ver!r}")

    flags = data[5]
    has_crc = bool(flags & 0x01)
    has_hmac = bool(flags & 0x02)

    # ── Determinar dónde termina el payload real ──
    decode_data = data
    if has_hmac:
        if len(data) < 6 + _HMAC_DIGEST_SIZE:
            raise MSCDecodeError("Datos truncados: falta HMAC")
        stored_mac = data[-_HMAC_DIGEST_SIZE:]
        payload_for_mac = data[:-_HMAC_DIGEST_SIZE]
        if hmac_key is None:
            raise MSCSecurityError(
                "Payload firmado con HMAC pero no se proporcionó hmac_key"
            )
        computed_mac = hmac.new(hmac_key, payload_for_mac, hashlib.sha256).digest()
        if not hmac.compare_digest(stored_mac, computed_mac):
            raise MSCSecurityError("HMAC-SHA256 no coincide: payload manipulado o clave incorrecta")
        decode_data = payload_for_mac
    elif hmac_key is not None:
        raise MSCSecurityError(
            "Se proporcionó hmac_key pero el payload no tiene flag HMAC. "
            "Posible ataque de downgrade."
        )

    if has_crc:
        if len(decode_data) < 10:  # 6 header + 4 crc minimum
            raise MSCDecodeError("Datos truncados: falta CRC")
        crc_payload = decode_data[:-4]
        stored_crc = struct.unpack('<I', decode_data[-4:])[0]
        computed_crc = zlib.crc32(crc_payload) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise MSCDecodeError(
                f"CRC32 no coincide: almacenado={stored_crc:#010x}, "
                f"calculado={computed_crc:#010x}"
            )
        # El decoder no debe leer los 4 bytes del CRC
        end_pos = len(decode_data) - 4
    else:
        end_pos = len(decode_data)

    buf = io.BytesIO(decode_data)
    buf.seek(6)  # skip header
    dec = _Decoder(buf, strict=strict)
    result = dec.decode()

    # ── Validar que no hay trailing bytes ──
    consumed = buf.tell()
    if consumed != end_pos:
        raise MSCDecodeError(
            f"Trailing bytes: se consumieron {consumed} de {end_pos} bytes. "
            f"Payload posiblemente corrupto o manipulado."
        )

    return result


def dump(obj: Any, file, *, with_crc: bool = False,
         hmac_key: Optional[bytes] = None) -> None:
    """Serializa obj al archivo (modo binario)."""
    file.write(dumps(obj, with_crc=with_crc, hmac_key=hmac_key))


def load(file, *, strict: bool = True,
         hmac_key: Optional[bytes] = None) -> Any:
    """Deserializa desde archivo (modo binario)."""
    return loads(file.read(), strict=strict, hmac_key=hmac_key)


def dump_compressed(obj: Any, file, level: int = 6, **kwargs) -> None:
    """Serializa con compresión zlib."""
    raw = dumps(obj, **kwargs)
    compressed = zlib.compress(raw, level)
    file.write(struct.pack('<I', len(raw)))
    file.write(compressed)


def load_compressed(file, **kwargs) -> Any:
    """Deserializa desde archivo comprimido."""
    orig_size = struct.unpack('<I', file.read(4))[0]
    if orig_size > MAX_SIZE:
        raise MSCDecodeError(f"Tamaño original excede límite: {orig_size:,}")
    compressed = file.read(MAX_COMPRESSED + 1)
    if len(compressed) > MAX_COMPRESSED:
        raise MSCDecodeError(
            f"Datos comprimidos exceden límite: {len(compressed):,} > {MAX_COMPRESSED:,}"
        )
    raw = zlib.decompress(compressed, bufsize=orig_size)
    if len(raw) > MAX_SIZE:
        raise MSCDecodeError(
            f"Datos descomprimidos exceden límite: {len(raw):,} > {MAX_SIZE:,}"
        )
    return loads(raw, **kwargs)


def copy(obj: Any) -> Any:
    """Deep copy vía serialización round-trip. Más seguro que copy.deepcopy."""
    return loads(dumps(obj), strict=False)


# ────────────────────── UTILIDADES ────────────────────────────────

def inspect(data: bytes) -> dict:
    """Retorna metadata del payload sin deserializar el objeto."""
    if len(data) < 5 or data[:4] != MAGIC:
        return {'valid': False, 'error': 'Magic bytes inválidos'}

    ver = data[4]
    info = {
        'valid': True,
        'version': ver,
        'size_bytes': len(data),
    }

    root_tag = None
    if ver == 1:
        root_tag = data[5] if len(data) > 5 else None
    elif ver == 2:
        if len(data) > 6:
            flags = data[5]
            info['has_crc'] = bool(flags & 0x01)
            root_tag = data[6]
        else:
            root_tag = None
    else:
        info['valid'] = False
        info['error'] = f'Versión desconocida: {ver}'
        return info

    if root_tag is not None:
        info['root_tag'] = hex(root_tag)
        info['root_type'] = _TAG_NAMES.get(root_tag, 'unknown')

    return info


def benchmark(obj: Any, rounds: int = 100) -> dict:
    """Mide rendimiento de serialización/deserialización."""
    import time as _time

    # Encode
    t0 = _time.perf_counter()
    for _ in range(rounds):
        data = dumps(obj)
    encode_time = (_time.perf_counter() - t0) / rounds

    # Decode
    t0 = _time.perf_counter()
    for _ in range(rounds):
        loads(data, strict=False)
    decode_time = (_time.perf_counter() - t0) / rounds

    # Compressed
    raw_size = len(data)
    buf = io.BytesIO()
    dump_compressed(obj, buf)
    comp_size = len(buf.getvalue())

    return {
        'encode_ms': round(encode_time * 1000, 3),
        'decode_ms': round(decode_time * 1000, 3),
        'raw_bytes': raw_size,
        'compressed_bytes': comp_size,
        'compression_ratio': round(raw_size / comp_size, 2) if comp_size else float('inf'),
        'rounds': rounds,
    }



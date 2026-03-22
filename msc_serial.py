"""
MSC Serial v2.1
===============
Reemplazo personal y seguro de pickle.

Soporta: dict, list, tuple, set, frozenset, str, int, float, complex,
         bool, None, bytes, bytearray, datetime, date, time, timedelta,
         Decimal, UUID, Path, Enum, numpy arrays, dataclasses,
         objetos con __slots__, objetos custom registrados,
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
import inspect as _inspect_mod
import dataclasses
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID
from typing import Any, Type, Dict, Optional, Set, List

__version__ = "2.1"
__all__ = [
    "dump", "load", "dumps", "loads",
    "dump_compressed", "load_compressed",
    "register", "register_module", "inspect", "benchmark", "copy",
    "MSCError", "MSCEncodeError", "MSCDecodeError", "MSCSecurityError",
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

_TAG_NAMES: Dict[int, str] = {
    0x00: 'None',    0x01: 'bool',      0x02: 'int',       0x03: 'float',
    0x04: 'str',     0x05: 'bytes',     0x06: 'list',      0x07: 'tuple',
    0x08: 'dict',    0x09: 'set',       0x0A: 'ndarray',   0x0B: 'object',
    0x0C: 'complex', 0x0D: 'frozenset', 0x0E: 'datetime',  0x0F: 'date',
    0x10: 'time',    0x11: 'timedelta', 0x12: 'Decimal',   0x13: 'Enum',
    0x14: 'bytearray', 0x15: 'ref',    0x16: 'UUID',      0x17: 'Path',
}

MAGIC   = b'MSCS'
VERSION = b'\x02'  # formato binario sigue siendo v2; cambios son aditivos

# ─────────────────────── LIMITS ───────────────────────────────────

MAX_DEPTH       = 256
MAX_SIZE        = 512 * 1024 * 1024  # 512 MB
MAX_COLLECTION  = 10_000_000
MAX_STRING      = 100 * 1024 * 1024  # 100 MB

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


def _is_safe_dtype(dtype_str: str) -> bool:
    """Valida que un dtype string sea seguro (no structured/object/void)."""
    clean = dtype_str.strip().lower()
    # Rechazar explícitamente tipos peligrosos
    if clean in ('object', 'O', 'void', 'V'):
        return False
    # Tipos simples directos
    if clean in _SAFE_NUMPY_DTYPES:
        return True
    # Con prefijo de byteorder: <f4, >i8, =f8, |b1, etc.
    if len(clean) > 1 and clean[0] in '<>=|!':
        clean = clean[1:]
    # Numpy shorthand: f4, f8, i4, i8, u2, b1, c8, c16, etc.
    import re
    if re.fullmatch(r'[fiubcUSV]\d+', clean):
        # Rechazar V (void) y O (object) — ya cubierto arriba
        if clean[0] in ('V',):
            return False
        return True
    # Nombre completo con bitsize: float32, int64, etc.
    if re.fullmatch(r'(int|uint|float|complex|bool)\d*_?', clean):
        return True
    return False


# ─────────────────────── REGISTRY ─────────────────────────────────

_registry: Dict[str, Type] = {}


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
    """
    key = _class_key(cls)
    _registry[key] = cls
    return cls


def register_module(module) -> List[Type]:
    """
    Registra todas las clases definidas en un módulo.
    Retorna lista de clases registradas.

        import my_models
        msc.register_module(my_models)
    """
    registered = []
    for name, obj in _inspect_mod.getmembers(module, _inspect_mod.isclass):
        # Solo clases definidas EN el módulo (no importadas de stdlib, etc.)
        if obj.__module__ == module.__name__:
            register(obj)
            registered.append(obj)
    return registered


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
    __slots__ = ('buf', 'depth', 'refs', 'ref_counter', 'use_refs')

    def __init__(self, buf: io.BytesIO, *, use_refs: bool = True):
        self.buf = buf
        self.depth = 0
        self.refs: Dict[int, int] = {}   # id(obj) -> ref_id
        self.ref_counter = 0
        self.use_refs = use_refs

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
            # v2.1: encode days, seconds, microseconds por separado
            # para preservar precisión completa sin float round-trip
            buf.write(_TIMEDELTA)
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

        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
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
        except ImportError:
            pass

        # ── Objeto registrado ──

        if self._assign_ref(obj):
            return

        buf.write(_OBJ)
        cls_path = _class_key(type(obj))
        self._encode_str(cls_path)

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            state = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        elif hasattr(obj, '__slots__') and not hasattr(obj, '__dict__'):
            # __slots__ sin __dict__ → extraer slots del MRO completo.
            # Priorizado sobre __getstate__ porque Python 3.11+ añade un
            # __getstate__ por defecto que retorna (None, slots_dict) — un
            # formato que complica la deserialización innecesariamente.
            state = {}
            for cls in type(obj).__mro__:
                for s in getattr(cls, '__slots__', ()):
                    if hasattr(obj, s) and s not in state:
                        state[s] = getattr(obj, s)
        elif '__getstate__' in type(obj).__dict__ or any(
            '__getstate__' in c.__dict__ for c in type(obj).__mro__[:-1]
            if c is not object
        ):
            # Solo usar __getstate__ si fue definido explícitamente por el
            # usuario, no el default de object.
            state = obj.__getstate__()
        elif hasattr(obj, '__dict__'):
            state = obj.__dict__
        else:
            raise MSCEncodeError(f"No se puede serializar: {type(obj)!r}")

        self.encode(state)

    def _encode_str(self, s: str):
        """Encode string directamente sin ref tracking (para metadata interna)."""
        self.buf.write(_STR)
        raw = s.encode('utf-8')
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

        if tag == _TIMEDELTA:
            # v2.1: 12 bytes = int days + int seconds + uint microseconds
            # v2.0: 12 bytes = int days + double total_seconds
            # Disambiguación: intentamos v2.1 primero; si el payload viene de
            # v2.0, el caller puede forzar _decode_timedelta_v20 vía flag,
            # pero en la práctica el formato v2.0 no se producirá más.
            # Como ambos son 12 bytes, usamos heurística:
            #   v2.1: struct '<iiI' = (days:i4, seconds:i4, microseconds:U4)
            #   v2.0: struct '<id'  = (days:i4, total_seconds:f8)
            # Leemos los 12 bytes y probamos v2.1 primero.
            raw12 = self._read(12)
            days_21, secs_21, us_21 = struct.unpack('<iiI', raw12)
            # Validación: seconds debe estar en [0, 86400) y us en [0, 1_000_000)
            if 0 <= secs_21 < 86400 and us_21 < 1_000_000:
                return timedelta(days=days_21, seconds=secs_21, microseconds=us_21)
            # Fallback a formato v2.0
            days_20, total_20 = struct.unpack('<id', raw12)
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
            items = []
            for i in range(n):
                self.path.append(f'({i})')
                items.append(self.decode())
                self.path.pop()
            t = tuple(items)
            return self._store_ref(t)

        if tag == _FROZENSET:
            n = self._read_length()
            items = frozenset(self.decode() for _ in range(n))
            return self._store_ref(items)

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
            try:
                import numpy as np
            except ImportError:
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
            arr = np.frombuffer(raw, dtype=np.dtype(dtype_str)).copy().reshape(shape)
            return self._store_ref(arr)

        if tag == _OBJ:
            class_path = self._decode_str()
            self.path.append(class_path.rsplit('.', 1)[-1])
            state = self.decode()
            self.path.pop()

            if self.strict:
                cls = _get_registered(class_path)
            elif _is_registered(class_path):
                cls = _registry[class_path]
            else:
                return {'__class__': class_path, '__state__': state}

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
                # Fallback: intentar setattr (cubre slots con __dict__ mixto)
                for k, v in state.items():
                    setattr(obj, k, v)
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

def dumps(obj: Any, *, with_crc: bool = False) -> bytes:
    """Serializa obj a bytes."""
    buf = io.BytesIO()
    buf.write(MAGIC + VERSION)
    flags = 0x01 if with_crc else 0x00
    buf.write(struct.pack('B', flags))
    enc = _Encoder(buf)
    enc.encode(obj)
    data = buf.getvalue()
    if with_crc:
        crc = zlib.crc32(data) & 0xFFFFFFFF
        data += struct.pack('<I', crc)
    return data


def loads(data: bytes, *, strict: bool = True) -> Any:
    """
    Deserializa bytes a objeto.
    strict=True: solo reconstruye clases registradas (lanza MSCSecurityError).
    strict=False: clases no registradas retornan dict fallback.
    """
    if len(data) < 6:
        raise MSCDecodeError("Datos demasiado cortos para ser MSC Serial")
    buf = io.BytesIO(data)
    magic = buf.read(4)
    if magic != MAGIC:
        raise MSCDecodeError(f"Magic bytes inválidos: {magic!r}")
    ver = buf.read(1)
    if ver == b'\x01':
        # Retrocompatibilidad con v1.0 (sin flags)
        dec = _Decoder(buf, strict=False)
        return dec.decode()
    if ver != VERSION:
        raise MSCDecodeError(f"Versión no soportada: {ver!r}")
    flags = struct.unpack('B', buf.read(1))[0]
    has_crc = bool(flags & 0x01)
    if has_crc:
        payload = data[:-4]
        stored_crc = struct.unpack('<I', data[-4:])[0]
        computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise MSCDecodeError(
                f"CRC32 no coincide: almacenado={stored_crc:#010x}, "
                f"calculado={computed_crc:#010x}"
            )
    dec = _Decoder(buf, strict=strict)
    return dec.decode()


def dump(obj: Any, file, **kwargs) -> None:
    """Serializa obj al archivo (modo binario)."""
    file.write(dumps(obj, **kwargs))


def load(file, **kwargs) -> Any:
    """Deserializa desde archivo (modo binario)."""
    return loads(file.read(), **kwargs)


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
    compressed = file.read()
    raw = zlib.decompress(compressed, bufsize=orig_size)
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


# ─────────────────────────── TESTS ────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  MSC Serial v2.1 — Test Suite")
    print("=" * 60)

    errors = []
    passed = 0
    import math

    def check(name, original, use_crc=False):
        global passed
        try:
            encoded = dumps(original, with_crc=use_crc)
            decoded = loads(encoded, strict=False)
            if isinstance(original, float) and math.isnan(original):
                ok = isinstance(decoded, float) and math.isnan(decoded)
            else:
                ok = decoded == original
            status = "✅" if ok else "❌"
            crc_flag = " [CRC]" if use_crc else ""
            print(f"  {status} {name:<40} {len(encoded):>7} bytes{crc_flag}")
            if ok:
                passed += 1
            else:
                errors.append(name)
                print(f"     original: {original!r}")
                print(f"     decoded : {decoded!r}")
        except Exception as e:
            print(f"  ❌ {name:<40} ERROR: {e}")
            errors.append(name)

    # ── Primitivos ──
    print("\n  ── Primitivos ──")
    check("None",                   None)
    check("bool True",              True)
    check("bool False",             False)
    check("int cero",               0)
    check("int pequeño",            42)
    check("int negativo",           -99999)
    check("int gigante",            2**128 + 7)
    check("float",                  3.14159265358979)
    check("float NaN",              float('nan'))
    check("float inf",              float('inf'))
    check("complex",                3+4j)
    check("str simple",             "hola mundo")
    check("str unicode",            "∑∞ → quantum 🧬")
    check("str vacío",              "")
    check("bytes",                  b"\x00\xff\xab")
    check("bytearray",             bytearray(b"\x01\x02\x03"))
    check("Decimal",               Decimal("3.14159265358979323846"))

    # ── Nuevos tipos v2.1 ──
    print("\n  ── Nuevos tipos v2.1 ──")
    check("UUID",                   UUID("12345678-1234-5678-1234-567812345678"))
    check("UUID v4",                UUID("550e8400-e29b-41d4-a716-446655440000"))
    check("Path simple",            Path("/home/user/data.txt"))
    check("Path relativo",          Path("models/v2/weights.pt"))

    # ── Temporales ──
    print("\n  ── Temporales ──")
    check("datetime",               datetime(2025, 6, 15, 10, 30, 45))
    check("date",                   date(2025, 6, 15))
    check("time",                   time(10, 30, 45))
    check("timedelta simple",       timedelta(days=5, hours=3, minutes=30))
    check("timedelta microsegundos", timedelta(days=1, seconds=3661, microseconds=123456))
    check("timedelta negativo",     timedelta(days=-3, seconds=100))
    check("timedelta cero",         timedelta())

    # ── Colecciones ──
    print("\n  ── Colecciones ──")
    check("list mixta",             [1, "dos", 3.0, None, True])
    check("tuple",                  (1, 2, (3, 4)))
    check("set",                    {1, 2, 3, 4})
    check("frozenset",              frozenset({1, 2, 3}))
    check("dict anidado",           {"a": [1, 2], "b": {"c": True}})
    check("lista vacía",            [])
    check("dict vacío",             {})
    check("nesting profundo",       {"a": {"b": {"c": {"d": [1, 2, 3]}}}})

    # ── CRC32 ──
    print("\n  ── Integridad CRC32 ──")
    check("dict con CRC",           {"x": 42, "y": [1, 2, 3]}, use_crc=True)
    check("str con CRC",            "integridad verificada", use_crc=True)

    data_crc = dumps({"test": 123}, with_crc=True)
    corrupted = data_crc[:-1] + bytes([(data_crc[-1] + 1) % 256])
    try:
        loads(corrupted)
        print(f"  ❌ {'CRC corrupción detectada':<40} (no lanzó error)")
        errors.append("CRC corruption")
    except MSCDecodeError:
        print(f"  ✅ {'CRC corrupción detectada':<40}")
        passed += 1

    # ── Referencias circulares ──
    print("\n  ── Referencias ──")
    shared_list = [1, 2, 3]
    data_refs = {"a": shared_list, "b": shared_list}
    enc_refs = dumps(data_refs)
    dec_refs = loads(enc_refs, strict=False)
    refs_ok = dec_refs["a"] is dec_refs["b"]
    print(f"  {'✅' if refs_ok else '❌'} {'refs compartidas (a is b)':<40}")
    if refs_ok:
        passed += 1
    else:
        errors.append("shared refs")

    circ = [1, 2]
    circ.append(circ)
    try:
        enc_circ = dumps(circ)
        dec_circ = loads(enc_circ, strict=False)
        circ_ok = dec_circ[2] is dec_circ
        print(f"  {'✅' if circ_ok else '❌'} {'ref circular list':<40}")
        if circ_ok:
            passed += 1
        else:
            errors.append("circular ref")
    except Exception as e:
        print(f"  ❌ {'ref circular list':<40} ERROR: {e}")
        errors.append("circular ref")

    # ── Enum ──
    print("\n  ── Enum ──")

    @register
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    enc_enum = dumps(Color.GREEN)
    dec_enum = loads(enc_enum)
    enum_ok = dec_enum == Color.GREEN
    print(f"  {'✅' if enum_ok else '❌'} {'Enum Color.GREEN':<40}")
    if enum_ok:
        passed += 1
    else:
        errors.append("enum")

    # ── Seguridad ──
    print("\n  ── Seguridad ──")

    class SecretObj:
        def __init__(self):
            self.data = "secret"

    enc_secret = dumps(SecretObj())
    try:
        loads(enc_secret, strict=True)
        print(f"  ❌ {'strict bloquea no registrados':<40} (no lanzó error)")
        errors.append("security strict")
    except MSCSecurityError:
        print(f"  ✅ {'strict bloquea no registrados':<40}")
        passed += 1

    dec_fallback = loads(enc_secret, strict=False)
    fb_ok = isinstance(dec_fallback, dict) and '__class__' in dec_fallback
    print(f"  {'✅' if fb_ok else '❌'} {'strict=False retorna dict':<40}")
    if fb_ok:
        passed += 1
    else:
        errors.append("security fallback")

    # ── Numpy ──
    print("\n  ── Numpy ──")
    try:
        import numpy as np
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        encoded = dumps(arr)
        decoded = loads(encoded, strict=False)
        ok = np.array_equal(arr, decoded)
        print(f"  {'✅' if ok else '❌'} {'numpy float32 array':<40} {len(encoded):>7} bytes")
        if ok:
            passed += 1
        else:
            errors.append("numpy")

        arr_big = np.random.randn(100, 100).astype(np.float64)
        enc_big = dumps(arr_big)
        dec_big = loads(enc_big, strict=False)
        ok_big = np.allclose(arr_big, dec_big)
        print(f"  {'✅' if ok_big else '❌'} {'numpy 100x100 float64':<40} {len(enc_big):>7} bytes")
        if ok_big:
            passed += 1
        else:
            errors.append("numpy big")

        # numpy dtype security test
        print("\n  ── Numpy dtype seguridad ──")
        ok_safe = _is_safe_dtype('float32') and _is_safe_dtype('<f8') and _is_safe_dtype('int64')
        ok_block = not _is_safe_dtype('object') and not _is_safe_dtype('O') and not _is_safe_dtype('void')
        print(f"  {'✅' if ok_safe else '❌'} {'dtype whitelist (safe acepta)':<40}")
        print(f"  {'✅' if ok_block else '❌'} {'dtype whitelist (bloquea object/void)':<40}")
        if ok_safe:
            passed += 1
        else:
            errors.append("dtype safe")
        if ok_block:
            passed += 1
        else:
            errors.append("dtype block")
    except ImportError:
        print("  ⚠️  numpy no instalado — tests omitidos")

    # ── Dataclass ──
    print("\n  ── Dataclass ──")

    @register
    @dataclasses.dataclass
    class Punto:
        x: float
        y: float
        label: str = "p"

    p = Punto(1.5, -2.3, "origen")
    enc_p = dumps(p)
    dec_p = loads(enc_p)
    dc_ok = isinstance(dec_p, Punto) and dec_p.x == 1.5 and dec_p.label == "origen"
    print(f"  {'✅' if dc_ok else '❌'} {'dataclass Punto (registrado)':<40} {len(enc_p):>7} bytes")
    if dc_ok:
        passed += 1
    else:
        errors.append("dataclass")

    # ── Slots con herencia ──
    print("\n  ── Slots con herencia ──")

    @register
    class Base:
        __slots__ = ('x',)
        def __init__(self, x):
            self.x = x

    @register
    class Child(Base):
        __slots__ = ('y',)
        def __init__(self, x, y):
            super().__init__(x)
            self.y = y

    child = Child(10, 20)
    enc_child = dumps(child)
    dec_child = loads(enc_child)
    slots_ok = isinstance(dec_child, Child) and dec_child.x == 10 and dec_child.y == 20
    print(f"  {'✅' if slots_ok else '❌'} {'__slots__ con herencia':<40}")
    if slots_ok:
        passed += 1
    else:
        errors.append("slots inheritance")

    # ── inspect mejorado ──
    print("\n  ── Inspect ──")
    info = inspect(dumps({"hello": "world"}))
    inspect_ok = info['valid'] and info.get('root_type') == 'dict'
    print(f"  {'✅' if inspect_ok else '❌'} {'inspect root_type=dict':<40}")
    if inspect_ok:
        passed += 1
    else:
        errors.append("inspect root_type")

    # ── copy() ──
    print("\n  ── Copy ──")
    original = {"a": [1, 2, 3], "b": {"c": True}}
    copied = copy(original)
    copy_ok = copied == original and copied is not original and copied["a"] is not original["a"]
    print(f"  {'✅' if copy_ok else '❌'} {'copy() deep copy':<40}")
    if copy_ok:
        passed += 1
    else:
        errors.append("copy")

    # ── Retrocompatibilidad v1 ──
    print("\n  ── Retrocompatibilidad ──")
    v1_data = b'MSCS\x01' + b'\x02' + struct.pack('<H', 1) + b'\x2a'
    try:
        v1_dec = loads(v1_data, strict=False)
        v1_ok = v1_dec == 42
        print(f"  {'✅' if v1_ok else '❌'} {'carga formato v1.0':<40}")
        if v1_ok:
            passed += 1
        else:
            errors.append("v1 compat")
    except Exception as e:
        print(f"  ❌ {'carga formato v1.0':<40} ERROR: {e}")
        errors.append("v1 compat")

    # ── Compresión ──
    print("\n  ── Compresión ──")
    big = {"data": list(range(1000)), "texto": "abc" * 500}
    raw_size = len(dumps(big))
    buf = io.BytesIO()
    dump_compressed(big, buf)
    comp_size = len(buf.getvalue())
    buf.seek(0)
    restored = load_compressed(buf, strict=False)
    ok = restored == big
    ratio = raw_size / comp_size if comp_size else float('inf')
    print(f"  {'✅' if ok else '❌'} Compresión zlib")
    print(f"     Sin comprimir : {raw_size:>10,} bytes")
    print(f"     Comprimido    : {comp_size:>10,} bytes  (ratio {ratio:.1f}x)")
    if ok:
        passed += 1
    else:
        errors.append("compression")

    # ── Benchmark ──
    print("\n  ── Benchmark ──")
    bench_obj = {"numbers": list(range(100)), "text": "hello " * 50, "nested": {"a": [1, 2, 3]}}
    results = benchmark(bench_obj, rounds=500)
    print(f"     Encode: {results['encode_ms']:.3f} ms")
    print(f"     Decode: {results['decode_ms']:.3f} ms")
    print(f"     Size:   {results['raw_bytes']:,} bytes → {results['compressed_bytes']:,} bytes "
          f"({results['compression_ratio']}x)")

    # ── Resumen ──
    total = passed + len(errors)
    print()
    print("─" * 60)
    if errors:
        print(f"  ⚠️  {passed}/{total} pasaron. Fallaron: {errors}")
        sys.exit(1)
    else:
        print(f"  🎉 {passed}/{total} tests pasaron.")
    print("=" * 60)

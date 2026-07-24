"""
MSC Serial v2.5
===============
Reemplazo personal y seguro de pickle.

Soporta: dict, list, tuple, set, frozenset, deque, str, int, float,
         complex, bool, None, bytes, bytearray, datetime, date, time,
         timedelta, Decimal, UUID, Path, Enum, numpy arrays, torch.Tensor,
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

Changelog v2.5.0:
  - SECURITY: MAX_SIZE ahora acota el blob TOTAL, no solo cada campo. loads()
              no comprobaba el tamaño total: los topes por campo (MAX_STRING,
              MAX_COLLECTION, MAX_SIZE por array) se cumplían mientras su SUMA
              quedaba sin acotar, y load() materializaba el archivo entero con
              file.read() antes de cualquier chequeo. loads() rechaza ahora
              len(data)>max_size antes del dispatch de versión (v1 y v2); load()
              lee como máximo max_size+1 bytes; load_compressed() acota TANTO
              la salida descomprimida como la lectura comprimida a max_size. El
              tope es sobre el blob total (datos + framing): copy() acota su
              round-trip confiable a la longitud exacta del blob. El pico de
              memoria sigue siendo un múltiplo de max_size (overhead de objetos
              Python ~6x): ajústalo bajo para entrada no confiable.
  - SECURITY: load_compressed valida la integridad del contenedor con la
              misma estrictez que loads() aplica al payload. Antes no verificaba
              el fin del stream zlib (decompressor.eof), no rechazaba bytes tras
              el stream (unused_data), no comparaba el tamaño real con orig_size,
              un header truncado filtraba struct.error y un stream corrupto
              filtraba zlib.error crudo. Así un stream truncado (sin parte del
              ADLER32), un orig_size forjado, basura final o dos contenedores
              concatenados pasaban como válidos. Ahora cada caso falla cerrado
              con MSCDecodeError (zlib.error envuelto). La lectura comprimida se
              acota por compressBound(max_size) — los datos incompresibles
              crecen al comprimirse, así que acotar al tamaño crudo rechazaba
              round-trips legítimos con max_size ajustado.
  - SECURITY: límites configurables de verdad, por llamada. dumps/dump aceptan
              max_depth; loads/load/load_compressed aceptan max_size y max_depth
              (default = constante del módulo). Reasignar mscs.MAX_DEPTH /
              mscs.MAX_SIZE no tenía efecto (nombres re-exportados, no los
              globals que lee _core); los parámetros per-call son el knob
              soportado.
  - FIX: pérdida silenciosa de atributos __slots__. Tres formas fallaban sin
         error: __slots__ declarado como string (iterado carácter a carácter
         → estado vacío), clases híbridas (base con slots + subclase con
         __dict__ → los slots heredados se perdían y el decoder los escribía
         en __dict__, donde el descriptor los sombrea), y slots privados
         (__nombre buscado sin name-mangling). _collect_slot_names() recorre
         el MRO con el __slots__ propio de cada clase (str/iterable/dict),
         aplica el mangling de CPython y excluye __dict__/__weakref__; el
         encoder fusiona slots + __dict__ en híbridas y el decoder enruta
         cada clave a su almacén: slots vía descriptor (setattr) y el resto
         directo al __dict__ de instancia — setattr indiscriminado dejaría
         que una clave espuria '__dict__' reemplazara el dict del objeto.
  - FIX: corrupción silenciosa de referencias circulares a través de tuplas
         y de objetos registrados. El decoder reservaba None como placeholder
         antes de decodificar hijos y _REF lo devolvía tal cual: los ciclos
         vía tupla y las auto/mutuas referencias de objetos decodificaban con
         None incrustado, sin error. Ahora los objetos se crean y publican
         ANTES de decodificar su estado (identidad primero, como pickle) y
         las refs hacia tuplas en construcción se resuelven con fix-ups
         diferidos que parchean el sitio de inserción en cascada.
  - SECURITY: payloads forjados con _REF hacia slots en construcción en
              posiciones no parcheables (clave de dict, set/frozenset, tupla
              raíz auto-referente, Enum value, estado bajo __setstate__
              custom) fallan cerrado con MSCDecodeError en vez de corromper.
              loads() verifica en ambas ramas de versión (v1 y v2) que
              ningún placeholder sobreviva al decode.
  - SECURITY: el código de usuario (__setattr__ sobrescrito, properties/
              descriptors, __setstate__) jamás observa el sentinela interno:
              un valor pendiente nunca se asigna — el fix-up realiza la
              primera y única asignación con el valor real.
  - Retrocompatible: formato intacto (encoder sin cambios); payloads v2.x y
    v1.0 válidos siguen cargando — los grafos cíclicos que antes cargaban
    corruptos ahora cargan correctos desde los mismos bytes.

Changelog v2.4.1:
  - SECURITY: loads() rechaza payloads v1 cuando se pasa hmac_key (fail-closed).
              Antes despachaba v1 antes de la verificación HMAC, ignorando la
              clave por completo y evadiendo la autenticación (downgrade). v1
              ahora respeta el strict del llamante en vez de forzar strict=False.
  - SECURITY: load_compressed descomprime incrementalmente con tope MAX_SIZE.
              Antes materializaba toda la salida antes de validar el tamaño,
              permitiendo una zip-bomb que agotaba memoria pese al límite.
  - SECURITY: el tag ENUM verifica issubclass(cls, Enum) antes de instanciar.
              Antes hacía cls(value) para cualquier clase registrada (confusión
              de tipos que invocaba constructores arbitrarios).
  - FIX: round-trip de dataclass frozen — el decoder usa object.__setattr__ en
         vez de setattr (que FrozenInstanceError prohíbe en instancias frozen).
  - Retrocompatible con payloads v2.4, v2.3, v2.2, v2.1, v2.0 y v1.0

Changelog v2.4.0:
  - FIX: __getstate__/__setstate__ ahora tiene prioridad sobre dataclass
         field walking — antes, dataclasses que definían __getstate__ eran
         serializados recorriendo fields directamente, lo que causaba
         MSCEncodeError si algún field contenía tipos no soportados (ej. deque).
         La prioridad ahora es: __getstate__/__setstate__ > dataclass fields > __slots__ > __dict__
  - ADD: Soporte nativo collections.deque (tag 0x1A) — preserva maxlen
         y soporta referencias circulares
  - Retrocompatible con payloads v2.3, v2.2, v2.1, v2.0 y v1.0

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
from collections import deque
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

__version__ = "2.5.0"
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
_DEQUE      = b'\x1A'  # v2.4: collections.deque nativo

_TAG_NAMES: Dict[int, str] = {
    0x00: 'None',    0x01: 'bool',      0x02: 'int',       0x03: 'float',
    0x04: 'str',     0x05: 'bytes',     0x06: 'list',      0x07: 'tuple',
    0x08: 'dict',    0x09: 'set',       0x0A: 'ndarray',   0x0B: 'object',
    0x0C: 'complex', 0x0D: 'frozenset', 0x0E: 'datetime',  0x0F: 'date',
    0x10: 'time',    0x11: 'timedelta', 0x12: 'Decimal',   0x13: 'Enum',
    0x14: 'bytearray', 0x15: 'ref',    0x16: 'UUID',      0x17: 'Path',
    0x18: 'tensor',  0x19: 'timedelta2', 0x1A: 'deque',
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
MAX_NDARRAY_DIMS = 64              # límite de numpy (NPY_MAXDIMS); un shape con
                                   # más dimensiones se rechaza SIN materializar
                                   # el split (anti-DoS de amplificación)
MAX_DIM_DIGITS  = 20               # una dimensión real cabe en un int64 (≤19
                                   # dígitos); acota la longitud de cada token
                                   # antes de int() — sin esto un solo token
                                   # gigante fuerza int() O(n²) (DoS de CPU),
                                   # sin depender de sys.int_max_str_digits

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
# Solo dtypes numéricos seguros: f(loat), i(nt), u(int), b(ool), c(omplex)
# NO incluir S(tring), U(nicode), V(oid) — estos permiten datos arbitrarios.
_RE_DTYPE_SHORT = _re.compile(r'[fiubc]\d+')
_RE_DTYPE_LONG  = _re.compile(r'(int|uint|float|complex|bool)\d*_?')


# Dtypes no-numéricos: S(tring), U(nicode), V(oid) en notación corta.
# Solo las versiones MAYÚSCULAS y 'v' minúscula son peligrosas.
# 'u' minúscula es uint (u1=uint8, u2=uint16, etc.) — SEGURO.
# 's' minúscula no existe como dtype válido en numpy, pero no es peligroso.
_RE_UNSAFE_SHORTHAND = _re.compile(r'[<>=|!]?[SUV]\d+')
_RE_UNSAFE_VOID_LOW  = _re.compile(r'[<>=|!]?v\d+')


def _is_safe_dtype(dtype_str: str) -> bool:
    """Valida que un dtype string sea seguro (no structured/object/void/string)."""
    clean = dtype_str.strip()
    # Rechazar explícitamente tipos peligrosos (case-insensitive)
    low = clean.lower()
    if low in ('object', 'o', 'void', 'v', 's', 'u'):
        return False
    # Rechazar string/unicode/void shorthand: S<n>, U<n>, V<n>
    # (con o sin prefijo byteorder: <U8, >S16, |V32, etc.)
    # Esto previene que un payload con dtype='U8' sea aceptado como
    # 'u8' (uint64) tras lowercase pero interpretado como Unicode por numpy.
    if _RE_UNSAFE_SHORTHAND.fullmatch(clean):
        return False
    # Rechazar void minúscula: v<n> (ej: v8)
    if _RE_UNSAFE_VOID_LOW.fullmatch(clean):
        return False
    # Con prefijo de byteorder: <f4, >i8, =f8, |b1, etc.
    stripped = low
    if len(stripped) > 1 and stripped[0] in '<>=|!':
        stripped = stripped[1:]
    # Tipos simples directos (en minúsculas)
    if low in _SAFE_NUMPY_DTYPES:
        return True
    # Numpy shorthand numérico: f4, f8, i4, i8, u2, b1, c8, c16
    if _RE_DTYPE_SHORT.fullmatch(stripped):
        return True
    # Nombre completo con bitsize: float32, int64, etc.
    if _RE_DTYPE_LONG.fullmatch(stripped):
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


def _collect_slot_names(cls: Type) -> tuple:
    """Nombres reales (post name-mangling) de todos los __slots__ del MRO,
    excluyendo __dict__ y __weakref__. Acepta __slots__ declarado como str
    (un solo slot), como iterable o como dict (se usan las claves)."""
    names: List[str] = []
    seen: Set[str] = set()
    for klass in cls.__mro__:
        slots = klass.__dict__.get('__slots__', ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            if s in ('__dict__', '__weakref__'):
                continue
            if s.startswith('__') and not s.endswith('__'):
                # Regla de CPython: _NombreClase__slot, con los underscores
                # iniciales del nombre de la clase eliminados; si el nombre
                # queda vacío (clase '_'), no se aplica mangling.
                stripped = klass.__name__.lstrip('_')
                if stripped:
                    s = f'_{stripped}{s}'
            if s not in seen:
                seen.add(s)
                names.append(s)
    return tuple(names)


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
    __slots__ = ('buf', 'depth', 'refs', 'ref_counter', 'use_refs', '_pinned',
                 'max_depth')

    def __init__(self, buf: io.BytesIO, *, use_refs: bool = True,
                 max_depth: Optional[int] = None):
        self.buf = buf
        self.depth = 0
        self.refs: Dict[int, int] = {}   # id(obj) -> ref_id
        self.ref_counter = 0
        self.use_refs = use_refs
        self._pinned: list = []  # prevent GC of temporary objects (id reuse)
        # Capturado por instancia: el default es el módulo, pero un llamante
        # puede acotarlo por llamada (reasignar mscs.MAX_DEPTH no tendría
        # efecto — es un nombre re-exportado, no el global que lee _core).
        self.max_depth = MAX_DEPTH if max_depth is None else max_depth

    def encode(self, obj: Any):
        self.depth += 1
        if self.depth > self.max_depth:
            raise MSCEncodeError(
                f"Profundidad máxima excedida ({self.max_depth}). "
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

        if isinstance(obj, deque):
            if self._assign_ref(obj):
                return
            buf.write(_DEQUE)
            maxlen = obj.maxlen
            buf.write(struct.pack('<i', -1 if maxlen is None else maxlen))
            self._write_length(len(obj))
            for item in obj:
                self.encode(item)
            return

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

        if '__getstate__' in type(obj).__dict__ or any(
            '__getstate__' in c.__dict__ for c in type(obj).__mro__[:-1]
            if c is not object
        ):
            state = obj.__getstate__()
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            state = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        else:
            slot_names = _collect_slot_names(type(obj))
            if slot_names:
                # Clases con __slots__ (puras o híbridas con __dict__):
                # nombres reales del MRO completo, post name-mangling. Los
                # slots pisan claves homónimas espurias del __dict__ (in-
                # accesibles: el descriptor gana) — se serializa el valor
                # efectivo. Slots sin asignar se omiten (quedan unset).
                state = dict(obj.__dict__) if hasattr(obj, '__dict__') else {}
                for s in slot_names:
                    if hasattr(obj, s):
                        state[s] = getattr(obj, s)
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

# Marcador para un slot de ref reservado cuyo contenedor aún se está
# construyendo (la ventana entre reservar el slot y materializar una
# tupla/frozenset, o crear la instancia de un objeto). Solo vive dentro
# de _Decoder.refs; nadie más lo ve.
_PENDING_SLOT = object()


class _Pending:
    """Sentinela que _REF devuelve al apuntar a un slot aún en ventana
    (ciclo hacia una tupla ancestro en construcción). Nunca escapa del
    decoder: o se sustituye vía fix-ups o el decode aborta."""
    __slots__ = ('ref_id',)

    def __init__(self, ref_id: int):
        self.ref_id = ref_id


class _Decoder:
    __slots__ = ('buf', 'depth', 'refs', 'strict', 'path', '_fixups',
                 '_open_windows', 'max_depth')

    def __init__(self, buf: io.BytesIO, *, strict: bool = True,
                 max_depth: Optional[int] = None):
        self.buf = buf
        self.depth = 0
        self.refs: Dict[int, Any] = {}
        self.strict = strict
        self.path: List[str] = []  # breadcrumbs para errores
        self._fixups: Dict[int, List] = {}  # ref_id pendiente -> [callable(valor)]
        self._open_windows = 0              # slots _PENDING_SLOT vivos en refs
        self.max_depth = MAX_DEPTH if max_depth is None else max_depth

    def decode(self) -> Any:
        self.depth += 1
        if self.depth > self.max_depth:
            raise MSCDecodeError(
                f"Profundidad máxima excedida ({self.max_depth}) en {self._path_str()}"
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

    def _parse_shape(self, shape_str: str) -> tuple:
        """Parsea 'AxBxC' a tuple de ints con dos cotas, ambas ANTES de la
        conversión cara: (1) el NÚMERO de dimensiones (`count('x')`, O(n) en C
        sin crear objetos) — sin él un shape_str de millones de 'x' dentro de
        MAX_STRING materializaría millones de substrings (DoS de memoria); y
        (2) la LONGITUD de cada token antes de int() — sin ella un único token
        gigante (ndim=1, que pasa la cota anterior) fuerza una conversión
        int() O(n²) (DoS de CPU), y no dependemos de sys.int_max_str_digits
        (global mutable, ausente en Python <3.11). Es la misma cota de tamaño
        que _INT ya aplica con MAX_INT_BYTES. Aplicado por igual a _NDARRAY y
        _TENSOR (paridad entre ramas gemelas)."""
        if not shape_str:
            return ()
        ndim = shape_str.count('x') + 1
        if ndim > MAX_NDARRAY_DIMS:
            raise MSCDecodeError(
                f"shape excede el número máximo de dimensiones: "
                f"{ndim:,} > {MAX_NDARRAY_DIMS} en {self._path_str()}"
            )
        dims = []
        for tok in shape_str.split('x'):
            if len(tok) > MAX_DIM_DIGITS:
                raise MSCDecodeError(
                    f"dimensión de shape con demasiados dígitos: "
                    f"{len(tok):,} > {MAX_DIM_DIGITS} en {self._path_str()}"
                )
            dims.append(int(tok))
        return tuple(dims)

    def _store_ref(self, obj: Any) -> Any:
        self.refs[len(self.refs)] = obj
        return obj

    def _reserve_ref(self) -> int:
        """Reserva el próximo slot de ref con el marcador de ventana (el
        orden de slots debe coincidir con el orden de asignación del
        encoder)."""
        ref_id = len(self.refs)
        self.refs[ref_id] = _PENDING_SLOT
        self._open_windows += 1
        return ref_id

    def _resolve_ref(self, ref_id: int, obj: Any) -> None:
        """Publica el valor final de un slot pendiente y aplica los fix-ups
        que esperaban por él (huecos dejados por _REF durante la ventana)."""
        self.refs[ref_id] = obj
        self._open_windows -= 1
        if self._fixups:
            for fixup in self._fixups.pop(ref_id, ()):
                fixup(obj)

    def _defer(self, pending: _Pending, fixup) -> None:
        """Programa fixup(valor) para cuando pending.ref_id se resuelva."""
        self._fixups.setdefault(pending.ref_id, []).append(fixup)

    def _contains_pending(self, node: Any, _seen: Optional[Set[int]] = None) -> bool:
        """True si node alcanza algún _Pending vía contenedores planos
        (dict/list/tuple/deque). Solo se llama con ventanas abiertas
        (ciclos a través de tuplas), nunca en el camino común."""
        if type(node) is _Pending:
            return True
        if isinstance(node, (list, tuple, deque)):
            items = node
        elif isinstance(node, dict):
            items = node.values()
        else:
            return False
        if _seen is None:
            _seen = set()
        if id(node) in _seen:
            return False
        _seen.add(id(node))
        return any(self._contains_pending(x, _seen) for x in items)

    def assert_fully_resolved(self, result: Any) -> Any:
        """Rechaza payloads que dejaron refs sin resolver. Un payload
        legítimo siempre termina con todos los slots materializados; uno
        forjado puede dejar placeholders colgando (p. ej. una tupla que
        se referencia a sí misma sin ancestro que la resuelva)."""
        if type(result) is _Pending or self._open_windows or self._fixups:
            raise MSCDecodeError(
                "Payload con referencias sin resolver hacia contenedores "
                "nunca completados — corrupto o forjado."
            )
        return result

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
            val = self.refs[ref_id]
            if val is _PENDING_SLOT:
                # Ref hacia un ancestro aún en construcción: se entrega un
                # sentinela y cada sitio de inserción registra un fix-up
                # (contenedores mutables) o falla cerrado (posiciones
                # hasheadas, donde parchear es imposible).
                return _Pending(ref_id)
            return val

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
            if type(value) is _Pending:
                # Sin este chequeo, en strict=False el sentinela escaparía
                # dentro del dict {'__enum__', '__value__'} sin fix-up
                # registrado (fuga silenciosa de un placeholder al usuario).
                raise MSCDecodeError(
                    f"Enum value referencia un contenedor en construcción "
                    f"en {self._path_str()} — payload corrupto o forjado"
                )
            if self.strict:
                cls = _get_registered(class_path)
                # El tag ENUM solo debe reconstruir Enums. Sin este chequeo,
                # un payload puede apuntar a CUALQUIER clase registrada y
                # forzar cls(value) con value del atacante — confusión de
                # tipos que invoca el constructor fuera del modelo previsto.
                if not (isinstance(cls, type) and issubclass(cls, Enum)):
                    raise MSCSecurityError(
                        f"Tag ENUM referencia clase no-Enum: {class_path!r}. "
                        f"Posible confusión de tipos."
                    )
                return cls(value)
            else:
                return {'__enum__': class_path, '__value__': value}

        if tag == _LIST:
            n = self._read_length()
            result = []
            self._store_ref(result)
            for i in range(n):
                self.path.append(f'[{i}]')
                item = self.decode()
                self.path.pop()
                result.append(item)
                if type(item) is _Pending:
                    # Hueco hacia una tupla ancestro en construcción:
                    # se parchea cuando esta se materialice.
                    self._defer(item, lambda v, c=result, k=i: c.__setitem__(k, v))
            return result

        if tag == _TUPLE:
            n = self._read_length()
            # Reserva el slot ANTES de decodificar hijos (mismo orden que
            # el encoder). La tupla no existe hasta tener todos sus items:
            # los _REF hacia ella durante esta ventana reciben un sentinela
            # y se resuelven vía fix-ups al materializarla.
            ref_id = self._reserve_ref()
            items = []
            has_pending = False
            for i in range(n):
                self.path.append(f'({i})')
                item = self.decode()
                self.path.pop()
                items.append(item)
                if type(item) is _Pending:
                    has_pending = True
            if not has_pending:
                t = tuple(items)
                self._resolve_ref(ref_id, t)
                return t
            # Algún item referencia un ancestro aún en construcción: esta
            # tupla queda pendiente y se materializa (en cascada) cuando
            # el último ancestro pendiente se resuelva.
            pending_slots = [i for i, item in enumerate(items)
                             if type(item) is _Pending]
            remaining = [len(pending_slots)]

            def _fill_slot(idx):
                def _fixup(value):
                    items[idx] = value
                    remaining[0] -= 1
                    if remaining[0] == 0:
                        self._resolve_ref(ref_id, tuple(items))
                return _fixup

            for idx in pending_slots:
                self._defer(items[idx], _fill_slot(idx))
            return _Pending(ref_id)

        if tag == _FROZENSET:
            n = self._read_length()
            # Reserva el slot ANTES de decodificar hijos (mismo orden que
            # el encoder).
            ref_id = self._reserve_ref()
            items = []
            for _ in range(n):
                item = self.decode()
                if type(item) is _Pending:
                    # Un frozenset legítimo no puede referenciar un contenedor
                    # en construcción: exigiría hashear un ciclo (inconstruible).
                    raise MSCDecodeError(
                        f"frozenset referencia un contenedor en construcción en "
                        f"{self._path_str()} — payload corrupto o forjado"
                    )
                items.append(item)
            result = frozenset(items)
            self._resolve_ref(ref_id, result)
            return result

        if tag == _SET:
            n = self._read_length()
            result = set()
            self._store_ref(result)
            for _ in range(n):
                item = self.decode()
                if type(item) is _Pending:
                    raise MSCDecodeError(
                        f"set contiene una referencia a un contenedor en "
                        f"construcción en {self._path_str()} — payload "
                        f"corrupto o forjado"
                    )
                result.add(item)
            return result

        if tag == _DICT:
            n = self._read_length()
            result = {}
            self._store_ref(result)
            for _ in range(n):
                k = self.decode()
                if type(k) is _Pending:
                    raise MSCDecodeError(
                        f"clave de dict referencia un contenedor en "
                        f"construcción en {self._path_str()} — payload "
                        f"corrupto o forjado"
                    )
                self.path.append(f'.{k!r}' if isinstance(k, str) else f'[{k!r}]')
                v = self.decode()
                self.path.pop()
                result[k] = v
                if type(v) is _Pending:
                    self._defer(v, lambda val, c=result, key=k: c.__setitem__(key, val))
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
            shape = self._parse_shape(shape_str)
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
            shape = self._parse_shape(shape_str)
            n = self._read_length(MAX_SIZE)
            raw = self._read(n)
            arr = _np.frombuffer(raw, dtype=_np.dtype(dtype_str)).copy().reshape(shape)
            t = _torch.from_numpy(arr)
            if requires_grad:
                t = t.requires_grad_(True)
            return self._store_ref(t)

        if tag == _DEQUE:
            maxlen_raw = struct.unpack('<i', self._read(4))[0]
            if maxlen_raw < -1:
                raise MSCDecodeError(
                    f"maxlen inválido para deque: {maxlen_raw} en {self._path_str()}"
                )
            maxlen = None if maxlen_raw == -1 else maxlen_raw
            n = self._read_length()
            # Prevenir CPU exhaustion: no decodear más items de los que
            # el deque puede retener. Un payload con maxlen=1, count=10M
            # forzaría decodear 10M items descartando 9,999,999.
            if maxlen is not None and n > maxlen:
                raise MSCDecodeError(
                    f"Deque count ({n:,}) excede maxlen ({maxlen:,}) "
                    f"en {self._path_str()} — posible payload adversarial"
                )
            result = deque(maxlen=maxlen)
            self._store_ref(result)
            for i in range(n):
                self.path.append(f'[{i}]')
                item = self.decode()
                self.path.pop()
                result.append(item)
                if type(item) is _Pending:
                    # Índices estables: n <= maxlen ya está validado, así
                    # que ningún append desplaza items ya insertados.
                    self._defer(item, lambda v, c=result, k=i: c.__setitem__(k, v))
            return result

        if tag == _OBJ:
            # Reserva el slot ANTES de decodificar hijos (mismo orden que
            # el encoder).
            ref_id = self._reserve_ref()

            class_path = self._decode_str()

            if self.strict:
                cls = _get_registered(class_path)
            else:
                cls = _registry.get(class_path)

            if cls is None:
                # strict=False + clase no registrada: dict fallback. Se
                # publica ANTES de decodificar el estado para que los
                # ciclos hacia el objeto resuelvan contra el fallback.
                fallback = {'__class__': class_path, '__state__': None}
                self._resolve_ref(ref_id, fallback)
                self.path.append(class_path.rsplit('.', 1)[-1])
                state = self.decode()
                self.path.pop()
                if type(state) is _Pending:
                    self._defer(state, lambda v, c=fallback: c.__setitem__('__state__', v))
                else:
                    fallback['__state__'] = state
                return fallback

            # La instancia se crea y publica ANTES de decodificar el
            # estado: los _REF del estado hacia el objeto reciben la
            # instancia real (identidad primero, estado después — misma
            # semántica que pickle en grafos cíclicos).
            obj = cls.__new__(cls)
            self._resolve_ref(ref_id, obj)
            self.path.append(class_path.rsplit('.', 1)[-1])
            state = self.decode()
            self.path.pop()

            if type(state) is _Pending:
                raise MSCDecodeError(
                    f"El estado de {class_path} referencia un contenedor "
                    f"en construcción en {self._path_str()} — irresoluble"
                )
            if '__setstate__' in type(obj).__dict__ or any(
                '__setstate__' in c.__dict__ for c in type(obj).__mro__[:-1]
                if c is not object
            ):
                if self._open_windows and self._contains_pending(state):
                    raise MSCDecodeError(
                        f"El estado de {class_path} contiene referencias a "
                        f"una tupla en construcción y la clase define "
                        f"__setstate__: los huecos no pueden parchearse a "
                        f"través de él. Rompe el ciclo por un contenedor "
                        f"mutable o elimina __setstate__."
                    )
                obj.__setstate__(state)
            elif dataclasses.is_dataclass(cls):
                # object.__setattr__ (no setattr) para que las dataclasses
                # frozen — cuyo __setattr__ lanza FrozenInstanceError — hagan
                # round-trip. Es lo que usa el __init__ generado de la dataclass.
                # Un valor pendiente NUNCA se asigna: la primera (y única)
                # asignación la hace el fix-up con el valor real, para que
                # descriptors/__setattr__ de usuario jamás vean el sentinela.
                # Las claves se filtran contra los fields declarados (fail-
                # closed): una clave espuria del payload no es un field, y sin
                # este filtro '__dict__' reemplazaría el dict de instancia
                # entero (clobbering/aliasing) y el nombre de una @property
                # invocaría su setter con datos del atacante — fuera del modelo
                # "solo __setstate__ se ejecuta". Paridad con la rama slots.
                field_names = {f.name for f in dataclasses.fields(cls)}
                for k, v in state.items():
                    if k not in field_names:
                        raise MSCDecodeError(
                            f"Clave '{k}' no es un field de la dataclass "
                            f"{class_path} en {self._path_str()} — payload "
                            f"manipulado"
                        )
                    if type(v) is _Pending:
                        self._defer(v, lambda val, o=obj, key=k: object.__setattr__(o, key, val))
                    else:
                        object.__setattr__(obj, k, v)
            else:
                slot_names = _collect_slot_names(cls)
                has_dict = hasattr(obj, '__dict__')
                if not slot_names and has_dict:
                    # Camino común: clase solo-__dict__.
                    if self._open_windows:
                        for k, v in state.items():
                            if type(v) is _Pending:
                                self._defer(v, lambda val, c=obj.__dict__, key=k: c.__setitem__(key, val))
                            else:
                                obj.__dict__[k] = v
                    else:
                        obj.__dict__.update(state)
                else:
                    # Clases con __slots__ (puras o híbridas) o sin __dict__:
                    # cada clave va a su almacén. Los slots se asignan vía su
                    # descriptor (setattr; escribirlos en obj.__dict__ los
                    # dejaría sombreados e inaccesibles) y el resto va
                    # directo al __dict__ de instancia — nunca por setattr,
                    # que con una clave espuria '__dict__' REEMPLAZARÍA el
                    # dict del objeto entero. En slots puros, una clave
                    # no-slot cae a setattr y falla cerrado (AttributeError).
                    slot_set = set(slot_names)
                    for k, v in state.items():
                        if k in slot_set or not has_dict:
                            if type(v) is _Pending:
                                self._defer(v, lambda val, o=obj, key=k: setattr(o, key, val))
                            else:
                                setattr(obj, k, v)
                        elif type(v) is _Pending:
                            self._defer(v, lambda val, c=obj.__dict__, key=k: c.__setitem__(key, val))
                        else:
                            obj.__dict__[k] = v
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
          hmac_key: Optional[bytes] = None,
          max_depth: Optional[int] = None) -> bytes:
    """
    Serializa obj a bytes.

    with_crc: añade CRC32 para detectar corrupción accidental.
    hmac_key: si se proporciona, añade HMAC-SHA256 para autenticación
              criptográfica. Verificado en loads() con la misma clave.
              Mutuamente exclusivo con with_crc (HMAC es estrictamente
              superior).
    max_depth: profundidad máxima de anidamiento (default MAX_DEPTH). Este
               parámetro es el knob soportado; reasignar mscs.MAX_DEPTH no
               tiene efecto.
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
    enc = _Encoder(buf, max_depth=max_depth)
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
          hmac_key: Optional[bytes] = None,
          max_size: Optional[int] = None,
          max_depth: Optional[int] = None) -> Any:
    """
    Deserializa bytes a objeto.

    strict=True: solo reconstruye clases registradas (lanza MSCSecurityError).
    strict=False: clases no registradas retornan dict fallback.
    hmac_key: si se proporciona, verifica HMAC-SHA256 antes de deserializar.
              Lanza MSCSecurityError si la firma no coincide.
    max_size: tope del blob TOTAL (default MAX_SIZE). Acota la suma de todos
              los campos, no solo cada campo individual; el blob se rechaza
              antes de decodificar si lo excede. Ajústalo bajo para entrada
              no confiable (el pico de memoria es un múltiplo por el overhead
              de objetos Python).
    max_depth: profundidad máxima de anidamiento (default MAX_DEPTH).

    max_size y max_depth son los knobs soportados; reasignar mscs.MAX_SIZE /
    mscs.MAX_DEPTH no tiene efecto (son nombres re-exportados, no los globals
    que lee _core).
    """
    size_limit = MAX_SIZE if max_size is None else max_size
    if len(data) < 6:
        raise MSCDecodeError("Datos demasiado cortos para ser MSC Serial")
    # Tope total ANTES del dispatch de versión: acota v1 y v2 por igual y
    # rechaza el payload entero (muchos campos individualmente válidos cuya
    # suma excede el límite) sin construir estructuras encima.
    if len(data) > size_limit:
        raise MSCDecodeError(
            f"Payload excede el tamaño máximo: {len(data):,} > {size_limit:,} bytes. "
            f"Sube max_size si el dato es confiable."
        )
    if data[:4] != MAGIC:
        raise MSCDecodeError(f"Magic bytes inválidos: {data[:4]!r}")
    ver = data[4:5]

    if ver == b'\x01':
        # Retrocompatibilidad con v1.0 (sin flags, sin integridad in-band).
        # El formato v1 no puede transportar HMAC: si el llamante exige
        # autenticación, un payload v1 nunca la satisface. Rechazarlo
        # (fail-closed) cierra el downgrade de un v2 firmado a un v1 sin
        # firma, que de otro modo evade por completo la verificación HMAC.
        if hmac_key is not None:
            raise MSCSecurityError(
                "Se proporcionó hmac_key pero el payload es v1 (sin soporte HMAC). "
                "Posible ataque de downgrade."
            )
        buf = io.BytesIO(data)
        buf.seek(5)
        # Respeta el strict del llamante: forzar strict=False sería otro
        # downgrade silencioso de la política de seguridad solicitada.
        dec = _Decoder(buf, strict=strict, max_depth=max_depth)
        result = dec.assert_fully_resolved(dec.decode())
        # Trailing bytes: mismo control que la rama v2 (paridad de versión).
        # v1 no tiene CRC/HMAC in-band, así que el payload real acaba en
        # len(data); cualquier byte de más es corrupción o smuggling.
        if buf.tell() != len(data):
            raise MSCDecodeError(
                f"Trailing bytes: se consumieron {buf.tell()} de "
                f"{len(data)} bytes. Payload posiblemente corrupto o "
                f"manipulado."
            )
        return result

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
    dec = _Decoder(buf, strict=strict, max_depth=max_depth)
    result = dec.assert_fully_resolved(dec.decode())

    # ── Validar que no hay trailing bytes ──
    consumed = buf.tell()
    if consumed != end_pos:
        raise MSCDecodeError(
            f"Trailing bytes: se consumieron {consumed} de {end_pos} bytes. "
            f"Payload posiblemente corrupto o manipulado."
        )

    return result


def dump(obj: Any, file, *, with_crc: bool = False,
         hmac_key: Optional[bytes] = None,
         max_depth: Optional[int] = None) -> None:
    """Serializa obj al archivo (modo binario)."""
    file.write(dumps(obj, with_crc=with_crc, hmac_key=hmac_key,
                     max_depth=max_depth))


def load(file, *, strict: bool = True,
         hmac_key: Optional[bytes] = None,
         max_size: Optional[int] = None,
         max_depth: Optional[int] = None) -> Any:
    """Deserializa desde archivo (modo binario).

    Lee como máximo max_size+1 bytes: un archivo mayor se rechaza sin
    materializarlo entero, en vez de asignar gigabytes antes del chequeo.
    """
    size_limit = MAX_SIZE if max_size is None else max_size
    data = file.read(size_limit + 1)
    if len(data) > size_limit:
        raise MSCDecodeError(
            f"Archivo excede el tamaño máximo: >{size_limit:,} bytes. "
            f"Sube max_size si el dato es confiable."
        )
    return loads(data, strict=strict, hmac_key=hmac_key,
                 max_size=size_limit, max_depth=max_depth)


def dump_compressed(obj: Any, file, level: int = 6, **kwargs) -> None:
    """Serializa con compresión zlib."""
    raw = dumps(obj, **kwargs)
    compressed = zlib.compress(raw, level)
    file.write(struct.pack('<I', len(raw)))
    file.write(compressed)


def _zlib_compress_bound(n: int) -> int:
    """Cota superior del tamaño comprimido de n bytes — la misma fórmula
    que `compressBound()` de zlib. Los datos incompresibles crecen al
    comprimirse (cabecera + ADLER32 + overhead por bloque), así que esta
    cota es el máximo legítimo que un stream que descomprime a n bytes
    puede ocupar comprimido."""
    return n + (n >> 12) + (n >> 14) + (n >> 25) + 13


def load_compressed(file, *, max_size: Optional[int] = None, **kwargs) -> Any:
    """Deserializa desde archivo comprimido.

    max_size acota la lectura comprimida (además de MAX_COMPRESSED), la salida
    descomprimida y el blob que se pasa a loads(); una zip-bomb se aborta
    durante la descompresión (pico ~max_size, no el tamaño real de la bomba).

    Integridad del contenedor (tan estricta como loads() con el payload):
    rechaza encabezado truncado, stream zlib corrupto o incompleto (checksum
    ADLER32 no verificado), bytes tras el stream, y un orig_size que no
    coincida con el tamaño real descomprimido. Todo error crudo de zlib se
    envuelve en MSCDecodeError.
    """
    size_limit = MAX_SIZE if max_size is None else max_size
    header = file.read(4)
    if len(header) < 4:
        raise MSCDecodeError(
            f"Encabezado comprimido truncado: se esperaban 4 bytes, "
            f"se obtuvieron {len(header)}"
        )
    orig_size = struct.unpack('<I', header)[0]
    if orig_size > size_limit:
        raise MSCDecodeError(f"Tamaño original excede límite: {orig_size:,}")
    # La lectura del blob comprimido se acota por el MENOR de MAX_COMPRESSED
    # (techo absoluto) y la cota superior de zlib para size_limit bytes: bajar
    # max_size reduce así el pico de memoria del lado comprimido. Se usa
    # compressBound (no size_limit a secas) porque los datos incompresibles
    # (aleatorios, cifrados, ya comprimidos) crecen ~0.03% + overhead de
    # bloques al comprimirse — acotar al tamaño crudo rechazaría un
    # dump_compressed legítimo cuyo max_size se ajusta al tamaño real.
    comp_limit = min(MAX_COMPRESSED, _zlib_compress_bound(size_limit))
    compressed = file.read(comp_limit + 1)
    if len(compressed) > comp_limit:
        raise MSCDecodeError(
            f"Datos comprimidos exceden límite: {len(compressed):,} > {comp_limit:,}"
        )
    # Descompresión incremental con tope duro. zlib.decompress() materializa
    # TODA la salida antes de que se pueda medir (bufsize es una pista, no un
    # límite), así que una zip-bomb agota memoria pese al chequeo posterior.
    # Descomprimir acotando la salida y abortar al cruzar size_limit mantiene
    # el pico de memoria en ~size_limit en vez del tamaño real de la bomba.
    # zlib.error (magic inválido, deflate corrupto, ADLER32 que no cuadra) se
    # envuelve en MSCDecodeError: fail-closed, sin filtrar el error crudo.
    decompressor = zlib.decompressobj()
    out = bytearray()
    buf = compressed
    try:
        while buf:
            out += decompressor.decompress(buf, size_limit + 1 - len(out))
            if len(out) > size_limit:
                raise MSCDecodeError(
                    f"Datos descomprimidos exceden límite: >{size_limit:,}"
                )
            buf = decompressor.unconsumed_tail
        out += decompressor.flush()
    except zlib.error as e:
        raise MSCDecodeError(f"Stream zlib inválido o corrupto: {e}") from e
    if len(out) > size_limit:
        raise MSCDecodeError(
            f"Datos descomprimidos exceden límite: >{size_limit:,}"
        )
    # ── Integridad del stream zlib (fail-closed, como el trailing-bytes de
    # loads()) ──
    # eof: el marcador de fin de stream se alcanzó y el ADLER32 se verificó.
    # Un stream truncado (p. ej. sin el checksum) deja eof=False con datos
    # parciales que de otro modo pasarían como válidos.
    if not decompressor.eof:
        raise MSCDecodeError(
            "Stream zlib incompleto o truncado (checksum no verificado). "
            "Contenedor comprimido corrupto."
        )
    # unused_data: bytes tras el final del stream — basura o contenedores
    # concatenados. Rechazar es coherente con la validación de trailing bytes
    # que loads() aplica al payload descomprimido.
    if decompressor.unused_data:
        raise MSCDecodeError(
            f"Bytes tras el stream comprimido: {len(decompressor.unused_data):,}. "
            f"Contenedor corrupto o manipulado."
        )
    # orig_size es el tamaño declarado por dump_compressed; debe coincidir con
    # el real. Una discrepancia señala corrupción o un encabezado forjado.
    if len(out) != orig_size:
        raise MSCDecodeError(
            f"orig_size no coincide: declarado {orig_size:,}, "
            f"real {len(out):,}. Contenedor corrupto o manipulado."
        )
    return loads(bytes(out), max_size=size_limit, **kwargs)


def copy(obj: Any) -> Any:
    """Deep copy vía serialización round-trip. Más seguro que copy.deepcopy."""
    data = dumps(obj)
    # El tope de tamaño protege contra ENTRADA no confiable; copiar un objeto
    # propio no lo es. Se acota a los bytes recién producidos para no romper
    # copias grandes: MAX_SIZE limita el blob TOTAL, y un único campo cercano
    # a MAX_SIZE (p. ej. un ndarray) suma el framing y cruzaría el default.
    return loads(data, strict=False, max_size=len(data))


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



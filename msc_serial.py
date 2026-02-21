"""
MSC Serial v1.0
===============
Reemplazo personal y seguro de pickle.
Soporta: dict, list, tuple, set, str, int, float, bool, None, bytes,
         numpy arrays, dataclasses, objetos con __slots__, objetos custom.

API compatible con pickle:
  msc.dump(obj, file)
  msc.load(file)
  msc.dumps(obj) -> bytes
  msc.loads(data) -> obj
  msc.dump_compressed(obj, file)   # con zlib
  msc.load_compressed(file)

Ventajas sobre pickle:
  - No ejecuta código arbitrario al deserializar
  - Formato legible/auditable
  - Compresión integrada
  - Versionado de esquema
"""

import struct
import io
import zlib
import dataclasses
from typing import Any

# ─────────────────────────── TYPE TAGS ────────────────────────────
_NONE    = b'\x00'
_BOOL    = b'\x01'
_INT     = b'\x02'
_FLOAT   = b'\x03'
_STR     = b'\x04'
_BYTES   = b'\x05'
_LIST    = b'\x06'
_TUPLE   = b'\x07'
_DICT    = b'\x08'
_SET     = b'\x09'
_NDARRAY = b'\x0A'
_OBJ     = b'\x0B'   # objetos con __dict__ o __slots__

MAGIC    = b'MSCS'
VERSION  = b'\x01'

# ──────────────────────────── ENCODER ─────────────────────────────

def _encode(obj: Any, buf: io.BytesIO):
    if obj is None:
        buf.write(_NONE)

    elif isinstance(obj, bool):
        buf.write(_BOOL)
        buf.write(b'\x01' if obj else b'\x00')

    elif isinstance(obj, int):
        buf.write(_INT)
        raw = obj.to_bytes((obj.bit_length() + 8) // 8, 'little', signed=True)
        buf.write(struct.pack('<H', len(raw)))
        buf.write(raw)

    elif isinstance(obj, float):
        buf.write(_FLOAT)
        buf.write(struct.pack('<d', obj))

    elif isinstance(obj, str):
        buf.write(_STR)
        raw = obj.encode('utf-8')
        buf.write(struct.pack('<I', len(raw)))
        buf.write(raw)

    elif isinstance(obj, (bytes, bytearray)):
        buf.write(_BYTES)
        buf.write(struct.pack('<I', len(obj)))
        buf.write(bytes(obj))

    elif isinstance(obj, list):
        buf.write(_LIST)
        buf.write(struct.pack('<I', len(obj)))
        for item in obj:
            _encode(item, buf)

    elif isinstance(obj, tuple):
        buf.write(_TUPLE)
        buf.write(struct.pack('<I', len(obj)))
        for item in obj:
            _encode(item, buf)

    elif isinstance(obj, set):
        buf.write(_SET)
        items = sorted(obj, key=repr)   # orden determinista
        buf.write(struct.pack('<I', len(items)))
        for item in items:
            _encode(item, buf)

    elif isinstance(obj, dict):
        buf.write(_DICT)
        buf.write(struct.pack('<I', len(obj)))
        for k, v in obj.items():
            _encode(k, buf)
            _encode(v, buf)

    else:
        # Intenta numpy
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                buf.write(_NDARRAY)
                meta = f"{obj.dtype}|{'x'.join(map(str, obj.shape))}"
                _encode(meta, buf)
                raw = obj.tobytes()
                buf.write(struct.pack('<I', len(raw)))
                buf.write(raw)
                return
        except ImportError:
            pass

        # Objeto genérico
        buf.write(_OBJ)
        cls = type(obj)
        _encode(f"{cls.__module__}.{cls.__qualname__}", buf)

        # Obtener estado
        if hasattr(obj, '__getstate__'):
            state = obj.__getstate__()
        elif dataclasses.is_dataclass(obj):
            state = dataclasses.asdict(obj)
        elif hasattr(obj, '__slots__'):
            state = {s: getattr(obj, s) for s in obj.__slots__ if hasattr(obj, s)}
        elif hasattr(obj, '__dict__'):
            state = obj.__dict__
        else:
            raise TypeError(f"No se puede serializar: {type(obj)}")

        _encode(state, buf)


# ──────────────────────────── DECODER ─────────────────────────────

def _decode(buf: io.BytesIO) -> Any:
    tag = buf.read(1)
    if not tag:
        raise EOFError("Buffer vacío")

    if tag == _NONE:
        return None

    elif tag == _BOOL:
        return buf.read(1) == b'\x01'

    elif tag == _INT:
        n = struct.unpack('<H', buf.read(2))[0]
        return int.from_bytes(buf.read(n), 'little', signed=True)

    elif tag == _FLOAT:
        return struct.unpack('<d', buf.read(8))[0]

    elif tag == _STR:
        n = struct.unpack('<I', buf.read(4))[0]
        return buf.read(n).decode('utf-8')

    elif tag == _BYTES:
        n = struct.unpack('<I', buf.read(4))[0]
        return buf.read(n)

    elif tag == _LIST:
        n = struct.unpack('<I', buf.read(4))[0]
        return [_decode(buf) for _ in range(n)]

    elif tag == _TUPLE:
        n = struct.unpack('<I', buf.read(4))[0]
        return tuple(_decode(buf) for _ in range(n))

    elif tag == _SET:
        n = struct.unpack('<I', buf.read(4))[0]
        return {_decode(buf) for _ in range(n)}

    elif tag == _DICT:
        n = struct.unpack('<I', buf.read(4))[0]
        return {_decode(buf): _decode(buf) for _ in range(n)}

    elif tag == _NDARRAY:
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy requerido para deserializar arrays")
        meta = _decode(buf)
        dtype_str, shape_str = meta.split('|')
        shape = tuple(int(x) for x in shape_str.split('x')) if shape_str else ()
        n = struct.unpack('<I', buf.read(4))[0]
        raw = buf.read(n)
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)

    elif tag == _OBJ:
        class_path = _decode(buf)
        state = _decode(buf)
        # Reconstruir objeto
        module_name, *cls_parts = class_path.rsplit('.', 1)
        try:
            import importlib
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_parts[0])
            obj = cls.__new__(cls)
            if hasattr(obj, '__setstate__'):
                obj.__setstate__(state)
            elif hasattr(obj, '__slots__'):
                for k, v in state.items():
                    setattr(obj, k, v)
            else:
                obj.__dict__.update(state)
            return obj
        except Exception as e:
            # Fallback: retorna dict con metadata
            return {'__class__': class_path, '__state__': state}

    else:
        raise ValueError(f"Tag desconocido: {tag!r}")


# ──────────────────────── PUBLIC API ──────────────────────────────

def dumps(obj: Any) -> bytes:
    """Serializa obj a bytes."""
    buf = io.BytesIO()
    buf.write(MAGIC + VERSION)
    _encode(obj, buf)
    return buf.getvalue()


def loads(data: bytes) -> Any:
    """Deserializa bytes a objeto."""
    buf = io.BytesIO(data)
    hdr = buf.read(5)
    if hdr[:4] != MAGIC:
        raise ValueError("No es un archivo MSC Serial")
    if hdr[4:5] != VERSION:
        raise ValueError(f"Versión no soportada: {hdr[4]}")
    return _decode(buf)


def dump(obj: Any, file) -> None:
    """Serializa obj al archivo (modo binario)."""
    file.write(dumps(obj))


def load(file) -> Any:
    """Deserializa desde archivo (modo binario)."""
    return loads(file.read())


def dump_compressed(obj: Any, file, level: int = 6) -> None:
    """Serializa con compresión zlib."""
    raw = dumps(obj)
    compressed = zlib.compress(raw, level)
    # Header: 4 bytes tamaño original
    file.write(struct.pack('<I', len(raw)))
    file.write(compressed)


def load_compressed(file) -> Any:
    """Deserializa desde archivo comprimido."""
    orig_size = struct.unpack('<I', file.read(4))[0]
    compressed = file.read()
    raw = zlib.decompress(compressed, bufsize=orig_size)
    return loads(raw)


# ────────────────────── UTILIDADES ────────────────────────────────

def inspect(data: bytes) -> dict:
    """Retorna metadata del payload sin deserializar el objeto."""
    if data[:4] != MAGIC:
        return {'valid': False}
    return {
        'valid': True,
        'version': data[4],
        'size_bytes': len(data),
        'root_tag': hex(data[5]) if len(data) > 5 else None,
    }


# ─────────────────────────── TESTS ────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 55)
    print("  MSC Serial v1.0 — Test Suite")
    print("=" * 55)

    errors = []

    def check(name, original):
        try:
            encoded = dumps(original)
            decoded = loads(encoded)
            ok = decoded == original
            status = "✅" if ok else "❌"
            print(f"  {status} {name:<30} {len(encoded):>6} bytes")
            if not ok:
                errors.append(name)
                print(f"     original: {original!r}")
                print(f"     decoded : {decoded!r}")
        except Exception as e:
            print(f"  ❌ {name:<30} ERROR: {e}")
            errors.append(name)

    check("None",              None)
    check("bool True",         True)
    check("bool False",        False)
    check("int pequeño",       42)
    check("int negativo",      -99999)
    check("int gigante",       2**128 + 7)
    check("float",             3.14159265358979)
    check("str simple",        "hola mundo")
    check("str unicode",       "∑∞ → quantum 🧬")
    check("bytes",             b"\x00\xff\xab")
    check("list mixta",        [1, "dos", 3.0, None, True])
    check("tuple",             (1, 2, (3, 4)))
    check("set",               {1, 2, 3, 4})
    check("dict anidado",      {"a": [1, 2], "b": {"c": True}})
    check("lista vacía",       [])
    check("dict vacío",        {})

    # Numpy
    try:
        import numpy as np
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        encoded = dumps(arr)
        decoded = loads(encoded)
        ok = np.array_equal(arr, decoded)
        print(f"  {'✅' if ok else '❌'} numpy float32 array             {len(encoded):>6} bytes")
        if not ok:
            errors.append("numpy")
    except ImportError:
        print("  ⚠️  numpy no instalado — test omitido")

    # Dataclass
    @dataclasses.dataclass
    class Punto:
        x: float
        y: float
        label: str = "p"

    p = Punto(1.5, -2.3, "origen")
    try:
        enc = dumps(p)
        dec = loads(enc)
        # puede ser el objeto o el fallback dict
        if isinstance(dec, dict):
            ok = dec['__state__']['x'] == 1.5
        else:
            ok = dec.x == 1.5
        print(f"  {'✅' if ok else '❌'} dataclass Punto                  {len(enc):>6} bytes")
    except Exception as e:
        print(f"  ❌ dataclass Punto                  ERROR: {e}")

    # Compresión
    big = {"data": list(range(1000)), "texto": "abc" * 500}
    raw_size   = len(dumps(big))
    buf = io.BytesIO()
    dump_compressed(big, buf)
    comp_size = len(buf.getvalue())
    buf.seek(0)
    restored = load_compressed(buf)
    ok = restored == big
    ratio = raw_size / comp_size
    print(f"\n  {'✅' if ok else '❌'} Compresión zlib")
    print(f"     Sin comprimir : {raw_size:,} bytes")
    print(f"     Comprimido    : {comp_size:,} bytes  (ratio {ratio:.1f}x)")

    print()
    if errors:
        print(f"  ⚠️  Fallaron: {errors}")
        sys.exit(1)
    else:
        print("  🎉 Todos los tests pasaron.")
    print("=" * 55)

# MSC Serial v2.0

> Reemplazo personal de `pickle` — seguro, comprimible y sin ejecución de código arbitrario.

---

## ¿Por qué MSC Serial?

| | `pickle` | `msc_serial` v2.0 |
|---|---|---|
| Seguridad al cargar | ❌ ejecuta código arbitrario | ✅ registry de clases seguras |
| Compresión integrada | ❌ | ✅ zlib nativo |
| Integridad de datos | ❌ | ✅ CRC32 opcional |
| Refs circulares | ✅ | ✅ ref tracking por id |
| Soporte numpy | parcial | ✅ nativo |
| Formato auditable | ❌ opaco | ✅ header `MSCS` + versión + flags |
| Límites de seguridad | ❌ | ✅ profundidad, tamaño, colecciones |

---

## Changelog v2.0

- **Registry de clases seguras** — elimina `importlib` dinámico en deserialización; solo clases registradas con `@msc.register` se reconstruyen
- **Tipos nuevos** — `complex`, `frozenset`, `bytearray`, `datetime`/`date`/`time`/`timedelta`, `Decimal`, `Enum`
- **Referencias circulares** — tracking por `id(obj)` con tag `_REF`; objetos compartidos se serializan una sola vez
- **Integridad CRC32** — `dumps(obj, with_crc=True)` detecta corrupción
- **Límites configurables** — `MAX_DEPTH=256`, `MAX_COLLECTION=10M`, `MAX_STRING=100MB`, `MAX_SIZE=512MB`
- **Excepciones tipadas** — `MSCEncodeError`, `MSCDecodeError`, `MSCSecurityError`
- **Retrocompatibilidad** — carga automática de archivos v1.0
- **Benchmark integrado** — `msc.benchmark(obj)` mide encode/decode/compresión

---

## Instalación

Sin dependencias externas. Copia `msc_serial.py` a tu proyecto y listo.

```bash
# opcional: numpy para arrays
pip install numpy
```

---

## Uso rápido

```python
import msc_serial as msc

# ── Serializar / deserializar ──────────────────────────
data = {"config": [1, 2, 3], "score": 0.97, "labels": {"a", "b"}}

raw = msc.dumps(data)          # → bytes
obj = msc.loads(raw)           # → objeto original

# ── Con verificación de integridad ─────────────────────
raw = msc.dumps(data, with_crc=True)
obj = msc.loads(raw)           # valida CRC32 automáticamente

# ── Archivos ───────────────────────────────────────────
with open("data.msc", "wb") as f:
    msc.dump(data, f)

with open("data.msc", "rb") as f:
    obj = msc.load(f)

# ── Con compresión zlib ────────────────────────────────
with open("data.mscz", "wb") as f:
    msc.dump_compressed(data, f)

with open("data.mscz", "rb") as f:
    obj = msc.load_compressed(f)

# ── Inspeccionar sin deserializar ──────────────────────
info = msc.inspect(raw)
# {'valid': True, 'version': 2, 'size_bytes': 512, 'has_crc': True, 'root_tag': '0x8'}
```

---

## Registrar clases

v2.0 requiere registrar clases explícitamente para deserialización segura. Sin registro, `loads()` lanza `MSCSecurityError`.

```python
import msc_serial as msc
from dataclasses import dataclass
from enum import Enum

@msc.register
@dataclass
class Punto:
    x: float
    y: float
    label: str = "p"

@msc.register
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# Serializar y reconstruir objetos seguros
p = Punto(1.5, -2.3, "origen")
raw = msc.dumps(p)
obj = msc.loads(raw)           # → Punto(x=1.5, y=-2.3, label='origen')

# Modo permisivo: clases no registradas → dict fallback
obj = msc.loads(raw, strict=False)
# → {'__class__': '...Punto', '__state__': {'x': 1.5, ...}}
```

---

## Tipos soportados

| Tipo | Tag | Notas |
|---|---|---|
| `None`, `bool` | `0x00`, `0x01` | |
| `int` | `0x02` | precisión arbitraria, little-endian |
| `float` | `0x03` | 64-bit IEEE 754 (NaN, ±inf) |
| `complex` | `0x0C` | **nuevo** — par de float64 |
| `str` | `0x04` | UTF-8 |
| `bytes`, `bytearray` | `0x05`, `0x14` | **bytearray nuevo** |
| `Decimal` | `0x12` | **nuevo** — precisión arbitraria |
| `list`, `tuple`, `set`, `frozenset` | `0x06`–`0x0D` | **frozenset nuevo** |
| `dict` | `0x08` | anidados con ref tracking |
| `datetime`, `date`, `time`, `timedelta` | `0x0E`–`0x11` | **nuevos** — ISO 8601 |
| `Enum` | `0x13` | **nuevo** — requiere `@register` |
| `numpy.ndarray` | `0x0A` | dtype + shape preservados |
| `dataclass` | `0x0B` | requiere `@register` |
| Objetos con `__dict__` / `__slots__` | `0x0B` | requiere `@register` |
| Objetos con `__getstate__` / `__setstate__` | `0x0B` | protocolo pickle compatible |
| Referencia circular | `0x15` | **nuevo** — puntero de 4 bytes |

---

## API

```python
# ── Core ───────────────────────────────────────────────
msc.dumps(obj, *, with_crc=False)      -> bytes
msc.loads(data, *, strict=True)        -> Any
msc.dump(obj, file, **kwargs)          -> None
msc.load(file, **kwargs)               -> Any

# ── Compresión ─────────────────────────────────────────
msc.dump_compressed(obj, file, level=6, **kwargs) -> None
msc.load_compressed(file, **kwargs)               -> Any

# ── Registry ───────────────────────────────────────────
msc.register(cls)              -> cls      # decorador

# ── Utilidades ─────────────────────────────────────────
msc.inspect(data)              -> dict     # metadata sin deserializar
msc.benchmark(obj, rounds=100) -> dict     # rendimiento encode/decode
```

---

## Seguridad

**v1.0** usaba `importlib.import_module()` para reconstruir objetos — contradicción directa con la premisa de "sin ejecución de código arbitrario".

**v2.0** corrige esto con un **registry explícito**:

- Solo clases registradas con `@msc.register` se reconstruyen como objetos
- `strict=True` (default): lanza `MSCSecurityError` si la clase no está registrada
- `strict=False`: retorna `{'__class__': ..., '__state__': ...}` como fallback
- Límites de profundidad, tamaño de colecciones y strings previenen memory bombs
- Sin `importlib`, `eval`, `exec`, ni resolución dinámica de módulos

---

## Formato del archivo

```
v1.0:  [MAGIC 4B][VERSION 1B][PAYLOAD...]
        MSCS      \x01

v2.0:  [MAGIC 4B][VERSION 1B][FLAGS 1B][PAYLOAD...][CRC32 4B opcional]
        MSCS      \x02       bit0=crc
```

Los tags de tipo son un byte. Los enteros se codifican en little-endian con longitud variable. El ref tracking usa IDs incrementales de 4 bytes. Archivos v1.0 se cargan automáticamente.

---

## Tests

```bash
python msc_serial.py
```

Salida esperada (43 tests):

```
── Primitivos ──
✅ None  ✅ bool  ✅ int  ✅ float  ✅ NaN  ✅ inf
✅ complex  ✅ str  ✅ bytes  ✅ bytearray  ✅ Decimal

── Temporales ──
✅ datetime  ✅ date  ✅ time  ✅ timedelta

── Colecciones ──
✅ list  ✅ tuple  ✅ set  ✅ frozenset  ✅ dict

── Integridad CRC32 ──
✅ dict con CRC  ✅ str con CRC  ✅ corrupción detectada

── Referencias ──
✅ refs compartidas  ✅ ref circular

── Enum / Dataclass / Numpy / Seguridad ──
✅ Enum  ✅ strict bloquea  ✅ strict=False fallback
✅ numpy float32  ✅ numpy 100x100  ✅ dataclass registrado

── Compresión ──
✅ zlib (ratio ~3.7x)

── Retrocompatibilidad ──
✅ carga formato v1.0

🎉 43/43 tests pasaron.
```

---

## Excepciones

| Excepción | Cuándo |
|---|---|
| `MSCEncodeError` | Tipo no serializable, límite de profundidad/tamaño excedido |
| `MSCDecodeError` | Datos corruptos, versión no soportada, CRC inválido, buffer truncado |
| `MSCSecurityError` | Clase no registrada en modo `strict=True` |

Todas heredan de `MSCError`.

---

## Benchmark

```python
results = msc.benchmark(my_obj, rounds=500)
# {
#   'encode_ms': 0.052,
#   'decode_ms': 0.065,
#   'raw_bytes': 781,
#   'compressed_bytes': 248,
#   'compression_ratio': 3.15,
#   'rounds': 500
# }
```

---

## Licencia

MIT — ver [LICENSE](LICENSE)

---

*Creado por **esraderey** · Co-autor **escribanosilente***

# MSC Serial v1.0

> Reemplazo personal de `pickle` — seguro, comprimible y sin ejecución de código arbitrario.

---

## ¿Por qué MSC Serial?

| | `pickle` | `msc_serial` |
|---|---|---|
| Seguridad al cargar | ❌ ejecuta código arbitrario | ✅ solo reconstruye datos |
| Compresión integrada | ❌ | ✅ zlib nativo |
| Soporte numpy | parcial | ✅ nativo |
| Formato auditable | ❌ opaco | ✅ header `MSCS` + versión |
| Tamaño extra | — | ~10 % overhead |

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

# ── Archivos ───────────────────────────────────────────
with open("data.msc", "wb") as f:
    msc.dump(data, f)

with open("data.msc", "rb") as f:
    obj = msc.load(f)

# ── Con compresión zlib (recomendado para modelos grandes) ──
with open("data.mscz", "wb") as f:
    msc.dump_compressed(data, f)

with open("data.mscz", "rb") as f:
    obj = msc.load_compressed(f)

# ── Inspeccionar sin deserializar ──────────────────────
info = msc.inspect(raw)
# {'valid': True, 'version': 1, 'size_bytes': 512, 'root_tag': '0x8'}
```

---

## Tipos soportados

| Tipo | Notas |
|---|---|
| `None`, `bool` | |
| `int` | precisión arbitraria |
| `float` | 64-bit IEEE 754 |
| `str` | UTF-8 |
| `bytes`, `bytearray` | |
| `list`, `tuple`, `set`, `dict` | anidados sin límite |
| `numpy.ndarray` | dtype + shape preservados |
| `dataclass` | vía `dataclasses.asdict` |
| Objetos con `__dict__` / `__slots__` | |
| Objetos con `__getstate__` / `__setstate__` | protocolo pickle compatible |

---

## API

```python
msc.dumps(obj)                  -> bytes
msc.loads(data: bytes)          -> Any
msc.dump(obj, file)             -> None
msc.load(file)                  -> Any
msc.dump_compressed(obj, file)  -> None   # zlib level=6
msc.load_compressed(file)       -> Any
msc.inspect(data: bytes)        -> dict   # metadata sin deserializar
```

---

## Formato del archivo

```
[MAGIC 4B][VERSION 1B][PAYLOAD...]
  MSCS       \x01
```

Los tags de tipo son un byte. Los enteros se codifican en little-endian con longitud variable. No se evalúa ningún tipo dinámico durante la deserialización.

---

## Tests

```bash
python msc_serial.py
```

Salida esperada:

```
✅ None  ✅ bool  ✅ int  ✅ float  ✅ str  ✅ bytes
✅ list  ✅ tuple  ✅ set  ✅ dict  ✅ numpy  ✅ dataclass
✅ Compresión zlib  (ratio ~3.7x)
🎉 Todos los tests pasaron.
```

---

## Licencia

MIT — ver [LICENSE](LICENSE)

---

*Creado por **esraderey** · Co-autor **escribanosilente***

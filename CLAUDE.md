# CLAUDE.md — MSC Serial (mscs)

Reemplazo seguro de `pickle`. Núcleo: `src/mscs/_core.py`. API pública en `src/mscs/__init__.py`.

## Gates de verificación

- **G0 (import):** `python -c "import sys; sys.path.insert(0,'src'); import mscs"`
- **G3 (tests):** `python -m pytest -q`  (suite: `tests/test_mscs.py`, `tests/test_fuzz.py`)
- Auditoría adversarial ad-hoc: `python tests/security_audit.py`
- No hay ruff/mypy configurados; el estilo se sigue por convención del archivo (UTF-8, acentos en mensajes, `_TAG` en bytes).

## Modelo de seguridad (leer antes de tocar el decoder)

- La garantía es "no ejecuta código arbitrario": solo reconstruye clases **registradas** (`register`). `__setstate__` de clases registradas **sí** se ejecuta; el registro implica confianza.
- `loads()` despacha por byte de versión. **Todo control de seguridad (HMAC, CRC, trailing bytes, strict) debe aplicarse en cada rama de versión.** El formato v1 (legacy) no tiene flags ni integridad in-band.
- `_is_safe_dtype()` es la única barrera entre un `dtype` del payload y `numpy.frombuffer`; un `object`/`void` que la sortee = lectura de punteros arbitrarios. No relajarla sin differential contra `numpy.dtype(...).kind`.

## Post-mortems (lecciones que persisten)

- **2026-07-06 — Bypass de HMAC vía downgrade a v1 (CRÍTICO):** la rama `ver == b'\x01'` de `loads()` retornaba antes de la lógica HMAC, ignorando `hmac_key` por completo y forzando `strict=False`. Un payload v1 forjado evadía toda la autenticación. **Causa:** el despacho por versión ocurría fuera de los controles de la ruta v2. **Fix:** rechazar v1 cuando `hmac_key is not None` (fail-closed) y respetar el `strict` del llamante. **Detectarlo antes:** todo control de seguridad nuevo debe probarse contra *cada* versión de formato aceptada, no solo la actual. Anclas: `test_hmac_v1_downgrade_rejected`, `test_v1_respects_strict_true`.
- **2026-07-06 — Zip-bomb en `load_compressed` (MEDIO):** `zlib.decompress(compressed, bufsize=orig_size)` materializaba toda la salida antes del chequeo `len(raw) > MAX_SIZE` (`bufsize` es una pista, no un tope), así que una bomba agotaba memoria pese al límite. **Fix:** descompresión incremental con `decompressobj()` acotando la salida a `MAX_SIZE + 1` y abortando al cruzarlo — pico ≈ `MAX_SIZE` en vez del tamaño de la bomba. **Detectarlo antes:** validar tamaño *después* de asignar no protege de DoS de memoria; acotar durante. Ancla: `test_zip_bomb_bounded_memory`.
- **2026-07-06 — Confusión de tipos con tag ENUM (MEDIO):** el decoder hacía `cls(value)` para cualquier clase registrada sin verificar que fuera un `Enum`, invocando el constructor con `value` del atacante (fuera del modelo documentado, que solo cubre `__setstate__`). **Fix:** `issubclass(cls, Enum)` antes de instanciar. **Detectarlo antes:** un tag no debe habilitar una operación (llamar al constructor) que su tipo no justifica. Ancla: `test_enum_tag_rejects_non_enum_class`.

## Hallazgos de auditoría pendientes (peritaje 2026-07-06)

- [BAJO] dataclass `frozen`: round-trip roto (decode usa `setattr`). Fix: `object.__setattr__`.

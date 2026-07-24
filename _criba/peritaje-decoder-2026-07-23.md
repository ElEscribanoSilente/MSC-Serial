# Auditoría: mscs decoder (`src/mscs/_core.py`) — 2026-07-23

Peritaje A3 (deserialización de entrada no confiable) sembrado por la criba del
mismo día. Cuadrilla de 4 peritos en paralelo sobre clases ortogonales
(deserialización, parsing/placeholders, recursos/DoS, criptografía) + reproductor.
Cada hallazgo alto re-disparado en persona por el director. Suite base: **284
passed** antes de tocar nada (commit `2d75cc8`, rama `release/2.5.0`).

---

## Alcance y modelo de amenazas

**Auditado:** `src/mscs/_core.py` (decoder completo: `loads`, `load`,
`load_compressed`, `decode`, todas las ramas de tag, registro de clases,
HMAC/CRC, límites de recursos) y `src/mscs/__init__.py` (solo re-exports).

**Fuera de alcance:** encoder salvo donde se necesitó para forjar PoCs; código de
test (cubierto por la criba); `benchmark.py`.

**Activos** (por dolor): (1) no ejecución de código fuera del modelo documentado
(`__setstate__` de clases registradas); (2) integridad del objeto reconstruido
(sin clobbering/aliasing/inyección por payload forjado); (3) autenticación HMAC
sin downgrade; (4) disponibilidad (sin DoS de memoria/CPU); (5) confidencialidad
de la clave HMAC.

**Atacante:** controla por completo el blob pasado a `loads`/`load`/
`load_compressed`. No puede llamar `register()` ni conoce la clave HMAC. Modelo de
entrada totalmente no confiable.

**Límite de confianza crítico:** la frontera `loads()` donde bytes del atacante
cruzan a reconstrucción de objetos Python; y la barrera `_is_safe_dtype ↔
numpy.frombuffer`.

---

## Resumen ejecutivo

**5 hallazgos: 2 ALTOS, 1 MEDIO, 1 BAJO, 1 informativo. Cero críticos.** Todos los
altos y el medio confirmados con PoC que dispara (E3), re-verificados por el
director. La garantía núcleo "no ejecuta código arbitrario" **se mantiene** (no se
encontró RCE); lo que se rompe es el sub-modelo "solo `__setstate__` se ejecuta" y
la integridad del objeto reconstruido.

Lo que de verdad importa, en una línea:

1. **La rama de reconstrucción de dataclass no filtra las claves del `state`**
   (ALTO) — un payload forjado inyecta atributos arbitrarios vía `'__dict__'` e
   invoca setters de `@property` con valores del atacante. El fix ya existe en la
   rama hermana de slots; nunca se portó.
2. **El número de dimensiones del shape de NDARRAY/TENSOR no tiene techo** (ALTO) —
   DoS de amplificación de memoria ~10× medido; la única "colección" del formato
   sin el cap `MAX_COLLECTION` que todas las demás sí aplican.
3. **La rama v1 de `loads()` no valida trailing bytes** (MEDIO) — diferencial de
   parser / smuggling; la rama v2 sí lo hace.

**Meta-patrón que gobierna los tres:** cada uno es "un control aplicado a unos
caminos pero no a su gemelo" (dataclass vs. slots; shape vs. resto de colecciones;
v1 vs. v2). Sumados a los dos post-mortems históricos de la misma forma (downgrade
HMAC v1, clobbering slots), son **cinco instancias del mismo defecto de proceso**.

---

## Hallazgos

### [ALTO] La rama dataclass del decoder asigna claves del `state` sin filtrar

- **Ubicación:** `src/mscs/_core.py:1272-1283`
- **Clase:** deserialización / confusión de tipos
- **Descripción:** para una clase registrada que es dataclass, el decoder hace
  `for k, v in state.items(): object.__setattr__(obj, k, v)` sin comparar `k`
  contra `{f.name for f in dataclasses.fields(cls)}`. El atacante controla las
  claves del `state`. La rama hermana de slots/`__dict__` (1284-1316) **sí** filtra
  —fue endurecida en el post-mortem 2026-07-13 con tres tests ancla— pero la rama
  dataclass, de idéntica forma, quedó fuera del fix. Dos vectores confirmados:

  **Vector A — clobbering + aliasing de `__dict__`.** Una clave forjada `'__dict__'`
  hace `object.__setattr__(obj, '__dict__', valor_atacante)`, que **reemplaza el
  dict de instancia entero**. Los fields legítimos desaparecen; el atacante inyecta
  atributos no declarados; con un `_REF` a un dict compartido, dos instancias
  independientes terminan con el **mismo** `__dict__` (aliasing de identidad).

  **Vector B — invocación de setters de `@property`.** `object.__setattr__` respeta
  los descriptores de datos de la *clase* (solo evita un `__setattr__` de
  instancia). Una clave que coincide con el nombre de una `@property` —que nunca fue
  field— invoca su **setter** con el valor del atacante. Contradice el modelo
  documentado "solo `__setstate__` se ejecuta"; no depende de `__dict__`.

- **PoC (re-disparado por el director):**
  ```
  # Vector A — dataclass Point(x, y) registrada, state forjado {'__dict__': {...}}
  [ataque1] obj.__dict__ = {'x':1,'y':2,'injected_by_attacker':'PWNED','is_admin':True}
  [ataque1] getattr(obj,'is_admin') = True   <- atributo que NO es field de Point
  [control ] obj benigno tiene solo fields: True

  # Vector B — dataclass Account con @property admin (no field), state {'admin':True}
  [benigno] llamadas al setter 'admin': []            <- round-trip real no lo toca
  [ataque ] llamadas al setter 'admin': [('admin_setter', True)]   <- setter del atacante
  ```
  Variante no-dict (`{'__dict__': 12345}`) → `MSCDecodeError` (falla cerrado; no es
  vector adicional).
- **Impacto:** corrupción de integridad del objeto reconstruido, inyección de
  atributos consultables por nombre (riesgo de escalada si la app lee flags como
  `obj.is_admin`), aliasing entre objetos asumidos independientes, y ejecución de
  setters del desarrollador con argumentos del atacante fuera del punto confiado.
  No es RCE (`object.__setattr__` no ejecuta código arbitrario del atacante).
- **Severidad ALTA:** explotabilidad trivial (cualquier dataclass registrada;
  `strict=True` no protege), entrada no confiable, y el propio proyecto clasificó
  el gemelo de slots como ALTA. Confirmado E3.
- **Remediación (forja):** filtrar `state.items()` contra los field names de la
  dataclass antes de asignar, fail-closed para claves no reconocidas — misma
  política que ya usa la rama slots pura. Un solo filtro cierra ambos vectores.
  Tests ancla: portar `test_hybrid_spurious_dict_key_no_clobbering`,
  `_no_aliasing` y un nuevo `test_dataclass_property_setter_not_invoked`.

### [ALTO] El conteo de dimensiones del shape (NDARRAY/TENSOR) no tiene techo

- **Ubicación:** `src/mscs/_core.py:1162` (NDARRAY) y `:1179` (TENSOR)
- **Clase:** manejo de recursos / amplificación (DoS de memoria)
- **Descripción:** `shape = tuple(int(x) for x in shape_str.split('x'))` se ejecuta
  antes de leer los datos crudos, y el **número de dimensiones** no pasa por
  `_read_length(MAX_COLLECTION)` como `LIST`/`TUPLE`/`DICT`/`SET`/`FROZENSET`/
  `DEQUE`. Solo lo acota indirectamente `MAX_STRING` (100MB) sobre el string `meta`,
  permitiendo decenas de millones de "dimensiones". El `.split('x')` materializa N
  substrings antes de que numpy rechace por `NPY_MAXDIMS=64`.
- **PoC (re-disparado por el director, escala segura):**
  ```
  [control lista] lista de 11M elementos: rechazada en 0.0000s O(1) (MAX_COLLECTION)
  [ataque ndarray] shape de 11M dimensiones (mismo N):
      numpy rechazó tras 4.49s | PICO 234.0 MB con 22.0 MB de input | amplificación 10.6x
  ```
  A escala real (perito C): input de ~100MB (20% del `MAX_SIZE` default de 512MB) →
  **~1.1GB de pico y 21s**. Con el default, el techo de amplificación por request
  ronda los ~5GB.
- **Impacto:** un solo request mediano fuerza memoria transitoria ~10× y segundos de
  CPU; varios en paralelo agotan RAM. Acotado (no OOM ilimitado) y no encadenable
  (numpy aborta al primer shape), pero un DoS de memoria realista con el `max_size`
  por defecto.
- **Severidad ALTA:** calibración honesta — el atenuante es que requiere input
  grande y que bajar `max_size` lo reduce; el agravante es la amplificación medida
  y que es la única colección sin el cap que el resto del formato sí aplica.
  Confirmado E3.
- **Remediación (forja):** acotar el número de dimensiones (contar tokens de
  `shape_str.split('x')`, o validar la longitud del `meta` de shape, contra
  `MAX_COLLECTION` o un cap de dimensiones específico) antes de materializar el
  tuple. Test ancla: shape con >`MAX_COLLECTION` dimensiones → `MSCDecodeError` O(1).

### [MEDIO] La rama v1 de `loads()` no valida trailing bytes

- **Ubicación:** `src/mscs/_core.py:1413-1429` (v1) vs. `:1480-1486` (v2)
- **Clase:** lógica de parsing / diferencial de parser
- **Descripción:** la rama v1 hace `return dec.assert_fully_resolved(dec.decode())`
  sin comprobar `buf.tell()` contra `len(data)`; la v2 sí (`consumed != end_pos` →
  `MSCDecodeError`). Un blob v1 con basura anexada decodifica ignorándola.
- **PoC (re-disparado por el director):**
  ```
  [v1] loads('MSCS\x01...' + 42 bytes de basura) = 42   *** ignora la basura ***
  [v2] mismo ataque contra dumps() por defecto: rechazado (Trailing bytes: 10 de 52)
  [smuggling] dos mensajes v1 concatenados: solo el primero se procesa; el 2.º desaparece
  ```
- **Impacto:** diferencial de parser / smuggling — dos blobs distintos producen el
  mismo objeto; contenido no examinado se cuela tras un v1 válido. No es bypass de
  auth (v1 con `hmac_key` ya se rechaza) ni RCE. Rompe el invariante propio de
  CLAUDE.md ("trailing bytes en cada rama de versión").
- **Severidad MEDIA.** Confirmado E3.
- **Remediación (forja):** portar la comprobación de trailing a la rama v1. Test
  ancla: `loads(v1_blob + b'X')` → `MSCDecodeError`.

### [BAJO] `max_depth` alto pierde significado pero degrada seguro

- **Ubicación:** chequeo de profundidad en `decode` (~819) vs. límite de recursión
  de Python.
- **Descripción:** con `max_depth` por encima de ~500, el `RecursionError` nativo
  dispara antes que el chequeo propio — pero se captura y envuelve como
  `MSCDecodeError` (sin fuga cruda, sin crash; probado hasta 100k niveles). Gap de
  claridad, no vulnerabilidad.
- **Severidad BAJA / robustez.**
- **Remediación (opcional):** documentar que `max_depth` efectivo está limitado por
  `sys.getrecursionlimit()`, o convertir la recursión en iterativa si se quiere un
  techo real configurable.

### [Informativo] CRC32 comparado con `!=` (no constant-time)

`_core.py:~1465`. No protege ningún secreto (el CRC no lleva clave; un atacante
recalcula uno válido) y el código lo trata coherentemente como corrupción
accidental (`MSCDecodeError`, nunca `MSCSecurityError`; mutuamente excluyente con
HMAC). No es vulnerabilidad; se anota para que ningún camino futuro trate un CRC
válido como autenticación.

---

## Superficie sin hallazgos confirmados

Examinada activamente con PoC, quedó limpia:

- **Criptografía (HMAC) — 33/33 comprobaciones adversariales fallaron cerrado:**
  `hmac.compare_digest` es el único punto de comparación de MAC (no hay `==`);
  downgrade v1+`hmac_key` rechazado (confirmado con `__setstate__` instrumentado que
  nunca se ejecutó); v2 sin flag + key, y flag sin key, ambos rechazados; orden
  verificar-antes-de-decodificar correcto; HMAC cubre el blob completo (cut-and-
  paste y flip del byte de flags rechazados; usa `hmac.new`, no vulnerable a
  length-extension); `hmac_key=b''` tratado como clave provista; sin fuga de clave
  ni MAC en mensajes de error.
- **Resolución de placeholders `_Pending`:** cada punto de emisión (LIST, TUPLE,
  FROZENSET, SET, DICT clave/valor, DEQUE, ENUM, OBJ en todas sus variantes) o
  difiere con fix-up o falla cerrado; ninguno almacena el sentinela crudo.
  `assert_fully_resolved` atrapa un pending enterrado 4 niveles vía chequeo global.
- **Zip-bomb / `load_compressed`:** descompresión incremental acota el pico a
  ~`size_limit` frente a bombas de 32MB y 500MB (picos medidos 1.4KB–2.1MB);
  integridad de contenedor (`eof`, `unused_data`, `orig_size`, header truncado,
  stream corrupto) falla cerrado.
- **`MAX_SIZE` como tope total:** `load()` nunca lee más de `max_size+1` bytes;
  ningún campo preasigna memoria proporcional a una longitud declarada mayor que los
  bytes reales.
- **Conversiones bytes→longitud/índice:** `_read_length` unsigned (sin negativos);
  `MAX_COLLECTION` rechazado O(1); `_REF` contra dict (no `IndexError` crudo);
  `_DEQUE` maxlen signed rechazado en los bordes; overflow del producto de shape
  rechazado por numpy sin asignar.
- **Reconstrucción de objetos (otras ramas):** guard `issubclass(cls, Enum)` en
  ENUM no sorteable; `__setstate__` dentro del modelo documentado con guard
  `_contains_pending`; rama slots general protegida contra `'__dict__'`/`'__weakref__'`
  espurios; fallback `strict=False` no instancia ni ejecuta nada.
- **Amplificación por `_REF`:** 500k referencias a un objeto grande decodifican como
  aliasing barato (identidad compartida), no como N copias.

---

## Sospechas sin confirmar

- **Shape `(0, N)` con `raw` vacío** produce un array cuyo `.shape` reporta una
  dimensión gigante sin memoria detrás — footgun de integridad de metadata, no de
  recursos; no se armó PoC de impacto.
- **`cls(value)` en ENUM** podría en teoría invocar dunders maliciosos de otra clase
  registrada, pero no es un vector nuevo (SET/DICT-KEY ya lo exponen vía hashing).
- **`max_depth` en otra plataforma/build:** no se descarta que un SO/Python distinto
  alcance un stack overflow real en el escenario compuesto de PC.2; no se disparó
  aquí pese a intentarlo.
- **Cobertura de `security_audit.py`:** el script ad-hoc no tiene sección de HMAC/
  downgrade (la cobertura vive en `pytest`); brecha de cobertura del script, no del
  código.

---

## Escalada → forja

Remediaciones ordenadas por retorno (el sistémico primero):

1. **Filtro de claves en la rama dataclass** (cierra el ALTO PA, ambos vectores) —
   portar la política de la rama slots. Mayor retorno: es el hallazgo de más
   severidad y el fix ya existe de referencia.
2. **Cap del número de dimensiones del shape** (cierra el ALTO PC.1) — aplicar
   `MAX_COLLECTION` al conteo de dimensiones antes de materializar el tuple.
3. **Validación de trailing en la rama v1** (cierra el MEDIO PB.1).
4. Sistémico: un **test de paridad** que ejercite cada control (filtro de claves,
   caps de colección, trailing, downgrade) contra cada rama/branch — es la defensa
   contra la sexta instancia de este meta-patrón.

El PoC de cada hallazgo (en `scratchpad/peritaje/`) es el test ancla de forja:
rojo antes del fix, verde después.

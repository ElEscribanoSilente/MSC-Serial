"""
MSCS — Safe serialization for Python. A secure, fast replacement for pickle.

Usage:
    import mscs

    data = mscs.dumps(obj)
    obj = mscs.loads(data)

    mscs.register(MyClass)  # allow deserialization of custom classes
"""
from mscs._core import (
    # Version
    __version__,
    # Public API
    dump,
    load,
    dumps,
    loads,
    dump_compressed,
    load_compressed,
    register,
    register_alias,
    register_module,
    inspect,
    benchmark,
    copy,
    # Exceptions
    MSCError,
    MSCEncodeError,
    MSCDecodeError,
    MSCSecurityError,
    # Constants (for advanced users)
    MAGIC,
    VERSION,
    MAX_DEPTH,
    MAX_SIZE,
    MAX_COMPRESSED,
    MAX_COLLECTION,
    MAX_STRING,
)

__all__ = [
    "__version__",
    "dump", "load", "dumps", "loads",
    "dump_compressed", "load_compressed",
    "register", "register_alias", "register_module",
    "inspect", "benchmark", "copy",
    "MSCError", "MSCEncodeError", "MSCDecodeError", "MSCSecurityError",
]

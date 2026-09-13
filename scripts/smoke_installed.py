"""Check the installed distribution with: python -I scripts/smoke_installed.py."""
from dataclasses import dataclass
from enum import IntEnum
import importlib.metadata
import io
from pathlib import Path
import sys

import mscs


def main():
    root = Path(__file__).resolve().parents[1]
    package_path = Path(mscs.__file__).resolve()
    if root / "src" in package_path.parents:
        raise SystemExit("Expected an installed wheel, not checkout/src")
    assert mscs.__version__ == importlib.metadata.version("mscs")

    @mscs.register
    class State(IntEnum):
        READY = 1

    @mscs.register
    @dataclass
    class Record:
        state: State
        values: list

    key = b"public-smoke-test-key-not-a-credential"
    shared = [1, 2]
    source = Record(State.READY, [shared, shared])
    encoded = mscs.dumps(source, hmac_key=key)
    result = mscs.loads(encoded, hmac_key=key, max_hash_work=10_000)
    assert result.state is State.READY
    assert result.values[0] is result.values[1]
    for bad in (encoded[:-1] + bytes([encoded[-1] ^ 1]), mscs.dumps(source)):
        try:
            mscs.loads(bad, hmac_key=key)
        except mscs.MSCSecurityError:
            pass
        else:
            raise AssertionError("Authentication accepted modified/unsigned input")
    file = io.BytesIO()
    mscs.dump_compressed(source, file, hmac_key=key)
    file.seek(0)
    assert mscs.load_compressed(file, hmac_key=key).state is State.READY
    cycle = []
    cycle.append(cycle)
    restored = mscs.loads(mscs.dumps(cycle))
    assert restored[0] is restored
    print("Installed smoke OK:", mscs.__version__, sys.version.split()[0], package_path)


if __name__ == "__main__":
    main()

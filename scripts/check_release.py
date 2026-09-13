"""Validate release versions and archive contents without importing the package."""
import argparse
import ast
from email.parser import BytesParser
import hashlib
import os
from pathlib import Path
import re
import tarfile
import zipfile

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--tag", default=os.environ.get("RELEASE_TAG", ""))
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    require(re.fullmatch(r"\d+\.\d+\.\d+", version), "Expected a stable x.y.z release")
    tree = ast.parse((ROOT / "src/mscs/_core.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    require(versions == [version], "Source and project versions differ")
    require("**" + version + "**" in (ROOT / "README.md").read_text(encoding="utf-8"),
            "README version differs")
    require("## [" + version + "]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
            "Missing changelog section")
    if args.tag:
        require(args.tag == "v" + version, "Release tag differs from package version")
    if args.dist is None:
        print("Release metadata OK:", version)
        return

    wheel_name = "mscs-" + version + "-py3-none-any.whl"
    sdist_name = "mscs-" + version + ".tar.gz"
    expected = {wheel_name, sdist_name}
    actual = {p.name for p in args.dist.iterdir() if p.is_file()}
    require(expected <= actual, "Wheel or sdist missing")
    require(actual <= expected | {"SHA256SUMS"}, "Unexpected files in distribution directory")
    sources = {p.relative_to(ROOT).as_posix(): p.read_bytes()
               for p in (ROOT / "src/mscs").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    require(set(sources) == {"src/mscs/_core.py", "src/mscs/__init__.py", "src/mscs/py.typed"},
            "Review package allowlist after changing its files")

    def metadata_ok(raw):
        metadata = BytesParser().parsebytes(raw)
        require(metadata["Name"] == "mscs" and metadata["Version"] == version,
                "Archive metadata differs")
        require(metadata["Requires-Python"] == project["requires-python"],
                "Requires-Python differs")
        require(metadata.get_content_type() == "text/plain", "Unexpected metadata encoding")
        require(metadata["Description-Content-Type"] == "text/markdown",
                "README content type differs")

    with zipfile.ZipFile(args.dist / wheel_name) as archive:
        names = archive.namelist()
        prefix = "mscs-" + version + ".dist-info/"
        required = {n.removeprefix("src/") for n in sources} | {
            prefix + "METADATA", prefix + "WHEEL", prefix + "RECORD", prefix + "licenses/LICENSE"}
        require(set(names) == required and len(names) == len(required), "Unexpected wheel contents")
        for name, data in sources.items():
            require(archive.read(name.removeprefix("src/")) == data, "Wheel source differs: " + name)
        require(archive.read(prefix + "licenses/LICENSE") == (ROOT / "LICENSE").read_bytes(),
                "Wheel license differs")
        metadata_ok(archive.read(prefix + "METADATA"))

    docs = {".gitignore", "README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml", "RELEASING.md", "SECURITY.md"}
    public = dict(sources)
    for folder in ("tests", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            public[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    public.update({name: (ROOT / name).read_bytes() for name in docs})
    with tarfile.open(args.dist / sdist_name, "r:gz") as archive:
        prefix = "mscs-" + version + "/"
        members = [m for m in archive.getmembers() if not m.isdir()]
        require(all(m.isfile() and m.name.startswith(prefix) for m in members),
                "Unsafe sdist member")
        names = [m.name[len(prefix):] for m in members]
        require(set(names) == set(public) | {"PKG-INFO"} and len(names) == len(set(names)),
                "Unexpected or missing sdist contents")
        for name, data in public.items():
            require(archive.extractfile(prefix + name).read() == data, "sdist source differs: " + name)
        metadata_ok(archive.extractfile(prefix + "PKG-INFO").read())

    sums = "".join(hashlib.sha256((args.dist / name).read_bytes()).hexdigest() + "  " + name + "\n"
                   for name in sorted(expected))
    manifest = args.dist / "SHA256SUMS"
    if manifest.exists():
        require(manifest.read_text(encoding="utf-8") == sums, "Existing SHA256SUMS differs")
    else:
        manifest.write_text(sums, encoding="utf-8")
    print("Release archives OK:", version)
    print(sums, end="")


if __name__ == "__main__":
    main()

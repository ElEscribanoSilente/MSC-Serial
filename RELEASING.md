# Releasing MSCS

The prepared version is **2.6.0**. Creating artifacts or running the manual
workflow does not publish a release.

## Validate and build locally

Run in a fresh checkout or isolated virtual environment:

```bash
python -m pip install -e ".[all,test]"
python -m pytest -q
python tests/security_audit.py
python -m pip install build twine "tomli; python_version < '3.11'"
python scripts/check_release.py --tag v2.6.0
python -m build --outdir dist/2.6.0
python -m twine check --strict dist/2.6.0/*.whl dist/2.6.0/*.tar.gz
python scripts/check_release.py --dist dist/2.6.0 --tag v2.6.0
```

Start with an empty version-specific output directory. The checker rejects
unrelated artifacts and checks the package version, tag, file allowlists,
source bytes, metadata and license. It writes a SHA256SUMS manifest for the
wheel and sdist. Keep SHA256SUMS out of commands that upload packages to PyPI.

Create a separate clean virtual environment, install the wheel, and run:

```bash
python -m pip install dist/2.6.0/mscs-2.6.0-py3-none-any.whl
python -I scripts/smoke_installed.py
```

The smoke check rejects imports from checkout/src. Also build/install the
sdist on Python 3.9, the supported minimum. Tests intentionally import local
source, so a source pytest run alone does not validate the installed wheel.

## Prepare GitHub

The canonical repository in package metadata is
[ElEscribanoSilente/MSC-Serial](https://github.com/ElEscribanoSilente/MSC-Serial).
The local fork may be used for a pull request; check `git remote -v` before
pushing. Do not change the canonical package URLs just to prepare a fork.

Stage the release files explicitly so unrelated working changes stay separate:

```bash
git add .gitignore .github/workflows/ci.yml .github/workflows/release.yml
git add README.md CHANGELOG.md RELEASING.md SECURITY.md pyproject.toml
git add src/mscs/_core.py src/mscs/__init__.py scripts/check_release.py scripts/smoke_installed.py
git add tests/test_audit_repairs.py tests/test_enum_diagnostics.py tests/test_security_invariants.py
git diff --cached --check
git diff --cached --stat
```

Review the staged diff, commit it and push the intended branch. Open/merge a
pull request as appropriate. Local caches, distributions, credentials, agent
state and audit scratch files are ignored. Previously tracked files remain
tracked even if a new ignore rule matches them.

The CI workflow runs source tests on Python 3.9–3.14 on Ubuntu and Windows,
checks optional NumPy/PyTorch support, builds the sdist/wheel, validates their
metadata and contents, and smoke-tests installed wheels on both systems.
Run the **Release** workflow manually on the intended commit for a validation
run. Manual dispatch never uploads to PyPI or creates a GitHub release.

## Configure publishing once

In PyPI's `mscs` project, add a GitHub
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
with these exact values:

| Setting | Value |
| --- | --- |
| Owner | `ElEscribanoSilente` |
| Repository | `MSC-Serial` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Create the GitHub environment `pypi` and restrict it to the intended release
tags/reviewers. The publish job requests `id-token: write`; no PyPI token is
stored in the repository. These account-side settings must be configured by a
maintainer with access. The repository guard intentionally prevents a fork's
release from publishing to PyPI. If publishing from a different repository is
intended, update that guard and the Trusted Publisher registration together.

See [PyPI's publishing instructions](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
for OIDC setup and troubleshooting.

## Publish the reviewed commit

Confirm CI is green and that version 2.6.0 is still available on PyPI immediately
before publication. PyPI release files cannot be replaced with different bytes.

Create a `v2.6.0` tag on the reviewed commit, push that tag to the canonical
repository, and create a draft GitHub release for it. Use the 2.6.0 changelog
entry as release notes. Publishing that draft as a stable release triggers:

1. All CI checks on the tagged source, including tag/version agreement.
2. Publication of the validated wheel and sdist to PyPI via Trusted Publishing.
3. Attachment of those same packages and SHA256SUMS to the GitHub release.

The workflow does not publish prereleases, manual validation runs or releases
from forks. It uses narrowly scoped permissions and pinned action commits.
If PyPI upload succeeds but GitHub asset attachment fails, rerun only the failed
asset job; do not rebuild or attempt to overwrite an existing PyPI version.

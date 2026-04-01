# PyPI Release Guide

This guide describes the recommended way to publish `spatho` to PyPI.

## Recommended Publishing Model

Use PyPI Trusted Publishing with GitHub Actions.

Why:

- no long-lived PyPI API token needs to be stored in GitHub
- the release is tied to a specific GitHub Actions workflow
- PyPI automatically produces provenance attestations for Trusted Publishing uploads

Official references:

- [PyPI: creating a project with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
- [PyPI: publishing with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)

## One-Time Setup

1. Create or sign in to your PyPI account at [pypi.org](https://pypi.org/).
2. Confirm the project name is the one you want to publish.
3. On PyPI, create the project through Trusted Publishing or add a trusted publisher for an existing project.
4. Use these GitHub settings:

- Repository owner: `hutaobo`
- Repository name: `AI-Driven-Spatial-Pathologist`
- Workflow file: `publish-pypi.yml`
- Environment name: `pypi`

5. In GitHub, create an environment named `pypi`.
6. Optionally add manual approval rules to the `pypi` environment.

## Release Process

1. Bump `spatho.__version__` in [src/spatho/__init__.py](../src/spatho/__init__.py).
2. Commit the version bump.
3. Tag the release:

```bash
git tag v0.1.0
git push origin main --tags
```

4. GitHub Actions will:

- build the sdist and wheel
- upload them as workflow artifacts
- publish them to PyPI through Trusted Publishing

## Local Preflight Checks

Run these before tagging:

```bash
python -m pip install -e .[dev]
python -m pytest tests
python -m build
python -m twine check dist/*
```

## TestPyPI

If you want a dress rehearsal, use TestPyPI first.

Official guide:

- [Packaging guide: using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/)

You can either:

- create a separate TestPyPI publishing workflow
- or upload locally with `twine` for a one-off check

## Notes for This Project

- `spatho` currently depends on `histoseg`, so public installation will expect `histoseg` to be resolvable.
- If `histoseg` itself is not on PyPI, then `spatho` cannot be installed cleanly from PyPI alone yet.
- Before the first real public release, either publish `histoseg` as a dependency or vendor/migrate the required runtime modules into `spatho`.

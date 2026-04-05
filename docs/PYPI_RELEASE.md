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
3. Commit and push the release branch.
4. Create a Git tag that matches the package version.
5. Create a GitHub Release from that tag and publish it.

Example:

```bash
git tag v0.1.0
git push origin main
git push origin v0.1.0
```

Then publish a GitHub Release for `v0.1.0`.

GitHub Actions will:

- build the sdist and wheel
- upload them as workflow artifacts
- publish them to PyPI through Trusted Publishing

The PyPI workflow is now triggered by a formal GitHub Release, not by tag push alone.

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

- `spatho` depends on `histoseg`, so releases should be coordinated.
- When `spatho` begins using new runtime APIs from `histoseg`, publish the required `histoseg` version first and then raise the lower bound in `pyproject.toml`.
- After updating the `histoseg` lower bound, create and publish a matching GitHub Release for `spatho`.

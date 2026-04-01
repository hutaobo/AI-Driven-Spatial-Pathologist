# Artifact Manifest Examples

`spatho` writes an `artifact_manifest.json` after a workflow run and can also rebuild one later:

```bash
spatho build-manifest --config /path/to/workflow.json
```

The manifest includes:

- a stable manifest version
- organ pack metadata
- provider metadata
- a file-level artifact inventory with relative paths, media types, sizes, and SHA256 hashes

This folder is intentionally kept free of machine-specific result snapshots.
Generate a fresh manifest locally for your own case outputs.

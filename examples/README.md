# Examples

This folder is for public-safe starter materials.

Use `spatho init-workflow` to generate a real workflow JSON for your own machine:

```bash
spatho init-workflow \
  --organ breast \
  --case-name breast_case_01 \
  --dataset-root /path/to/Xenium_outs \
  --base-pipeline-config /path/to/project/configs/breast_case_01.json \
  --output /path/to/workflows/breast_case_01_full_auto_openai.json
```

Reference templates live in `examples/workflows/`.
They intentionally use placeholders instead of local machine paths.

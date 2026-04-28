# Xenium RNA+Protein + H&E Alignment

This note documents the intended modeling for Xenium Gene + Protein data when the downstream goal is polygon-level pathology analysis.

## Core idea

The smallest pathology-facing unit is a polygon, not a single centroid.

That polygon should be able to carry:

- RNA evidence from transcript points and cell-level summaries
- protein evidence from the same-cell table and protein image channels
- H&E or morphology evidence from the image patch covering the same region

## Spaces

- Canonical internal analysis space: `physical_um`
- External export/viewer space: `xenium_explorer_pixel`

The package defaults to `0.2125 um/pixel` for Xenium export, but explicit metadata should override the fallback constant whenever available.

## Why polygon export matters

The existing Gradio surface in [main.py](../main.py) already exports polygons to Xenium Explorer.

That export path is more than a visualization detail:

- internal structure assignment is computed in micron space
- polygon coordinates are transformed into Xenium Explorer pixel space
- the same polygon can then be used to localize an H&E or morphology image patch
- pathology AI can operate on that polygon-linked image region while preserving traceability back to RNA and protein signals

## Recommended object model

- Same-cell table:
  - keep `feature_modality` so RNA and protein features are not conflated
  - keep cell centroids in `physical_um`
- Transcript points:
  - keep in `physical_um`
- Protein images, autofluorescence, H&E, morphology:
  - preserve as first-class image objects
  - keep their native image-space metadata
- Labels:
  - keep cell and nucleus segmentations as independent spatial objects
- Shapes:
  - treat structure polygons and ROI polygons as stable pathology units

## Minimum alignment checks

- RNA transcript points map back to the segmentation used for quantification.
- Protein image signal remains spatially consistent with centroids and exported polygons.
- H&E or morphology image patches extracted for pathology AI stay locked to the same polygon geometry.
- `um -> pixel -> um` round-trips remain within tolerance.

## Practical use in this repo

Use `spatho write-xenium-alignment-fixtures` to generate:

- a short alignment note
- a fixture manifest
- transform cases for Xenium RNA+protein + H&E workflows

These assets are intended to support hackathon planning, interoperability discussions, and future conformance-style tests.

# Atera WTA Breast Topology Reproducibility Bundle

Sample ID: `atera_wta_breast_pdc_20260429`
Dataset root: `/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs`
t_and_c / StructureMap anchor source: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/tbc_anchor_placeholder`

## Core Summary

- Cells loaded: `170057`
- RNA features loaded: `18028`
- Cluster count: `20`
- Topology celltype count: `20`
- Unassigned cells: `12`
- panel_num_targets_predesigned: `18028`
- median_transcripts_per_cell: `2116`
- Runtime (s): `180.27`

## LR Smoke Panel

- `CSF1-CSF1R`: top `CAFs, DCIS Associated -> Macrophages` (`0.5300`)
- `CXCL12-CXCR4`: top `CAFs, DCIS Associated -> T Lymphocytes` (`0.6617`)
- `TGFB1-TGFBR2`: top `Endothelial Cells -> Endothelial Cells` (`0.5413`)
- `JAG1-NOTCH1`: top `11q13 Invasive Tumor Cells -> Basal-like Structured DCIS Cells` (`0.5195`)
- `DLL4-NOTCH3`: top `Endothelial Cells -> Pericytes` (`0.6695`)

## LR Acceptance

- `PASS` CSF1-CSF1R top sender should not be Mast Cells
- `PASS` CXCL12-CXCR4 should keep CAFs, DCIS Associated -> T Lymphocytes high-ranking
- `PASS` DLL4-NOTCH3 top hit should be Endothelial Cells -> Pericytes

## Pathway Primary Results

- `MacrophageProgram` -> `Macrophages` (`distance=0.0430`)
- `PlasmaProgram` -> `Plasma Cells` (`distance=0.0684`)
- `VascularProgram` -> `Endothelial Cells` (`distance=0.0457`)
- `BasalDCISProgram` -> `Basal-like Structured DCIS Cells` (`distance=0.0430`)
- `ApocrineProgram` -> `Apocrine Cells` (`distance=0.0737`)
- `LuminalAmorphousProgram` -> `Luminal-like Amorphous DCIS Cells` (`distance=0.1967`)

## Pathway Acceptance

- `PASS` `MacrophageProgram` expected `Macrophages`, observed `Macrophages`
- `PASS` `PlasmaProgram` expected `Plasma Cells`, observed `Plasma Cells`
- `PASS` `VascularProgram` expected `Endothelial Cells, Pericytes`, observed `Endothelial Cells`
- `PASS` `BasalDCISProgram` expected `Basal-like Structured DCIS Cells`, observed `Basal-like Structured DCIS Cells`
- `PASS` `ApocrineProgram` expected `Apocrine Cells`, observed `Apocrine Cells`
- `PASS` `LuminalAmorphousProgram` expected `Luminal-like Amorphous DCIS Cells`, observed `Luminal-like Amorphous DCIS Cells`

## Fixed Output Files

- `ligand_to_cell`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/ligand_to_cell.csv`
- `receptor_to_cell`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/receptor_to_cell.csv`
- `lr_sender_receiver_scores`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/lr_sender_receiver_scores.csv`
- `lr_component_diagnostics`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/lr_component_diagnostics.csv`
- `lr_summary_heatmap`: `2` file(s)
- `lr_hotspot_cells_csv`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/figures/lr_hotspot_cells.csv`
- `lr_hotspot_cells_parquet`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/figures/lr_hotspot_cells.parquet`
- `lr_hotspot_overlay`: `2` file(s)
- `pathway_to_cell`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/pathway_to_cell.csv`
- `pathway_structuremap`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/pathway_structuremap.csv`
- `pathway_activity_to_cell`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/pathway_activity_to_cell.csv`
- `pathway_activity_structuremap`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/pathway_activity_structuremap.csv`
- `pathway_mode_comparison`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/pathway_mode_comparison.csv`
- `pathway_to_cell_heatmap`: `2` file(s)
- `pathway_activity_to_cell_heatmap`: `2` file(s)
- `pathway_hotspot_cells_csv`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/figures/pathway_hotspot_cells.csv`
- `pathway_hotspot_overlay`: `2` file(s)
- `summary_json`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/summary.json`
- `report_md`: `/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429/pyxenium/topology/report.md`

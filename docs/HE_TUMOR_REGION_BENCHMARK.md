# H&E Tumor-Region Benchmark

Use this when you want two things from the same protocol:

1. confirm the current strongest pixel-level tumor-region methods on a public set
2. rank those same methods on your own H&E tiles

The comparison is a frozen protocol, not a one-off notebook. Public Dice only tells you the implementation is sane. **Private Dice decides which model to keep.**

## What to test

Two tracks. Do not average them.

| Track | Question | Models |
| --- | --- | --- |
| `pixel_tumor_bulk` | Where is the invasive tumor bulk? | `stain_threshold`, `plip_fulltile`, `dino_nested_unet`, `uni2_upernet`, optional `uni2_unetr` |
| `pixel_neoplastic_cells` | Which pixels are neoplastic cells? | `segtme_uni2` |

Required bulk models:

- `stain_threshold`: always-on weak baseline. If a foundation model loses to this, the protocol or checkpoint loading is wrong.
- `plip_fulltile`: current `spatho` H&E backend, converted to a coarse full-tile mask.
- `dino_nested_unet`: published tumor-bulk SOTA (DINOv3 + nested dense decoder).
- `uni2_upernet`: strongest pathology-domain encoder for dense prediction.

`segtme_uni2` is required, but it is a **cell/TME** model. Report it in its own table.

Print the live catalog:

```bash
spatho he-benchmark catalog
```

## How to run it

### 1. Create a workspace

```bash
spatho he-benchmark init \
  --output-dir /path/to/he_benchmark \
  --with-synthetic-fixture
```

`--with-synthetic-fixture` writes one 64×64 purple-blob case so you can smoke-test the harness without GPU weights.

### 2. Add your H&E

Preferred: paired tiles and pathologist masks, same filename stem.

```bash
spatho he-benchmark ingest \
  --dataset-id private_he \
  --kind private \
  --images /path/to/he_tiles \
  --masks /path/to/tumor_masks \
  --output-dir /path/to/he_benchmark/datasets/private_he \
  --organ breast \
  --pixel-size-um 0.5
```

If you only have a slide plus tumor GeoJSON (QuPath / Xenium Explorer contours):

```bash
spatho he-benchmark ingest \
  --dataset-id private_he \
  --kind private \
  --image /path/to/slide.png \
  --geojson /path/to/tumor_regions.geojson \
  --output-dir /path/to/he_benchmark/datasets/private_he
```

No masks is allowed. The run then writes overlays and inter-model agreement only. You can still see which model looks better; you cannot rank by Dice.

### 3. Optional public sanity set

Put CAMELYON16 (or TIGER WSIBULK) tiles and masks in `datasets/public_camelyon16/` and ingest with `--kind public --dataset-id public_camelyon16`. Enable that dataset in `protocol.json`. Do this before trusting private ranking.

### 4. Attach foundation-model predictions

This repo does not download gated UNI2/DINOv3 checkpoints. The fair way to include them is:

1. run each model on the **same** `cases.jsonl` tiles on a GPU box
2. write one mask per case, named `{case_id}.png` or `{image_stem}.npy`
3. point `prediction_dirs` at those folders

```json
{
  "name": "he_tumor_region_v1",
  "models": ["stain_threshold", "dino_nested_unet", "uni2_upernet", "segtme_uni2"],
  "prediction_dirs": {
    "dino_nested_unet": "preds/dino_nested_unet",
    "uni2_upernet": "preds/uni2_upernet",
    "segtme_uni2": "preds/segtme_uni2"
  },
  "datasets": [
    {
      "dataset_id": "private_he",
      "cases_path": "datasets/private_he/cases.jsonl",
      "kind": "private",
      "enabled": true
    }
  ],
  "allow_partial": true
}
```

Missing models are skipped with a reason. `stain_threshold` always runs.

### 5. Doctor, then run

```bash
spatho he-benchmark doctor --protocol /path/to/he_benchmark/protocol.json
spatho he-benchmark run \
  --protocol /path/to/he_benchmark/protocol.json \
  --output-dir /path/to/he_benchmark/runs/v1
```

Outputs:

- `leaderboard.md` / `leaderboard.json`: private table first, then public
- `case_metrics.csv`: per-tile Dice, IoU, precision, recall, HD95
- `model_agreement.json`: pairwise Dice when ground truth is missing
- `overlays/`: green = true positive, red = false positive, blue = false negative

## Protocol rules

- Same tiles, same tile size, same tissue foreground for every model.
- Slide-level splits. Never train on `private_he` test tiles.
- Rank `private_he` first. Public numbers are a reproduction check.
- Report mean **and** std, not only mean Dice.
- Keep SegTME neoplastic-cell Dice out of the tumor-bulk leaderboard.
- If you later fine-tune UNI2/Dino-NestedUNet on your lab’s slides, put those runs in a separate `finetuned_*` model id. Zero-shot and fine-tuned must not share a row.

## Suggested GPU jobs

Dump masks with whatever training code you already use. The only contract is: one binary tumor mask per `cases.jsonl` row, aligned to the image pixels.

| Model | Encoder | Head | Dump as |
| --- | --- | --- | --- |
| Dino-NestedUNet | frozen DINOv3 | nested dense decoder | `preds/dino_nested_unet/{stem}.png` |
| UNI2-UperNet | frozen UNI2-h | UperNet | `preds/uni2_upernet/{stem}.png` |
| UNI2-UNETR | frozen UNI2-h | UNETR | `preds/uni2_unetr/{stem}.png` |
| SegTME-UNI2 | frozen UNI2-h | dual UperNet | neoplastic class only |

If you cannot get UNI2 weights, still run `stain_threshold` + `dino_nested_unet` + your current PLIP masks. The private ranking remains valid for the models you actually have.

"""What to test for pixel-level H&E tumor-region segmentation.

The catalog is the protocol, not a leaderboard: it names the methods that
must be compared, why each one is in the set, and what it is *not*.
"""

from __future__ import annotations

from typing import Any


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "stain_threshold": {
        "track": "pixel_tumor_bulk",
        "role": "weak_baseline",
        "required": True,
        "always_runnable": True,
        "encoder": "H&E stain heuristic",
        "decoder": "none",
        "weights": "none",
        "why": "Sanity check and lower bound. If a foundation model cannot beat this on your slides, the evaluation protocol is broken or the model is not loaded.",
        "notes": "Not a published SOTA method. Ships with spatho so the harness runs without GPU weights.",
    },
    "plip_fulltile": {
        "track": "pixel_tumor_bulk",
        "role": "current_product_baseline",
        "required": True,
        "always_runnable": False,
        "encoder": "vinid/plip",
        "decoder": "full-tile label fill",
        "weights": "Hugging Face vinid/plip",
        "why": "Current spatho H&E contour backend. Converts a patch-level zero-shot label into a coarse full-tile mask so you can see how far the existing product is from pixel SOTA.",
        "notes": "Not a dense segmenter. Expect over-smooth masks. Skip if PLIP weights are unavailable.",
    },
    "dino_nested_unet": {
        "track": "pixel_tumor_bulk",
        "role": "published_sota_bulk",
        "required": True,
        "always_runnable": False,
        "encoder": "DINOv3 ViT-S/16 (frozen)",
        "decoder": "Nested Dense Decoder + FAPM",
        "weights": "DINOv3 LVD-1689M plus a trained nested decoder checkpoint",
        "why": "Strongest published method specifically for H&E tumor-bulk boundaries (CHTN Dice 0.946, CAMELYON16 Dice 0.846, TIGER WSIBULK zero-shot mDice 0.818).",
        "paper": "arXiv:2605.00894",
        "notes": "Prefer eval-only masks dumped from a GPU box if the nested decoder checkpoint is not local.",
    },
    "uni2_upernet": {
        "track": "pixel_tumor_bulk",
        "role": "pathology_fm_dense",
        "required": True,
        "always_runnable": False,
        "encoder": "UNI2-h ViT-Giant / ViT-h/14",
        "decoder": "UperNet",
        "weights": "MahmoodLab/UNI2-h plus a trained UperNet head",
        "why": "Best pathology-domain encoder for dense prediction in 2025–2026 benchmarks. Tests whether a histology FM beats generic DINOv3 on your stain/scanner.",
        "paper": "UNI2 (Mahmood Lab, 2025); SegTME-UNI2 arXiv:2606.17702",
        "notes": "Gated weights. Frozen encoder + trained head.",
    },
    "uni2_unetr": {
        "track": "pixel_tumor_bulk",
        "role": "pathology_fm_dense",
        "required": False,
        "always_runnable": False,
        "encoder": "UNI2 / UNI2-h",
        "decoder": "UNETR",
        "weights": "MahmoodLab/UNI2-h plus a trained UNETR head",
        "why": "Won multi-cancer epithelium segmentation (Dice 0.893) with a frozen UNI2 backbone. Strong ablation against UperNet on the same encoder.",
        "notes": "Optional but cheap to add once UNI2 embeddings exist.",
    },
    "segtme_uni2": {
        "track": "pixel_neoplastic_cells",
        "role": "cell_tme",
        "required": True,
        "always_runnable": False,
        "encoder": "UNI2-h ViT-Giant",
        "decoder": "dual UperNet (semantic 6-class + HV nuclei)",
        "weights": "SegTME-UNI2-UperHoVer checkpoints on Hugging Face",
        "why": "Maps the neoplastic class to a tumor mask. Use this when you care about tumor *cells*, not the coarse invasive bulk.",
        "paper": "arXiv:2606.17702",
        "notes": "Do not average this track with tumor-bulk Dice. Report it as a separate table.",
    },
}

DATASET_CATALOG: dict[str, dict[str, Any]] = {
    "public_camelyon16": {
        "kind": "public",
        "role": "sanity_and_reproducibility",
        "required": False,
        "task": "pixel_tumor_bulk",
        "why": "Standard breast lymph-node metastasis bulk/region benchmark. Confirms your reimplementation matches published Dice before you trust private ranking.",
        "pixel_size_um": 0.243,
        "source": "CAMELYON16 tumor-positive WSIs",
    },
    "public_tiger_wsibulk": {
        "kind": "public",
        "role": "zero_shot_external",
        "required": False,
        "task": "pixel_tumor_bulk",
        "why": "Coarse breast tumor-bulk annotations (TIGER WSIBULK). Use as a held-out public set, not for training the private comparison.",
        "pixel_size_um": 0.5,
        "source": "TIGER challenge WSIBULK",
    },
    "public_chtn": {
        "kind": "public",
        "role": "multi_cancer_bulk",
        "required": False,
        "task": "pixel_tumor_bulk",
        "why": "14-cancer pixel tumor annotations. Only needed if you want to match Dino-NestedUNet numbers.",
        "pixel_size_um": 0.5,
        "source": "CHTN annotated subset",
    },
    "private_he": {
        "kind": "private",
        "role": "decision",
        "required": True,
        "task": "pixel_tumor_bulk",
        "why": "Your own H&E. This is the ranking that matters for deployment. Prefer pathologist masks; contour GeoJSON is acceptable as coarse GT.",
        "pixel_size_um": None,
        "source": "user-provided images plus masks or tumor GeoJSON",
    },
}

REQUIRED_PIXEL_MODELS = tuple(
    model_id for model_id, spec in MODEL_CATALOG.items() if spec.get("required")
)
DECISION_DATASETS = tuple(
    dataset_id for dataset_id, spec in DATASET_CATALOG.items() if spec.get("role") == "decision"
)

DEFAULT_TUMOR_TOKENS = (
    "tumor",
    "tumour",
    "neoplastic",
    "carcinoma",
    "invasive",
    "metastasis",
    "adenocarcinoma",
    "dcis",
    "idc",
    "malignant",
    "cancer",
)


def catalog_payload() -> dict[str, Any]:
    return {
        "tracks": {
            "pixel_tumor_bulk": "Invasive tumor bulk / tumor region masks. Primary ranking track.",
            "pixel_neoplastic_cells": "Neoplastic cell pixels from multiclass TME models. Report separately.",
        },
        "models": MODEL_CATALOG,
        "datasets": DATASET_CATALOG,
        "required_pixel_models": list(REQUIRED_PIXEL_MODELS),
        "decision_datasets": list(DECISION_DATASETS),
        "protocol_rules": [
            "Compare models on the same tiles, tile size, and tissue foreground.",
            "Split public and private tables. Never train on private_he test tiles.",
            "Rank private_he first; public sets only confirm the implementation.",
            "Report mean and std Dice, IoU, precision, recall, and HD95.",
            "If a case has no mask, run qualitative overlays and inter-model agreement only.",
            "Do not mix SegTME neoplastic-cell Dice into the tumor-bulk leaderboard.",
        ],
    }

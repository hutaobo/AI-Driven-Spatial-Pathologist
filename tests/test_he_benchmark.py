from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spatho.he_benchmark import (
    catalog_payload,
    doctor_benchmark,
    init_benchmark,
    ingest_dataset,
    run_benchmark,
)
from spatho.he_benchmark.datasets import save_array, write_synthetic_fixture
from spatho.he_benchmark.metrics import case_metrics
from spatho.he_benchmark.models import StainThresholdSegmenter


def test_catalog_names_required_pixel_models() -> None:
    payload = catalog_payload()
    models = payload["models"]
    assert "dino_nested_unet" in models
    assert "uni2_upernet" in models
    assert "segtme_uni2" in models
    assert "stain_threshold" in models
    assert payload["decision_datasets"] == ["private_he"]
    assert "Never train on private_he test tiles." in " ".join(payload["protocol_rules"])


def test_identical_masks_have_perfect_dice() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    metrics = case_metrics(mask, mask)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["hd95"] == 0.0


def test_disjoint_masks_have_zero_dice() -> None:
    gt = np.zeros((16, 16), dtype=bool)
    pred = np.zeros((16, 16), dtype=bool)
    gt[0:4, 0:4] = True
    pred[12:16, 12:16] = True
    metrics = case_metrics(pred, gt)
    assert metrics["dice"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_stain_threshold_recovers_synthetic_purple_tumor(tmp_path: Path) -> None:
    fixture = write_synthetic_fixture(tmp_path / "synth", dataset_id="private_he")
    case = json.loads(Path(fixture["cases_path"]).read_text(encoding="utf-8").splitlines()[0])
    image = np.load(case["image"])
    mask = np.load(case["mask"])
    pred = StainThresholdSegmenter().predict(image, case={})
    metrics = case_metrics(pred, mask)
    assert metrics["dice"] > 0.7
    assert metrics["recall"] > 0.9


def test_init_doctor_and_run_rank_oracle_above_heuristic(tmp_path: Path) -> None:
    bench_dir = tmp_path / "bench"
    init = init_benchmark(bench_dir, with_synthetic_fixture=True)
    protocol_path = Path(init["protocol_json"])
    cases = json.loads((bench_dir / "datasets" / "private_he" / "cases.jsonl").read_text().splitlines()[0])
    gt = np.load(cases["mask"])
    pred_dir = bench_dir / "preds" / "dino_nested_unet"
    save_array(pred_dir / f"{Path(cases['image']).stem}.npy", gt)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["models"] = ["stain_threshold", "dino_nested_unet", "uni2_upernet"]
    protocol["prediction_dirs"] = {"dino_nested_unet": str(pred_dir)}
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    doctor = doctor_benchmark(protocol_path)
    assert doctor["ready_to_run"] is True
    available = {row["model_id"]: row["available"] for row in doctor["models"]}
    assert available["stain_threshold"] is True
    assert available["dino_nested_unet"] is True
    assert available["uni2_upernet"] is False

    result = run_benchmark(protocol_path, output_dir=bench_dir / "runs" / "test")
    leaderboard = json.loads(Path(result["leaderboard_json"]).read_text(encoding="utf-8"))
    private = [row for row in leaderboard if row["kind"] == "private"]
    assert private[0]["model_id"] == "dino_nested_unet"
    assert private[0]["dice_mean"] == 1.0
    heuristic = next(row for row in private if row["model_id"] == "stain_threshold")
    assert heuristic["dice_mean"] < 1.0
    skipped = {row["model_id"] for row in result["skipped_models"]}
    assert "uni2_upernet" in skipped
    overlay_dir = Path(result["output_dir"]) / "overlays"
    assert any(overlay_dir.glob("*.npy"))


def test_ingest_pairs_images_and_masks_by_stem(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    save_array(images / "tile_a.npy", rgb)
    save_array(masks / "tile_a.npy", mask)
    save_array(images / "tile_b.npy", rgb)
    save_array(masks / "tile_b.npy", mask)
    meta = ingest_dataset(
        dataset_id="private_he",
        output_dir=tmp_path / "private_he",
        images_dir=images,
        masks_dir=masks,
    )
    assert meta["n_cases"] == 2
    assert meta["n_with_masks"] == 2


def test_cli_catalog_and_synthetic_run(tmp_path: Path, capsys, monkeypatch) -> None:
    from spatho.cli import main

    monkeypatch.setattr("sys.argv", ["spatho", "he-benchmark", "catalog"])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert "dino_nested_unet" in payload["models"]

    bench_dir = tmp_path / "cli-bench"
    monkeypatch.setattr(
        "sys.argv",
        [
            "spatho",
            "he-benchmark",
            "init",
            "--output-dir",
            str(bench_dir),
            "--with-synthetic-fixture",
        ],
    )
    main()
    capsys.readouterr()
    protocol_path = bench_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["models"] = ["stain_threshold"]
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["spatho", "he-benchmark", "run", "--protocol", str(protocol_path)],
    )
    main()
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["n_runnable_models"] == 1
    assert Path(run_out["leaderboard_md"]).exists()

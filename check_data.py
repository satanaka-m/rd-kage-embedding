import argparse
import os
from pathlib import Path
import tempfile
import yaml
import mlflow
import torch
from torchvision.io import write_png
import pytorch_lightning as pl

from data import StrokeImageDataModule
from kage_util import strokes_to_svg


def load_config(config_file):
    """YAML設定ファイルをロードする"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    v = value.lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main(config, output_strokes=False):
    seed = 42
    pl.seed_everything(seed)

    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 8)

    dataset_config = config.get("dataset")
    val_dataset_config = config.get("val_dataset")
    if dataset_config is None:
        raise ValueError("Config key `dataset` is required.")
    if val_dataset_config is None:
        val_dataset_config = dataset_config

    dm = StrokeImageDataModule(
        dataset_config,
        val_dataset_config=val_dataset_config,
        batch_size=batch_size,
    )
    # dl = dm.train_dataloader()
    dl = dm.val_dataloader()
    dataset = dl.dataset
    max_batches = 10

    with tempfile.TemporaryDirectory(prefix="check_data_") as tmpdir:
        artifact_dir = Path(tmpdir) / "check_data"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for i, xs in enumerate(dl):
            if i >= max_batches:
                break
            x_target, stroke_labels, stroke_controls = xs
            print(x_target.shape, stroke_labels.shape, stroke_controls.shape)

            for b in range(x_target.shape[0]):
                global_idx = i * batch_size + b
                if global_idx >= len(dataset.target_cid_list):
                    break

                cid = dataset.target_cid_list[global_idx]
                stem = f"{i:03d}_{b:02d}_cid{cid:05d}"

                # target image
                target_png_path = artifact_dir / f"{stem}_target.png"
                write_png((x_target[b] * 255).to(torch.uint8), target_png_path.as_posix())

                # KAGE stroke SVG
                if output_strokes:
                    glyph_key = f"aj1-{cid:05d}"
                    if glyph_key in dataset.kage:
                        svg_text = strokes_to_svg(dataset.kage[glyph_key])
                        (artifact_dir / f"{stem}_strokes.svg").write_text(svg_text, encoding="utf-8")

        mlflow.log_artifacts(artifact_dir.as_posix(), artifact_path="check_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check StrokeImageDataModule")
    parser.add_argument("--config", type=str, default="conf/config.yaml",
                        help="Path to config file")
    parser.add_argument("--output_strokes", type=_str2bool, default=False,
                        help="Output stroke visualization")

    args = parser.parse_args()
    config = load_config(args.config)

    if mlflow.active_run() is not None:
        raise RuntimeError(
            "Unexpected active MLflow run detected before startup. "
            "Run this script only via `mlflow run . -e check_data`."
        )

    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError(
            "No MLflow project run context found. Run this script via `mlflow run . -e check_data`."
        )

    with mlflow.start_run(run_id=run_id):
        main(config, output_strokes=args.output_strokes)

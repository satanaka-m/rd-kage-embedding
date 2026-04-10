import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from fontTools.misc.transform import Transform

from data import StrokeImageDataset, custom_collate_fn
from model import ImageEncoder, StrokeSetEncoder
from rasterize import CidFace, rasterize


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_config(base_config: dict, checkpoint: dict) -> dict:
    model_config = copy.deepcopy(base_config["model"])

    hyper_parameters = checkpoint.get("hyper_parameters")
    if isinstance(hyper_parameters, dict):
        ckpt_model_config = hyper_parameters.get("model")
        if isinstance(ckpt_model_config, dict):
            model_config.update(ckpt_model_config)
            return model_config

    return model_config


def build_dataset_config(config):
    dataset_config = copy.deepcopy(config["dataset"])
    dataset_config.pop("augmentation", None)
    return dataset_config


def load_unicode_map(kage_dump_dir: str | Path) -> dict[int, str]:
    dump_path = Path(kage_dump_dir) / "dump_newest_only.txt"
    if not dump_path.exists():
        return {}

    cid_to_unicode = {}
    with open(dump_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            left, middle, *_ = [part.strip() for part in line.split("|")]
            if not left.startswith("aj1-") or not left[4:9].isdigit():
                continue
            cid = int(left[4:9])
            cid_to_unicode[cid] = middle
    return cid_to_unicode


def format_cid_label(cid: int, unicode_name: str | None) -> str:
    if unicode_name and unicode_name != "u3013":
        if unicode_name.startswith("u") and len(unicode_name) > 1:
            try:
                char = chr(int(unicode_name[1:], 16))
                return f"CID {cid} / {char} / {unicode_name.upper()}"
            except ValueError:
                pass
        return f"CID {cid} / {unicode_name}"
    return f"CID {cid}"


def load_encoders(config, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = resolve_model_config(config, checkpoint)

    dim_emb = int(model_config["dim_emb"])
    tf_dim_model = int(model_config["tf_dim_model"])
    tf_dim_ff = int(model_config["tf_dim_ff"])
    image_encoder = ImageEncoder(dim_emb, model_config.get("vit_layers", 4))
    stroke_encoder = StrokeSetEncoder(
        dim_emb,
        tf_dim_model,
        model_config.get("tf_layers", 4),
        model_config.get("tf_heads", 8),
        model_config.get("tf_dropout", 0.1),
        tf_dim_ff,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)

    image_state = {}
    stroke_state = {}
    for key, value in state_dict.items():
        if key.startswith("image_encoder."):
            image_state[key[len("image_encoder."):]] = value
        elif key.startswith("stroke_encoder."):
            stroke_state[key[len("stroke_encoder."):]] = value

    missing_image, unexpected_image = image_encoder.load_state_dict(image_state, strict=False)
    missing_stroke, unexpected_stroke = stroke_encoder.load_state_dict(stroke_state, strict=False)
    if missing_image or unexpected_image or missing_stroke or unexpected_stroke:
        raise RuntimeError(
            "Failed to load checkpoint cleanly.\n"
            f"image missing={missing_image}, image unexpected={unexpected_image}, "
            f"stroke missing={missing_stroke}, stroke unexpected={unexpected_stroke}"
        )

    image_encoder.eval().to(device)
    stroke_encoder.eval().to(device)
    return image_encoder, stroke_encoder


@torch.no_grad()
def compute_embeddings(dataset, image_encoder, stroke_encoder, batch_size: int, device: torch.device):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    image_embs = []
    stroke_embs = []
    for images, stroke_labels, stroke_controls in dataloader:
        images = images.to(device)
        stroke_labels = stroke_labels.to(device)
        stroke_controls = stroke_controls.to(device)
        image_embs.append(image_encoder(images).cpu())
        stroke_embs.append(stroke_encoder(stroke_labels, stroke_controls).cpu())

    image_embs = torch.cat(image_embs, dim=0)
    stroke_embs = torch.cat(stroke_embs, dim=0)
    image_embs = torch.nn.functional.normalize(image_embs, dim=1)
    stroke_embs = torch.nn.functional.normalize(stroke_embs, dim=1)
    return image_embs, stroke_embs


class GlyphRenderer:
    def __init__(self, dataset_config):
        self.width = int(dataset_config["width"])
        self.height = int(dataset_config["height"])
        self.scale = float(dataset_config["scale"])
        font_path = Path(dataset_config["font_dir"]) / dataset_config["font_filename"]
        self.font = CidFace(font_path.resolve().as_posix())

    def render(self, cid: int) -> Image.Image:
        transform = (
            Transform()
            .translate(self.width / 2, self.height / 2)
            .scale(self.scale)
            .translate(-self.width / 2, -self.height / 2)
        )
        arr = rasterize(self.font, cid, self.width, self.height, transform)
        rgb = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(rgb, mode="RGB")


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    resized = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", size, "white")
    offset = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def draw_panel(draw, x0, y0, panel_w, panel_h, title, subtitle, glyph_image, score=None, font=None, small_font=None):
    border = "#d0d0d0"
    text = "#111111"
    score_text = "#444444"
    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=10, outline=border, width=2, fill="white")
    draw.text((x0 + 12, y0 + 10), title, fill=text, font=font)
    draw.text((x0 + 12, y0 + 38), subtitle, fill=score_text, font=small_font)
    if score is not None:
        draw.text((x0 + 12, y0 + 60), f"score={score:.4f}", fill=score_text, font=small_font)
        img_top = y0 + 84
    else:
        img_top = y0 + 66
    img_box = fit_image(glyph_image, (panel_w - 24, panel_h - (img_top - y0) - 12))
    panel = Image.new("RGB", (panel_w - 24, panel_h - (img_top - y0) - 12), "white")
    panel.paste(img_box, (0, 0))
    return panel, (x0 + 12, img_top)


def compose_knn_figure(
    query_title: str,
    query_subtitle: str,
    query_image: Image.Image,
    neighbors: list[dict],
    output_path: Path,
):
    panel_w = 220
    panel_h = 260
    columns = 4
    rows = 1 + math.ceil(len(neighbors) / columns)
    canvas_w = 40 + columns * panel_w + (columns - 1) * 16 + 40
    canvas_h = 40 + rows * panel_h + (rows - 1) * 16 + 40

    image = Image.new("RGB", (canvas_w, canvas_h), "#f6f6f2")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    query_panel, query_pos = draw_panel(
        draw,
        40,
        40,
        panel_w,
        panel_h,
        query_title,
        query_subtitle,
        query_image,
        score=None,
        font=font,
        small_font=small_font,
    )
    image.paste(query_panel, query_pos)

    for rank, item in enumerate(neighbors, start=1):
        col = (rank - 1) % columns
        row = 1 + (rank - 1) // columns
        x0 = 40 + col * (panel_w + 16)
        y0 = 40 + row * (panel_h + 16)
        panel, pos = draw_panel(
            draw,
            x0,
            y0,
            panel_w,
            panel_h,
            f"Top {rank}",
            item["label"],
            item["image"],
            score=item["score"],
            font=font,
            small_font=small_font,
        )
        image.paste(panel, pos)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def select_indices(length: int, num: int, seed: int) -> list[int]:
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    num = min(num, length)
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), k=num))


def build_neighbor_records(indices, scores, cids, renderer, unicode_map):
    records = []
    for idx in indices:
        cid = cids[idx]
        records.append(
            {
                "index": idx,
                "cid": cid,
                "label": format_cid_label(cid, unicode_map.get(cid)),
                "score": float(scores[idx]),
                "image": renderer.render(cid),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Visualize bidirectional kNN from a trained cross-modal model")
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/knn")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num", type=int, default=8, help="Number of random query samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, mps; default auto")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = build_dataset_config(config)
    dataset = StrokeImageDataset(dataset_config)
    unicode_map = load_unicode_map(Path(dataset_config["kage_pkl"]).parent / "kage_dump")
    cids = dataset.target_cid_list

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    image_encoder, stroke_encoder = load_encoders(config, args.checkpoint, device)
    image_embs, stroke_embs = compute_embeddings(
        dataset=dataset,
        image_encoder=image_encoder,
        stroke_encoder=stroke_encoder,
        batch_size=args.batch_size,
        device=device,
    )

    similarity = image_embs @ stroke_embs.T
    renderer = GlyphRenderer(dataset_config)
    sampled_indices = select_indices(len(cids), args.num, args.seed)
    output_dir = Path(args.output_dir)

    top_k = min(args.top_k, len(cids))
    for idx in sampled_indices:
        cid = cids[idx]
        query_label = format_cid_label(cid, unicode_map.get(cid))
        query_image = renderer.render(cid)

        glyph_to_shape_indices = torch.topk(similarity[idx], k=top_k).indices.tolist()
        glyph_to_shape_scores = similarity[idx].tolist()
        glyph_to_shape_neighbors = build_neighbor_records(
            glyph_to_shape_indices,
            glyph_to_shape_scores,
            cids,
            renderer,
            unicode_map,
        )
        compose_knn_figure(
            query_title="Glyph query",
            query_subtitle=query_label,
            query_image=query_image,
            neighbors=glyph_to_shape_neighbors,
            output_path=output_dir / f"glyph_to_shape_cid{cid:05d}.png",
        )

        shape_to_glyph_indices = torch.topk(similarity[:, idx], k=top_k).indices.tolist()
        shape_to_glyph_scores = similarity[:, idx].tolist()
        shape_to_glyph_neighbors = build_neighbor_records(
            shape_to_glyph_indices,
            shape_to_glyph_scores,
            cids,
            renderer,
            unicode_map,
        )
        compose_knn_figure(
            query_title="Shape query",
            query_subtitle=query_label,
            query_image=query_image,
            neighbors=shape_to_glyph_neighbors,
            output_path=output_dir / f"shape_to_glyph_cid{cid:05d}.png",
        )

    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": len(sampled_indices),
        "top_k": top_k,
        "sampled_cids": [cids[idx] for idx in sampled_indices],
    }
    (output_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"Saved kNN visualizations to {output_dir}")


if __name__ == "__main__":
    main()

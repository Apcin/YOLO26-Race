"""Diagnose whether background low-pass filtering reduces YOLO26 race false positives.

This is a frozen-model causal diagnostic, not an HBS implementation for training. It intercepts
the P3/P4/P5 features immediately before Detect and optionally applies a fixed local low-pass
filter to GT-defined background regions. The normal competition validator is reused so standard
metrics remain comparable with previous runs.

Example:
    python hbs_frequency_diagnostic.py --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou


DEFAULT_MODEL = "runs/detect/runs/competition25/yolo26m-strip-reg/weights/best.pt"
DEFAULT_DATA = "prepare_data/competition25_all.yaml"
VARIANT_LEVELS = {
    "baseline": (),
    "p3": (0,),
    "p3p4": (0, 1),
    "p3p4p5": (0, 1, 2),
    "full": (0, 1, 2),
}


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse a comma-separated tuple of positive odd integers."""
    values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not values or any(x < 1 or x % 2 == 0 for x in values):
        raise argparse.ArgumentTypeError("kernel sizes must be positive odd integers, e.g. 3,5,5")
    return values


def build_foreground_mask(batch: dict[str, torch.Tensor], height: int, width: int) -> torch.Tensor:
    """Rasterize augmented normalized xywh GT boxes into a BCHW foreground mask."""
    batch_size = batch["img"].shape[0]
    mask = torch.zeros((batch_size, 1, height, width), device=batch["img"].device, dtype=batch["img"].dtype)
    boxes = batch["bboxes"]
    batch_idx = batch["batch_idx"].long()
    for box, image_index in zip(boxes, batch_idx):
        cx, cy, bw, bh = box
        x1 = max(0, min(width - 1, int(torch.floor((cx - bw / 2) * width).item())))
        y1 = max(0, min(height - 1, int(torch.floor((cy - bh / 2) * height).item())))
        x2 = max(x1 + 1, min(width, int(torch.ceil((cx + bw / 2) * width).item())))
        y2 = max(y1 + 1, min(height, int(torch.ceil((cy + bh / 2) * height).item())))
        mask[image_index, :, y1:y2, x1:x2] = 1
    return mask


def masked_low_pass(feature: torch.Tensor, foreground: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Average only valid background neighbors and preserve foreground values exactly."""
    background = 1 - foreground
    padding = kernel_size // 2
    smoothed_sum = F.avg_pool2d(
        feature * background, kernel_size, stride=1, padding=padding, count_include_pad=True
    )
    valid_fraction = F.avg_pool2d(
        background, kernel_size, stride=1, padding=padding, count_include_pad=True
    ).clamp_min(torch.finfo(feature.dtype).eps)
    smoothed_background = smoothed_sum / valid_fraction
    return feature * foreground + smoothed_background * background


class FeatureLowPassHook:
    """Modify selected Detect input levels using the validator's current GT batch."""

    def __init__(
        self,
        validator: "FrequencyDiagnosticValidator",
        levels: tuple[int, ...],
        kernels: tuple[int, ...],
        full_feature: bool,
    ) -> None:
        self.validator = validator
        self.levels = levels
        self.kernels = kernels
        self.full_feature = full_feature

    def __call__(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        """Return modified feature inputs; leave warmup calls unchanged because they have no current batch."""
        if not self.levels or self.validator.current_batch is None:
            return None
        features = list(inputs[0])
        if max(self.levels) >= len(features):
            raise RuntimeError(f"requested feature level {max(self.levels)} but Detect received {len(features)} levels")
        if len(self.kernels) < len(features):
            raise RuntimeError(f"received {len(features)} feature levels but only {len(self.kernels)} kernels")

        batch = self.validator.current_batch
        image_h, image_w = batch["img"].shape[-2:]
        foreground_image = None if self.full_feature else build_foreground_mask(batch, image_h, image_w)
        for level in self.levels:
            feature = features[level]
            kernel = self.kernels[level]
            if self.full_feature:
                features[level] = F.avg_pool2d(feature, kernel, stride=1, padding=kernel // 2)
            else:
                foreground = F.interpolate(foreground_image, size=feature.shape[-2:], mode="nearest")
                features[level] = masked_low_pass(feature, foreground, kernel)
        return (features, *inputs[1:])


def init_extended_stats(nc: int) -> dict[str, np.ndarray]:
    """Create exhaustive subcategories for the validator's combined background-or-IoU bucket."""
    return {
        "pure_background": np.zeros(nc, dtype=np.int64),
        "low_iou_same_class": np.zeros(nc, dtype=np.int64),
        "low_iou_other_class": np.zeros(nc, dtype=np.int64),
    }


def split_background_or_iou(
    detections: torch.Tensor,
    labels: torch.Tensor,
    nc: int,
    pure_background_iou: float,
    vehicle_cls: int = 24,
    vehicle_iou: float = 0.35,
    default_iou: float = 0.50,
) -> dict[str, np.ndarray]:
    """Repeat competition matching and exhaustively split its background-or-IoU fallback cases."""
    stats = init_extended_stats(nc)
    nl, nd = labels.shape[0], detections.shape[0]
    if nd == 0:
        return stats
    if nl == 0:
        for cls in detections[:, 5].int().tolist():
            stats["pure_background"][cls] += 1
        return stats

    order = detections[:, 4].argsort(descending=True)
    ious = box_iou(labels[:, 1:], detections[:, :4])
    matched = torch.zeros(nl, dtype=torch.bool, device=labels.device)
    thresholds = torch.where(
        labels[:, 0] == vehicle_cls,
        torch.tensor(vehicle_iou, device=labels.device),
        torch.tensor(default_iou, device=labels.device),
    )

    for detection_index in order.tolist():
        predicted_class = int(detections[detection_index, 5].item())
        same_class = labels[:, 0] == predicted_class
        candidates = torch.where(same_class & ~matched & (ious[:, detection_index] >= thresholds))[0]
        if candidates.numel():
            best = candidates[ious[candidates, detection_index].argmax()]
            matched[best] = True
            continue

        wrong_class = torch.where(
            (labels[:, 0] != predicted_class) & ~matched & (ious[:, detection_index] >= thresholds)
        )[0]
        if wrong_class.numel() or torch.any(same_class & matched & (ious[:, detection_index] >= thresholds)):
            continue

        overlap = ious[:, detection_index]
        max_any_iou = float(overlap.max().item())
        if max_any_iou < pure_background_iou:
            stats["pure_background"][predicted_class] += 1
        elif same_class.any() and float(overlap[same_class].max().item()) >= pure_background_iou:
            stats["low_iou_same_class"][predicted_class] += 1
        else:
            stats["low_iou_other_class"][predicted_class] += 1
    return stats


class FrequencyDiagnosticValidator(DetectionValidator):
    """Race validator that exposes the current batch and saves refined FP attribution."""

    def __init__(self, *args: Any, pure_background_iou: float = 0.1, **kwargs: Any) -> None:
        self.current_batch: dict[str, torch.Tensor] | None = None
        self.pure_background_iou = pure_background_iou
        self.extended_stats: dict[str, np.ndarray] | None = None
        super().__init__(*args, **kwargs)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Store the device-ready batch so the Detect pre-hook can construct GT masks."""
        batch = super().preprocess(batch)
        self.current_batch = batch
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize standard and extended competition metrics."""
        super().init_metrics(model)
        self.extended_stats = init_extended_stats(self.nc)

    def _update_race_metrics(self, pred: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        """Update normal race metrics and the refined background/low-IoU attribution."""
        super()._update_race_metrics(pred, batch)
        labels = (
            torch.cat((batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            if batch["cls"].shape[0]
            else torch.zeros((0, 5), device=self.device)
        )
        detections = (
            torch.cat((pred["bboxes"], pred["conf"].view(-1, 1), pred["cls"].view(-1, 1)), 1)
            if pred["cls"].shape[0]
            else torch.zeros((0, 6), device=self.device)
        )
        update = split_background_or_iou(detections, labels, self.nc, self.pure_background_iou)
        for key, value in update.items():
            self.extended_stats[key] += value

    def print_race_results(self) -> None:
        """Save standard race outputs followed by the refined attribution CSV."""
        super().print_race_results()
        output = self.save_dir / "race_background_iou_split.csv"
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["class_id", "class_name", "pure_background", "low_iou_same_class", "low_iou_other_class", "total"]
            )
            for class_id in range(self.nc):
                values = [int(self.extended_stats[key][class_id]) for key in self.extended_stats]
                class_name = self.names.get(class_id, str(class_id)) if isinstance(self.names, dict) else self.names[class_id]
                writer.writerow([class_id, class_name, *values, sum(values)])


def summarize_variant(variant: str, validator: FrequencyDiagnosticValidator, metrics: dict[str, float]) -> dict[str, Any]:
    """Build a compact JSON-serializable summary for cross-variant comparison."""
    race = validator.race_stats
    extended = validator.extended_stats
    tp, fp, fn = (int(race[key].sum()) for key in ("tp", "fp", "fn"))
    summary: dict[str, Any] = {
        "variant": variant,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fdr": fp / (tp + fp) if tp + fp else 0.0,
        "wrong_class": int(race["fp_wrong_class"].sum()),
        "duplicate": int(race["fp_duplicate"].sum()),
        "background_or_iou": int(race["fp_background_or_iou"].sum()),
        **{key: int(value.sum()) for key, value in extended.items()},
        "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "classes": {},
    }
    for class_id in (3, 24):
        class_name = validator.names.get(class_id, str(class_id)) if isinstance(validator.names, dict) else validator.names[class_id]
        class_tp, class_fp, class_fn = (int(race[key][class_id]) for key in ("tp", "fp", "fn"))
        summary["classes"][class_name] = {
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "recall": class_tp / (class_tp + class_fn) if class_tp + class_fn else 0.0,
            "background_or_iou": int(race["fp_background_or_iou"][class_id]),
            **{key: int(value[class_id]) for key, value in extended.items()},
        }
    aircraft = np.arange(4, 24)
    aircraft_tp, aircraft_fp, aircraft_fn = (int(race[key][aircraft].sum()) for key in ("tp", "fp", "fn"))
    summary["aircraft"] = {
        "tp": aircraft_tp,
        "fp": aircraft_fp,
        "fn": aircraft_fn,
        "recall": aircraft_tp / (aircraft_tp + aircraft_fn) if aircraft_tp + aircraft_fn else 0.0,
        "fdr": aircraft_fp / (aircraft_tp + aircraft_fp) if aircraft_tp + aircraft_fp else 0.0,
    }
    return summary


def write_comparison(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    """Write machine-readable JSON and a compact CSV comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    columns = [
        "variant", "tp", "fp", "fn", "recall", "fdr", "wrong_class", "duplicate", "background_or_iou",
        "pure_background", "low_iou_same_class", "low_iou_other_class", "map50", "map50_95",
        "ms_fp", "ms_bg_iou", "fsc_fp", "fsc_bg_iou", "aircraft_recall", "aircraft_fdr",
    ]
    with (output_dir / "diagnostic_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **{key: summary[key] for key in columns[:14]},
                    "ms_fp": summary["classes"]["MS"]["fp"],
                    "ms_bg_iou": summary["classes"]["MS"]["background_or_iou"],
                    "fsc_fp": summary["classes"]["FSC"]["fp"],
                    "fsc_bg_iou": summary["classes"]["FSC"]["background_or_iou"],
                    "aircraft_recall": summary["aircraft"]["recall"],
                    "aircraft_fdr": summary["aircraft"]["fdr"],
                }
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kernels", type=parse_int_tuple, default=(3, 5, 5))
    parser.add_argument(
        "--variants", default="baseline,p3,p3p4,p3p4p5,full", help=f"comma-separated subset of {tuple(VARIANT_LEVELS)}"
    )
    parser.add_argument("--pure-background-iou", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("runs/hbs_frequency_diagnostic"))
    parser.add_argument(
        "--expect-baseline-bg-iou", type=int, default=177, help="warn if baseline does not reproduce this count"
    )
    return parser.parse_args()


def main() -> None:
    """Run all requested frozen-model diagnostic variants."""
    args = parse_args()
    variants = tuple(value.strip().lower() for value in args.variants.split(",") if value.strip())
    unknown = sorted(set(variants) - VARIANT_LEVELS.keys())
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {tuple(VARIANT_LEVELS)}")
    if not 0 <= args.pure_background_iou <= 1:
        raise ValueError("--pure-background-iou must be between 0 and 1")

    model = YOLO(args.model).model
    head = model.model[-1]
    summaries = []
    for variant in variants:
        LOGGER.info(f"\nRunning HBS frequency diagnostic variant: {variant}")
        validator = FrequencyDiagnosticValidator(
            save_dir=args.output / variant,
            args={
                "model": args.model,
                "data": args.data,
                "device": args.device,
                "split": args.split,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "rect": args.rect,
                "end2end": args.end2end,
                "plots": False,
                "verbose": False,
                "exist_ok": True,
            },
            pure_background_iou=args.pure_background_iou,
        )
        levels = VARIANT_LEVELS[variant]
        hook = FeatureLowPassHook(validator, levels, args.kernels, full_feature=variant == "full")
        handle = head.register_forward_pre_hook(hook) if levels else None
        try:
            metrics = validator(model=model)
        finally:
            if handle is not None:
                handle.remove()
            validator.current_batch = None
        summary = summarize_variant(variant, validator, metrics)
        summaries.append(summary)
        write_comparison(args.output, summaries)

        if variant == "baseline" and summary["background_or_iou"] != args.expect_baseline_bg_iou:
            LOGGER.warning(
                f"Baseline background_or_iou={summary['background_or_iou']} does not reproduce "
                f"expected {args.expect_baseline_bg_iou}. Check conf/imgsz/split/model before interpreting variants."
            )

    LOGGER.info(f"Diagnostic comparison saved to {args.output / 'diagnostic_summary.csv'}")


if __name__ == "__main__":
    main()

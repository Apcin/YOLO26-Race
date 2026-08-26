# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


RACE_GROUPS = {
    "ship": range(0, 4),
    "aircraft": range(4, 24),
    "vehicle": range(24, 25),
}
RACE_VEHICLE_CLASS = 24
RACE_VEHICLE_IOU = 0.35
RACE_DEFAULT_IOU = 0.50


def init_race_stats(nc: int) -> dict[str, np.ndarray]:
    """Create competition metric accumulators."""
    return {
        "tp": np.zeros(nc, dtype=np.int64),
        "fp": np.zeros(nc, dtype=np.int64),
        "fn": np.zeros(nc, dtype=np.int64),
        "gt": np.zeros(nc, dtype=np.int64),
        "fp_wrong_class": np.zeros(nc, dtype=np.int64),
        "fp_duplicate": np.zeros(nc, dtype=np.int64),
        "fp_background_or_iou": np.zeros(nc, dtype=np.int64),
        "confusion": np.zeros((nc, nc), dtype=np.int64),
    }


def add_race_stats(total: dict[str, np.ndarray], update: dict[str, np.ndarray]) -> None:
    """Accumulate competition statistics in-place."""
    for k, v in update.items():
        total[k] += v


def race_summary(tp: int, fp: int, fn: int) -> tuple[int, float, float]:
    """Return ground-truth count, recall and false discovery rate."""
    gt = tp + fn
    recall = tp / gt if gt else 0.0
    fdr = fp / (tp + fp) if tp + fp else 0.0
    return gt, recall, fdr


def race_match_stats(
    detections: torch.Tensor,
    labels: torch.Tensor,
    nc: int,
    vehicle_cls: int = RACE_VEHICLE_CLASS,
    vehicle_iou: float = RACE_VEHICLE_IOU,
    default_iou: float = RACE_DEFAULT_IOU,
) -> dict[str, np.ndarray]:
    """Count competition TP/FP/FN with class-exact, confidence-ordered one-to-one matching."""
    stats = init_race_stats(nc)
    nl, nd = labels.shape[0], detections.shape[0]

    for cls in labels[:, 0].int().tolist() if nl else []:
        stats["gt"][cls] += 1

    if nl == 0:
        for cls in detections[:, 5].int().tolist() if nd else []:
            stats["fp"][cls] += 1
            stats["fp_background_or_iou"][cls] += 1
        return stats
    if nd == 0:
        for cls in labels[:, 0].int().tolist():
            stats["fn"][cls] += 1
        return stats

    order = detections[:, 4].argsort(descending=True)
    ious = box_iou(labels[:, 1:], detections[:, :4])
    matched = torch.zeros(nl, dtype=torch.bool, device=labels.device)
    thresholds = torch.where(
        labels[:, 0] == vehicle_cls,
        torch.tensor(vehicle_iou, device=labels.device),
        torch.tensor(default_iou, device=labels.device),
    )

    for di in order.tolist():
        det_cls = int(detections[di, 5].item())
        same_cls = labels[:, 0] == det_cls
        candidates = torch.where(same_cls & ~matched & (ious[:, di] >= thresholds))[0]
        if candidates.numel():
            best = candidates[ious[candidates, di].argmax()]
            matched[best] = True
            stats["tp"][det_cls] += 1
            continue

        stats["fp"][det_cls] += 1
        wrong_cls = torch.where((labels[:, 0] != det_cls) & ~matched & (ious[:, di] >= thresholds))[0]
        if wrong_cls.numel():
            best = wrong_cls[ious[wrong_cls, di].argmax()]
            true_cls = int(labels[best, 0].item())
            stats["fp_wrong_class"][det_cls] += 1
            stats["confusion"][true_cls, det_cls] += 1
        elif torch.any(same_cls & matched & (ious[:, di] >= thresholds)):
            stats["fp_duplicate"][det_cls] += 1
        else:
            stats["fp_background_or_iou"][det_cls] += 1

    for cls in labels[~matched, 0].int().tolist():
        stats["fn"][cls] += 1
    return stats


class DetectionValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        jdict (list[dict[str, Any]]): List for storing JSON detection results.
        stats (dict[str, list[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        conf = args.get("conf") if isinstance(args, dict) else getattr(args, "conf", None)
        self.confusion_matrix_conf = 0.25 if conf is None else conf
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
        self.race_enabled = False
        self.race_stats = None

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith((f"{os.sep}val2017.txt", f"{os.sep}test-dev2017.txt")))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.metrics.clear_stats()
        self.metrics.clear_image_metrics()
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)
        self.race_enabled = self._is_race_dataset()
        self.race_stats = init_race_stats(self.nc) if self.race_enabled else None

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains 'bboxes', 'conf',
                'cls', and 'extra' tensors.
        """
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics with new predictions and ground truth.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            batch (dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.confusion_matrix_conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(
                        batch["img"][si],
                        pbatch["im_file"],
                        self.save_dir,
                        self.args.show_labels,
                        self.args.show_conf,
                    )

            if not self.training and self.args.split in {"val", "test"} and self.race_enabled:
                self._update_race_metrics(predn, pbatch)

            if no_pred:
                continue

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def _gather_image_metrics(self, metric) -> None:
        """Gather per-image metrics from all GPUs for a single metric object."""
        if RANK == 0:
            gathered_image_metrics = [None] * dist.get_world_size()
            dist.gather_object(metric.image_metrics, gathered_image_metrics, dst=0)
            metric.clear_image_metrics()
            for image_metrics in gathered_image_metrics:
                if image_metrics:
                    metric.image_metrics.update(image_metrics)
        elif RANK > 0:
            dist.gather_object(metric.image_metrics, None, dst=0)
            metric.clear_image_metrics()

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in self.metrics.stats}
            for stats_dict in gathered_stats:
                for key, value in stats_dict.items():
                    merged_stats[key].extend(value)
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            self.metrics.stats = merged_stats
            self._gather_image_metrics(self.metrics.box)
            self.seen = len(self.dataloader.dataset)  # total image count from dataset
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            dist.gather_object(self.jdict, None, dst=0)
            self._gather_image_metrics(self.metrics.box)
            self.jdict = []
            self.metrics.clear_stats()
        if self.args.plots and RANK > -1:
            matrix = torch.as_tensor(self.confusion_matrix.matrix, device=self.device)
            dist.reduce(matrix, dst=0, op=dist.ReduceOp.SUM)
            if RANK == 0:
                self.confusion_matrix.matrix = matrix.cpu().numpy()
        if self.race_enabled and RANK > -1:
            self._gather_race_stats()

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

        if not self.training and self.args.split in {"val", "test"} and self.race_enabled:
            self.print_race_results()

        if self.args.verbose and self.nc > 1:
            self._print_class_results()

    def _print_class_results(self) -> None:
        """Print detailed per-class metrics for the current validation run."""
        box = self.metrics.box
        if not len(box.p):
            return

        pf = "%20s" + "%12.3g" * 7
        header_pf = "%20s" + "%12s" * 7
        all_ap = box.all_ap
        map90 = all_ap[:, 8].mean() if len(all_ap) else 0.0
        LOGGER.info(
            header_pf % ("Class", "P", "R", "f1", "mAP@0.5", "mAP@0.75", "mAP@.90", "mAP@.5:.95")
        )
        LOGGER.info(pf % ("all", box.mp, box.mr, box.f1.mean(), box.map50, box.map75, map90, box.map))
        for i, c in enumerate(self.metrics.ap_class_index):
            LOGGER.info(
                pf
                % (
                    self.names[c],
                    box.p[i],
                    box.r[i],
                    box.f1[i],
                    box.ap50[i],
                    box.all_ap[i, 5],
                    box.all_ap[i, 8],
                    box.ap[i],
                )
            )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for
                10 IoU levels.
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def _is_race_dataset(self) -> bool:
        """Return True for the 25-class competition dataset layout."""
        names = self.names if isinstance(self.names, dict) else dict(enumerate(self.names))
        return self.nc == 25 and names.get(0) == "HM" and names.get(24) == "FSC"

    def _update_race_metrics(self, pred: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        """Accumulate competition metrics for one image."""
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
        add_race_stats(self.race_stats, race_match_stats(detections, labels, self.nc))

    def _gather_race_stats(self) -> None:
        """Gather competition metrics from all ranks."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.race_stats, gathered_stats, dst=0)
            self.race_stats = init_race_stats(self.nc)
            for stats in gathered_stats:
                add_race_stats(self.race_stats, stats)
        elif RANK > 0:
            dist.gather_object(self.race_stats, None, dst=0)

    def print_race_results(self) -> None:
        """Print and save competition Recall/FDR metrics."""
        stats = self.race_stats
        race_tp, race_fp, race_fn = (int(stats[k].sum()) for k in ("tp", "fp", "fn"))
        race_gt, race_recall, race_fdr = race_summary(race_tp, race_fp, race_fn)
        infer_ms = float(self.speed.get("inference", 0.0))
        post_ms = float(self.speed.get("postprocess", 0.0))
        competition_ms = infer_ms + post_ms
        race_metrics = {
            "tp": race_tp,
            "fp": race_fp,
            "fn": race_fn,
            "gt": int(race_gt),
            "recall": race_recall,
            "fdr": race_fdr,
            "inference_ms_per_image": infer_ms,
            "postprocess_ms_per_image": post_ms,
            "competition_inference_ms_per_image": competition_ms,
            "vehicle_iou": RACE_VEHICLE_IOU,
            "default_iou": RACE_DEFAULT_IOU,
            "vehicle_class": RACE_VEHICLE_CLASS,
        }
        LOGGER.info(
            f"Race metrics: Recall={race_recall:.6f}, FDR={race_fdr:.6f}, "
            f"TP={race_tp}, FP={race_fp}, FN={race_fn}, GT={race_gt}, "
            f"Forward={infer_ms:.3f}ms/img, Decode/NMS={post_ms:.3f}ms/img, "
            f"Competition inference={competition_ms:.3f}ms/img"
        )

        group_rows = []
        LOGGER.info("Race metrics by group:")
        LOGGER.info(("%12s" + "%10s" * 7) % ("Group", "TP", "FP", "FN", "GT", "Recall", "FDR", "Pred"))
        for group, cls_ids in RACE_GROUPS.items():
            cls_ids = [c for c in cls_ids if c < self.nc]
            tp_g = int(stats["tp"][cls_ids].sum()) if cls_ids else 0
            fp_g = int(stats["fp"][cls_ids].sum()) if cls_ids else 0
            fn_g = int(stats["fn"][cls_ids].sum()) if cls_ids else 0
            gt_g, recall_g, fdr_g = race_summary(tp_g, fp_g, fn_g)
            pred_g = tp_g + fp_g
            row = (group, tp_g, fp_g, fn_g, int(gt_g), recall_g, fdr_g, pred_g)
            group_rows.append(row)
            LOGGER.info(("%12s" + "%10i" * 4 + "%10.4f" * 2 + "%10i") % row)

        class_rows = []
        LOGGER.info("Race metrics by class:")
        LOGGER.info(
            ("%4s %20s" + "%8s" * 10)
            % ("ID", "Class", "TP", "FP", "FN", "GT", "Recall", "FDR", "Wrong", "Dup", "Bg/IoU", "Pred")
        )
        for c in range(self.nc):
            tp_c, fp_c, fn_c = (int(stats[k][c]) for k in ("tp", "fp", "fn"))
            gt_c, recall_c, fdr_c = race_summary(tp_c, fp_c, fn_c)
            wrong_c = int(stats["fp_wrong_class"][c])
            dup_c = int(stats["fp_duplicate"][c])
            bg_iou_c = int(stats["fp_background_or_iou"][c])
            pred_c = tp_c + fp_c
            class_name = self.names.get(c, str(c)) if isinstance(self.names, dict) else self.names[c]
            row = (c, class_name, tp_c, fp_c, fn_c, int(gt_c), recall_c, fdr_c, wrong_c, dup_c, bg_iou_c, pred_c)
            class_rows.append(row)
            LOGGER.info(("%4i %20s" + "%8i" * 4 + "%8.4f" * 2 + "%8i" * 4) % row)

        with open(self.save_dir / "race_metrics.json", "w", encoding="utf-8") as f:
            json.dump(race_metrics, f, indent=2)
        with open(self.save_dir / "race_metrics.csv", "w", encoding="utf-8") as f:
            f.write(
                "tp,fp,fn,gt,recall,fdr,inference_ms_per_image,postprocess_ms_per_image,"
                "competition_inference_ms_per_image,vehicle_iou,default_iou,vehicle_class\n"
            )
            f.write(
                f"{race_tp},{race_fp},{race_fn},{race_gt},{race_recall:.10f},{race_fdr:.10f},"
                f"{infer_ms:.10f},{post_ms:.10f},{competition_ms:.10f},"
                f"{RACE_VEHICLE_IOU},{RACE_DEFAULT_IOU},{RACE_VEHICLE_CLASS}\n"
            )
        with open(self.save_dir / "race_metrics_by_group.csv", "w", encoding="utf-8") as f:
            f.write("group,tp,fp,fn,gt,recall,fdr,pred\n")
            for row in group_rows:
                f.write("%s,%i,%i,%i,%i,%.10f,%.10f,%i\n" % row)
        with open(self.save_dir / "race_metrics_by_class.csv", "w", encoding="utf-8") as f:
            f.write(
                "class_id,class_name,tp,fp,fn,gt,recall,fdr,"
                "fp_wrong_class,fp_duplicate,fp_background_or_iou,pred\n"
            )
            for row in class_rows:
                f.write("%i,%s,%i,%i,%i,%i,%.10f,%.10f,%i,%i,%i,%i\n" % row)
        with open(self.save_dir / "race_confusion_matrix.csv", "w", encoding="utf-8") as f:
            f.write("true_class_id,true_class_name,pred_class_id,pred_class_name,count\n")
            for true_c, pred_c in zip(*np.nonzero(stats["confusion"])):
                if isinstance(self.names, dict):
                    true_name = self.names.get(int(true_c), str(true_c))
                    pred_name = self.names.get(int(pred_c), str(pred_c))
                else:
                    true_name = self.names[int(true_c)]
                    pred_name = self.names[int(pred_c)]
                f.write(f"{true_c},{true_name},{pred_c},{pred_name},{int(stats['confusion'][true_c, pred_c])}\n")

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
            device=self.device,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (int | None): Maximum number of detections to plot.
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics for object detection. Updates the
        provided stats dictionary with computed metrics including mAP50, mAP50-95, and LVIS-specific metrics if
        applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]): IoU type(s) for evaluation. Can be single string or list of strings. Common
                values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]): Suffix to append to metric names in stats dictionary. Should correspond to
                iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]
                    # record mAP for small, medium, large objects as well
                    stats["metrics/mAP_small(B)"] = val.stats_as_dict["AP_small"]
                    stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
                    stats["metrics/mAP_large(B)"] = val.stats_as_dict["AP_large"]
                    # update fitness
                    stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats

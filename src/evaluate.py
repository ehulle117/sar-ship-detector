"""Evaluate a trained checkpoint: precision/recall/mAP at IoU thresholds."""

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import SSDDDataset
from model import build_model
from train import collate_fn, to_tensor_transform


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import box_iou
    return box_iou(boxes1, boxes2)


@torch.no_grad()
def evaluate(data_root: str, checkpoint: str, iou_threshold: float = 0.5, score_threshold: float = 0.5, full_dataset: bool = False):
    # MPS skipped -- see note in train.py.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SSDDDataset(data_root, transforms=to_tensor_transform)

    if full_dataset:
        eval_set = dataset
        print("Evaluating on the FULL dataset (includes training images -- optimistic).")
    else:
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - val_size
        split_generator = torch.Generator().manual_seed(42)
        _, eval_set = random_split(dataset, [train_size, val_size], generator=split_generator)
        print(f"Evaluating on held-out val split only ({len(eval_set)} images not seen during training).")

    loader = DataLoader(eval_set, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    total_tp, total_fp, total_fn = 0, 0, 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, target in zip(predictions, targets):
            keep = pred["scores"] >= score_threshold
            pred_boxes = pred["boxes"][keep].cpu()
            gt_boxes = target["boxes"]

            if len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            ious = box_iou_matrix(pred_boxes, gt_boxes)
            matched_gt = set()
            for i in range(len(pred_boxes)):
                best_iou, best_j = ious[i].max(dim=0)
                if best_iou >= iou_threshold and best_j.item() not in matched_gt:
                    total_tp += 1
                    matched_gt.add(best_j.item())
                else:
                    total_fp += 1
            total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    print(f"IoU >= {iou_threshold}, score >= {score_threshold}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--full-dataset", action="store_true", help="Evaluate on all images instead of the held-out val split (optimistic; includes training data).")
    args = parser.parse_args()

    evaluate(args.data_root, args.checkpoint, args.iou_threshold, args.score_threshold, args.full_dataset)

"""Draw predicted vs ground-truth boxes on sample images for visual QA."""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from dataset import SSDDDataset
from model import build_model
from train import to_tensor_transform


@torch.no_grad()
def visualize_samples(data_root: str, checkpoint: str, output_dir: str, num_samples: int = 6, score_threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SSDDDataset(data_root, transforms=to_tensor_transform)
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(min(num_samples, len(dataset))):
        image, target = dataset[idx]
        prediction = model([image.to(device)])[0]

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image.permute(1, 2, 0).cpu())

        for box in target["boxes"]:
            x0, y0, x1, y1 = box.tolist()
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="lime", facecolor="none", label="ground truth"))

        keep = prediction["scores"] >= score_threshold
        for box, score in zip(prediction["boxes"][keep], prediction["scores"][keep]):
            x0, y0, x1, y1 = box.tolist()
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none"))
            ax.text(x0, y0 - 5, f"{score:.2f}", color="red", fontsize=8)

        ax.set_title(f"Sample {idx} — green: ground truth, red: prediction")
        ax.axis("off")
        fig.savefig(f"{output_dir}/sample_{idx}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {num_samples} visualizations to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="../outputs/visualizations")
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    visualize_samples(args.data_root, args.checkpoint, args.output_dir, args.num_samples)

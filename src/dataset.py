"""PyTorch Dataset for SSDD: loads SAR image chips + VOC-style ship bounding boxes."""

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class SSDDDataset(Dataset):
    def __init__(self, root: str, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.image_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"
        self.image_files = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        ann_path = self.ann_dir / f"{image_path.stem}.xml"

        image = Image.open(image_path).convert("RGB")
        boxes, labels = self._parse_annotation(ann_path)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def _parse_annotation(ann_path: Path):
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # single class: ship

        return boxes, labels

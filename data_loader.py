import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

class CosmosDataset(Dataset):
    def __init__(self, json_file, img_dir, transform_full=None):
        """
        Args:
            json_file (str): Path to the JSON file with samples.
            img_dir (str): Directory containing images.
            transform_full (callable, optional): Transformation for the full image.
        """
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.img_dir = img_dir
        self.transform_full = transform_full

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get sample dictionary
        sample = self.data[idx]
        # Load full image
        img_path = os.path.join(self.img_dir, sample["img_local_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image at {img_path}: {e}")

        if self.transform_full:
            image = self.transform_full(image)  # Tensor shape: (3, H, W)
        # Retrieve bounding boxes; each box is [[x1, y1], [x2, y2]]
        bboxes = sample.get("maskrcnn_bboxes", [])
        if len(bboxes) == 0:
            # If no box exists, use full image box.
            _, H, W = image.shape
            bboxes = [ [(0, 0), (W, H)] ]
        # Use one of the caption variants (e.g., caption1_modified).
        caption_match = sample.get("caption1_modified", "")
        return {
            "idx": idx,
            "image": image,   # full image tensor (3, H, W)
            "bboxes": bboxes[:10], # list of bounding boxes
            "caption_match": caption_match
        }
        

if __name__ == "__main__":
    dataset = CosmosDataset(json_file='data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json', img_dir="data/train")
    print(dataset.data[0])
    
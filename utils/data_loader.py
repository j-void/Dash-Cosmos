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
        example = self.data[idx]
        img_path = os.path.join(self.img_dir, example["img_local_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image at {img_path}: {e}")

        if self.transform_full:
            image = self.transform_full(image)  # Tensor shape: (3, H, W)
            
        bboxes = example.get("maskrcnn_bboxes", [])[:10]
        _, H, W = image.shape
        bboxes.append([0, 0, W, H])
        bboxes = [torch.tensor(bbox) for bbox in bboxes]
        img_captions = img_data['articles']
        caption_match = img_captions[random.randint(0, len(img_captions) - 1)]['caption_modified']
        
        caption_diff = ""
        while True:
            ridx = random.randint(0, len(self.data) - 1)
            random_captions = self.data[ridx]['articles']
            caption_diff = random_captions[random.randint(0, len(random_captions) - 1)]['caption_modified']
            if caption_match != caption_diff:
                break
                
        return image, bboxes, caption_match, caption_diff




if __name__ == "__main__":
    dataset = CosmosDataset(json_file='data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json', img_dir="data/train")
    print(dataset.data[0])
    
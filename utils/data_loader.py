import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from utils.data_aug import Resize

import cv2

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
        # self.data = self.data[:10000]
        self.img_dir = img_dir
        self.transform_full = transform_full
        self.resize = Resize(512)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        img_path = os.path.join(self.img_dir, example["img_local_path"])
        try:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error loading image at {img_path}: {e}")
        bboxes = np.array(example.get("maskrcnn_bboxes", []))
        image, bboxes = self.resize(image, bboxes)
        _, H, W = image.shape
        bboxes = list(bboxes)[:10]
        bboxes.append([0, 0, W, H])
        bboxes = [torch.tensor(bbox) for bbox in bboxes]

        if self.transform_full:
            image = self.transform_full(image)        

        img_captions = example['articles']
        caption_match = img_captions[random.randint(0, len(img_captions) - 1)]['caption_modified']
        
        caption_diff = ""
        while True:
            ridx = random.randint(0, len(self.data) - 1)
            random_captions = self.data[ridx]['articles']
            caption_diff = random_captions[random.randint(0, len(random_captions) - 1)]['caption_modified']
            if caption_match != caption_diff:
                break

        return image.float(), bboxes, caption_match, caption_diff




if __name__ == "__main__":
    dataset = CosmosDataset(json_file='data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json', img_dir="data/train")
    print(dataset.data[0])
    
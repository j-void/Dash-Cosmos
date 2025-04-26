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
import nltk
import utils.misc as misc
import nlpaug.augmenter.word as naw
from transformers import logging
logging.set_verbosity_error()

import cv2

try:
    nltk.find('wordnet')
    nltk.find('averaged_perceptron_tagger_eng')
    nltk.find('omw-1.4')
except:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('omw-1.4')

class CosmosDataset(Dataset):
    def __init__(self, json_file, img_dir, transform_full=None, size=512):
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
        # self.data = self.data[:len(self.data)//10]
        self.img_dir = img_dir
        self.transform_full = transform_full
        self.resize = Resize(size)

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
    
class CosmosDataset_Syth(Dataset):
    def __init__(self, json_file, img_dir, transform_full=None, size=512):
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
        # self.data = self.data[:len(self.data)//2000]
        self.img_dir = img_dir
        self.transform_full = transform_full
        self.resize = Resize(size)

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
        
        all_caption_info = img_captions[random.randint(0, len(img_captions) - 1)]
        caption_match_orig = all_caption_info['caption']        
        caption_match = all_caption_info['caption_modified']
        
        caption_diff = ""
        while True:
            ridx = random.randint(0, len(self.data) - 1)
            random_captions = self.data[ridx]['articles']
            caption_diff = random_captions[random.randint(0, len(random_captions) - 1)]['caption_modified']
            if caption_match != caption_diff:
                break

        return image.float(), bboxes, caption_match, caption_diff, caption_match_orig

class CosmosTestDataset(Dataset):
    def __init__(self, json_file, img_dir, transform_full=None, size=512):
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
        self.resize = Resize(size)

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

        caption1 = example['caption1_modified']
        caption2 = example['caption2_modified']
        
        label = torch.tensor(example['context_label']).long()        
        bert_score = float(example['bert_base_score'])
        bert_score = torch.tensor(bert_score).float()

        return image.float(), bboxes, caption1, caption2, label, bert_score, example['caption1'], example['caption2']


class SyntheticNegatives:
    def __init__(self, antonym_prob=0.9):
        self.context_aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action='substitute', aug_p=0.6, device='cuda')
        self.antonym_aug = naw.AntonymAug(aug_p=0.3, aug_min=1, aug_max=3)
        self.antonym_prob = antonym_prob
    
    def get_augmented(self, caption):
        caption_augmented = self.context_aug.augment(caption)
        if self.antonym_prob > np.random.rand():
            caption_augmented = self.antonym_aug.augment(caption_augmented)
        return caption_augmented

if __name__ == "__main__":
    dataset = CosmosDataset(json_file='data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json', img_dir="data/train")
    print(dataset.data[0])

    
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
import nltk
from nltk.corpus import wordnet

import cv2

nltk.download('wordnet')

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


class SpacyNegatives:
    def __init__(self):
        nlp = spacy.load("en_core_web_sm")
    
    def get_wordnet_pos(self, spacy_pos):
        return {
            'ADJ': wordnet.ADJ,
            'ADV': wordnet.ADV,
            'VERB': wordnet.VERB,
            'NOUN': wordnet.NOUN
        }.get(spacy_pos, None)
    
    def get_simple_negation(self, caption):
        doc = nlp(caption)

        NEG_TOKENS = {'not', 'never', 'no', 'none', 'nobody', 'nothing', 'neither'}
        contains_neg = any(tok.lower_ in NEG_TOKENS for tok in doc)

        NEG_PREFIXES = ('un', 'in', 'im', 'il', 'ir', 'dis', 'non')
        for tok in doc:
            wn_pos = get_wordnet_pos(tok.pos_)
            if wn_pos is None:
                continue
            if tok.lower_ in NEG_TOKENS:
                continue
            if contains_neg:
                break
            synsets = wordnet.synsets(tok.text, pos=wn_pos)
            for syn in synsets:
                for lemma in syn.lemmas():
                    for ant in lemma.antonyms():
                        ant_name = ant.name().replace('_', ' ')
                        if any(ant_name.startswith(pref) for pref in NEG_PREFIXES):
                            continue
                        words = [t.text for t in doc]
                        words[tok.i] = ant_name
                        return " ".join(words)
        return caption

class SyntheticOppositeGenerator:
    def __init__(self):
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            
    def diff_caption_t5(self, caption):
        
        prompts = [
            f"paraphrase but contradict: {caption}",
            f"generate opposite meaning of: {caption}",
            f"rewrite with different meaning: {caption}"
        ]
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.paraphrase_model.generate(
            inputs["input_ids"], 
            max_length=100,
            do_sample=True,
            top_p=0.9
        )
        contradictory_caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return contradictory_caption
    

if __name__ == "__main__":
    #dataset = CosmosDataset(json_file='data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json', img_dir="data/train")
    #print(dataset.data[0])
    sn = SpacyNegatives()
    print(sn("Freeland defends Canada\u2019s position, says it will always speak up for human rights"))
    
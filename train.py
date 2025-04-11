
import os
import json
import torch
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.data_loader import *
from networks.cosmos import CosmosFullModel
import utils.misc as misc

def collate_fn(batch):
    return batch


if __name__=="__main__":
    ## define config 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EMBEDDING_DIM = 300
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    
    train_dataset = CosmosDataset(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json", img_dir="data/train")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
	)
    
    model = CosmosFullModel(embedding_dim=EMBEDDING_DIM, device=DEVICE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    optimizer.zero_grad()
    
    margin_rank_loss = nn.MarginRankingLoss(margin=1)
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        
        for idx, (image, bboxes, caption_match, caption_diff) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            object_embeddings, match_embeddings, diff_embeddings = model(image, bboxes, caption_match, caption_diff)
            
            match_scores, diff_scores = misc.get_scores(object_embeddings, match_embeddings, diff_embeddings)
            loss = margin_rank_loss(match_scores, diff_scores)

            loss.backward()
            optimizer.step()
            
            train_acc = misc.compute_accuracy(match_scores, diff_scores)

    
    
    
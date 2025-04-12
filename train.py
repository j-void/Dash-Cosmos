
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
from torch.utils.tensorboard import SummaryWriter
import math

def collate_fn(batch):
    return batch

def run_eval(model, dataloader, loss_fn):
    model.eval()
    correct = 0
    count = 0
    total_loss = 0
    with torch.no_grad():
        for image, bboxes, caption_match, caption_diff in dataloader:
            object_embeddings, match_embeddings, diff_embeddings = model(image, bboxes, caption_match, caption_diff)
            match_scores, diff_scores = misc.get_scores(object_embeddings, match_embeddings, diff_embeddings)
            ceval = misc.compute_matches(match_scores, diff_scores)
            loss = loss_fn(match_scores, diff_scores)
            total_loss += loss
            correct += ceval[0]
            count += ceval[1]
    return correct/count, total_loss/count

if __name__=="__main__":
    ## define config 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EMBEDDING_DIM = 300
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    CHECKPOINT_DIR = "checkpoints/cosmos"
    
    save_dir = os.path.join(CHECKPOINT_DIR, "save")
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(CHECKPOINT_DIR, "logs"))
    
    train_dataset = CosmosDataset(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json", img_dir="data/train")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
	)
    
    val_dataset = CosmosDataset(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/val_data.json", img_dir="data/val")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
	)
    
    model = CosmosFullModel(embedding_dim=EMBEDDING_DIM, device=DEVICE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    optimizer.zero_grad()
    
    margin_rank_loss = nn.MarginRankingLoss(margin=1)
    
    min_val_loss = math.inf
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        count = 0
        total_loss = 0
        correct = 0
        for idx, (image, bboxes, caption_match, caption_diff) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            object_embeddings, match_embeddings, diff_embeddings = model(image, bboxes, caption_match, caption_diff)
            
            match_scores, diff_scores = misc.get_scores(object_embeddings, match_embeddings, diff_embeddings)
            loss = margin_rank_loss(match_scores, diff_scores)

            loss.backward()
            optimizer.step()
            
            train_eval = misc.compute_matches(match_scores, diff_scores)
            correct += train_eval[0]
            count += train_eval[1]
            total_loss += loss.item()
        
        train_acc = correct/count
        train_loss = total_loss/count
        val_acc, val_loss = run_eval(model, val_loader, margin_rank_loss)
        writer.add_scalars("Loss", {'val':val_loss, 'train': train_loss}, epoch)
        writer.add_scalars("Accuracy", {'val':val_acc, 'train': train_acc}, epoch)
        tqdm.write(f'Epoch {epoch}: Loss- (train={train_loss}, val={val_loss}) | Acc- (train={train_acc}, val={val_acc}))')
        
        if min_val_loss < val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "maskrcnn_use.torch"))
            tqdm.write("Saving best model.....")
    
    

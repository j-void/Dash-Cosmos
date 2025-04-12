
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import torch
import torch.optim as optim
import torchvision
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.data_loader import *
from networks.cosmos import CosmosFullModel
import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

def collate_fn(batch):
    images, bboxes, caption_match, caption_diff = zip(*batch)
    return list(images), list(bboxes), list(caption_match), list(caption_diff)

def run_eval(model, dataloader, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    count = 0
    with torch.no_grad():
        for image, bboxes, caption_match, caption_diff in tqdm(dataloader, desc="Val Batch:"):
            object_embeddings, match_embeddings, diff_embeddings = model(image, bboxes, caption_match, caption_diff)
            match_scores, diff_scores = misc.get_scores(object_embeddings, match_embeddings, diff_embeddings)
            
            # Binary prediction: 1 if match > diff, else 0
            preds = (match_scores > diff_scores).int().cpu().numpy()
            labels = np.ones_like(preds)  # All match cases are positives
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Loss (assumes margin ranking loss)
            loss = loss_fn(match_scores, diff_scores, torch.ones(match_scores.shape[0]).to(match_scores.device))
            total_loss += loss.item()
            count += match_scores.size(0)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "average_precision": ap,
        "loss": total_loss / len(dataloader)
    }

if __name__=="__main__":
    ## define config 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EMBEDDING_DIM = 300
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 100
    CHECKPOINT_DIR = "checkpoints/cosmos"
    
    save_dir = os.path.join(CHECKPOINT_DIR, "save")
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(CHECKPOINT_DIR, "logs"))
    
    transform_full = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])    
    
    train_dataset = CosmosDataset(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json", \
        img_dir="data", transform_full=transform_full)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
	)
    
    val_dataset = CosmosDataset(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/val_data.json", \
        img_dir="data", transform_full=transform_full)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
	)
    
    model = CosmosFullModel(embedding_dim=EMBEDDING_DIM, device=DEVICE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    optimizer.zero_grad()
    
    margin_rank_loss = torch.nn.MarginRankingLoss(margin=1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    
    min_val_loss = math.inf
    max_val_acc = 0
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        count = 0
        total_loss = 0
        correct = 0
        for idx, batch in enumerate((tqdm(train_loader, desc="Train Batch: "))):
            image, bboxes, caption_match, caption_diff = batch
            optimizer.zero_grad()
            object_embeddings, match_embeddings, diff_embeddings = model(image, bboxes, caption_match, caption_diff)
            
            match_scores, diff_scores = misc.get_scores(object_embeddings, match_embeddings, diff_embeddings)
            loss = margin_rank_loss(match_scores, diff_scores, torch.ones(match_scores.shape[0]).to(DEVICE))

            loss.backward()
            optimizer.step()
            
            train_eval = misc.compute_matches(match_scores, diff_scores)
            correct += train_eval[0]
            count += train_eval[1]
            total_loss += loss.item()
            
            
        
        train_acc = correct/count
        train_loss = total_loss/len(train_loader)
        val_metrics = run_eval(model, val_loader, margin_rank_loss)
        writer.add_scalars("Loss", {'val':val_metrics["loss"], 'train': train_loss}, epoch)
        writer.add_scalars("Accuracy", {'val':val_metrics["accuracy"], 'train': train_acc}, epoch)
        writer.add_scalar("Val-F1", val_metrics["f1_score"], epoch)
        writer.add_scalar("Val-AP", val_metrics["average_precision"], epoch)
        tqdm.write(f'Epoch {epoch}: Loss-> (train={train_loss:.4f}, val={val_metrics["loss"]:.4f}) | Acc-> (train={train_acc}, val={val_metrics["accuracy"]}))')
        
        scheduler.step(val_metrics["loss"])
        
        if min_val_loss > val_metrics["loss"]:
            min_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(save_dir, "maskrcnn_use_loss.torch"))
            tqdm.write("Saving best loss model.....")
        
        if max_val_acc < val_metrics["accuracy"]:
            max_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(save_dir, "maskrcnn_use_acc.torch"))
            tqdm.write("Saving best accuracy model.....")
    

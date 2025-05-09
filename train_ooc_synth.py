
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.data_loader import *
from networks.ooc_models import OOCBasic
import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, BatchEncoding


def collate_fn(batch):
    images, bboxes, caption_match, caption_diff, caption_match_orig = zip(*batch)
    return list(images), list(caption_match), list(caption_diff), list(caption_match_orig)


def get_caption_tokens(caption_match, caption_diff, caption_orig, tokenizer, augmenter, synthetic_prob):
    n = len(caption_match)
    synth_mask = np.random.rand(n) < synthetic_prob

    new_diffs = list(caption_diff)

    idxs = np.nonzero(synth_mask)[0].tolist()
    if idxs:
        to_aug = [caption_orig[i] for i in idxs]
        raw_ctx = augmenter.context_aug.augment(to_aug)
        ctx_aug = [c[0] if isinstance(c, list) else c for c in raw_ctx]

        raw_ant = augmenter.antonym_aug.augment(ctx_aug)
        ant_aug = [a[0] if isinstance(a, list) else a for a in raw_ant]

        new_texts = misc.batch_replace_entities(ant_aug)
        for i, aug in zip(idxs, new_texts):
            new_diffs[i] = aug

    ## speed up
    # all_texts = caption_match + new_diffs
    # batch_text = tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    # bs = len(caption_match)
    # pos_tokens = BatchEncoding({k: v[:bs] for k, v in batch_text.items()})
    # neg_tokens = BatchEncoding({k: v[bs:] for k, v in batch_text.items()})

    pos_tokens = tokenizer(caption_match, return_tensors='pt', padding=True, truncation=True, max_length=32)
    neg_tokens = tokenizer(new_diffs, return_tensors='pt', padding=True, truncation=True, max_length=32)

    return pos_tokens, neg_tokens


def run_eval(model, dataloader, loss_fn, tokenizer, augmenter):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    count = 0
    total_match_distance = 0.0
    total_diff_distance = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Batch:"):
            image, caption_match, caption_diff, caption_match_orig = batch
            image = torch.stack(image, dim=0).to(DEVICE)
            caption_match, caption_diff = get_caption_tokens(caption_match, caption_diff, caption_match_orig, tokenizer, augmenter, 0.8)
            caption_match = caption_match.to(DEVICE)
            caption_diff = caption_diff.to(DEVICE)
            object_embeddings, match_embeddings, diff_embeddings = model(image, caption_match, caption_diff)
            
            # Compute distances
            match_distances = torch.norm(object_embeddings - match_embeddings, dim=1)
            diff_distances = torch.norm(object_embeddings - diff_embeddings, dim=1)
            total_match_distance += match_distances.sum().item()
            total_diff_distance += diff_distances.sum().item()
            
            # 1 if match is closer than diff, else 0
            preds = (match_distances < diff_distances).int().cpu().numpy()
            labels = np.ones_like(preds)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Loss (assumes margin ranking loss)
            loss = loss_fn(object_embeddings, match_embeddings, diff_embeddings)
            total_loss += loss.item()
            count += object_embeddings.size(0)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    ap = 0#average_precision_score(all_labels, all_preds)
    
    avg_match_distance = total_match_distance / count
    avg_diff_distance = total_diff_distance / count

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "average_precision": ap,
        "loss": total_loss / len(dataloader),
        "avg_match_distance": avg_match_distance,
        "avg_diff_distance": avg_diff_distance
    }

if __name__=="__main__":
    
    ## define config 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 100
    CHECKPOINT_DIR = "checkpoints/ooc_basic_synth"
    
    save_dir = os.path.join(CHECKPOINT_DIR, "save")
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(CHECKPOINT_DIR, "logs"))
    
    transform_full = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])    
    
    augmenter = SyntheticNegatives()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", use_fast=True)
    
    train_dataset = CosmosDataset_Syth(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/train_data.json", \
        img_dir="data", transform_full=transform_full, size=224)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_dataset = CosmosDataset_Syth(json_file="data/cosmos_anns_acm/cosmos_anns_acm/acm_anns/val_data.json", \
        img_dir="data", transform_full=transform_full, size=224)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
	)
    
    start_epoch = 0
    
    model = OOCBasic(img_model="vit_base_patch16_224", txt_model="sentence-transformers/all-mpnet-base-v2",
                 embed_dim=768, proj_dim=256, num_heads=8)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    optimizer.zero_grad()
    
    margin_rank_loss = torch.nn.MarginRankingLoss(margin=1)
    triplet_loss  = torch.nn.TripletMarginLoss(margin=0.2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    
    checkpoint_path = os.path.join(save_dir, "ooc_acc.torch")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1 
    
    min_val_loss = math.inf
    max_val_acc = 0
    
    for epoch in tqdm(range(start_epoch, NUM_EPOCHS), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        count = 0
        total_loss = 0
        correct = 0
        for idx, batch in enumerate((tqdm(train_loader, desc="Train Batch: "))):
            optimizer.zero_grad()
            image, caption_match, caption_diff, caption_match_orig = batch
            image = torch.stack(image, dim=0).to(DEVICE)
            caption_match, caption_diff = get_caption_tokens(caption_match, caption_diff, caption_match_orig, tokenizer, augmenter, 0.8)
            caption_match = caption_match.to(DEVICE)
            caption_diff = caption_diff.to(DEVICE)
            object_embeddings, match_embeddings, diff_embeddings = model(image, caption_match, caption_diff)
            
            loss = triplet_loss(object_embeddings, match_embeddings, diff_embeddings)

            loss.backward()
            optimizer.step()
            
            train_eval = misc.compute_accuracy_cl(object_embeddings, match_embeddings, diff_embeddings)
            correct += train_eval[0]
            count += train_eval[1]
            total_loss += loss.item()
            
            
        
        train_acc = correct/count
        train_loss = total_loss/len(train_loader)
        val_metrics = run_eval(model, val_loader, triplet_loss, tokenizer, augmenter)
        writer.add_scalars("Loss", {'val':val_metrics["loss"], 'train': train_loss}, epoch)
        writer.add_scalars("Accuracy", {'val':val_metrics["accuracy"], 'train': train_acc}, epoch)
        writer.add_scalar("Val-F1", val_metrics["f1_score"], epoch)
        writer.add_scalar("Val-AP", val_metrics["average_precision"], epoch)
        tqdm.write(f'Epoch {epoch}: Loss-> (train={train_loss:.4f}, val={val_metrics["loss"]:.4f}) | Acc-> (train={train_acc:.4f}, val={val_metrics["accuracy"]:.4f})')
        
        scheduler.step(val_metrics["loss"])
        
        if min_val_loss > val_metrics["loss"]:
            min_val_loss = val_metrics["loss"]
            misc.save_model(epoch, model, optimizer, scheduler, os.path.join(save_dir, "ooc_loss.torch"))
            tqdm.write("Saving best loss model.....")
            # tqdm.write(f'Avg dist -> (match={val_metrics["avg_match_distance"]:.4f}, diff={val_metrics["avg_diff_distance"]:.4f})')
        
        if max_val_acc < val_metrics["accuracy"]:
            max_val_acc = val_metrics["accuracy"]
            misc.save_model(epoch, model, optimizer, scheduler, os.path.join(save_dir, "ooc_acc.torch"))
            tqdm.write("Saving best accuracy model.....")
            # tqdm.write(f'Avg dist -> (match={val_metrics["avg_match_distance"]:.4f}, diff={val_metrics["avg_diff_distance"]:.4f})')
    

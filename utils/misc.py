
import os
import torch
import numpy as np

def get_scores(object_embeddings, match_embeddings, diff_embeddings):
    match_scores_all = torch.einsum("bkd,bd->bk", object_embeddings, match_embeddings)
    diff_scores_all = torch.einsum("bkd,bd->bk", object_embeddings, diff_embeddings)

    # Take the max over the K objects in each image
    match_scores = match_scores_all.max(dim=1).values  # [B]
    diff_scores = diff_scores_all.max(dim=1).values    # [B]
    
    return match_scores, diff_scores

def compute_matches(match_scores, diff_scores):

    # Predict 1 if match score > diff score
    predictions = (match_scores > diff_scores).long()   # (B,)
    correct = predictions.sum().item()
    total = predictions.size(0)
    return correct, total

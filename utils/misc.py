
import os
import torch
import numpy as np
import spacy
from functools import partial

spacy.require_gpu() 
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm", disable=["tagger","parser","attribute_ruler","lemmatizer"])

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

def save_model(epoch, model, optimizer, scheduler, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)
    
def top_bbox_from_scores(bboxes, scores):
    bbox_scores = [(bbox, score) for bbox, score in zip(bboxes, scores)]
    sorted_bbox_scores = sorted(bbox_scores, key=lambda x: x[1], reverse=True)
    matched_bbox = sorted_bbox_scores[0][0]
    return matched_bbox

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of both boxes
    # intersection area / areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def is_bbox_overlap(bbox1, bbox2, iou_overlap_threshold):
    iou = bb_intersection_over_union(boxA=bbox1, boxB=bbox2)
    if iou >= iou_overlap_threshold:
        return True
    return False

def compute_accuracy_cl(anchor, positive, negative):
    # Calculate distances
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    
    # Determine correct predictions
    correct = (pos_dist < neg_dist).sum().item()
    total = anchor.size(0)
    
    return correct, total

def modify_caption_replace_entities(caption_text):

    doc = nlp(caption_text)
    caption_modified = caption_text
    caption_entity_list = []
    for ent in doc.ents:
        caption_entity_list.append((ent.text, ent.label_))
        caption_modified = caption_modified.replace(ent.text, ent.label_, 1)
    return caption_modified


pipe_cfg = {"batch_size": 256, "n_process": 1, 
            "disable": ["tagger", "parser", "attribute_ruler", "lemmatizer"]}

def batch_replace_entities(texts):
    out_texts = []
    for doc in nlp.pipe(texts, **pipe_cfg):
        text = doc.text
        for ent in doc.ents:
            text = text.replace(ent.text, ent.label_, 1)
        out_texts.append(text)
    return out_texts
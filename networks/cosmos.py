
import os
import json
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import roi_align
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class ObjectEncoder(nn.Module):
    def __init__(self, embedding_dim, device):
        super(ObjectEncoder, self).__init__()
        # Using ResNet-50 feature extractor from Mask-RCNN
        self.embedding_dim = embedding_dim
        self.device = device
        self.feature_extractor = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone
        # Freeze weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Additional layers for embedding
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)  # After RoIAlign with output size 7x7
        self.fc2 = nn.Linear(1024, self.embedding_dim)
        self.relu = nn.ReLU()
        
    def forward(self, images, bboxes):
        # Extract features from backbone
        features = self.feature_extractor(torch.stack(images).to(self.device))
        
        # Get the feature map
        feature_map = features["0"]  # Use the highest resolution feature map
        
        # Prepare bboxes for roi_align
        bboxes = [torch.stack(b) for b in bboxes]
        roi_boxes = []
        for batch_idx, boxes in enumerate(bboxes):
            batch_ids = torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
            roi_boxes.append(torch.cat([batch_ids, boxes], dim=1))

        roi_boxes = torch.cat(roi_boxes, dim=0).to(self.device).float()  # (K, 5)

        # Step 3: Apply RoI Align
        roi_features = roi_align(feature_map, roi_boxes, output_size=(7, 7))
        
        # Flatten
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # Feed through fully connected layers
        x = self.relu(self.fc1(roi_features))
        object_embeddings = self.fc2(x)
        object_embeddings = object_embeddings.view(len(bboxes), bboxes[0].shape[0], -1)
            
        return object_embeddings

# Text Encoder: Processes captions using Universal Sentence Encoder
class USE_TextEncoder(nn.Module):
    def __init__(self, embedding_dim, device):
        super(USE_TextEncoder, self).__init__()
        # Additional layer after USE embedding
        self.embedding_dim = embedding_dim
        self.device = device
        self.fc = nn.Linear(512, embedding_dim)
        self.relu = nn.ReLU()
        
        # Load Universal Sentence Encoder
        self.use_module = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
    def forward(self, captions):
        # Get embeddings from USE 
        with torch.no_grad():
            use_embeddings = self.use_module(captions).numpy()
            use_embeddings = torch.tensor(use_embeddings, device=self.device)
        
        # Transform to match object embedding dimension
        text_embeddings = self.fc(self.relu(use_embeddings))
        return text_embeddings

# Complete Image-Text Matching Model
class CosmosFullModel(nn.Module):
    def __init__(self, embedding_dim, device):
        super(CosmosFullModel, self).__init__()
        self.object_encoder = ObjectEncoder(embedding_dim, device)
        self.text_encoder = USE_TextEncoder(embedding_dim, device)
        
    def forward(self, images, bboxes, captions_match, captions_diff):
        # Get object embeddings for each image
        object_embeddings = self.object_encoder(images, bboxes)
        
        # Get caption embeddings
        match_embeddings = self.text_encoder(captions_match)
        diff_embeddings = self.text_encoder(captions_diff)
        
        return object_embeddings, match_embeddings, diff_embeddings
    
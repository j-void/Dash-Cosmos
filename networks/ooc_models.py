import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
import random
import os

class ImageEncoder(nn.Module):
    """
    ViT-based image encoder.
    """
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.vit = create_model(model_name, pretrained=pretrained)
        self.vit.head = nn.Identity()  # Remove classification head
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, images):
        # images: [B, 3, 224, 224]
        # forward_features returns [B, N+1, D]
        feats = self.vit.forward_features(images)
        return feats[:, 1:, :]  # discard CLS token, return [B, N, D]
    
class TextEncoder(nn.Module):
    """
    SBERT-based text encode.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False  # freeze

    def forward(self, tokens):
        out = self.model(**tokens)
        return out.last_hidden_state

class CrossAttention_TextBased(nn.Module):
    """
    Cross-attention: all text tokens (incl. CLS) attend to image patch tokens.
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, patch_feats, text_feats):
        """
        patch_feats: [B, N, D] - image tokens (ViT)
        text_feats:  [B, T, D] - all text tokens incl. CLS (SBERT)

        Returns:
            attn_out: [B, T, D] - updated text tokens after attending to image
        """
        attn_out, _ = self.attn(text_feats, patch_feats, patch_feats)
        return attn_out[:, 0, :]

class CrossAttention_ImageBased(nn.Module):
    """
    Cross-attention: image tokens attend to all text tokens 
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, patch_feats, text_feats):
        """
        patch_feats: [B, N, D] - image patch tokens (ViT)
        text_feats:  [B, T, D] - all text tokens incl. CLS (SBERT)

        Returns:
            attn_out: [B, N, D] - updated image tokens after attending to text
        """
        attn_out, _ = self.attn(patch_feats, text_feats, text_feats)
        return attn_out[:, 0, :]


class ProjectionHead(nn.Module):
    """
    MLP projection head to shared latent space.
    """
    def __init__(self, dim, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x, normalize_toks=True):
        if normalize_toks:
            return F.normalize(self.proj(x), dim=-1)
        else:
            return self.proj(x)

class OOCBasic(nn.Module):
    """
    Self-supervised model with cross-attention for OOC detection.
    """
    def __init__(self, img_model="vit_base_patch16_224", txt_model="sentence-transformers/all-mpnet-base-v2",
                 embed_dim=768, proj_dim=256, num_heads=8):
        super().__init__()
        self.image_encoder = ImageEncoder(img_model)
        self.text_encoder  = TextEncoder(txt_model)
        self.cross_attn    = CrossAttention_TextBased(embed_dim, num_heads)
        self.image_proj    = ProjectionHead(embed_dim, proj_dim)
        self.text_proj     = ProjectionHead(embed_dim, proj_dim)

    def forward(self, images, pos_tokens, neg_tokens, normalize_toks=True):
        patch_feats = self.image_encoder(images)          # [B, N, D]

        pos_txt_tokens = self.text_encoder(pos_tokens)    # [B, T, D]
        neg_txt_tokens = self.text_encoder(neg_tokens)    # [B, T, D]

        pos_context = self.cross_attn(patch_feats, pos_txt_tokens)  # [B, D]
        neg_context = self.cross_attn(patch_feats, neg_txt_tokens)  # [B, D]

        img_emb = self.image_proj(pos_context)            # [B, P]
        pos_emb = self.text_proj(pos_txt_tokens[:, 0])    # [B, P]  — CLS token from positive text
        neg_emb = self.text_proj(neg_txt_tokens[:, 0])    # [B, P]  — CLS token from negative text

        return img_emb, pos_emb, neg_emb

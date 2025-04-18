# Cheapfake Detection Using Image-Caption Triplets

## Instalation
```
conda create -n dash python=3.10
conda activate dash
pip install -r requirements.txt
```

## Models

## Log
- [x] **Baseline COSMOS implementation** (Week 1)  
  Code for training and validation implemented. Training is still in progress.

- [x] **COSMOS testing** (Week 1)
  Code for testing completed.

- [x] **OOCBasic Model implementation: ViT + SBERT + Cross-Attention + Triplet Loss** (Week 2)  
    Code for training and validation implemented.

- [ ] **OOCBasic testing** (Week 2)  
    In progress.

## Upcoming Tasks

- [ ] **Grounding to provide better context for OOC**  
  - [ ] Via loss - predicting top 10 bboxes  
  - [ ] Explicit bbox-based like COSMOS (using bbox for feature extraction)  

- [ ] **Utilizing Attention / Grounding Similarity**  
  - [ ] Implement attention-based similarity

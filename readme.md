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
  Code for training and validation implemented. 

- [x] **COSMOS testing** (Week 1)
  Code for testing completed. (Test Metrics: Accuracy = 0.768, F1 Score = 0.782, Average Precision = 0.697)

- [x] **OOCBasic Model implementation: ViT + SBERT + Cross-Attention + Triplet Loss** (Week 2)  
    Code for training and validation implemented.

- [x] **OOCBasic testing** (Week 2)  
    (Test Metrics: Accuracy = 0.745, F1 Score = 0.769, Average Precision = 0.671)

## Upcoming Tasks

- [ ] **Grounding to provide better context for OOC**  
  - [ ] Via loss - predicting top 10 bboxes  
  - [ ] Explicit bbox-based like COSMOS (using bbox for feature extraction)  

- [ ] **Utilizing Attention / Grounding Similarity**  
  - [ ] Implement attention-based similarity

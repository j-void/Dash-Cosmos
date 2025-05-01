# Cheapfake Detection Using Image-Caption Triplets

## Instalation
```
conda create -n dash python=3.10
conda activate dash
pip install -r requirements.txt
```

## Dataset
You will need to download two separate components:

- **Training Images:**  
  [MEGA](https://mega.nz/file/EDlCkZqS#bOKW4ezrkeuTqL3TJDznYZwYB4FDJyN1tEpqC3nQTRM)

- **Validation, Test Images & Annotations:**  
  [Google Drive](https://drive.google.com/drive/folders/1wDGX9PE0y8bPeepa-tIS_5MHnfncADEc?usp=sharing)

After downloading and extracting the datasets, arrange the files in the following format:

```
data/
│── cosmos_anns_acm
│    └── cosmos_anns_acm
│       └── acm_anns
│           ├── public_test_acm.json
│           ├── train_data.json
│           └── val_data.json
├── test
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── .....
├── train
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── .....
├── val
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── .....
```
*Note: Make sure the JSON annotation files are placed under `data/cosmos_anns_acm/cosmos_anns_acm/acm_anns`, and the image files are organized into `train`, `val`, and `test` folders accordingly.*

## Training
- **COSMOS Training:** `python train.py`
- **OOCBasic Training:** `python train_ooc.py`
- **OOCBasic_Synth:** `python train_ooc_synth.py`
  
## Checkpoints and Testing
Saved corresponding best loss and accuracy models for validation in the following structure:
```
checkpoints/
├── cosmos
│   └── save
│       ├── maskrcnn_use_acc.torch
│       └── maskrcnn_use_loss.torch
├── ooc_basic
│   └── save
│       ├── ooc_acc.torch
│       └── ooc_loss.torch
├── ooc_basic_synth
│   └── save
│       ├── ooc_acc.torch
│       └── ooc_loss.torch
```

**Testing:** Open and run `test.ipynb` in Jupyter Notebook.


## Log
- [x] **Baseline COSMOS implementation** (Week 1)  
  Code for training and validation implemented. 

- [x] **COSMOS testing** (Week 1)
  Code for testing completed. (Test Metrics: Accuracy = 0.76, F1 Score = 0.78, Average Precision = 0.56)

- [x] **OOCBasic Model implementation: ViT + SBERT + Cross-Attention + Triplet Loss** (Week 2)  
    Code for training and validation implemented.

- [x] **OOCBasic testing** (Week 2)  
    (Test Metrics: Accuracy = 0.74, F1 Score = 0.76, Average Precision = 0.57)

- [x] **Synthetic different generator** (Week 3)
  Code implemented. (Traning very slow, can be optimized pushing spacy to gpu and proper nlpaug batching)

## Future Work

- [ ] **Synthetic Negatives**
  - [ ] Better synthetic negatives generation, try combination of multiple augmentations
  - [ ] Discard too similar negatives based on sentence similarity score
  - [ ] Try burnin to slowly increase the probability on synthetic caption selection

- [ ] **Grounding to provide better context for OOC**  
  - [ ] Attention masking based on bounding boxes
  - [ ] Via loss - predicting top 10 bboxes  
  - [ ] Explicit bbox-based like COSMOS (using bbox for feature extraction)  


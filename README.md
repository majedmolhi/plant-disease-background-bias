# Plant Disease Detection with Background Bias Elimination

This project presents a deep learning pipeline for plant disease classification with a specific focus on reducing background bias in leaf images.  
Using EfficientNet as the backbone, the proposed approach explicitly separates leaf features from background cues through a two-stage training strategy and controlled data transformations.

The framework is designed to be fully reproducible and experimentally validated under three evaluation conditions: original images, leaf-only images, and background-only images, enabling a clear and quantitative analysis of background dependency.

**EfficientNet-based approach with systematic background bias mitigation and quantitative validation.**

---

## Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test Accuracy (Original)** | 98.97% | Standard performance |
| **Test Accuracy (Leaf-only)** | 96.62% | Feature learning quality |
| **Test Accuracy (Background-only)** | 23.58% | Background dependency |
| **Background Bias Coefficient (BBC)** | **0.762** | **76.2% genuine leaf features** |

---

##  Dataset

* **Source**: PlantVillage ([Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease))
* **Images**: 20,638
* **Classes**: 15 plant diseases
* **Split**: 70% train / 15% val / 15% test (stratified)

---

## Quick Start

### 1. Setup
```bash
git clone https://github.com/majedmolhi/plant-disease-background-bias.git
cd plant-disease-background-bias
pip install -r requirements.txt
```


### 2. Run Pipeline

**Option A: Jupyter Notebook**
```bash
jupyter notebook main.ipynb
# Run all cells sequentially
```

**Option B: Command Line**
```bash
python src/download_data.py      # Download dataset
python src/split_data.py         # Create train/val/test split
python -m src.stage1_train       # Baseline training (15 epochs)
python -m src.stage2_train       # Bias mitigation (10 epochs)
```

## Project Structure

```
plant-disease-background-bias/
├── src/
│   ├── download_data.py       # Dataset acquisition
│   ├── split_data.py          # Stratified splitting
│   ├── models.py              # CNN & EfficientNet architectures
│   ├── utils.py               # Leaf segmentation & preprocessing
│   ├── stage1_train.py        # Baseline training
│   └── stage2_train.py        # Two-phase bias mitigation
├── main.ipynb                 # Complete workflow
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .gitignore                 # Git exclusions
```

---

## Methodology

### Stage 1: Baseline Training
* **Model**: EfficientNet-B0 (ImageNet pretrained)
* **Training**: 15 epochs on original images
* **Optimizer**: Adam (lr=3e-4)
* **Result**: 98.5% validation accuracy

### Stage 2: Background Bias Mitigation

**Phase A (Warm-start, 2 epochs)**
* Fine-tune last 3 layers
* Train on original images
* Optimizer: AdamW (lr=3e-4, wd=1e-5)

**Phase B (Mixed refinement, 8 epochs)**
* Unfreeze last 40 layers
* **Mixed training batches**:
  - 33% original images
  - 33% leaf-only (background removed via HSV masking)
  - 33% random synthetic backgrounds
* Optimizer: AdamW (lr=1e-4, wd=1e-5)

### Evaluation: Three-Condition Protocol
1. **Original**: Standard test images → 98.97%
2. **Leaf-only**: Background removed → 96.62%
3. **Background-only**: Leaf removed → 23.58%

**Background Bias Coefficient (BBC)**:
```
BBC = (Acc_leaf - Acc_bg) / Acc_original
    = (96.62 - 23.58) / 98.97
    = 0.762
```

---

## Outputs

After training, outputs are saved to:

```
outputs/
├── stage1/
│   ├── models/
│   │   ├── baseline.h5           # Baseline CNN
│   │   └── effnet_finetune.h5    # EfficientNet Stage-1
│   └── history/
│       ├── baseline.json         # Training logs
│       └── effnet.json
└── stage2/
    ├── M_final.keras             # Final trained model
    ├── history_stage2.json       # Phase A+B training logs
    └── training_curve.png        # Visualization
```

---

## Technical Details

**Leaf Segmentation**: HSV color thresholding (green range: [25,40,40] to [95,255,255])

**Class Imbalance**: Computed sample weights using `sklearn.utils.class_weight`

**Regularization**: 
* Label smoothing (ε=0.05)
* Weight decay (1e-5)
* Dropout (0.5)

**Callbacks**: ReduceLROnPlateau, EarlyStopping (patience=4), ModelCheckpoint

---

## Citation

If you use this work, please cite:

```bibtex
@software{molhi_2026_18258119,
  author       = {Majed Molhi},
  title        = {plant-disease-background-bias: Initial Release},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.0.0},
  doi          = {10.5281/zenodo.18258119},
  url          = {https://doi.org/10.5281/zenodo.18258119}
}

```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## Links

* **Dataset**: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Paper**: [Link to paper when published]
* **DOI**: [Zenodo DOI](https://doi.org/10.5281/zenodo.18258119)

---

##  Contributing

Contributions welcome! Open an issue or submit a pull request.

---

## Contact

For questions: [majedmolhi@gmail.com] or open an issue on GitHub.

---

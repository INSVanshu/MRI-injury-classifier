# MRI Knee Injury Classifier

> Deep learning model for automated classification of knee injuries from MRI scans — detecting **normal tissue**, **sprains** (ligament tears), and **fractures** with clinically interpretable Grad-CAM heatmaps.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Problem Statement

Musculoskeletal injuries — particularly knee sprains and fractures — are among the most common presentations in emergency and orthopedic care. Accurate and timely diagnosis is critical, yet:

- **Sprains** involve ligament/soft tissue damage and are **invisible on X-rays** — MRI is the gold standard
- **Fractures** visible on MRI include both cortical breaks and bone marrow edema
- Misclassification leads to delayed treatment, unnecessary immobilization, or missed injuries

This project builds an end-to-end deep learning pipeline to automatically classify knee MRI volumes into three categories: **Normal**, **Sprain**, and **Fracture** — with explainability built in from day one.

---

##  Objectives

- [ ] Build a robust MRI preprocessing pipeline (DICOM → normalized tensor)
- [ ] Train a transfer-learned CNN (ResNet50) on the Stanford MRNet dataset
- [ ] Achieve clinically acceptable sensitivity and specificity per class
- [ ] Implement Grad-CAM heatmaps to highlight the injury region on MRI slices
- [ ] Deploy an interactive Gradio demo for real-time inference
- [ ] Track all experiments with Weights & Biases

---

##  Dataset

**Primary: [Stanford MRNet](https://stanfordmlgroup.github.io/projects/mrnet/)**

| Property | Details |
|---|---|
| Modality | Knee MRI (sagittal, coronal, axial planes) |
| Size | 1,370 exams (1,104 train / 120 valid) |
| Labels | Abnormal / ACL tear / Meniscal tear |
| Format | NumPy arrays (pre-extracted from DICOM) |
| Access | Free — requires registration |

> Raw MRI files are never committed to this repository. See `data/README.md` for download instructions.

---

## Project Architecture

```
mri-injury-classifier/
│
├── data/
│   ├── raw/                  # Downloaded MRNet files (gitignored)
│   ├── processed/            # Preprocessed tensors (gitignored)
│   └── README.md             # Download & setup instructions
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_grad_cam.ipynb
│
├── src/
│   ├── dataset.py            # PyTorch Dataset class
│   ├── model.py              # Model architecture
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Metrics & evaluation
│   ├── gradcam.py            # Grad-CAM implementation
│   └── utils.py              # Helper functions
│
├── app/
│   └── gradio_app.py         # Interactive demo
│
├── configs/
│   └── config.yaml           # Hyperparameters & paths
│
├── requirements.txt
├── PROJECT_PLAN.md
└── README.md
```

---

##  Model Architecture

```
Input: MRI volume (N slices × 3 × 224 × 224)
         ↓
Slice-level feature extraction
  └── ResNet50 (ImageNet pretrained, fine-tuned)
         ↓
Per-slice predictions aggregated (max pooling)
         ↓
Fully connected classifier
         ↓
Output: [Normal | Sprain | Fracture]
         ↓
Grad-CAM heatmap on most informative slice
```

---

##  Evaluation Metrics

Standard accuracy is insufficient for medical AI. This project uses:

| Metric | Why it matters |
|---|---|
| **Sensitivity (Recall)** | Minimizing missed injuries (false negatives) |
| **Specificity** | Minimizing unnecessary interventions (false positives) |
| **ROC-AUC** | Overall discriminative performance per class |
| **F1-Score** | Balance under class imbalance |
| **Confusion Matrix** | Per-class error breakdown |

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision |
| MRI Handling | pydicom, nibabel, SimpleITK |
| Explainability | pytorch-grad-cam |
| Data & Viz | NumPy, pandas, matplotlib, OpenCV |
| Experiment Tracking | Weights & Biases |
| Demo | Gradio |
| Environment | Python 3.10+, CUDA 11.8+ |

---

##  Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mri-injury-classifier.git
cd mri-injury-classifier

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Follow instructions in data/README.md

# 5. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```


## Disclaimer

This project is for **educational and research purposes only**. It is not intended for clinical use, medical diagnosis, or as a substitute for professional medical advice. Always consult a qualified radiologist or physician for medical decisions.

---

## License

This project is licensed under the MIT License 

---

## Acknowledgements

- [Stanford ML Group](https://stanfordmlgroup.github.io/) for the MRNet dataset
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) by Jacob Gildenblat

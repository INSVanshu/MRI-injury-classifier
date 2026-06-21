# Data Setup

## Dataset: Stanford MRNet

This project uses the **MRNet dataset** from the Stanford ML Group.

### Download Instructions

1. Go to: https://stanfordmlgroup.github.io/projects/mrnet/
2. Fill in the registration form (approval takes 24–48 hours)
3. Once approved, download the dataset and extract it here

### Expected folder structure after download

data/
├── raw/
│   ├── train/
│   │   ├── axial/       # .npy files, one per exam
│   │   ├── coronal/
│   │   └── sagittal/
│   ├── valid/
│   │   ├── axial/
│   │   ├── coronal/
│   │   └── sagittal/
│   └── labels/
│       ├── train-abnormal.csv
│       ├── train-acl.csv
│       ├── train-meniscus.csv
│       ├── valid-abnormal.csv
│       ├── valid-acl.csv
│       └── valid-meniscus.csv
└── processed/             # Auto-generated during preprocessing

## Label Mapping

| CSV column | Meaning in this project |
|---|---|
| abnormal = 0 | Normal knee |
| acl = 1 | Sprain (ACL ligament tear) |
| meniscus = 1 | Fracture proxy (bone/structural damage) |

> Note: MRNet doesn't have a "fracture" label directly.
> We use meniscal tears as a structural injury proxy.
> This mapping is documented and justified in notebooks/01_eda.ipynb

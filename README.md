# Pose-Based Badminton Stroke Recognition

A pose-based badminton stroke classification project using human joint sequences extracted from video and Transformer-based models.

## Project status

The project currently provides the data preprocessing and human pose extraction pipeline. Model training and evaluation will be added in later stages.

## Overview

The pipeline converts badminton video clips into skeleton sequences that can be used as input for stroke classification models.

```text
Video clips
    ↓
Dataset manifest
    ↓
Train / validation / test split
    ↓
Human pose extraction with RTMPose
    ↓
Skeleton arrays (.npy)
    ↓
Transformer-based stroke classification
```

## Pose representation

Each processed video is stored as a NumPy array with the following shape:

```text
(T, 2, 17, 2)
```

Where:

- `T`: number of video frames
- `2`: maximum number of players
- `17`: COCO human body joints
- `2`: joint coordinates `(x, y)`

The pose extraction pipeline uses:

- [RTMLib](https://github.com/Tau-J/rtmlib)
- RTMPose
- ONNX Runtime GPU
- OpenCV

## Current preprocessing result

| Item | Count |
|---|---:|
| Total video clips | 7,821 |
| Successfully processed | 7,818 |
| Unreadable videos | 3 |
| Valid joint files | 7,818 |
| Invalid joint files | 0 |

All generated joint files were validated for the expected shape `(T, 2, 17, 2)`.

## Repository structure

```text
.
├── notebook/
│   └── data-preprocessing.ipynb
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

Large files are excluded from Git:

```text
data/
processed/
*.zip
*.mp4
*.npy
.venv/
```

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU
- CUDA-compatible NVIDIA driver

The current environment uses a Tesla V100 GPU. cuDNN is pinned to version `9.7.1.26` to retain support for the Volta architecture.

## Installation

Clone the repository:

```bash
git clone git@github.com:shinbkb/pose-based-badminton-stroke-recognition.git
cd pose-based-badminton-stroke-recognition
```

Create and synchronize the environment:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Register the Jupyter kernel:

```bash
python -m ipykernel install \
  --user \
  --name badminton-stroke \
  --display-name "Python (Badminton Stroke)"
```

Start Jupyter:

```bash
uv run jupyter lab
```

## Data preparation

Place the video dataset under the local `data/` directory. The dataset is not included in this repository because of its size and distribution terms.

The videos are expected to be grouped by stroke class:

```text
data/
└── VideoBadminton_Dataset/
    ├── 00_Short Serve/
    │   ├── clip_001.mp4
    │   └── ...
    ├── 01_.../
    └── ...
```

Run the cells in:

```text
notebook/data-preprocessing.ipynb
```

The notebook performs:

1. Dataset discovery
2. Video validation
3. Class mapping
4. Manifest generation
5. Train, validation and test splitting
6. RTMPose initialization
7. Joint extraction
8. Extraction status tracking
9. Joint file validation

## Resume support

Joint extraction can continue safely after interruption. If a valid output file already exists, the notebook skips that video instead of processing it again.

Extraction progress is recorded in:

```text
processed/joint_extraction_status.csv
```

Generated skeleton data is stored under:

```text
processed/joints_npy_balanced/
```

## Planned work

- Player position extraction
- Shuttlecock trajectory extraction
- Sequence normalization
- Transformer model implementation
- Training and evaluation
- Inference on unseen videos

## Author

**Khac Binh**

GitHub: [@shinbkb](https://github.com/shinbkb)
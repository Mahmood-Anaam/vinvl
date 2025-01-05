# VinVL VisualBackbone

<a href="https://colab.research.google.com/github/Mahmood-Anaam/vinvl/blob/main/notebooks/vinvl_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Overview

VinVL Visual Backbone provides a simplified API for feature extraction, bounding boxes, and object detection, enabling you to achieve these tasks with minimal code. This implementation is based on [microsoft/scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark). Refer to their repository for additional details.

## Installation

#### Option 1: Install Directly via pip
```bash
pip install git+https://github.com/Mahmood-Anaam/vinvl.git --quiet
```

#### Option 2: Clone Repository and Install in Editable Mode
```bash
git clone https://github.com/Mahmood-Anaam/vinvl.git
cd vinvl
pip install -e .
```

#### Option 3: Use Conda Environment
```bash
git clone https://github.com/Mahmood-Anaam/vinvl.git
cd vinvl
conda env create -f environment.yml
conda activate vinvl
pip install -e .
```

## Features
- Simplified feature extraction with pretrained VinVL models.
- Support for multiple input types (file path, URL, PIL.Image, NumPy array, or tensor).
- Scalable batch processing and seamless PyTorch integration.
- Predefined configurations for fast setup and customization.
- High performance on GPUs and CPUs.

## Quick Start

### Code Example

```python
import torch
from PIL import Image
import requests
from vinvl.scene_graph_benchmark.wrappers import VinVLVisualBackbone

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = VinVLVisualBackbone(device=device, config_file=None, opts=None)

# Single Image Feature Extraction
img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(img_url, stream=True).raw)

image_features = feature_extractor(image)
# Output: List of dictionaries with keys:
"boxes", "classes", "scores", "img_feats", "spatial_features".

# Batch Image Feature Extraction
batch = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://farm1.staticflickr.com/26/53573290_1d167223e8_z.jpg"
]

batch_features = feature_extractor(batch)
for feature in batch_features:
    print("\n", feature['classes'])
```

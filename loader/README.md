# Loader Module

## Overview

The `loader` module is a core component of the GraPLUS (Graph-based Placement Using Semantics) framework, responsible for data loading, preprocessing, and augmentation. It provides specialized support for the Object Placement Assessment (OPA) dataset with scene graph integration, enabling the semantic-first approach to object placement that distinguishes GraPLUS from pixel-based methods.

The module contains two parallel dataset hierarchies:

1. The `OPABasicDataset` hierarchy (used by other models):
   - Handles traditional image-based processing without scene graphs
   - Used as a baseline for comparison with GraPLUS

2. The `SG_OPABasicDataset` hierarchy (used by GraPLUS):
   - Integrates scene graph data with image data
   - Processes both semantic relationships and spatial information
   - Enables GraPLUS's unique semantic-first approach to object placement

## Directory Structure

```
loader/
│
├── __init__.py        # Entry point with exported classes and functions
├── base.py            # Base dataset and sampler classes
├── datasets.py        # Specific dataset implementations with augmentation
├── sg_utils.py        # Scene graph processing utilities
└── utils.py           # General utilities for image and data manipulation
```

## Key Components

### Dataset Classes

#### Base Classes

- **OPABasicDataset**: Base dataset class for the OPA dataset
  - Handles loading of background, foreground, and composite images
  - Supports different data modes (train, trainpos, sample, eval, evaluni)
  - Used by models other than GraPLUS

- **SG_OPABasicDataset**: Specialized dataset class with scene graph support
  - Directly inherits from torch.utils.data.Dataset (not from OPABasicDataset)
  - Implements scene graph loading and processing functionality
  - Maintains separate indices for positive and negative samples
  - Specifically designed for GraPLUS

#### Implementation Classes

- **OPADst1** and **OPADst3**: Specialized implementations inheriting from OPABasicDataset
  - Used by models other than GraPLUS
  - Provide different image processing and transformation approaches

- **SG_OPADst**: Primary dataset implementation for GraPLUS
  - Inherits from SG_OPABasicDataset
  - Integrates scene graph data with image data
  - Implements comprehensive data augmentation techniques
  - Converts bounding boxes into transformation parameters
  - Core dataset class that enables GraPLUS's semantic-first approach

### Samplers

The module provides specialized samplers to address data imbalance in the OPA dataset:

- **BalancedSampler**: Ensures equal representation of positive and negative samples in each batch
  - Uses all data in each epoch
  - Balances positive and negative samples with potential reuse of positive samples

- **TwoToOneSampler**: Creates batches with a 2:1 ratio of positive to negative samples
  - Useful for training models to focus more on positive examples

- **PositiveSampler**: Yields only positive samples
  - Useful for specialized training stages or evaluation

### Scene Graph Utilities

The `sg_utils.py` file provides essential functionality for scene graph processing:

- **dict_extract**: Extracts scene graph components (bounding boxes, labels, relation pairs, relation labels)
- **adjust_bbox**: Adjusts bounding boxes for different image sizes
- **bbox_to_trans**: Converts bounding boxes to transformation parameters [scale, x, y]
- **get_size**: Calculates resized dimensions while maintaining aspect ratio

### Data Augmentation

The `SG_OPADst` class implements multiple data augmentation techniques:

- **Horizontal flipping**: Flips images with corresponding adjustments to bounding boxes
- **Color jittering**: Adjusts brightness, contrast, saturation, and hue
- **Grayscale conversion**: Converts images to grayscale while maintaining 3 channels
- **Gaussian blur**: Applies blur with random radius

## Usage

### Loading the OPA Dataset with Scene Graphs

```python
from loader import get_sg_loader, get_sg_dataset

# Create a dataloader for training
train_loader = get_sg_loader(
    name="SG_OPADst",
    batch_size=64,
    num_workers=8,
    image_size=32,
    mode_type="train",
    data_root="path/to/OPA_dataset",
    sg_root="path/to/OPA_SG",
    num_nodes=20,
    augment_flag=True,
    sampler_type="balance_sampler"
)

# Create a dataset for sampling/evaluation
sample_dataset = get_sg_dataset(
    name="SG_OPADst",
    image_size=32,
    mode_type="sample",
    data_root="path/to/OPA_dataset",
    sg_root="path/to/OPA_SG",
    num_nodes=20,
    augment_flag=False
)
```

### Custom Collation

The module provides a custom collation function (`custom_collate_fn`) for handling batches of scene graph data, which:

- Combines node attributes, edge indices, and edge attributes from multiple graphs
- Properly adjusts edge indices for batched graphs
- Handles different return formats based on mode type (eval/sample vs. train)

## Integration with GraPLUS

In the GraPLUS framework, this loader module provides the foundation for the semantic-first approach by:

1. Loading scene graphs that represent the semantic structure of background scenes
2. Processing node attributes (object categories) and edge attributes (relationships) from scene graphs
3. Converting spatial coordinates from bounding boxes into transformation parameters
4. Implementing data balancing strategies via specialized samplers
5. Employing data augmentation to improve model robustness

The `SG_OPADst` class is specifically designed for GraPLUS and works in tandem with the model architecture, particularly the Graph Transformer Network (GTN) and cross-modal attention mechanisms. It prepares the structured data needed for GraPLUS to understand both the semantic relationships and spatial configurations in scenes.

Unlike traditional pixel-based approaches that process background and foreground images directly, GraPLUS relies on the scene graph structure provided by this loader module to make contextually appropriate object placement decisions.

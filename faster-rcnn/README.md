# Faster R-CNN Model Pre-trained on Visual Genome

This directory contains the Faster R-CNN model that serves as a backbone for the SimOPA binary classifier used in GraPLUS to evaluate object placement plausibility.

## Overview

The Faster R-CNN model pre-trained on Visual Genome provides powerful object detection capabilities that enable:
* Extracting rich visual features from images
* Identifying objects and their characteristics
* Supporting the SimOPA binary classifier's evaluation capabilities

## Model Setup

1. Download the pre-trained model from Google Drive
2. Place the downloaded file in this directory as:`/faster-rcnn/models/faster_rcnn_res101_vg.pth`

## Usage in GraPLUS

This model is essential for the evaluation process in GraPLUS:
1. The SimOPA binary classifier (located in the `BINARY_CLASSIFIER` directory) uses this Faster R-CNN model to extract features from composite images
2. These features help determine if object placements are plausible or implausible
3. The accuracy metric reported in our paper is computed using this classifier

## Model Details

* **Architecture**: Faster R-CNN with ResNet-101 backbone
* **Pre-training**: Trained on Visual Genome dataset
* **Input**: Composite images from object placement models
* **Output**: Features that enable the classifier to assess placement plausibility

## Acknowledgements

This component is adapted and modified from Faster-RCNN-VG. We thank the original authors for making their implementation and pre-trained models publicly available.

## Related Components

* **SimOPA Binary Classifier**: Located in the `BINARY_CLASSIFIER` directory, this classifier uses the Faster R-CNN features to determine placement plausibility
* **Evaluation Scripts**: Located in the `script` directory, these scripts integrate the model for **accuracy** evaluation





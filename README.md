# GraPLUS: Graph-based Placement Using Semantics for Image Composition

[![arXiv](https://img.shields.io/badge/arXiv-2503.15761-b31b1b.svg)](https://arxiv.org/abs/2503.15761)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4.1](https://img.shields.io/badge/pytorch-2.4.1-orange.svg)](https://pytorch.org/get-started/previous-versions/)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![torchvision 0.19.1](https://img.shields.io/badge/torchvision-0.19.1-orange.svg)](https://pytorch.org)
[![OpenCV 4.11](https://img.shields.io/badge/opencv-4.11-5C3EE8.svg)](https://opencv.org)

**Mir Mohammad Khaleghi, Mehran Safayani, Abdolreza Mirzaei**  
Department of Electrical and Computer Engineering, Isfahan University of Technology, Isfahan, Iran

## 📑 Table of Contents
- [Abstract](#abstract)  
- [Key Innovations](#-key-innovations)
- [Model Architecture](#-model-architecture)
- [Visual Comparisons](#-visual-comparisons)
- [Pre-trained Models](#-pre-trained-models)
- [Environment Setup](#-environment-setup)
- [Hardware Requirements](#-hardware-requirements)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Inference](#-inference)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

  
## Abstract

We present GraPLUS, a novel framework for plausible object placement in images that leverages scene graphs and large language models. Our approach uniquely combines graph-structured scene representation with semantic understanding to determine contextually appropriate object positions. The framework employs GPT-2 to transform categorical node and edge labels into rich semantic embeddings that capture both definitional characteristics and typical spatial contexts, enabling nuanced understanding of object relationships and placement patterns.

GraPLUS achieves placement accuracy of **92.1%** and an FID score of 28.83 on the OPA dataset, outperforming state-of-the-art methods by 8.1% while maintaining competitive visual quality. In human evaluation studies involving 964 samples assessed by 19 participants, our method was preferred in **52.1%** of cases, significantly outperforming previous approaches.

## ✨ Key Innovations

- **Semantic-First Approach**: We determine optimal placement using only the foreground object's category without requiring the actual foreground image, significantly reducing computational complexity.

- **Transfer Learning**: We leverage pre-trained scene graph extraction models that incorporate cross-domain knowledge of common object relationships and spatial arrangements.

- **Edge-Aware Graph Neural Networks**: Our model processes scene semantics through structured relationships, preserving and enhancing semantic connections.

- **Cross-Modal Attention**: We align categorical embeddings with enhanced scene features through a dedicated attention mechanism.

- **Multi-Objective Training**: Our approach incorporates semantic consistency constraints alongside adversarial learning.

## 📊 Model Architecture
![GraPLUS](.images/framework.png)
GraPLUS consists of four principal components:

1. **Scene Graph Processing**: Transforms background images into structured graph representations using a pre-trained Neural-MOTIFS model.

2. **Semantic Enhancement**: Maps nodes and edges to rich embeddings using GPT-2 and augments them with spatial information for more comprehensive scene understanding.

3. **Graph Transformer Network (GTN)**: Processes object-object interactions through edge-aware attention with a configurable number of heads and layers.

4. **Cross-Attention Module (MHA)**: Computes attention weights between foreground object category and scene features to identify optimal placement locations.

This semantic-first design enables contextually appropriate object placements with improved coherence and accuracy compared to pixel-based methods.

## 🧠 GPT-2 Semantic Embeddings

A key innovation in GraPLUS is our use of rich semantic embeddings generated using GPT-2. Unlike traditional categorical embeddings, these representations capture nuanced semantic information about objects and their relationships:

- **Node Embeddings**: 768-dimensional vectors for 169 object categories that encode both definitional characteristics and typical spatial contexts
- **Edge Embeddings**: 768-dimensional vectors for 51 relationship types that define how objects interact spatially and semantically

These embeddings enable the model to understand that a "chair" is not just a categorical label, but a piece of furniture typically found at tables, desks, or in grouped arrangements - substantially improving placement decisions.

The `gpt2_embeddings/` directory contains:
- Pre-computed embeddings for efficient model training and inference
- A Jupyter notebook (`generate_embeddings.ipynb`) documenting the embedding generation process
- Detailed documentation of the embedding methodology

For more details, see [gpt2_embeddings/README.md](gpt2_embeddings/README.md).

## 🎯 Visual Comparisons

Below are some visual comparisons between our method and previous GAN-based approaches:

![Visual Comparisons](.images/visual_comparison.png)

Each column represents a different method, and each row represents a different test case. The red outline indicates the predicted placement boundaries.

## ⏬ Pre-trained Models 

We provide models for **TERSE** (CVPR 2019) [[arXiv]](https://arxiv.org/abs/1904.05475), **PlaceNet** (ECCV 2020) [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf), **GracoNet** (ECCV 2022) [[arXiv]](https://arxiv.org/abs/2207.11464), **CA-GAN** (ICME 2023, Oral) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219885), **CSANet** (BMVC 2024) [[paper]](https://papers.bmvc2024.org/0165.pdf), and our **GraPLUS**:

|     | Method   | User Study ↑ | Accuracy ↑ | FID ↓   | LPIPS ↑ | Model & Logs |
|-----|----------|------------|----------|--------|--------|---------------------|
| 0   | TERSE    | -          |   0.683  | 47.44  | 0.000  | [Google_Drive](https://drive.google.com/file/d/1L2R4J7nMoNhtg5a0dnGSVpLmxJCVXM3s/view?usp=sharing) |
| 1   | PlaceNet | -          |   0.684  | 37.63  | 0.160  | [Google_Drive](https://drive.google.com/file/d/1TDyLUt4Xc2anGVQZZYlvTmlayKCq4Mzx/view?usp=sharing) |
| 2   | GracoNet | 0.263 | 0.838 | 29.35 | 0.207 | [Google_Drive](https://drive.google.com/file/d/1LQEb3nX5oTd8RR1uEcA99SKC6u6fyiGS/view?usp=sharing) |
| 3   | CA-GAN   | -          |   0.734  | 25.54  | **0.267**  | [Google_Drive](https://drive.google.com/file/d/1fF9EG5TXX_mMMhF3Nz5qBTwbqQ8M-4wI/view?usp=sharing) |
| 4   | CSANet   | 0.216 | 0.803 | **22.42** | 0.264 | [Google_Drive](https://drive.google.com/file/d/1me7Ua67Pnwl9entWXgFRTP_ZyvAfWhJu/view?usp=sharing) |
| 5   | GraPLUS  | **0.521**  | **0.921**| 28.83  | 0.055  | [Google_Drive](https://drive.google.com/file/d/1TDyLUt4Xc2anGVQZZYlvTmlayKCq4Mzx/view?usp=sharing) |

See the [GracoNet repository](https://github.com/bcmi/GracoNet-Object-Placement) and [CSANet repository](https://github.com/CodeGoat24/CSANet) for the original model implementations and checkpoints.

## 🔧 Environment Setup
### Prerequisites
- Conda package manager
- CUDA capability (for GPU acceleration)

### Installation
1. Clone the repository (specifying the target directory name):
```bash
git clone https://github.com/mamadkhaleghi/GraPLUS.git GraPLUS
cd GraPLUS
```
2. Create and activate a conda environment using the provided `environment.yaml` file:
```bash
conda env create -f environment.yaml
conda activate graplus
```
3. To evaluate the `accuracy` metric, you will need to do the following first:
   
    **Binary Classifier**: We use the SimOPA binary classifier from the [OPA repository](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA) to measure placement plausibility
   - Download the pre-trained classifier from [BCMI Cloud](https://cloud.bcmi.sjtu.edu.cn/sharing/XPEgkSHdQ) or [Baidu Disk](https://pan.baidu.com/s/1skFRfLyczzXUpp-6tMHArA) (code: 0qty)
   - Place the file as `BINARY_CLASSIFIER/best-acc.pth` in your project directory
   
    **Faster R-CNN Model**: Required by the SimOPA classifier and originally provided by [Faster-RCNN-VG](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)
   - Download from [Google Drive](https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view)
   - Place the file as `faster-rcnn/models/faster_rcnn_res101_vg.pth`
   - Build faster-rcnn (require GCC 5 or later):
     ```bash
      cd faster-rcnn/lib
      python setup.py build develop
      cd ../..
     ```
## 💻 Hardware Requirements

Our model was developed and tested on the following hardware configuration:

### Training Setup
- **GPU**: NVIDIA GeForce GTX 1080 Ti (11GB VRAM)
- **CPU**: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (12 cores)
- **RAM**: 31GB

## 🌓 Data Preparation
GraPLUS requires two datasets:

**OPA Dataset**: Object Placement Assessment dataset with background/foreground images

**Scene Graph Data**: Pre-processed graph representations of OPA background images

### OPA Dataset

Download and extract the OPA dataset from [Google Drive](https://drive.google.com/file/d/133Wic_nSqfrIajDnnxwvGzjVti-7Y6PF/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1IzVLcXWLFgFR4GAbxZUPkw) (code: a982) and put it to `dataset/OPA/` directory after preprocessing:

```bash
python tool/preprocess.py --data_root /path/to/OPA_dataset
mkdir -p dataset/OPA
mv /path/to/OPA_dataset/* dataset/OPA/
```
### Scene Graph Data

Download and extract the scene graph data from [Google_Drive](https://drive.google.com/file/d/1fHjI_M6SwaHHEKcHZbHBn7dkSRTKjSXm/view?usp=sharing) and put it to `dataset/OPA_SG/` directory:
```bash
mkdir -p dataset/OPA_SG
mv -r /path/to/sg_opa_background_n20/ dataset/OPA_SG/
```
For more details about datasets, see [dataset/README.md](dataset/README.md).

## 💻 Training
### Basic Training
To train GraPLUS with default settings:
```bash
./train.sh graplus <experiment_name>  
```
### Advanced Training Options
You can customize various parameters:
```bash
./train.sh graplus <experiment_name>  \
  --batch_size 64 \
  --n_epochs 40 \
  --gtn_num_head 8 \
  --gtn_num_layer 5 \
  --gtn_hidden_dim 768 \
  --d_k 64 \
  --d_v 64 \
  --n_heads 8 \
  --data_augmentation True \
  --sampler_type "balance_sampler"
```

To train other models, simply replace `graplus` with the name of the model you want to train (model names correspond to subdirectories in the `models` directory, such as `cagan`, `csanet`, `graconet`, `placenet`, or `terse`):
```bash
./train.sh  <model_name> <experiment_name> 
```
Training files will be saved to `result/experiment_name/`.

### Training Monitoring
To monitor the training progress:
```bash
tensorboard --logdir result/experiment_name/tblog --port YOUR_SPECIFIED_PORT
```

## 🔥 Inference
To generate composite images from a trained model:
```bash
./infer.sh eval graplus <experiment_name>  <epoch> 
```
or
```bash
./infer.sh evaluni graplus <experiment_name>  <epoch> 
```
To infer other models, simply replace `graplus` with the name of the model you want to train (model names correspond to subdirectories in the `models` directory, such as `cagan`, `csanet`, `graconet`, `placenet`, or `terse`):
```bash
./infer.sh eval <model_name> <experiment_name> <epoch>
./infer.sh evaluni <model_name> <experiment_name>  <epoch>
```
The output files of inference will be saved in `result/model_name/eval/epoch` and `result/model_name/evaluni/epoch` respectively.

## Inference Using Pre-trained Models
1. Download the model from the provided link above
2. Extract it to the `result/` directory
3. Run:
```bash
./infer_checkpoint.sh eval <model_name>
```
or 
```bash
./infer_checkpoint.sh evaluni <model_name>
```
The output files will be saved in `result/model_name/eval/` and `result/model_name/evaluni/` respectively.
model names correspond to subdirectories in the `models` directory, such as `graplus`, `cagan`, `csanet`, `graconet`, `placenet`, or `terse`):


## 📊 Evaluation
Use the `evaluate.sh` script in the main directory to run evaluations on your trained models:

#### Usage
```bash
./evaluate.sh <experiment_name> <epoch> <metric> 
```

#### Examples
```bash
./evaluate.sh graplus 21 accuracy   # Evaluate Accuracy
./evaluate.sh graplus 21 fid        # Evaluate FID
./evaluate.sh graplus 21 lpips      # Evaluate LPIPS
./evaluate.sh graplus 21 sp         # Evaluate Spatial Precision
./evaluate.sh graplus 21 all        # Run all evaluations
```

This script will automatically evaluate:
- **Accuracy**: Uses a binary classifier to determine placement plausibility
- **FID**: Measures visual quality compared to ground truth
- **LPIPS**: Quantifies generation diversity
- **Spatial Precision**: Computes Spatial Precision metrics

Results will be saved to `result/experiment_name/eval_metrics_experiment_name.csv`.

## 📈 Results

### Quantitative Results
Our model outperforms previous GAN-based methods across multiple metrics:

| Method   | User Study ↑| Accuracy ↑| Mean IoU ↑| Center Dist. ↓| Scale Ratio ↑|
|----------|------------|----------|----------|--------------|-------------|
| TERSE    | -          | 0.683    | 0.171    | 172.04       | 9.1%        |
| PlaceNet | -          | 0.684    | 0.194    | 144.77       | 12.0%       |
| GracoNet | 0.263      | 0.838    | 0.192    | 166.95       | 14.7%       |
| CA-GAN   | -          | 0.734    | 0.165    | 190.37       | 12.5%       |
| CSANet   | 0.216      | 0.803    | 0.162    | 193.34       | 13.6%       |
| GraPLUS  | **0.521**  | **0.921**| **0.203**| **141.77**   | **16.5%**   |

### Ablation Studies
Our experiments validate key design choices:

| Component                         | Accuracy ↑| FID    ↓| LPIPS  ↑|
|-----------------------------------|----------|--------|--------|
| Full Model                        | **0.921**    | 28.83  | 0.055  |
| No Spatial Features on GTN output | 0.899    | **25.35**  | **0.070**  |
| No Position Encoding in MHA       | 0.812    | 31.70  | 0.054  |
| No Residual Connection in MHA     | 0.828    | 30.80  | 0.056  |
| No Balanced Sampling              | 0.873    | 29.14  | 0.053  |
| No Data Augmentation              | 0.842    | 32.60  | 0.039  |

## 🖊️ Citation
If you find **GraPLUS** useful, please cite our paper:

```bibtex
@article{khaleghi2025graplus,
  title={GraPLUS: Graph-based Placement Using Semantics for Image Composition},
  author={Khaleghi, Mir Mohammad and Safayani, Mehran and Mirzaei, Abdolreza},
  journal={arXiv preprint arXiv:2503.15761},
  year={2025}
}
```

## 🙏 Acknowledgements
Our codebase builds upon several excellent works:
- [GracoNet](https://github.com/bcmi/GracoNet-Object-Placement)
- [CSANet](https://github.com/CodeGoat24/CSANet)

Additional components are borrowed and modified from:
- [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [Faster-RCNN-VG](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)
- [OPA Dataset](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA)
- [FID-Pytorch](https://github.com/mseitzer/pytorch-fid)
- [Perceptual Similarity](https://github.com/richzhang/PerceptualSimilarity)

## 📧 Contact
If you have any technical questions or suggestions, please open a new issue or feel free to contact:

- Mir Mohammad Khaleghi (m.khaleghi@ec.iut.ac.ir, mamadkhaleghi1994@gmail.com)
- Mehran Safayani (safayani@iut.ac.ir) - Corresponding author

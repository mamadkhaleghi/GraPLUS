# GraPLUS: Graph-based Placement Using Semantics for Image Composition

[![arXiv](https://img.shields.io/badge/arXiv-2503.15761-b31b1b.svg)](https://arxiv.org/abs/2503.15761)

**Mir Mohammad Khaleghi, Mehran Safayani, Abdolreza Mirzaei**  
Department of Electrical and Computer Engineering, Isfahan University of Technology, Isfahan, Iran

![GraPLUS](images/framework.png)

## Abstract

We present GraPLUS, a novel framework for plausible object placement in images that leverages scene graphs and large language models. Our approach uniquely combines graph-structured scene representation with semantic understanding to determine contextually appropriate object positions. The framework employs GPT-2 to transform categorical node and edge labels into rich semantic embeddings that capture both definitional characteristics and typical spatial contexts, enabling nuanced understanding of object relationships and placement patterns.

GraPLUS achieves placement accuracy of 92.1% and an FID score of 28.83 on the OPA dataset, outperforming state-of-the-art methods by 8.1% while maintaining competitive visual quality. In human evaluation studies involving 964 samples assessed by 19 participants, our method was preferred in 52.1% of cases, significantly outperforming previous approaches.

## ⏬ Download Pre-trained Models 

We provide models for **TERSE** (CVPR 2019) [[arXiv]](https://arxiv.org/abs/1904.05475), **PlaceNet** (ECCV 2020) [[arXiv]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf), **GracoNet** (ECCV 2022) [[arXiv]](https://arxiv.org/abs/2207.11464), **CA-GAN** (ICME 2023, Oral) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219885), **CSANet** (BMVC 2024) [[paper]](https://papers.bmvc2024.org/0165.pdf), and our **GraPLUS**:

|     | method   | Accuracy | FID    | LPIPS  | url of model & logs |
|-----|----------|----------|--------|--------|---------------------|
| 0   | TERSE    | 0.683    | 47.44  | 0.000  | [Google Drive](https://drive.google.com/file/d/1xxxxxxxxxxxxx/view?usp=sharing) |
| 1   | PlaceNet | 0.684    | 37.63  | 0.160  | [Google Drive](https://drive.google.com/file/d/1xxxxxxxxxxxxx/view?usp=sharing) |
| 2   | GracoNet | 0.838    | 29.35  | 0.207  | [Google Drive](https://drive.google.com/file/d/1xxxxxxxxxxxxx/view?usp=sharing) |
| 3   | CA-GAN   | 0.734    | 25.54  | 0.267  | [Google Drive](https://drive.google.com/file/d/1xxxxxxxxxxxxx/view?usp=sharing) |
| 4   | CSANet   | 0.803    | 22.42  | 0.264  | [Google Drive](https://drive.google.com/file/d/1xxxxxxxxxxxxx/view?usp=sharing) |
| 5   | GraPLUS  | 0.921    | 28.83  | 0.055  | [Google Drive](https://drive.google.com/file/d/1xxxxxxxxxxxxx/view?usp=sharing) |

## 🔧 Environment Setup

Install Python 3.6 and PyTorch 1.9.1 (require CUDA >= 10.2):
```bash
conda create -n graplus python=3.6
conda activate graplus
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch

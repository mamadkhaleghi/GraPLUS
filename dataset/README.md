# GraPLUS Dataset Directory

This directory contains all the data required to train and evaluate the GraPLUS model. It should contain two main subdirectories:

1. `OPA/` - The Object Placement Assessment dataset
2. `OPA_SG/` - Scene graph representations of the OPA background images

Both datasets need to be downloaded separately as explained below.

## 🖼️ OPA Dataset
The Object Placement Assessment (OPA) dataset is a benchmark for evaluating object placement methods. It contains background images, foreground objects with masks, and composite images with annotations.
![opa](../.images/opa.png)

1. Download and extract the OPA dataset from [Google Drive](https://drive.google.com/file/d/133Wic_nSqfrIajDnnxwvGzjVti-7Y6PF/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1IzVLcXWLFgFR4GAbxZUPkw) (code: a982)

2. Expected directory structure:
```
OPA_dataset/
  background/       # background images
  foreground/       # foreground images with masks
  composite/        # composite images with masks
  train_set.csv     # train annotation
  test_set.csv      # test annotation
```

3. Preprocess the data and move it to ` dataset/OPA/` directory:
```bash
python tool/preprocess.py --data_root /path/to/OPA_dataset
mkdir -p dataset/OPA
mv /path/to/OPA_dataset/* dataset/OPA/
```

4. After preprocessing, you'll have:
```
GraPLUS-main/
  dataset/
    OPA/
      background/
      foreground/
      composite/
      com_pic_testpos299/      # test set positive composite images (resized to 299)
      train_data.csv           # transformed train annotation
      train_data_pos.csv       # train annotation for positive samples
      test_data.csv            # transformed test annotation
      test_data_pos.csv        # test annotation for positive samples
      test_data_pos_unique.csv # test annotation for positive samples with different fg/bg pairs
```
### 📜 Citation
If you use this dataset in your research, please cite the original paper:
```bibtex
@article{liu2022opa,
  title={OPA: Object Placement Assessment Dataset},
  author={Liu, Liu and Liu, Zhenchen and Zhang, Bo and Li, Jiangtong and Niu, Li and Liu, Qingyang and Zhang, Liqing},
  journal={arXiv preprint arXiv:2107.01889},
  year={2022}
}
```

## 🕸️ Scene Graph Dataset
The scene graph dataset (`OPA_SG/`) contains structured graph representations of all background images in the OPA dataset, capturing object relationships and their interactions.

1. Download and extract our generated and pre-processed scene graph data:
   - Scene Graphs with 10 nodes (3.10 MB)  : [Google_Drive](https://drive.google.com/file/d/1dJRUm2EIj6j4zdxODyzxNT0-zzJ12D6S/view?usp=sharing)
   - **Scene Graphs with 20 nodes (10.3 MB)** : [**Google_Drive**](https://drive.google.com/file/d/1fHjI_M6SwaHHEKcHZbHBn7dkSRTKjSXm/view?usp=sharing) (**Default**)
   - Scene Graphs with 30 nodes (21.7 MB)  : [Google_Drive](https://drive.google.com/file/d/1vv9k-3Ki_li5yH-azJKKRGwEs1W6TeMX/view?usp=sharing)

2. Move the extracted scene graph data to ` dataset/OPA_SG/` directory:
```bash
mkdir -p dataset/OPA_SG
mv -r /path/to/sg_opa_background_n20/ dataset/OPA_SG/

# mv -r /path/to/sg_opa_background_n10/ dataset/OPA_SG/
# mv -r /path/to/sg_opa_background_n30/ dataset/OPA_SG/
```

3. Each scene graph directory structure should be:
```
GraPLUS-main/
  dataset/
    OPA_SG/
      sg_opa_background_nX/      # Scene graphs with X nodes for background images
        <category>/               # Categories folders
          sg_<image_id>.json      # Scene graph for each background image
```

4. Each JSON file contains:

- **Nodes**: Object instances detected in the background image
- **Edges**: Relationships between objects
- **Bounding boxes**: Spatial coordinates for each detected object
- **Node and edge labels**: Category information for objects and relationships

These scene graphs are generated using a pre-trained **Neural-MOTIFS** model operating in **Total Direct Effect (TDE)** mode.

### 📊 Dataset Statistics

- **OPA Dataset**: 62,074 training samples (21,376 positive, 40,698 negative) and 11,396 test samples (3,588 positive, 7,808 negative)
- **Scene Graphs**: Each graph contains up to 20 object nodes with their relationships
- **Categories**: 47 object categories from COCO dataset
- **Scene Graph Vocabulary**: 151 object types and 51 relationship types


### Scene Graph Generation Details
![sgg](../.images/sgg.png)

The scene graphs were generated using the **Neural-MOTIFS** model (via the [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) implementation) operating in **Total Direct Effect (TDE)** mode. Each background image was processed to:

1. Detect up to 20 objects per image 
2. Identify relationships between these objects
3. Export structured graph data in JSON format

We used the following configuration:
- Predictor: CausalAnalysisPredictor
- Effect Type: TDE (Total Direct Effect)
- Fusion Type: sum
- Context Layer: motifs
- No ground truth boxes or labels (fully automatic detection)

Each JSON file contains:
- Nodes: Object instances detected in the background image
- Edges: Relationships between objects
- Bounding boxes: Spatial coordinates for each detected object
- Node and edge labels: Category information for objects and relationships

This approach mitigates dataset biases by effectively separating causal relationships from contextual correlations, resulting in more meaningful scene representations.

### 📜 Citation
If you use this scene graph dataset in your research, please cite ours and the original papers:
```bibtex
@article{khaleghi2025graplus,
  title={GraPLUS: Graph-based Placement Using Semantics for Image Composition},
  author={Khaleghi, Mir Mohammad and Safayani, Mehran and Mirzaei, Abdolreza},
  journal={arXiv preprint arXiv:2503.15761},
  year={2025}
}

@article{liu2022opa,
  title={OPA: Object Placement Assessment Dataset},
  author={Liu, Liu and Liu, Zhenchen and Zhang, Bo and Li, Jiangtong and Niu, Li and Liu, Qingyang and Zhang, Liqing},
  journal={arXiv preprint arXiv:2107.01889},
  year={2022}
}

@misc{tang2020unbiasedscenegraphgeneration,
      title={Unbiased Scene Graph Generation from Biased Training}, 
      author={Kaihua Tang and Yulei Niu and Jianqiang Huang and Jiaxin Shi and Hanwang Zhang},
      year={2020},
      eprint={2002.11949},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2002.11949}, 
}
@misc{zellers2018neuralmotifsscenegraph,
      title={Neural Motifs: Scene Graph Parsing with Global Context}, 
      author={Rowan Zellers and Mark Yatskar and Sam Thomson and Yejin Choi},
      year={2018},
      eprint={1711.06640},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1711.06640}, 
}
```

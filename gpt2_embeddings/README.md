# GPT-2 Semantic Embeddings for GraPLUS

This directory contains pre-computed embeddings generated using the GPT-2 language model that encode semantic information about object categories and spatial relationships. These embeddings are a core component of the GraPLUS framework, enabling the model to understand both object definitions and their typical spatial contexts.

## Contents

- `node_embeddings.npy`: Embeddings for object categories (768-dimensional vectors)
- `edge_embeddings.npy`: Embeddings for spatial relationships (768-dimensional vectors)

## Overview

The embeddings in this directory provide rich semantic representations of objects and their relationships by leveraging the pre-trained knowledge of GPT-2. Unlike traditional categorical embeddings that simply assign an arbitrary vector to each class, these embeddings capture nuanced semantic information by encoding:

- Object definitions (what the object is)
- Typical placement contexts (where the object is usually found)
- Spatial relationships (how objects relate to each other)

This semantic encoding enables GraPLUS to make more informed and contextually appropriate placement decisions.

## Node Embeddings

`node_embeddings.npy` contains embeddings for 169 object categories (151 from scene graph generation plus 18 from OPA dataset). Each object has multiple embedding types:

1. `category_embedding`: Basic encoding of the object name
2. `description_embedding`: Encoding of the object's definition
3. `cat_desc_embedding`: Combined category and description
4. `placement_embedding`: Encoding of typical placement contexts
5. `cat_place_embedding`: Combined category and placement information
6. `cat_desc_place_embedding`: Full integration of category, description, and placement

The model primarily uses `cat_desc_place_embedding` as configured in the framework, providing the most comprehensive semantic representation.

### Example Categories

Some examples of object categories include:
- Common objects: chair, table, car, bottle
- Animals: dog, cat, elephant, horse
- People: person, man, woman, child
- Structural elements: wall, floor, building
- Natural elements: tree, plant, mountain

## Edge Embeddings

`edge_embeddings.npy` contains embeddings for 51 relationship types that define how objects interact spatially or semantically. Each relationship is encoded as a 768-dimensional vector that captures the nature of the relationship.

### Example Relationships

- Spatial relationships: above, below, behind, in front of
- Functional relationships: using, holding, carrying
- Compositional relationships: part of, made of
- Associative relationships: with, belonging to

## How These Embeddings Were Generated

These embeddings were created using the following process:

1. For each object category, detailed descriptions and placement contexts were defined
2. For each relationship type, precise definitions were created
3. The GPT-2 model was used to encode these textual descriptions into 768-dimensional vectors
4. The resulting embeddings were saved in NumPy format for efficient loading

Each object category was processed to create multiple semantic perspectives, and the relationship embeddings were generated to capture the precise nature of spatial and semantic interactions.

## Usage in GraPLUS

The GraPLUS model loads these embeddings via the `GraphEmbeddings` class in `SG_Networks.py`, which:

1. Maps object categories and relationship types to their corresponding embeddings
2. Makes these embeddings available to the Graph Transformer Network (GTN)
3. Enables the model to understand semantic relationships for placement decisions

The embedding mode can be configured using the `--gpt2_node_mode` parameter during training and inference.

## Citation

If you use these embeddings in your research, please cite our paper:

```
@article{khaleghi2025graplus,
  title={GraPLUS: Graph-based Placement Using Semantics for Image Composition},
  author={Khaleghi, Mir Mohammad and Safayani, Mehran and Mirzaei, Abdolreza},
  journal={arXiv preprint arXiv:2503.15761},
  year={2025}
}
```

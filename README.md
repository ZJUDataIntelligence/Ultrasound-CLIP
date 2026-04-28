# Ultrasound-CLIP

## News

- Our paper **"Ultrasound-CLIP"** has been accepted to **CVPR 2026**.

## Resources

- **Dataset:** [US-365K on Hugging Face](https://huggingface.co/datasets/JJY-0823/US-365K)
- **Paper (arXiv):** [http://arxiv.org/abs/2604.01749](http://arxiv.org/abs/2604.01749)
- **pre-trained weights:**[here](https://1drv.ms/f/c/48250be4328ce7ab/IgCnIA0--KugSoW0japEO8HaAW6koYf9YVNnB9vQiqeq_O8?e=c7xe4J)

This repository contains a set of modules designed to enhance CLIP text representations using **heterogeneous tag-graph encoding** and **semantic similarity supervision**, improving fine-grained cross-modal alignment between images and text.

It is suitable for image-text retrieval, matching, or contrastive learning scenarios with structured labels (e.g., multi-task medical annotations).

## File Overview

The core files released in this module are:

- `open_clip_train/graph_model/enhanced_clip_model.py`  
  Implements `EnhancedCLIP` on top of a base CLIP model. It fuses graph embeddings into text features via `CrossAttentionFusion`, with numerical safety guards (NaN/Inf checks, gated residuals, and fallback behavior).

- `open_clip_train/graph_model/graph_builder.py`  
  Builds heterogeneous graphs from per-sample multi-task labels (`diagnosis` and `descriptor` node types with bidirectional relations), supporting both per-sample graph creation and batched graph assembly.

- `open_clip_train/graph_model/graph_encoder.py`  
  Defines `GraphEncoder`, a multi-layer heterogeneous graph encoder based on DGL `HeteroGraphConv(GraphConv)`, producing graph representations aligned with text feature dimensions.

- `open_clip_train/semanticLoss/semantic_loss.py`  
  Defines `SemanticLoss` (`L_SM`): computes a predicted similarity matrix from image/text features and matches it to a target semantic similarity matrix using a joint MSE + KL objective.

- `open_clip_train/semanticLoss/similarity_processor.py`  
  Defines `SimilarityMatrixProcessor`: loads precomputed task-level tag similarity matrices and computes batch-wise target semantic similarity matrices.

- `open_clip_train/graph_model/tag_vocab.py`  
  Maintains tag sets for 9 tasks and builds `diagnosis`/`descriptor` vocabularies with UNK indices for graph construction and graph encoding.

## Module Pipeline

1. `tag_vocab.py` defines label spaces and vocabularies.
2. `graph_builder.py` converts per-sample labels into heterogeneous graphs.
3. `graph_encoder.py` encodes heterogeneous graphs into graph features.
4. `enhanced_clip_model.py` fuses graph features into text representations.
5. `similarity_processor.py` generates batch-level target semantic similarity matrices.
6. `semantic_loss.py` enforces consistency between predicted and target similarities.

## Requirements

Python 3.9+ is recommended. Core dependencies:

- `torch`
- `dgl`
- `numpy`

Installation example:

```bash
pip install torch dgl numpy
```

## Quick Start Example

```python
from open_clip_train.graph_model.graph_builder import build_hetero_graph_from_data
from open_clip_train.graph_model.graph_encoder import GraphEncoder
from open_clip_train.graph_model.enhanced_clip_model import EnhancedCLIP
from open_clip_train.semanticLoss.semantic_loss import SemanticLoss

# 1) Build heterogeneous graphs
# full_data: list of records containing media_name and task labels
graphs = build_hetero_graph_from_data(full_data, image_keys=batch_image_keys)

# 2) Initialize graph encoder
graph_encoder = GraphEncoder(out_dim=text_feature_dim, hidden=128, n_layers=2)

# 3) Build enhanced CLIP
model = EnhancedCLIP(base_clip_model, graph_encoder, freeze_graph_encoder=False)
prediction = model(images, texts, graphs=graphs)

# 4) Compute semantic matching loss
# similarity_matrix is produced by SimilarityMatrixProcessor
semantic_loss_fn = SemanticLoss(args, similarity_weight=1.0, temperature=0.07)
loss_sm = semantic_loss_fn(prediction, similarity_matrix)
```

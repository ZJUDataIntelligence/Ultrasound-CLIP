# Ultrasound-CLIP

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

## Data Format Assumptions

- Each record in `full_data` should include at least:
  - `media_name`
  - `Diagnosis`
  - `Body_system_level`, `Organ_level`, `Shape`, `Margins`, `Echogenicity`,
    `InternalCharacteristics`, `PosteriorAcoustics`, `Vascularity`
- Each task field can be either a string or a list of strings (internally normalized to list format).

## Notes

- The code includes multiple robustness safeguards (zero-edge graph fallback, empty-graph placeholders, NaN/Inf cleanup, safe normalization thresholds) to improve training stability.
- If you plan to release only these 6 files, it is recommended to also provide a minimal runnable training script so others can reproduce the full pipeline.

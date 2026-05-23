# ConVis-GNN (Anonymous Submission Artifact) #

Anonymous research artifact for the submitted ConVis-GNN paper.

## Introduction ##

ConVis-GNN is built upon MariusGNN ([EuroSys '23](https://dl.acm.org/doi/abs/10.1145/3552326.3567501)).

Original MariusGNN repository: https://github.com/marius-team/marius

## Installation ##

To Install ConVis-GNN

### Dependencies ###

- CUDA >= 10.1 (recommended: 11.8)
- cuDNN >= 7
- PyTorch >= 1.8 (recommended: 2.0.1)
- Python >= 3.7 (recommended: 3.10.12 or 3.10.14)
- GCC >= 7 on Linux, or Clang >= 11.0 on macOS
- CMake >= 3.12
- Make >= 3.8

### Setup

From the root directory of this repository:
```bash
cd ConVis
pip3 install . --no-build-isolation

### Execution ###

The following example runs OGBN-Arxiv.

1. ** Preprocess **
   - marius_preprocess --dataset ogbn_arxiv --output_dir datasets/ogbn_arxiv --num_partitions 32
2. ** Training **
   - (Default) marius_train examples/configuration/ogbn_arxiv.yaml
   - (ConVis) marius_train examples/configuration/ogbn_arxiv_convis.yaml

```markdown
## Notes
This repository is anonymized for double-blind review and contains the core implementation path used in the submitted paper.

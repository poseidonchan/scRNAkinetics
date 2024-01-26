# scKinectics: inferring single-cell RNA kinetics
![scKinetics](https://img.shields.io/badge/scKinetics-v0.2.4-blue)![PyPI - Downloads](https://img.shields.io/pypi/dm/scKinetics)![GitHub](https://img.shields.io/github/license/poseidonchan/scKinetics)
![figure1](https://github.com/poseidonchan/scKinectics/blob/main/Figures/figure1.png)

### Installation

This package requires the jax as framework, please make sure your operation system is compatible with jax. Usually, all Linux systems are compatible with directly using PyPI installation. For Mac users, please make try to use conda to install jax first.

```bash
conda create -n kinetics python=3.9
conda activate kinetics
pip install scKinetics
```

### Usage

There are two modes for inferring kinetics, the "**coarse-grained**" mode offers the uniform kinetics estimation in a group of cells, the "**high-resolution**" mode offeres kinetics estimation for each cell.

Data requirements:

AnnData object, proper pseudo-time estimation in the adata.obs, KNN smoothed unspliced and spliced expression matrices (['Ms'] and ['Mu'] key in the adata.layers), KNN-graph stored in adata.obsp['connectivities']. group information in adata.obs if necessary. 

scKinetics output:

AnnData object, updated cell- and gene-wise kinetics parameters (['alpha'], ['beta'], ['gamma']) and reconstruced unspliced and spliced velocity (['unspliced_velocity'], and ['spliced_velocity']) in adata.layers

```python
import scvelo as scv
from sckinetics import kinetics_inference

# Read data
adata = scv.read('your_file_path.h5ad')
# Preprocessing
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, retain_genes=None) #select genes based on your flavor
scv.pp.moments(adata, n_neighbors=30)
# Optional: Run coarse-grained mode scKinetics
# adata = kinetics_inference(adata, mode='coarse-grained', pt_key='pseudotime_in_your_data', group_key='cell_groups_in_your_data', num_iter=2000, n_jobs=-1, optimizer='jax')
# Run high-resolution mode scKinetics
adata = kinetics_inference(adata, mode='high-resolution', pt_key='pseudotime_in_your_data', num_iter=2000, n_jobs=-1, optimizer='jax')
```

### Experiments
Reproducible experiments can be found in the Experiments folder. 

Archived data is available at Google Drive: https://drive.google.com/file/d/1NbVdMjWjvDbhJiaTH1_MDw73OqD_B8wK/view?usp=sharing.
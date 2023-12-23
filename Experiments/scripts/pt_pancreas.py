import scanpy as sc
import numpy as np
import scvelo as scv
import cellrank as cr
from sckinetics import kinetics_inference

adata = cr.datasets.pancreas()
n_neighbors = 2*int(np.sqrt(adata.shape[0]))

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, retain_genes=['Hhex','Sst', 'Cd24a'])
scv.pp.moments(adata, n_pcs=30, n_neighbors=n_neighbors)

adata = kinetics_inference(adata, mode='coarse-grained', group_key='clusters', pt_key='palantir_pseudotime', num_iter=500, n_jobs=-1)
adata.write('pancreas_lores.h5ad')

adata = kinetics_inference(adata, mode='high-resolution', pt_key='palantir_pseudotime', num_iter=500, n_jobs=-1)
adata.write('pancreas_hires.h5ad')
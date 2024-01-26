import scanpy as sc
import numpy as np
import scvelo as scv
import cellrank as cr
from cellrank.kernels import CytoTRACEKernel
from sckinetics import kinetics_inference


adata = scv.datasets.gastrulation_erythroid()
scv.pp.filter_and_normalize(adata, n_top_genes=None)
scv.pp.moments(adata,  n_neighbors=30)
ctk = CytoTRACEKernel(adata).compute_cytotrace()
pt = adata.obs['ct_pseudotime']
adata = scv.datasets.gastrulation_erythroid()
murk_genes = adata.var.index[adata.var['MURK_gene'] == True]
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, retain_genes=murk_genes)
scv.pp.moments(adata, n_neighbors=30)
adata.obs['ct_pseudotime'] = pt
adata = kinetics_inference(adata, mode='coarse-grained', pt_key='ct_pseudotime', group_key='celltype', num_iter=2000, n_jobs=-1, optimizer='jax')
adata.write('gastrulation_lores.h5ad')


adata = scv.datasets.gastrulation_erythroid()
scv.pp.filter_and_normalize(adata, n_top_genes=None)
scv.pp.moments(adata,  n_neighbors=30)
ctk = CytoTRACEKernel(adata).compute_cytotrace()
pt = adata.obs['ct_pseudotime']
adata = scv.datasets.gastrulation_erythroid()
murk_genes = adata.var.index[adata.var['MURK_gene'] == True]
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, retain_genes=murk_genes)
scv.pp.moments(adata, n_neighbors=30)
adata.obs['ct_pseudotime'] = pt
adata = kinetics_inference(adata, mode='high-resolution', pt_key='ct_pseudotime', num_iter=2000, n_jobs=15, optimizer='jax')
adata.write('gastrulation_hires.h5ad')

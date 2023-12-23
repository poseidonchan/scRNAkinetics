import scanpy as sc
import numpy as np
import scvelo as scv
import cellrank as cr
from cellrank.kernels import CytoTRACEKernel
from sckinetics import kinetics_inference

adata = scv.datasets.gastrulation_erythroid()
n_neighbors = int(np.sqrt(adata.shape[0]))

scv.pp.filter_and_normalize(adata, n_top_genes=None)
scv.pp.moments(adata,  n_neighbors=n_neighbors)
ctk = CytoTRACEKernel(adata).compute_cytotrace()
scv.pl.scatter(
    adata,
    c=["ct_pseudotime", "celltype"],
    legend_loc="right",
    color_map="gnuplot2",
)

pt = adata.obs['ct_pseudotime']
adata = scv.datasets.gastrulation_erythroid()
murk_genes = adata.var.index[adata.var['MURK_gene'] == True]
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, retain_genes=murk_genes)
scv.pp.moments(adata, n_neighbors=n_neighbors)
adata.obs['ct_pseudotime'] = pt

adata = kinetics_inference(adata, mode='coarse-grained', group_key='celltype', pt_key='palantir_pseudotime', num_iter=500, n_jobs=-1)
adata.write('gastrulation_lores.h5ad')

adata = kinetics_inference(adata, mode='high-resolution', pt_key='palantir_pseudotime', num_iter=500, n_jobs=-1)
adata.write('gastrulation_hires.h5ad')

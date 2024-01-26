#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import numpy as np
import scvelo as scv
from sckinetics import kinetics_inference
from scipy.sparse import csr_matrix


# In[2]:


adata = sc.read("./data/organoid.h5ad")
adata


# In[3]:


adata = adata[adata.obs["time"] != "dmso", :].copy()
adata.obs["labeling_time"] = adata.obs["time"].astype(float) / 60


# In[4]:


adata.obs['monocle_pseudotime'] = [float(t) for t in adata.obs['monocle_pseudotime'].values]


# In[5]:


adata.layers['spliced'] = adata.layers['sl']+adata.layers['su']
adata.layers['unspliced'] = adata.layers['ul']+adata.layers['uu']
# adata.layers['labeled'] = adata.layers['sl']+adata.layers['ul']
# adata.layers['unlabeled'] = adata.layers['su']+adata.layers['uu']
del adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']

# In[6]:

vt = sc.read("./processed_organoid.h5ad").obs['latent_time']
umap = sc.read("./processed_organoid.h5ad").obsm['X_umap']

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
adata.obs['unitvelo_time'] = vt
adata.obsm['X_umap'] = umap

# In[ ]:


adata = kinetics_inference(adata, mode='high-resolution', pt_key='unitvelo_time', num_iter=2000, n_jobs=-1, optimizer='jax')
adata.write('sceu_organoid_hires.h5ad')


# In[ ]:





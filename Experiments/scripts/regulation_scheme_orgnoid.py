#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scanpy as sc
import numpy as np
from scipy import stats
import scvelo as scv
from sckinetics import kinetics_inference


# In[2]:


adata = sc.read("./cellrank2/data/organoid.h5ad")
adata = adata[adata.obs["time"] != "dmso", :].copy()
adata.obs["labeling_time"] = adata.obs["time"].astype(float) / 60
adata.obs['monocle_pseudotime'] = [float(t) for t in adata.obs['monocle_pseudotime'].values]
adata.layers['spliced'] = adata.layers['sl']+adata.layers['su']
adata.layers['unspliced'] = adata.layers['ul']+adata.layers['uu']
del adata.layers['uu'], adata.layers['ul'], adata.layers['su'], adata.layers['sl']
vt = sc.read("./unitvelo/processed_organoid.h5ad").obs['latent_time']
umap = sc.read("./unitvelo/processed_organoid.h5ad").obsm['X_umap']
adata.obs['unitvelo_time'] = vt
adata.obsm['X_umap'] = umap
scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=None)
# sc.pp.neighbors(adata,n_neighbors=30, use_rep='X_umap')
scv.pp.moments(adata, n_neighbors=15, n_pcs=30)


# In[3]:


adata


# In[4]:


df = pd.read_csv('./metabolic_labeling/aax3072_table-s4.csv',index_col=0,header=1)


# In[5]:


gene_list = []
for i in range(df.shape[0]):
    if isinstance(df.iloc[i,]['strategy_group'],str):
        gene_list.append(df.iloc[i,]['gene_symbol'])
    # if df.iloc[i,]['strategy_group']=='A':
    #     gene_list.append(df.iloc[i,]['gene_symbol'])
print(len(gene_list))


# In[6]:


adata_selected = adata[:,gene_list].copy()


# In[ ]:


adata_selected = kinetics_inference(adata_selected, mode='high-resolution', pt_key='unitvelo_time', n_jobs=-1, optimizer='scipy')


# In[ ]:


adata_selected.write('./organoid_selected_genes_scipy.h5ad')


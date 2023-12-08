#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scvelo as scv
import dynamo as dyn
import numpy as np
import scanpy as sc
import cellrank as cr
import wandb
import magic
import anndata
from tqdm import tqdm


# In[2]:


adata = cr.datasets.pancreas()


# In[3]:


scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, retain_genes=['Hhex'])
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
magic_operator = magic.MAGIC(t='auto')
adata.layers['Mu'] = magic_operator.fit_transform(adata.layers['Mu'], genes='all_genes')
adata.layers['Ms'] = magic_operator.fit_transform(adata.layers['Ms'], genes='all_genes')


# In[4]:


adata


# In[14]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

class RNAvelo(nn.Module):
    def __init__(self, feature_size):
        super(RNAvelo, self).__init__()
        self.feature_size = feature_size
        self.log_alpha = nn.Parameter(torch.rand(feature_size))
        self.log_beta = nn.Parameter(torch.rand(feature_size))
        self.log_gamma = nn.Parameter(torch.rand(feature_size))
        
    def forward(self, t, x):
        # x shape as (n, 2m)
        if len(x.shape) == 1:
            du = x[:self.feature_size]
            ds = x[self.feature_size:]
            du_dt = torch.exp(self.log_alpha) - torch.exp(self.log_beta) * du
            ds_dt = torch.exp(self.log_beta) * du - torch.exp(self.log_gamma) * ds
            return torch.cat([du_dt, ds_dt])
            
        else:
            du = x[:,:self.feature_size]
            ds = x[:,self.feature_size:]
        
            du_dt = torch.exp(self.log_alpha) - torch.exp(self.log_beta) * du
            ds_dt = torch.exp(self.log_beta) * du - torch.exp(self.log_gamma) * ds
        
            return torch.cat([du_dt, ds_dt],dim=1)
    
    def params(self,):
        return torch.exp(self.log_alpha).detach().cpu().numpy(), torch.exp(self.log_beta).detach().cpu().numpy(), torch.exp(self.log_gamma).detach().cpu().numpy()
    
    def predict(self, t, x):
        v = self.forward(t,x)
        v = v.cpu().detach().numpy()
        return v[:,:self.feature_size], v[:,self.feature_size:]


    
def fit_neural_ode(u, s, t, num_epochs=300, device='cpu'):
    # Check if CUDA is available and set the device accordingly
    if device == 'cuda':
        device = device if torch.cuda.is_available() else 'cpu'
    else:
        device = device
    
    # print(f"Training on device: {device}")
    # Extract the expression matrix and pseudo-time
    
    u = torch.tensor(u, dtype=torch.float32).to(device)
    s = torch.tensor(s, dtype=torch.float32).to(device)
    expression_matrix = torch.cat([u, s],dim=1)
    pseudo_time = torch.tensor(t, dtype=torch.float32).to(device)

    # Feature size is the number of genes (columns) in the expression matrix
    n_samples, feature_size = u.shape[0], u.shape[1]

    # Initialize the ODE function with a neural network
    velo_model = RNAvelo(feature_size=feature_size).to(device)

    # Define an optimizer
    optimizer = torch.optim.Adam(velo_model.parameters(), lr=5e-2)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Predict the state at the next pseudo-time point
        predicted = odeint(velo_model, expression_matrix[0], pseudo_time, method='rk4')
        # Calculate loss (e.g., mean squared error between predicted and actual expression)
        loss = F.l1_loss(predicted, expression_matrix)
        # Backpropagation
        loss.backward()
        optimizer.step()
        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch}: loss = {loss.item()}')
    
    # After training, use the ODE function to estimate velocity
    # u_v, s_v = velo_model.predict(pseudo_time, expression_matrix)
    alpha, beta, gamma = velo_model.params()
    return alpha, beta, gamma


# In[15]:


def coarse_mode(adata, pt_key='palantir_pseudotime', num_epochs=100):
    # Sort adata based on pseudo-time
    pseudo_time = adata.obs[pt_key].values
    sorted_indices = np.argsort(pseudo_time)
    adata = adata[sorted_indices,]
    n_samples = adata.shape[0]
    
    alpha, beta, gamma = fit_neural_ode(adata.layers['Mu'], adata.layers['Ms'], 
                                        adata.obs[pt_key].values, num_epochs=num_epochs, device='cuda')
    
    alpha = np.repeat(alpha.reshape(1, -1), n_samples, axis=0)
    beta = np.repeat(beta.reshape(1, -1), n_samples, axis=0)
    gamma = np.repeat(gamma.reshape(1, -1), n_samples, axis=0)
    adata.layers['alpha'] = alpha
    adata.layers['beta'] = beta
    adata.layers['gamma'] = gamma
    adata.layers['velocity'] = adata.layers['Mu'] * adata.layers['beta'] - adata.layers['Ms'] * adata.layers['gamma']
    adata.layers['unspliced_velocity'] = adata.layers['alpha'] - adata.layers['Mu'] * adata.layers['beta']
    
    return adata

# Assume coarse_mode function is defined above or imported

# This function will process each group in parallel
def process_group(adata, group, group_value, pt_key='palantir_pseudotime', num_epochs=100):
    adata_i = adata[adata.obs[group] == group_value].copy()
    adata_i = coarse_mode(adata_i, pt_key=pt_key, num_epochs=num_epochs)
    return adata_i

# Modified coarse_grained_kinetics function to use joblib for parallel processing
def coarse_grained_kinetics(adata, group, num_epochs=100, n_jobs=-1):
    group_values = adata.obs[group].value_counts().index

    # Run the processing in parallel
    adata_list = Parallel(n_jobs=n_jobs)(delayed(process_group)(adata, group, group_value, num_epochs=num_epochs) for group_value in tqdm(group_values))
    
    # Concatenate the results
    adata = anndata.concat(adata_list, axis=0)
    
    return adata

# This function will be executed in parallel for each cell
def process_cell(cell_idx, adata, pt_key, num_epochs):
    adj = adata.obsp['connectivities']
    neighbor_indices = adj[cell_idx].nonzero()[1]
    s = adata.layers['Ms'][neighbor_indices, :]
    u = adata.layers['Mu'][neighbor_indices, :]
    t = adata.obs[pt_key][neighbor_indices].values
    
    sorted_indices = np.argsort(t)
    s = s[sorted_indices, ]
    u = u[sorted_indices, ]
    t = t[sorted_indices]
    
    # Assuming fit_neural_ode is a function that you have defined elsewhere
    alpha, beta, gamma = fit_neural_ode(u, s, t, num_epochs=num_epochs)
    
    return alpha, beta, gamma

# Modified high_resolution_kinetics function to use joblib for parallel processing
def high_resolution_kinetics(adata, pt_key='palantir_pseudotime', num_epochs=100, n_jobs=-1):
    # Initialize matrices
    alpha = np.zeros((adata.shape[0], adata.shape[1]))
    beta = np.zeros((adata.shape[0], adata.shape[1]))
    gamma = np.zeros((adata.shape[0], adata.shape[1]))

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_cell)(cell_idx, adata, pt_key, num_epochs) for cell_idx in tqdm(range(adata.shape[0])))

    # Unpack results
    for i, (alpha_i, beta_i, gamma_i) in enumerate(results):
        alpha[i,] = alpha_i
        beta[i,] = beta_i
        gamma[i,] = gamma_i

    # Store results in adata
    adata.layers['alpha'] = alpha
    adata.layers['beta'] = beta
    adata.layers['gamma'] = gamma
    adata.layers['velocity'] = adata.layers['Mu'] * beta - adata.layers['Ms'] * gamma
    adata.layers['unspliced_velocity'] = alpha - adata.layers['Mu'] * beta
    
    return adata


# In[16]:


# adata = coarse_grained_kinetics(adata, group='clusters', num_epochs=200)


# In[17]:


adata = high_resolution_kinetics(adata, num_epochs=300)


# In[ ]:


adata.write("adata_hires.h5ad")


# In[13]:


adata.obs['speed'] = np.linalg.norm(adata.layers['velocity'], ord=2, axis=1)


# In[9]:


scv.tl.velocity_graph(adata, n_jobs=-1)
scv.pl.velocity_embedding_stream(adata, basis='umap')


# In[14]:


sc.pl.umap(adata, color='speed')


# In[10]:


i = 910


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1
sns.scatterplot(x=np.array(adata.layers['Ms'][:, i]).reshape(-1),
                y=np.array(adata.layers['Mu'][:, i]).reshape(-1),
                hue=adata.obs['clusters'], marker='.', ax=axes[0])
axes[0].set_ylabel('unspliced')
axes[0].set_xlabel('spliced')
axes[0].get_legend().remove()

# Plot 2
sns.scatterplot(x=adata.obs['palantir_pseudotime'],
                y=np.array(adata.layers['Mu'][:, i]).reshape(-1),
                hue=adata.obs['clusters'], marker='.', ax=axes[1])
axes[1].set_ylabel('unspliced')
axes[1].set_xlabel('palantir_pseudotime')
axes[1].get_legend().remove()

# Plot 3
sns.scatterplot(x=adata.obs['palantir_pseudotime'],
                y=np.array(adata.layers['Ms'][:, i]).reshape(-1),
                hue=adata.obs['clusters'], marker='.', ax=axes[2])
axes[2].set_ylabel('spliced')
axes[2].set_xlabel('palantir_pseudotime')
axes[2].get_legend().remove()

# Remove top and right spines
for j in range(3):
    axes[j].spines['top'].set_visible(False)
    axes[j].spines['right'].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.suptitle(adata.var.iloc[i].name+' gene expression')
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)

plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1
sns.scatterplot(x=adata.obs['palantir_pseudotime'],
                y=adata.layers['alpha'][:, i],
                hue=adata.obs['clusters'], marker='.',edgecolor='none', ax=axes[0])
axes[0].set_ylabel('alpha')
axes[0].set_xlabel('palantir_pseudotime')
axes[0].get_legend().remove()

# Plot 2
sns.scatterplot(x=adata.obs['palantir_pseudotime'],
                y=adata.layers['beta'][:, i],
                hue=adata.obs['clusters'], marker='.',edgecolor='none', ax=axes[1])
axes[1].set_ylabel('beta')
axes[1].set_xlabel('palantir_pseudotime')
axes[1].get_legend().remove()

# Plot 3
sns.scatterplot(x=adata.obs['palantir_pseudotime'],
                y=adata.layers['gamma'][:, i],
                hue=adata.obs['clusters'], marker='.',edgecolor='none', ax=axes[2])
axes[2].set_ylabel('gamma')
axes[2].set_xlabel('palantir_pseudotime')
axes[2].get_legend().remove()

# Remove top and right spines
for j in range(3):
    axes[j].spines['top'].set_visible(False)
    axes[j].spines['right'].set_visible(False)

# Get handles and labels for the legend
handles, labels = axes[0].get_legend_handles_labels()

fig.suptitle(adata.var.iloc[i].name+' kinetic parameters')

# Create a single legend
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)

plt.show()


# In[ ]:





import anndata
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import diffeqsolve, ODETerm, Euler
from jax import random, grad, jit
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", message="An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.")

class RNAvelo_jax(eqx.Module):
    log_alpha: jnp.array
    log_beta: jnp.array
    log_gamma: jnp.array
    feature_size: int

    def __init__(self, feature_size, *, key, **kwargs):
        """
        define the RNA velocity function for all features, parameters are constrained to positive values through
        the exponential function.
        :param feature_size: number of features of the datasets
        :param key: necessary parameters for equinox
        :param kwargs: necessary parameters for equinox
        """
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.log_alpha = jax.random.normal(key, (feature_size,))
        self.log_beta = jax.random.normal(key, (feature_size,))
        self.log_gamma = jax.random.normal(key, (feature_size,))

    def __call__(self, t, x, args):
        if len(x.shape) == 1:
            u = x[:self.feature_size]
            s = x[self.feature_size:]
            du = jnp.exp(self.log_alpha) - jnp.exp(self.log_beta) * u
            ds = jnp.exp(self.log_beta) * u - jnp.exp(self.log_gamma) * s
            return jnp.concatenate([du, ds])

        else:
            u = x[:, :self.feature_size]
            s = x[:, self.feature_size:]
            du = jnp.exp(self.log_alpha) - jnp.exp(self.log_beta) * u
            ds = jnp.exp(self.log_beta) * u - jnp.exp(self.log_gamma) * s
            return jnp.concatenate([du, ds], axis=1)


class NeuralODE(eqx.Module):
    func: RNAvelo_jax
    
    def __init__(self, feature_size, *, key, **kwargs):
        """
        Define neuralODE modual for clean forward pass
        :param feature_size: number of features of the datasets
        :param key: necessary parameters for equinox
        :param kwargs: necessary parameters for equinox
        """
        super().__init__(**kwargs)
        self.func = RNAvelo_jax(feature_size, key=key)

    def __call__(self, ts, y0):
        """
        Forward pass to infer the future states
        :param ts: the future time point
        :param y0: the first state
        :return: future states corresponding to future time points
        """
        solution = diffeqsolve(
            ODETerm(self.func),
            Euler(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            max_steps=16**4,
            throw=False,
        )

        return solution.ys


def fit_neural_ode_jax(u, s, t, gene_i=None, num_iter=300):
    """
    fit the RNA velocity function to find the unknown coefficients $\alpha$, $\beta$, $\gamma$
    :param u: unspliced RNA expression feature matrix with shape as (n, m)
    :param s: spliced RNA expression feature matrix with shape as (n, m)
    :param t: time points for each sample, shape as (n,)
    :param num_iter: number of iterations in the optimization step
    :param solver: specific solver used in the initial value problem.
    :return: $\alpha$, $\beta$ and $\gamma$, each with shape as (1, m)
    """
    u = jnp.array(u, dtype=jnp.float32)  # Ensure u is float32
    s = jnp.array(s, dtype=jnp.float32)  # Ensure s is float32
    expression_matrix = jnp.concatenate([u, s], axis=1)
    pseudo_time = jnp.array(t, dtype=jnp.float32)  # Ensure t is float32

    feature_size = u.shape[1]

    model = NeuralODE(feature_size=feature_size, key=random.PRNGKey(0), )

    @eqx.filter_value_and_grad
    def grad_total_loss(model, t, y):
        y_pred = model(t, y[0])
        total_loss = jnp.mean(jnp.abs(y - y_pred))
        return total_loss

    def compute_feature_loss(y, y_pred, gene_i, feature_size):
        return jnp.mean(jnp.abs(y[:, gene_i] - y_pred[:, gene_i]) + jnp.abs(y[:, gene_i + feature_size] - y_pred[:, gene_i + feature_size]))

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, gene_i=None, feature_size=None):
        total_loss, grads = grad_total_loss(model, ti, yi)
        y_pred = model(ti, yi[0])  # You might need to adjust this line based on your model's specifics
        feature_loss = compute_feature_loss(yi, y_pred, gene_i, feature_size)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return total_loss, feature_loss, model, opt_state

    optim = optax.adam(learning_rate=5e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    
    for iter in range(num_iter):
        total_loss, feature_loss, model, opt_state = make_step(pseudo_time, expression_matrix, model, opt_state, gene_i, feature_size=feature_size)

    # After training
    log_alpha, log_beta, log_gamma = model.func.log_alpha, model.func.log_beta, model.func.log_gamma
    # Convert them to their exponential form if needed
    alpha, beta, gamma = jnp.exp(log_alpha), jnp.exp(log_beta), jnp.exp(log_gamma)
    # clear the cache
    # jax.clear_caches()
    return alpha, beta, gamma, feature_loss

def process_group(adata, group_key, group_value,
                  pt_key='palantir_pseudotime',  vkey=None,
                  num_iter=100):
    """
    helper function for parallelization
    :param adata: adata with necessary attributes
    :param group_key: the name of the group, should be stored in adata.obs[group]
    :param group_value: name for each group of cells
    :param pt_key: pseudo-time key
    :param num_iter: number of optimization iterations
    :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
    and "spliced_velocity" clearly.
    :return: adata stored with kinetics information
    """
    adata = adata[adata.obs[group_key] == group_value].copy()
    pseudo_time = adata.obs[pt_key].values
    sorted_indices = np.argsort(pseudo_time)
    adata = adata[sorted_indices,].copy()
    n_samples = adata.shape[0]

    alpha, beta, gamma = fit_neural_ode_jax(adata.layers['Mu'], adata.layers['Ms'], adata.obs[pt_key].values,
                                              num_iter=num_iter)

    alpha = np.repeat(alpha.reshape(1, -1), n_samples, axis=0)
    beta = np.repeat(beta.reshape(1, -1), n_samples, axis=0)
    gamma = np.repeat(gamma.reshape(1, -1), n_samples, axis=0)
    adata.layers['alpha'] = alpha
    adata.layers['beta'] = beta
    adata.layers['gamma'] = gamma
    adata.layers[vkey['spliced_velocity']] = adata.layers['Mu'] * adata.layers['beta'] - adata.layers['Ms'] * \
                                             adata.layers['gamma']
    adata.layers[vkey['unspliced_velocity']] = adata.layers['alpha'] - adata.layers['Mu'] * adata.layers['beta']
    return adata


def coarse_grained_kinetics(adata, group_key, pt_key='palantir_pseudotime',
                            vkey: dict = {'unspliced_velocity': "unspliced_velocity",
                                          'spliced_velocity': "spliced_velocity"},
                            num_iter=100, n_jobs=-1):
    """
    group-mode kinetics parameters inference. Assume each group of cells have the same kinetics parameters,
    and then solve the RNA velocity ODE with the neuralODE solver. All cells in the same group will have
    the same inferred kinetics parameters
    :param adata: anndata with necessary attributes like unspliced and spliced expression value saved in layers
    :param group_key: the key for the group, like "clusters", "cell_type", should be stored in adata.obs
    :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
    :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
    and "spliced_velocity" clearly.
    :param num_iter: number of iterations in the optimization steps
    :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
    :return: adata stored with group-level kinetics information
    """
    adata_obsp = adata.obsp.copy()
    adata_uns = adata.uns.copy()

    group_values = adata.obs[group_key].value_counts().index

    # Run the processing in parallel
    adata_list = Parallel(n_jobs=n_jobs, prefer="processes", backend='loky')(
        delayed(process_group)(adata, group_key, group_value, pt_key=pt_key, vkey=vkey, num_iter=num_iter) for group_value in
        tqdm(group_values))

    # Concatenate the results
    adata = anndata.concat(adata_list, axis=0)

    adata.obsp = adata_obsp
    adata.uns = adata_uns

    return adata


def process_cell(cell_idx, adata, pt_key, num_iter, adj, gene_i=None):
    """
    helper function for parallelization, each cell's neighbors are fetched and used to fit the RNA velocity.
    :param cell_idx: cell index used in this cell
    :param adata: adata for all cells
    :param pt_key: pseudo-time key in adata.obs
    :param num_iter: number of optimization iteration
    :param adj: predefined adjacency matrix, like the knn distances matrix
    :return: alpha, beta, and gamma for this cell
    """

    neighbor_indices = adj[cell_idx].nonzero()[1]
    s = adata.layers['Ms'][neighbor_indices, :]
    u = adata.layers['Mu'][neighbor_indices, :]
    t = adata.obs[pt_key][neighbor_indices].values

    sorted_indices = np.argsort(t)
    s = s[sorted_indices,]
    u = u[sorted_indices,]
    t = t[sorted_indices]

    # Assuming fit_neural_ode is a function that you have defined elsewhere
    alpha, beta, gamma, feature_loss = fit_neural_ode_jax(u, s, t, num_iter=num_iter, gene_i=gene_i)

    return alpha, beta, gamma, feature_loss


def high_resolution_kinetics(adata, pt_key='palantir_pseudotime',
                             gene_list = None,
                             vkey={'unspliced_velocity': "unspliced_velocity",
                                  'spliced_velocity': "spliced_velocity"},
                             num_iter=100, n_jobs=-1, gene_monitor=None):
    """
    cell-level kinetics parameters inference. Assume each cell and its neighbors have the same kinetics parameters,
    and then solve the RNA velocity ODE with the neuralODE solver. After calculation, each cell's kinetics parameters
    are different
    :param adata: anndata with necessary information
    :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
    :param gene_list: selected list of genes to calculate kinetics paramter, if None, all genes will be involved in calculation.
    :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
    and "spliced_velocity" clearly.
    :param num_iter: number of optimization iteration
    :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
    :return: adata stored with group-level kinetics information
    """
    adata_obsp = adata.obsp.copy()
    adata_uns = adata.uns.copy()

    if gene_list is None:
        adata = adata
    else:
        # choose special genes
        adata = adata[:,gene_list]

    for i in range(adata.shape[1]):
        if adata.var.index[i] == gene_monitor:
            gene_i = i
    
    # Initialize matrices
    alpha = np.zeros((adata.shape[0], adata.shape[1]))
    beta = np.zeros((adata.shape[0], adata.shape[1]))
    gamma = np.zeros((adata.shape[0], adata.shape[1]))

    results = Parallel(n_jobs=n_jobs, prefer="processes", backend='loky')(
        delayed(process_cell)(cell_idx, adata, pt_key, num_iter, adj=adata_obsp['connectivities'], gene_i=gene_i) for cell_idx in tqdm(range(adata.shape[0])))
    # results = []
    # for cell_idx in tqdm(range(adata.shape[0])):
    #     result = process_cell(cell_idx, adata, pt_key, num_iter, adj=adata_obsp['connectivities'], gene_i=gene_i)
    #     results.append(result)

    feature_losses = []
    # Unpack results
    for i, (alpha_i, beta_i, gamma_i, feature_loss_i) in enumerate(results):
        alpha[i,] = alpha_i
        beta[i,] = beta_i
        gamma[i,] = gamma_i
        feature_losses.append(feature_loss_i)

    # Store results in adata
    adata.layers['alpha'] = alpha
    adata.layers['beta'] = beta
    adata.layers['gamma'] = gamma
    
    for loss in feature_losses:
        print(loss)
    
    adata.obs[gene_monitor+' loss'] = feature_losses

    adata.layers[vkey['spliced_velocity']] = adata.layers['Mu'] * adata.layers['beta'] - adata.layers['Ms'] * \
                                             adata.layers['gamma']
    adata.layers[vkey['unspliced_velocity']] = adata.layers['alpha'] - adata.layers['Mu'] * adata.layers['beta']

    adata.obsp = adata_obsp
    adata.uns = adata_uns
    return adata


def kinetics_inference(adata: anndata.AnnData = None,
                       pt_key: str = None,
                       mode: str = 'high-resolution',
                       group_key = None,
                       gene_list = None,
                       vkey: dict = {'unspliced_velocity': "unspliced_velocity",
                                     'spliced_velocity': "spliced_velocity"},
                       n_jobs: int = 1,
                       num_iter: int = 200,
                       gene_monitor = None,
                       ) -> anndata.AnnData:
    """
    Infer the kinetics parameters of RNA velocity equations. It has two modes, "coarse-grained" or "high-resolution" mode,
    based on the assumption that each group of cell or each cell and its neighbors share the same kinetics parameters, respectively.
    For "coarse-grained" mode, the kinetic parameters for cells within a group will be the same but different between genes.
    For "high-resolution" mode, each cell and each gene's kinetic parameters will be different. The inference methods
    involve fitting a RNA velocity function using neuralODE solvers.
    :param adata: single-cell RNA-seq data with annotated pseudo-time.
    :param pt_key: pseudo-time key for retrieving pseudo-time in adata.obs.
    :param mode: support two modes "coarse-grained" and "high-resolution".
    :param group_key: only be used when the mode is "coarse-grained", which is necessary.
    :param gene_list: selected list of genes to calculate kinetics paramter, if None, all genes will be involved in calculation.
    :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
    and "spliced_velocity" clearly.
    :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
    :param num_iter: number of optimization iteration

    :return:
    """

    if mode == 'coarse-grained':
        adata = coarse_grained_kinetics(adata, group_key, pt_key=pt_key, vkey=vkey,
                                        num_iter=num_iter, n_jobs=n_jobs)
    elif mode == 'high-resolution':
        adata = high_resolution_kinetics(adata, pt_key=pt_key, gene_list=gene_list, vkey=vkey,
                                         num_iter=num_iter, n_jobs=n_jobs, gene_monitor=gene_monitor)
    else:
        raise ValueError("please make sure the mode is correct selected, only 'coarse-grained' and 'high-resolution' are valid and supported")

    return adata

"""
class RNAvelo_jax(eqx.Module):
    log_alpha_matrices: list
    log_beta_matrices: list
    log_gamma_matrices: list
    feature_size: int

    def __init__(self, feature_size, *, key, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        keys = jax.random.split(key, 9)
        # Initialize matrices for log_alpha, log_beta, log_gamma
        self.log_alpha_matrices = [
            jax.random.normal(keys[0], (1, 128)),
            jax.random.normal(keys[1], (128, feature_size)),
        ]

        self.log_beta_matrices = [
            jax.random.normal(keys[3], (1, 128)),
            jax.random.normal(keys[4], (128, feature_size)),
        ]

        self.log_gamma_matrices = [
            jax.random.normal(keys[6], (1, 128)),
            jax.random.normal(keys[7], (128, feature_size)),
        ]

    def _compute_parameter(self, matrices):
        return jnp.dot(matrices[0], matrices[1])

    def get_parameters(self):
        alpha = self._compute_parameter(self.log_alpha_matrices).squeeze()
        beta = self._compute_parameter(self.log_beta_matrices).squeeze()
        gamma = self._compute_parameter(self.log_gamma_matrices).squeeze()
        return alpha, beta, gamma

    def __call__(self, t, x, args):
        log_alpha, log_beta, log_gamma = self.get_parameters()

        if len(x.shape) == 1:
            u = x[:self.feature_size]
            s = x[self.feature_size:]
            du = jnp.exp(log_alpha) - jnp.exp(log_beta) * u
            ds = jnp.exp(log_beta) * u - jnp.exp(log_gamma) * s
            return jnp.concatenate([du, ds])
        else:
            u = x[:, :self.feature_size]
            s = x[:, self.feature_size:]
            du = jnp.exp(log_alpha) - jnp.exp(log_beta) * u
            ds = jnp.exp(log_beta) * u - jnp.exp(log_gamma) * s
            return jnp.concatenate([du, ds], axis=1)
"""
# def high_resolution_kinetics(adata, pt_key='palantir_pseudotime',
#                              gene_list = None,
#                              vkey={'unspliced_velocity': "unspliced_velocity",
#                                   'spliced_velocity': "spliced_velocity"},
#                              num_iter=100, n_jobs=-1):
#     """
#     cell-level kinetics parameters inference. Assume each cell and its neighbors have the same kinetics parameters,
#     and then solve the RNA velocity ODE with the neuralODE solver. After calculation, each cell's kinetics parameters
#     are different
#     :param adata: anndata with necessary information
#     :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
#     :param gene_list: selected list of genes to calculate kinetics paramter, if None, all genes will be involved in calculation.
#     :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
#     and "spliced_velocity" clearly.
#     :param num_iter: number of optimization iteration
#     :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
#     :return: adata stored with group-level kinetics information
#     """
#     adata_obsp = adata.obsp.copy()
#     adata_uns = adata.uns.copy()
#
#     if gene_list is None:
#         gene_list = adata.var.index
#
#     # Initialize matrices
#     alpha = np.zeros((adata.shape[0], len(gene_list)))
#     beta = np.zeros((adata.shape[0], len(gene_list)))
#     gamma = np.zeros((adata.shape[0], len(gene_list)))
#
#     # Initialize velocity layers with zero matrices
#     adata.layers[vkey['spliced_velocity']] = np.zeros((adata.shape[0], adata.shape[1]))
#     adata.layers[vkey['unspliced_velocity']] = np.zeros((adata.shape[0], adata.shape[1]))
#
#     # Process each gene
#     for j, gene in tqdm(enumerate(gene_list), total=len(gene_list), desc="Processing genes"):
#         adata_gene = adata[:, gene].copy()
#
#         results = Parallel(n_jobs=n_jobs, prefer="processes", backend='loky')(
#             delayed(process_cell)(cell_idx, adata_gene, pt_key, num_iter, adj=adata_obsp['connectivities'])
#             for cell_idx in tqdm(range(adata_gene.shape[0]), desc=f"Processing cells for gene {gene}", leave=False)
#         )
#
#         # Unpack results and update matrices
#         for i, (alpha_i, beta_i, gamma_i) in enumerate(results):
#             alpha[i, j] = alpha_i
#             beta[i, j] = beta_i
#             gamma[i, j] = gamma_i
#
#         # Update velocity layers for this gene
#         gene_idx = adata.var.index.get_loc(gene)
#         adata.layers[vkey['spliced_velocity']][:, gene_idx] = -adata.layers['Ms'][:, gene_idx] * adata.layers['gamma'][:,
#                                                                                                 gene_idx] + \
#                                                               adata.layers['Mu'][:, gene_idx] * adata.layers['beta'][:,
#                                                                                                 gene_idx]
#         adata.layers[vkey['unspliced_velocity']][:, gene_idx] = adata.layers['alpha'][:, gene_idx] - \
#                                                                 adata.layers['Mu'][:, gene_idx] * adata.layers['beta'][
#                                                                                                   :, gene_idx]
#
#     # Store the computed alpha, beta, and gamma matrices in adata
#     adata.layers['alpha'] = alpha
#     adata.layers['beta'] = beta
#     adata.layers['gamma'] = gamma
#
#     adata.obsp = adata_obsp
#     adata.uns = adata_uns
#
#     return adata


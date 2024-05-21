import anndata
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import diffeqsolve, Euler, Dopri8, Tsit5
from jax import random, grad, jit
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

import warnings

warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, minimize


def fit_ode_scipy(u_data, s_data, t_data, initial_guess=[1, 1, 1]):
    """
    Fit the RNA velocity ODE model to the data using scipy.optimize.minimize for vectorized parameters,
    ensuring alpha, beta, gamma are non-negative.

    :param u_data: Unspliced RNA expression feature matrix with shape (n_samples, n_features).
    :param s_data: Spliced RNA expression feature matrix with shape (n_samples, n_features).
    :param t_data: Time points corresponding to the data with shape (n_samples,).
    :param initial_guess: Initial guess for the parameters [alpha, beta, gamma].
    :return: Estimated parameters alpha, beta, gamma, each a vector with length n_features.
    """

    n_features = u_data.shape[1]
    alpha_est, beta_est, gamma_est = np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)

    # Define bounds for alpha, beta, gamma to be non-negative
    parameter_bounds = Bounds([0, 0, 0], [np.inf, np.inf, np.inf])

    for i in range(n_features):
        # Extract the data for the current feature
        u_feature_data = u_data[:, i]
        s_feature_data = s_data[:, i]

        # Define the cost function for the current feature
        def cost_function(params):
            alpha, beta, gamma = params

            # System of ODEs
            def system_of_odes(t, y):
                u, s = y
                du_dt = alpha - beta * u
                ds_dt = beta * u - gamma * s
                return [du_dt, ds_dt]

            # Initial conditions
            initial_conditions = [u_feature_data[0], s_feature_data[0]]

            # Solve the ODEs
            solution = solve_ivp(system_of_odes, (t_data[0], t_data[-1]), initial_conditions, t_eval=t_data,
                                 method='DOP853', max_step=16 ** 4)
            u_pred, s_pred = solution.y

            # Calculate the cost (sum of squared differences)
            cost = np.sum((u_feature_data - u_pred) ** 2) + np.sum((s_feature_data - s_pred) ** 2)
            # print(alpha, beta, gamma)
            return cost

        # Run the optimization for the current feature with bounds
        result = minimize(cost_function, initial_guess, method='Powell', bounds=parameter_bounds)

        # Extract the optimized parameters for the current feature
        alpha_est[i], beta_est[i], gamma_est[i] = result.x

    return alpha_est, beta_est, gamma_est


class RNAvelo_jax(eqx.Module):
    alpha: jnp.array
    beta: jnp.array
    gamma: jnp.array
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
        self.alpha = jnp.ones(feature_size)
        self.beta = jnp.ones(feature_size)
        self.gamma = jnp.ones(feature_size)

    def __call__(self, t, x, args):
        u = x[:self.feature_size]
        s = x[self.feature_size:]
        du = self.alpha - self.beta * u
        ds = self.beta * u - self.gamma * s
        return jnp.concatenate([du, ds])


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
            diffrax.ODETerm(self.func),
            Dopri8(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=16 ** 4,
            throw=False,
        )

        return solution.ys


def fit_ode_jax(u, s, t, num_iter=1000):
    """
    fit the RNA velocity function to find the unknown coefficients $\alpha$, $\beta$, $\gamma$
    :param u: unspliced RNA expression feature matrix with shape as (n, m)
    :param s: spliced RNA expression feature matrix with shape as (n, m)
    :param t: time points for each sample, shape as (n,)
    :param num_iter: number of iterations in the optimization step
    :param solver: specific solver used in the initial value problem.
    :return: $\alpha$, $\beta$ and $\gamma$, each with shape as (1, m)
    """
    u = jnp.array(u, dtype=jnp.float32)
    s = jnp.array(s, dtype=jnp.float32)
    expression_matrix = jnp.concatenate([u, s], axis=1)
    pseudo_time = jnp.array(t, dtype=jnp.float32)

    feature_size = u.shape[1]

    model = NeuralODE(feature_size=feature_size, key=random.PRNGKey(0), )

    @eqx.filter_value_and_grad
    def grad_loss(model, t, y):
        y_pred = model(t, y[0])
        loss = jnp.sum(jnp.abs(y - y_pred))
        return loss

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        total_loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        # Project the parameters to the feasible region [0, infinity)
        new_alpha = jnp.maximum(model.func.alpha, 0)
        new_beta = jnp.maximum(model.func.beta, 0)
        new_gamma = jnp.maximum(model.func.gamma, 0)

        # Update the model with the projected parameters
        model = eqx.tree_at(lambda m: m.func.alpha, model, new_alpha)
        model = eqx.tree_at(lambda m: m.func.beta, model, new_beta)
        model = eqx.tree_at(lambda m: m.func.gamma, model, new_gamma)

        return total_loss, model, opt_state

    optim = optax.adam(learning_rate=1e-1)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    for iter in range(num_iter):
        total_loss, model, opt_state = make_step(pseudo_time, expression_matrix, model, opt_state)

    # After training
    alpha, beta, gamma = model.func.alpha, model.func.beta, model.func.gamma
    # clear the cache
    jax.clear_caches()
    return alpha, beta, gamma


def process_group(adata, group_key, group_value,
                  pt_key='palantir_pseudotime',
                  gene_list=None,
                  vkey=None,
                  num_iter=100, optimizer=None):
    """
    helper function for parallelization
    :param adata: adata with necessary attributes
    :param group_key: the name of the group, should be stored in adata.obs[group]
    :param group_value: name for each group of cells
    :param pt_key: pseudo-time key
    :param gene_list: selected a list of genes to calculate kinetics parameters, if None, all genes will be involved in calculation.
    :param num_iter: number of optimization iterations
    :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
    and "spliced_velocity" clearly.
    :return: adata stored with kinetics information
    """
    # Subset for the specific group
    adata = adata[adata.obs[group_key] == group_value].copy()
    # make t strictly increased
    t = adata.obs[pt_key].values
    noise_scale = (t.max() - t.min()) * 0.005  # 0.5% noise
    noise = np.abs(np.random.normal(0, noise_scale, t.size))
    t = t + noise
    sorted_indices = np.argsort(t)
    adata = adata[sorted_indices,].copy()
    t = t[sorted_indices]
    # Identify indices of selected genes
    gene_indices = [i for i, gene in enumerate(adata.var.index) if
                    gene in gene_list] if gene_list is not None else range(adata.shape[1])

    n_samples = adata.shape[0]

    # Initialize the kinetics parameter arrays
    alpha = np.zeros((n_samples, adata.shape[1]))
    beta = np.zeros_like(alpha)
    gamma = np.zeros_like(alpha)

    # Initialize the velocity layers

    adata.layers[vkey['unspliced_velocity']] = np.zeros(adata.shape)
    adata.layers[vkey['spliced_velocity']] = np.zeros(adata.shape)

    # Kinetics parameter fitting
    if len(gene_indices) <= 100 or optimizer == 'scipy':

        alpha_fit, beta_fit, gamma_fit = fit_ode_scipy(adata.layers['Mu'][:, gene_indices],
                                                       adata.layers['Ms'][:, gene_indices],
                                                       t)
    elif len(gene_indices) > 100 or optimizer == 'jax':
        alpha_fit, beta_fit, gamma_fit = fit_ode_jax(adata.layers['Mu'][:, gene_indices],
                                                     adata.layers['Ms'][:, gene_indices],
                                                     t=t, num_iter=num_iter)

    # Fill the arrays with fitted values for selected genes
    for j, gene_idx in enumerate(gene_indices):
        alpha[:, gene_idx] = alpha_fit[j]
        beta[:, gene_idx] = beta_fit[j]
        gamma[:, gene_idx] = gamma_fit[j]

    # Store kinetics parameters in adata
    adata.layers['alpha'] = alpha
    adata.layers['beta'] = beta
    adata.layers['gamma'] = gamma

    # Calculate and store velocity layers for selected genes
    for gene_idx in gene_indices:
        adata.layers[vkey['spliced_velocity']][:, gene_idx] = adata.layers['Mu'][:, gene_idx] * beta[:, gene_idx] - \
                                                              adata.layers['Ms'][:, gene_idx] * gamma[:, gene_idx]
        adata.layers[vkey['unspliced_velocity']][:, gene_idx] = alpha[:, gene_idx] - adata.layers['Mu'][:,
                                                                                     gene_idx] * beta[:, gene_idx]

    return adata


def coarse_grained_kinetics(adata, group_key, pt_key='palantir_pseudotime',
                            gene_list=None,
                            vkey: dict = {'unspliced_velocity': "unspliced_velocity",
                                          'spliced_velocity': "spliced_velocity"},
                            optimizer=None,
                            num_iter=100, n_jobs=-1):
    """
    group-mode kinetics parameters inference. Assume each group of cells have the same kinetics parameters,
    and then solve the RNA velocity ODE with the neuralODE solver. All cells in the same group will have
    the same inferred kinetics parameters
    :param adata: anndata with necessary attributes like unspliced and spliced expression value saved in layers
    :param group_key: the key for the group, like "clusters", "cell_type", should be stored in adata.obs
    :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
    :param gene_list: selected a list of genes to calculate kinetics parameters, if None, all genes will be involved in calculation.
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
        delayed(process_group)(adata, group_key, group_value, gene_list=gene_list,
                               pt_key=pt_key, vkey=vkey, num_iter=num_iter, optimizer=optimizer) for
        group_value in
        tqdm(group_values))

    # Concatenate the results
    adata = anndata.concat(adata_list, axis=0)

    adata.obsp = adata_obsp
    adata.uns = adata_uns

    return adata


def process_cell(cell_idx, adata, pt_key, num_iter, adj, gene_list, optimizer='jax'):
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
    # print(gene_list)
    s = s[:, gene_list]
    u = u[:, gene_list]
    t = adata.obs[pt_key][neighbor_indices].values
    noise_scale = (t.max() - t.min()) * 0.005  # 0.5% noise
    noise = np.abs(np.random.normal(0, noise_scale, t.size))
    t = t + noise

    sorted_indices = np.argsort(t)
    s = s[sorted_indices,]
    u = u[sorted_indices,]
    t = t[sorted_indices]

    if optimizer == 'jax':
        alpha, beta, gamma = fit_ode_jax(u, s, t, num_iter=num_iter)
    elif optimizer == 'scipy':
        alpha, beta, gamma = fit_ode_scipy(u, s, t)
    else:
        raise ValueError('Please choose the optimizer from ["jax", "scipy"]')

    return alpha, beta, gamma


def high_resolution_kinetics(adata, pt_key='palantir_pseudotime',
                             gene_list=None,
                             vkey={'unspliced_velocity': "unspliced_velocity",
                                   'spliced_velocity': "spliced_velocity"},
                             num_iter=100, n_jobs=-1, optimizer=None):
    """
    cell-level kinetics parameters inference. Assume each cell and its neighbors have the same kinetics parameters,
    and then solve the RNA velocity ODE with the neuralODE solver. After calculation, each cell's kinetics parameters
    are different
    :param adata: anndata with necessary information
    :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
    :param gene_list: selected a list of genes to calculate kinetics parameters, if None, all genes will be involved in calculation.
    :param vkey: stored key of recalculated RNA velocity in adata.layers. Should specify the key for "unspliced_velocity"
    and "spliced_velocity" clearly.
    :param num_iter: number of optimization iteration
    :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
    :param optimizer: choose between ['jax', 'scipy'], 'jax' means using jax with gradient descent to optimize and is suitable for
    all genes calculation. 'scipy' will use scipy with L-BFGS-B optimizer which is suitable for selected small number of genes.
    :return: adata stored with group-level kinetics information
    """
    adata_obsp = adata.obsp.copy()
    adata_uns = adata.uns.copy()

    # Identify indices of selected genes
    gene_indices = [i for i, gene in enumerate(adata.var.index) if
                    gene in gene_list] if gene_list is not None else range(adata.shape[1])

    if len(gene_indices) <= 100 and optimizer is None:
        optimizer = 'scipy'

    # Initialize the velocity layers
    adata.layers[vkey['unspliced_velocity']] = np.zeros(adata.shape)
    adata.layers[vkey['spliced_velocity']] = np.zeros(adata.shape)

    # Prepare arrays for kinetics parameters
    alpha = np.zeros((adata.shape[0], adata.shape[1]))
    beta = np.zeros_like(alpha)
    gamma = np.zeros_like(alpha)

    # Parallel processing of cells for selected genes
    results = Parallel(n_jobs=n_jobs, prefer="processes", backend='loky')(
        delayed(process_cell)(cell_idx, adata, pt_key, num_iter, adj=adata_obsp['connectivities'],
                              gene_list=gene_indices, optimizer=optimizer) for
        cell_idx in tqdm(range(adata.shape[0])))

    # Store results only for selected genes
    for i, result in enumerate(results):
        for j, gene_idx in enumerate(gene_indices):
            alpha[i, gene_idx] = result[0][j]
            beta[i, gene_idx] = result[1][j]
            gamma[i, gene_idx] = result[2][j]

    # Store kinetics parameters in adata
    adata.layers['alpha'] = alpha
    adata.layers['beta'] = beta
    adata.layers['gamma'] = gamma

    # Calculate and store velocity layers for selected genes
    for gene_idx in gene_indices:
        adata.layers[vkey['spliced_velocity']][:, gene_idx] = adata.layers['Mu'][:, gene_idx] * beta[:, gene_idx] - \
                                                              adata.layers['Ms'][:, gene_idx] * gamma[:, gene_idx]
        adata.layers[vkey['unspliced_velocity']][:, gene_idx] = alpha[:, gene_idx] - adata.layers['Mu'][:,
                                                                                     gene_idx] * beta[:, gene_idx]

    # Restore original adata.obsp and adata.uns
    adata.obsp = adata_obsp
    adata.uns = adata_uns

    return adata


def process_raw(cell_idx, u=None, s=None, t=None, adj=None, num_iter=1000, optimizer='scipy'):
    neighbor_indices = adj[cell_idx].nonzero()[0]
    s = s[neighbor_indices, :]
    u = u[neighbor_indices, :]
    t = t[neighbor_indices]
    noise_scale = (t.max() - t.min()) * 0.005  # 0.5% noise
    noise = np.abs(np.random.normal(0, noise_scale, t.size))
    t = t + noise

    sorted_indices = np.argsort(t)
    s = s[sorted_indices,]
    u = u[sorted_indices,]
    t = t[sorted_indices]

    if optimizer == 'scipy':
        alpha, beta, gamma = fit_ode_scipy(u, s, t)
    elif optimizer == 'jax':
        alpha, beta, gamma = fit_ode_jax(u, s, t)

    return alpha, beta, gamma


def high_resolution_raw(u, s, t, adj, num_iter=100, n_jobs=-1, optimizer='scipy'):
    # Initialize matrices
    alpha = np.zeros((u.shape[0], u.shape[1]))
    beta = np.zeros((u.shape[0], u.shape[1]))
    gamma = np.zeros((u.shape[0], u.shape[1]))

    results = Parallel(n_jobs=n_jobs, prefer="processes", backend='loky')(
        delayed(process_raw)(cell_idx, u, s, t, adj, num_iter, optimizer) for cell_idx in tqdm(range(u.shape[0])))

    for i, (alpha_i, beta_i, gamma_i) in enumerate(results):
        alpha[i,] = alpha_i
        beta[i,] = beta_i
        gamma[i,] = gamma_i

    return alpha, beta, gamma


def kinetics_inference(adata: anndata.AnnData = None,
                       pt_key: str = None,
                       mode: str = 'high-resolution',
                       group_key=None,
                       gene_list=None,
                       vkey: dict = {'unspliced_velocity': "unspliced_velocity",
                                     'spliced_velocity': "spliced_velocity"},
                       n_jobs: int = 1,
                       num_iter: int = 200,
                       optimizer: str = None,
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
        adata = coarse_grained_kinetics(adata, group_key, pt_key=pt_key, vkey=vkey, gene_list=gene_list,
                                        num_iter=num_iter, n_jobs=n_jobs, optimizer=optimizer)
    elif mode == 'high-resolution':
        adata = high_resolution_kinetics(adata, pt_key=pt_key, gene_list=gene_list, vkey=vkey,
                                         num_iter=num_iter, n_jobs=n_jobs, optimizer=optimizer)
    else:
        raise ValueError(
            "please make sure the mode is correct selected, only 'coarse-grained' and 'high-resolution' are valid and supported")

    return adata
import anndata
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import diffeqsolve, ODETerm, Tsit5
from jax import random, grad, jit
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

class RNAvelo(eqx.Module):
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
            du = x[:self.feature_size]
            ds = x[self.feature_size:]
            du_dt = jnp.exp(self.log_alpha) - jnp.exp(self.log_beta) * du
            ds_dt = jnp.exp(self.log_beta) * du - jnp.exp(self.log_gamma) * ds
            return jnp.concatenate([du_dt, ds_dt])

        else:
            du = x[:, :self.feature_size]
            ds = x[:, self.feature_size:]
            du_dt = jnp.exp(self.log_alpha) - jnp.exp(self.log_beta) * du
            ds_dt = jnp.exp(self.log_beta) * du - jnp.exp(self.log_gamma) * ds
            return jnp.concatenate([du_dt, ds_dt], axis=1)


class NeuralODE(eqx.Module):
    func: RNAvelo

    def __init__(self, feature_size, *, key, **kwargs):
        """
        Define neuralODE modual for clean forward pass
        :param feature_size: number of features of the datasets
        :param key: necessary parameters for equinox
        :param kwargs: necessary parameters for equinox
        """
        super().__init__(**kwargs)
        self.func = RNAvelo(feature_size, key=key)

    def __call__(self, ts, y0):
        """
        Forward pass to infer the future states
        :param ts: the future time point
        :param y0: the first state
        :return: future states corresponding to future time points
        """
        solution = diffeqsolve(
            ODETerm(self.func),
            Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-4),
            max_steps=10000,
        )

        return solution.ys


def fit_neural_ode(u, s, t, num_iter=300):
    """
    fit the RNA velocity function to find the unknown coefficients $\alpha$, $\beta$, $\gamma$
    :param u: unspliced RNA expression feature matrix with shape as (n, m)
    :param s: spliced RNA expression feature matrix with shape as (n, m)
    :param t: time points for each sample, shape as (n,)
    :param num_iter: number of iterations in the optimization step
    :return: $\alpha$, $\beta$ and $\gamma$, each with shape as (1, m)
    """
    u = jnp.array(u, dtype=jnp.float32)  # Ensure u is float32
    s = jnp.array(s, dtype=jnp.float32)  # Ensure s is float32
    expression_matrix = jnp.concatenate([u, s], axis=1)
    pseudo_time = jnp.array(t, dtype=jnp.float32)  # Ensure t is float32

    feature_size = u.shape[1]

    model = NeuralODE(feature_size=feature_size, key=random.PRNGKey(0))

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = model(ti, yi[0])
        return jnp.mean(jnp.abs(yi - y_pred))

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate=5e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    for iter in range(num_iter):
        loss, model, opt_state = make_step(pseudo_time, expression_matrix, model, opt_state)

    # After training
    alpha, beta, gamma = model.func.log_alpha, model.func.log_beta, model.func.log_gamma
    # Convert them to their exponential form if needed
    alpha_exp, beta_exp, gamma_exp = jnp.exp(alpha), jnp.exp(beta), jnp.exp(gamma)

    return alpha_exp, beta_exp, gamma_exp


def process_group(adata, group_key, group_value,
                  pt_key='palantir_pseudotime', num_iter=100,
                  vkey={'unspliced_velocity': "unspliced_velocity",
                        'spliced_velocity': "spliced_velocity"}):
    """
    helper function for parallelization
    :param adata: adata with necessary attributes
    :param group_key: the name of the group, should be stored in adata.obs[group]
    :param group_value: name for each group of cells
    :param pt_key: pseudo-time key
    :param num_iter: number of optimization iterations
    :return: adata stored with kinetics information
    """
    adata = adata[adata.obs[group_key] == group_value].copy()
    pseudo_time = adata.obs[pt_key].values
    sorted_indices = np.argsort(pseudo_time)
    adata = adata[sorted_indices,]
    n_samples = adata.shape[0]

    alpha, beta, gamma = fit_neural_ode(adata.layers['Mu'], adata.layers['Ms'], adata.obs[pt_key].values,
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


def coarse_grained_kinetics(adata, group_key, pt_key='palantir_pseudotime', num_iter=100, n_jobs=-1):
    """
    group-mode kinetics parameters inference. Assume each group of cells have the same kinetics parameters,
    and then solve the RNA velocity ODE with the neuralODE solver. All cells in the same group will have
    the same inferred kinetics parameters
    :param adata: anndata with necessary attributes like unspliced and spliced expression value saved in layers
    :param group_key: the key for the group, like "clusters", "cell_type", should be stored in adata.obs
    :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
    :param num_iter: number of iterations in the optimization steps
    :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
    :return: adata stored with group-level kinetics information
    """
    adata_obsp = adata.obsp.copy()
    adata_uns = adata.uns.copy()

    group_values = adata.obs[group_key].value_counts().index

    # Run the processing in parallel
    adata_list = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_group)(adata, group_key, group_value, pt_key=pt_key, num_iter=num_iter) for group_value in
        tqdm(group_values))

    # Concatenate the results
    adata = anndata.concat(adata_list, axis=0)

    adata.obsp = adata_obsp
    adata.uns = adata_uns

    return adata


def process_cell(cell_idx, adata, pt_key, num_iter):
    """
    helper function for parallelization, each cell's neighbors are fetched and used to fit the RNA velocity.
    :param cell_idx: cell index used in this cell
    :param adata: adata for all cells
    :param pt_key: pseudo-time key in adata.obs
    :param num_iter: number of optimization iteration
    :return: alpha, beta, and gamma for this cell
    """
    adj = adata.obsp['connectivities']
    neighbor_indices = adj[cell_idx].nonzero()[1]
    s = adata.layers['Ms'][neighbor_indices, :]
    u = adata.layers['Mu'][neighbor_indices, :]
    t = adata.obs[pt_key][neighbor_indices].values

    sorted_indices = np.argsort(t)
    s = s[sorted_indices,]
    u = u[sorted_indices,]
    t = t[sorted_indices]

    # Assuming fit_neural_ode is a function that you have defined elsewhere
    alpha, beta, gamma = fit_neural_ode(u, s, t, num_iter=num_iter)

    return alpha, beta, gamma


def high_resolution_kinetics(adata, pt_key='palantir_pseudotime', num_iter=100, n_jobs=-1):
    """
    cell-level kinetics parameters inference. Assume each cell and its neighbors have the same kinetics parameters,
    and then solve the RNA velocity ODE with the neuralODE solver. After calculation, each cell's kinetics parameters
    are different
    :param adata: anndata with necessary information
    :param pt_key: pseudo-time key to retrieve the pseudo-time value, should be stored in adata.obs
    :param num_iter: number of optimization iteration
    :param n_jobs: number of cores for parallelization computation. -1 means use all cores.
    :return: adata stored with group-level kinetics information
    """
    adata_obsp = adata.obsp.copy()
    adata_uns = adata.uns.copy()
    # Initialize matrices
    alpha = np.zeros((adata.shape[0], adata.shape[1]))
    beta = np.zeros((adata.shape[0], adata.shape[1]))
    gamma = np.zeros((adata.shape[0], adata.shape[1]))

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_cell)(cell_idx, adata, pt_key, num_iter) for cell_idx in tqdm(range(adata.shape[0])))

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

    adata.obsp = adata_obsp
    adata.uns = adata_uns
    return adata

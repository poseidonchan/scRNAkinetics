import torch
import numpy as np
import torch.nn as nn
import lightning as pl
from lightning import seed_everything
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from .layers import AutoEncoder, UnS_Dataset, EpochProgressBar
import scipy

class KineticsKernel(pl.LightningModule):
    def __init__(self,
                 adata,
                 kernel = None,
                 lipschitz_bound: float = 10,
                 latent_dim=32,
                 hidden_dim=128,
                 lr: float = 1e-3,
                 variational: bool = True,
                 batch_size: int = 1024,
                 weight_decay: float = 0,
                 validation_size: float = 0.1,
                 lambda_kld: float = 1e-3,
                 seed: int = 0,
                 activation_fn = nn.CELU):
        super().__init__()


        if isinstance(adata.layers['Ms'], np.ndarray):
            self.Ms = adata.layers['Ms']
        else:
            self.Ms = np.array(self.adata.layers['Ms'].todense())

        if isinstance(adata.layers['Mu'], np.ndarray):
            self.Mu = adata.layers['Mu']
        else:
            self.Mu = np.array(adata.layers['Mu'].todense())

        self.u_v, self.s_v = kernel.velocity()

        self.variational = variational

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lambda_kld = lambda_kld


        # self.batch_id = onehot_batchid(get_batch_id(adata.obs, batch_key)) if batch_key is not None else None

        self.validation_size = validation_size


        seed_everything(seed)

        setattr(self, "model1", AutoEncoder(
            n_genes=self.Ms.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim,
            activation_fn=activation_fn, variational=variational, lipschitz_bound=lipschitz_bound,
        ))
        print("Created model1!")



    def fit(self, max_epochs=200, early_stopping=True, patience=20, min_delta=0.01, use_logger=False, project_name=None, devices=[1]):
        
        if use_logger:
            print("Please use wandb to visualize the metrics!")
            self.wandb_logger = WandbLogger(project=project_name)
        else:
            self.wandb_logger = None

        if early_stopping:
            if self.validation_size is not None:
                early_stop_callback = EarlyStopping(monitor="val_loss",
                                                    min_delta=min_delta, patience=patience,
                                                    verbose=False, mode="min")

            else:
                early_stop_callback = EarlyStopping(monitor="train_loss",
                                                    min_delta=min_delta, patience=patience,
                                                    verbose=False, mode="min")

        train_set = UnS_Dataset(self.Mu, self.Ms, self.u_v, self.s_v)
        if self.validation_size is not None:
            train_set_size = int(len(train_set) * (1 - self.validation_size))
            valid_set_size = len(train_set) - train_set_size
            train_set, val_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=len(val_set))
        else:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        callbacks = [EpochProgressBar()]

        if early_stopping:
            callbacks.append(early_stop_callback)
        else:
            pass

        trainer = pl.Trainer(accelerator="auto",
                             devices=devices,
                             max_epochs=max_epochs,
                             num_sanity_val_steps=1,
                             callbacks=callbacks,
                             logger=self.wandb_logger)

        if self.validation_size is not None:
            trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(model=self, train_dataloaders=train_loader)

    def validation_step(self, batch, batch_idx):
        u, s, u_v, s_v = batch

        uns = torch.concat([u, s], dim=1)

        if self.variational:
            mu, logvar, z = self.model1.encode(uns)
        else:
            z = self.model1.encode(uns)

        alpha, beta, gamma = self.model1.decode(z)

        u_v_pred = (alpha - beta * u)

        s_v_pred = (beta * u - gamma * s)  # (batch_size, n_feature)

        mse_loss = nn.functional.mse_loss(u_v_pred, u_v) + nn.functional.mse_loss(s_v_pred, s_v)

        if self.variational:
            kld = self.KLD_loss(mu, logvar)
            self.log("val_kld_loss", kld.mean())
            loss = mse_loss.mean() + kld.mean()

        else:
            loss = mse_loss.mean()

        self.log("val_loss", loss.mean())

        self.log("val_velocity_loss", mse_loss.mean())
        return loss

    def training_step(self, batch, batch_idx):
        u, s, u_v, s_v = batch
        
        uns = torch.concat([u, s], dim=1)

        if self.variational:
            mu, logvar, z = self.model1.encode(uns)
        else:
            z = self.model1.encode(uns)

        alpha, beta, gamma = self.model1.decode(z)

        u_v_pred = (alpha - beta * u)

        s_v_pred = (beta * u - gamma * s) # (batch_size, n_feature)

        mse_loss = nn.functional.mse_loss(u_v_pred, u_v) + nn.functional.mse_loss(s_v_pred, s_v)

        if self.variational:
            kld = self.KLD_loss(mu, logvar)
            self.log("kld_loss", kld.mean())
            loss = self.lambda_kld*kld.mean() + mse_loss

        else:
            loss = mse_loss
        
        self.log("train_loss", loss.mean())
        self.log("velocity_loss", mse_loss)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        u, s, u_v, s_v = batch

        uns = torch.concat([u, s], dim=1)

        if self.variational:
            mu, logvar, z = self.model1.encode(uns)
        else:
            z = self.model1.encode(uns)

        alpha, beta, gamma = self.model1.decode(z)

        u_v_pred = (alpha - beta * u)

        s_v_pred = (beta * u - gamma * s)  # (batch_size, n_feature)

        mse_loss = nn.functional.mse_loss(u_v_pred, u_v) + nn.functional.mse_loss(s_v_pred, s_v)

        return alpha, beta, gamma, u_v_pred, s_v_pred, mse_loss, z

    def KLD_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def predict(self,):
        train_set = UnS_Dataset(self.Mu, self.Ms, self.u_v, self.s_v)
        train_loader = DataLoader(train_set, batch_size=len(train_set))
        trainer = pl.Trainer(accelerator="auto")
        alpha, beta, gamma, u_v_pred, s_v_pred, mse_loss, z = trainer.predict(self, train_loader)[0]
        return alpha.numpy(), beta.numpy(), gamma.numpy(), s_v_pred.numpy(), mse_loss.numpy(), z.numpy()


class BRV:
    """
    Compute expected velocity of spliced or unspliced expression based on the pre-set pseudotime value.
    """
    def __init__(self,
                 adata,
                 pt_key: str = None,
                 knn_key: str = 'distances',
                 ):
        super().__init__()

        if isinstance(adata.layers['Ms'], np.ndarray):
            pass
        else:
            self.adata.layers['Ms'] = np.array(self.adata.layers['Ms'].todense())

        if isinstance(adata.layers['Mu'], np.ndarray):
            pass
        else:
            self.adata.layers['Mu'] = np.array(self.adata.layers['Mu'].todense())

        self.spliced = adata.layers['Ms']
        self.unspliced = adata.layers['Mu']

        self.pt = np.array(adata.obs[pt_key]).reshape(-1, 1)
        self.knn = adata.obsp[knn_key]
        print("Computing SNN based on KNN")
        self.compute_snn()

    def compute_snn(self):
        """
        Compute shared-nearest neighbor graph, only mutual neighbor in KNN graph is kept.
        """
        K = self.knn.todense()
        self.mask = (K != 0) & (K == K.T)
        self.snn = np.where(self.mask, K, 0)
        if (self.snn > 0).sum(axis=1).min() >= 0:
            print("Some cells don't have neighbors, maybe increase the numebr of nearest neighbor!")

    def velocity(self, graph_type='SNN'):
        if graph_type == 'KNN':
            adj = self.knn.todense()
        elif graph_type == 'SNN':
            adj = self.snn
        else:
            raise ValueError('Please choose only "KNN" or "SNN" as graph_type')
        self.dt = self.displacement_calculation(self.pt, adj)
        self.d_spliced = self.displacement_calculation(self.spliced, adj)
        self.d_unspliced = self.displacement_calculation(self.unspliced, adj)

        return self.expected_velocity(self.d_spliced, self.dt), self.expected_velocity(self.d_unspliced, self.dt)

    def expected_velocity(self,
                          d: np.ndarray = None,
                          dt: np.ndarray = None) -> np.ndarray:
        """
        calculate the expected velocity projection on neighbors displacement vector.
        assume we have all the displacement vector for each point i, d_{i,j}, j = 1, 2, ..., k
        assume we have all the dt_{i,j}, j = 1, 2, ..., k for each point i to describe the time difference
        we expect that the volocity v_i projected to d_{i,j} satisfy the basic rule:
        v_i * cos \theta * dt_{i,j} = d_{i,j}
        therefore, we can solve v_i using the least square solution
        :param d: shape as [n, k, m], displacement vector for point i to point j where j is the neighbor of point i
        :param dt: shape as [n, k, 1], time difference for point i to point j where j is the neighbor of point i
        :return: estimated velocity vector, shape as [n, m]
        """
        # n, k, m = d.shape
        # estimated_velocity = np.zeros((n, m))
        # for i in range(n):
        #     # Avoid division by zero for zero vectors
        #     norms = np.linalg.norm(d[i], axis=1, keepdims=True)
        #     norms[norms == 0] = 1
        #     unit_vectors = d[i] / norms
        #     # Set up the linear system
        #     A = unit_vectors * dt[i]  # Shape: (k, m)
        #     b = np.linalg.norm(d[i], axis=1)  # Shape: (k)
        #     # Solve the linear system in a least squares sense
        #     v_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        #     estimated_velocity[i] = v_i

        n = len(d)
        _, m = d[0].shape
        estimated_velocity = np.zeros((n, m))
        for i in range(n):
            unit_vectors = d[i] / np.linalg.norm(d[i], axis=1, keepdims=True)
            # Set up the linear system
            # Ax = b
            A = unit_vectors * dt[i]  # Shape: (k, m)
            b = np.linalg.norm(d[i], axis=1)  # Shape: (k)
            # Solve the linear system in a least squares sense
            # Note: np.linalg.lstsq returns several values; we only need the first one
            v_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            estimated_velocity[i] = v_i

        return estimated_velocity

    # def displacement_calculation(self,
    #                              X: np.ndarray,
    #                              adj: np.ndarray) -> np.ndarray:
    #     """
    #     calculate the displacement based on a graph,
    #     for each i, we enumerate its neighbor, then calculate displacement as X[j] - X[i],
    #     if this point doesn't have any neighbors, the displacement is a zero vector
    #     :param X: feature vector, shape as [n, m]
    #     :param knn: KNN matrix, it has K neighbors for each point, shape as [n, n]
    #     :return: displacement of the feature, shape as [n, k, m]
    #     """
    #     n, m = X.shape
    #     max_neighbors = np.max(np.sum(adj != 0, axis=1))
    #     displacement = np.zeros((n, max_neighbors, m))
    #
    #     for i in range(n):
    #         neighbors = np.where(adj[i] != 0)[0]
    #         for idx, j in enumerate(neighbors):
    #             displacement[i, idx] = X[j] - X[i]
    #
    #     return displacement

    def displacement_calculation(self,
                                 X: np.ndarray,
                                 adj: np.ndarray) -> np.ndarray:
        """
        calculate the displacement based on a graph,
        for each i, we enumerate its neighbor, then calculate displacement as X[j] - X[i],
        if this point doesn't have any neighbors, the displacement is a zero vector
        :param X: feature vector, shape as [n, m]
        :param adj: adjacency matrix, it has Ki neighbors for each point, shape as [n, n]
        :return: displacement of the feature, a list, each element has shape [Ki, m]
        """
        n, m = X.shape
        displacement = []

        for i in range(n):
            neighbors = np.where(adj[i] != 0)[1]  # get the indices of neighbors
            if len(neighbors) == 0:
                displacement.append(np.zeros((1, m)))  # if no neighbors, add a zero vector
            else:
                displace_vectors = [X[j] - X[i] for j in neighbors]
                displacement.append(np.vstack(displace_vectors))

        return displacement


class VelocityKernel:
    def __init__(self):
        pass

    def velocity_calculation(self):
        pass

    def scvelo_helper(self):
        pass

    def dynamo_helper(self):
        pass

    def velocyto_helper(self):
        pass
    def velocity(self):
        pass


import numpy as np



class Kernel:
    """
    Compute expected velocity of spliced or unspliced expression based on the pre-set pseudotime value.
    """
    def __init__(self,
                 adata,
                 pt_key: str = None,
                 knn_key: str = 'distances',
                 ):
        super().__init__()

        if isinstance(adata.layers['Ms'], np.ndarray):
            pass
        else:
            self.adata.layers['Ms'] = np.array(self.adata.layers['Ms'].todense())

        if isinstance(adata.layers['Mu'], np.ndarray):
            pass
        else:
            self.adata.layers['Mu'] = np.array(self.adata.layers['Mu'].todense())

        self.spliced = adata.layers['Ms']
        self.unspliced = adata.layers['Mu']

        self.pt = np.array(adata.obs[pt_key]).reshape(-1, 1)
        self.knn = adata.obsp[knn_key]
        print("Computing SNN based on KNN")
        self.compute_snn()

    def compute_snn(self):
        """
        Compute shared-nearest neighbor graph, only mutual neighbor in KNN graph is kept.
        """
        K = self.knn.todense()
        self.mask = (K != 0) & (K == K.T)
        self.snn = np.where(self.mask, K, 0)
        if (self.snn > 0).sum(axis=1).min() >= 0:
            print("Some cells don't have neighbors, maybe increase the numebr of nearest neighbor!")

    def displacement_calculation(self, X: np.ndarray,
                                 knn: np.ndarray) -> np.ndarray:
        # Ensure the input arrays are of correct shapes
        assert X.ndim == 2 and knn.ndim == 2, "Input arrays have incorrect shapes"

        # Find the non-zero entries in the KNN matrix
        non_zero_entries = np.nonzero(knn)
        unique_points = np.unique(non_zero_entries[0])

        # Initialize a list to store the displacement vectors
        displacement_list = []

        # Iterate over each point with non-zero entries in the KNN matrix
        for i in unique_points:
            # Get the indices of the non-zero entries for point i
            neighbor_indices = non_zero_entries[1][non_zero_entries[0] == i]
            # Compute the displacement vectors for point i and its neighbors
            displacement_i = X[neighbor_indices] - X[i]
            displacement_list.append(displacement_i)

        # Convert the list of displacement arrays to a single NumPy array
        displacement = np.array(displacement_list)

        return displacement

    def velocity_correction(self, v, mask, deltas):
        """
        v: original velocity
        mask: the mask show the SNN
        delta: the delta_{ij}
        """
        # check symmetric
        assert np.allclose(mask, mask.T)
        # check no diagonal
        assert np.allclose(np.diag(mask), np.zeros(mask.shape[0]))
        kis = mask.astype(int).sum(axis=1)
        N = mask.shape[0]
        nzrows = (kis != 0).sum()
        dec_ind = np.cumsum(kis != 0) - 1

        def make_inc_ind(dec_ind):
            inc_ind = np.zeros(nzrows, dtype=int)
            for i, d in enumerate(dec_ind):
                inc_ind[d] = i
            return inc_ind

        inc_ind = make_inc_ind(dec_ind)
        # check nan
        assert not any(np.any(np.isnan(d)) for d in deltas)
        #  assert not np.any(np.isnan(v))  # fail
        v = np.nan_to_num(v)
        assert not np.any(np.isnan(v))

        # transform dense_phi to flatten_v
        ukis = np.triu(mask).sum(axis=1)
        dim_dense_phi = ukis.sum()

        def get_dense_i_to_ij(mask):
            dense_i_to_ij = np.zeros((dim_dense_phi, 2), dtype=int)
            t = 0
            for i in range(mask.shape[0]):
                for j in range(i, mask.shape[1]):
                    if mask[i, j] == 1:
                        dense_i_to_ij[t] = [i, j]
                        t += 1
            assert t == dim_dense_phi
            return dense_i_to_ij

        dense_i_to_ij = get_dense_i_to_ij(mask)

        def get_ij_to_ik(mask):
            ij_to_ik = {}
            for i in range(mask.shape[0]):
                k = 0
                for j in range(0, mask.shape[1]):
                    if mask[i, j] == 1:
                        ij_to_ik[(i, j)] = k
                        k += 1
            return ij_to_ik

        ij_to_ik = get_ij_to_ik(mask)

        def dense_phi_to_vs(dphi, ukis, deltas):
            di = 0
            vs = np.zeros((N, deltas[0].shape[1]))
            for ki in ukis:
                if ki == 0:
                    continue
                else:
                    for _ in range(ki):
                        phi = dphi[di]
                        i, j = dense_i_to_ij[di]
                        k = ij_to_ik[(i, j)]
                        vs[i] += phi * deltas[dec_ind[i]][k]
                        i, j = j, i
                        k = ij_to_ik[(i, j)]
                        vs[i] += phi * deltas[dec_ind[i]][k]
                        di += 1
            return vs

        def transpose_vs_to_dense_phi(vs, ukis, deltas):
            dphi = np.zeros(dim_dense_phi)
            di = 0
            for ki in ukis:
                if ki == 0:
                    continue
                else:
                    for _ in range(ki):
                        i, j = dense_i_to_ij[di]
                        k = ij_to_ik[(i, j)]
                        dphi[di] += vs[i] @ deltas[dec_ind[i]][k]
                        i, j = j, i
                        k = ij_to_ik[(i, j)]
                        dphi[di] += vs[i] @ deltas[dec_ind[i]][k]
                        di += 1
            return dphi

        def dense_phi_to_fv(dphi, ukis, deltas):
            vs = dense_phi_to_vs(dphi, ukis, deltas)
            #  print("one forward pass")
            return vs.flatten()

        def transpose_fv_to_dense_phi(fv, ukis, deltas):
            vs = fv.reshape(N, deltas[0].shape[1])
            #  print("one backward pass")
            return transpose_vs_to_dense_phi(vs, ukis, deltas)

        def dense_phi_to_phi(dphi, ukis, kis):
            nphi = [np.zeros(kis[inc_ind[i]]) for i in range(nzrows)]
            di = 0
            for ki in ukis:
                if ki == 0:
                    continue
                else:
                    for _ in range(ki):
                        phi = dphi[di]
                        i, j = dense_i_to_ij[di]
                        k = ij_to_ik[(i, j)]
                        nphi[dec_ind[i]][k] = phi
                        i, j = j, i
                        k = ij_to_ik[(i, j)]
                        nphi[dec_ind[i]][k] = phi
            return nphi

        A = scipy.sparse.linalg.LinearOperator(
            (v.size, dim_dense_phi),
            matvec=lambda x: dense_phi_to_fv(x, ukis, deltas),
            rmatvec=lambda x: transpose_fv_to_dense_phi(x, ukis, deltas),
        )
        b = v.flatten()

        sol = scipy.sparse.linalg.lsmr(A, b, show=True)

        dphi = sol[0]

        nphi = dense_phi_to_phi(dphi, ukis, kis)
        v_recon = dense_phi_to_vs(dphi, ukis, deltas)

        return v_recon









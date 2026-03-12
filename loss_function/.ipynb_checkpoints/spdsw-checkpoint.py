import torch

import numpy as np
import torch.nn.functional as F

from geoopt import linalg

def isometric_vector_to_symmetric_matrix_torch(vec_batch, n):
    """
    将一个批次的向量等距映射到一批 n x n 的对称矩阵。
    这是高效的向量化版本。

    :param vec_batch: 输入的向量批次，形状为 (batch_size, n*(n+1)/2)。
    :param n: 目标矩阵的维度。
    :return: 一批 n x n 的对称矩阵，形状为 (batch_size, n, n)。
    """
    batch_size = vec_batch.shape[0]
    expected_len = n * (n + 1) // 2
    if vec_batch.shape[1] != expected_len:
        raise ValueError(f"输入向量的长度应为 {expected_len}，但实际为 {vec_batch.shape[1]}")

    # 创建一个 (batch_size, n, n) 的零矩阵批次
    matrix_batch = torch.zeros((batch_size, n, n), dtype=vec_batch.dtype, device=vec_batch.device)
    
    # 获取上三角索引
    iu_indices = torch.triu_indices(n, n, device=vec_batch.device)
    
    # --- 关键修改：缩放向量现在也是批处理的 ---
    scaling = torch.ones(expected_len, dtype=vec_batch.dtype, device=vec_batch.device)
    off_diag_mask = iu_indices[0] != iu_indices[1]
    scaling[off_diag_mask] = 1.0 / torch.sqrt(torch.tensor(2.0))
    # scaling 的形状是 (d,)
    
    # 应用缩放。利用广播机制，(bs, d) * (d,) -> (bs, d)
    scaled_vecs = vec_batch * scaling
    
    # --- 关键修改：使用高级索引来填充批次 ---
    # matrix_batch[:, iu_indices[0], iu_indices[1]] 是一种高级索引
    # 它会选择批次中每个矩阵的上三角位置
    matrix_batch[:, iu_indices[0], iu_indices[1]] = scaled_vecs
    
    # 完成对称矩阵
    # 需要获取转置，注意维度的变化 (1, 2) -> (2, 1)
    matrix_batch_transpose = matrix_batch.transpose(1, 2)
    # 获取对角线，需要注意保持维度
    diag = torch.diagonal(matrix_batch, dim1=-2, dim2=-1)
    
    matrix_batch = matrix_batch + matrix_batch_transpose - torch.diag_embed(diag)
    
    return matrix_batch


def power_matrix(pred,a):
    ###警告，需要check shape
    
    if isinstance(a, int) and a >= 0:
        result = torch.matrix_power(pred, a)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(pred)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        eigenvalues_power = eigenvalues ** a
        eigenvalues_diag = torch.diag_embed(eigenvalues_power)
        
        result = torch.bmm(
            torch.bmm(eigenvectors, eigenvalues_diag),
            torch.linalg.inv(eigenvectors)
        )
        result = result.real
    
   
    return result


def get_spdsw(pred,metric,a=1,alpha=0.05,beta=0.05):
    d= pred.shape[1]
    pred_power = power_matrix(pred,a)
    
    if metric == "ecm":
        manifold =CorEuclideanCholeskyMetric(d)
    elif metric == "lecm":
        manifold =CorLogEuclideanCholeskyMetric(d)
    elif metric == "olm":
        manifold = CorOffLogMetric(d)
    elif metric == "lsm":
        manifold = CorLogScaledMetric(d)
    corr= manifold.covariance_to_correlation(pred_power)
    Lp = manifold.deformation(corr)
    return alpha*Lp+beta*linalg.sym_logm(pred)

class SPDSW:
    """
        Class for computing SPDSW distance and embedding

        Parameters
        ----------
        shape_X : int
            dim projections
        num_projections : int
            Number of projections
        num_ts : int
            Number of timestamps for quantiles, default 20
        device : str
            Device for computations, default None
        dtype : type
            Data type, default torch.float
        random_state : int
            Seed, default 123456
        sampling : str
            Sampling type
                - "spdsw": symetric matrices + geodesic projection
                - "logsw": unit norm matrices + geodesic projection
                - "sw": unit norm matrices + euclidean projection
            Default "spdsw"
        """


    def __init__(
        self,
        shape_X,
        num_projections,
        device=None,
        dtype=torch.float,
        random_state=123456,
        sampling="lsm",
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if sampling not in ["lsm", "olm", "ecm", "lecm"]:
            raise Exception(
                "'sampling' should be in ['lsm', 'olm', 'ecm', 'lecm']"
            )

        self.generate_projections(
            shape_X, num_projections, 
            device, dtype, random_state
        )

        self.sampling = sampling
       
    def generate_projections(self, shape_X, num_projections, 
                             device, dtype, random_state):
        """
        Generate projections for sampling

        Parameters
        ----------
        shape_X : int
            dim projections
        num_projections : int
            Number of projections
        device : str
            Device for computations
        dtype : type
            Data type
        random_state : int
            Seed
        sampling : str
            Sampling type
                - "spdsw": symetric matrices + geodesic projection
                - "logsw": unit norm matrices + geodesic projection
                - "sw": unit norm matrices + euclidean projection
        """

        rng = np.random.default_rng(random_state)

      

        self.A = torch.tensor(
            rng.normal(size=(num_projections, shape_X, shape_X)),
            dtype=dtype,
            device=device
        )

        self.A /= torch.norm(self.A, dim=(1, 2), keepdim=True)

    def guass_distr(self,shape_X, device, dtype, random_state, batch_size=None): ###警告:batch_size维度需要修改
        rng = np.random.default_rng(random_state)
        self.B = torch.tensor(rng.normal(size=(batch_size,shape_X*(shape_X+1)//2)),dtype=dtype,device=device)
        self.B =  isometric_vector_to_symmetric_matrix_torch(self.B, shape_X)
        return self.B
        
    

    def emd1D(self, u_values, v_values, u_weights=None, v_weights=None, p=1):
        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

        if v_weights is None:
            v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

        # Sort
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

        # Compute CDF
        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

        u_index = torch.searchsorted(u_cdf, cdf_axis)
        v_index = torch.searchsorted(v_cdf, cdf_axis)

        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

        cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

        if p == 1:
            return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
        if p == 2:
            return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)

        return torch.sum(
            delta * torch.pow(torch.abs(u_icdf - v_icdf), p),
            axis=-1
        )

    def spdsw(self, pred,alpha,beta, u_weights=None, v_weights=None, p=2):
        """
            Parameters:
            Xs: ndarray, shape (n_batch, d, d)
                Samples in the source domain
            Xt: ndarray, shape (m_batch, d, d)
                Samples in the target domain
            device: str
            p: float
                Power of SW. Need to be >= 1.
        """
        n, _,_,  = pred.shape ####雷达信号(bs,C,C)复数域
        

        n_proj, d, _ = self.A.shape

    
        prod_Xs = (self.A * get_spdsw(pred,metric=self.sampling,alpha=alpha,beta=beta)).reshape(n, n_proj, -1)
        prod_Xt = (self.A * self.B).reshape(n, n_proj, -1)

        Xps = prod_Xs.sum(-1)
        Xpt = prod_Xt.sum(-1)

        return torch.mean(
            self.emd1D(Xps.T, Xpt.T, u_weights, v_weights, p)
        )

    def get_quantiles(self, x, ts, weights=None):
        """
            Inputs:
            - x: 1D values, size: n_projs * n_batch
            - ts: points at which to evaluate the quantile
        """
        n_projs, n_batch = x.shape

        if weights is None:
            X_weights = torch.full(
                (n_batch,), 1/n_batch, dtype=x.dtype, device=x.device
            )
            X_values, X_sorter = torch.sort(x, -1)
            X_weights = X_weights[..., X_sorter]

        X_cdf = torch.cumsum(X_weights, -1)

        X_index = torch.searchsorted(X_cdf, ts.repeat(n_projs, 1))
        X_icdf = torch.gather(X_values, -1, X_index.clip(0, n_batch-1))

        return X_icdf

    
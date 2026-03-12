import torch
import numpy as np
import torch.nn.functional as F
from .CorMatrix import (
    CorEuclideanCholeskyMetric,
    CorLogEuclideanCholeskyMetric,
    CorOffLogMetric,
    CorLogScaledMetric
)
from geoopt import linalg


def isometric_vector_to_symmetric_matrix_torch(vec_batch, n):
    """
    将一个批次的向量等距映射到一批 n x n 的对称矩阵。
    这是高效的向量化版本。

    :param vec_batch: 输入的向量批次,形状为 (batch_size, n*(n+1)/2)。
    :param n: 目标矩阵的维度。
    :return: 一批 n x n 的对称矩阵,形状为 (batch_size, n, n)。
    """
    batch_size = vec_batch.shape[0]
    expected_len = n * (n + 1) // 2
    if vec_batch.shape[1] != expected_len:
        raise ValueError(f"输入向量的长度应为 {expected_len},但实际为 {vec_batch.shape[1]}")

    # 创建一个 (batch_size, n, n) 的零矩阵批次
    matrix_batch = torch.zeros((batch_size, n, n), dtype=vec_batch.dtype, device=vec_batch.device)
    
    # 获取上三角索引
    iu_indices = torch.triu_indices(n, n, device=vec_batch.device)
    
    # 缩放向量现在也是批处理的
    scaling = torch.ones(expected_len, dtype=vec_batch.dtype, device=vec_batch.device)
    off_diag_mask = iu_indices[0] != iu_indices[1]
    scaling[off_diag_mask] = 1.0 / torch.sqrt(torch.tensor(2.0))
    
    # 应用缩放。利用广播机制,(bs, d) * (d,) -> (bs, d)
    scaled_vecs = vec_batch * scaling
    
    # 使用高级索引来填充批次
    matrix_batch[:, iu_indices[0], iu_indices[1]] = scaled_vecs
    
    # 完成对称矩阵
    matrix_batch_transpose = matrix_batch.transpose(1, 2)
    diag = torch.diagonal(matrix_batch, dim1=-2, dim2=-1)
    
    matrix_batch = matrix_batch + matrix_batch_transpose - torch.diag_embed(diag)
    
    return matrix_batch


# --- FIX: 支持批量维度，并避免 bmm 的 3D 限制 ---

def power_matrix(pred, a):
    pred = 0.5 * (pred + pred.transpose(-1, -2))
    if isinstance(a, (int, float)):
        a_float = float(a)
        if a_float >= 0 and abs(a_float - round(a_float)) < 1e-12:
            return torch.matrix_power(pred, int(round(a_float)))
    eigenvalues, eigenvectors = torch.linalg.eigh(pred)
    eigenvalues = eigenvalues.clamp(min=1e-10)
    eigenvalues_power = eigenvalues ** a
    eigenvalues_diag = torch.diag_embed(eigenvalues_power)
    result = eigenvectors @ eigenvalues_diag @ eigenvectors.transpose(-1, -2)
    return result


def get_spdsw(pred, metric, power=1, lamda=0.05, gamma=0.05):
    """
    计算SPD流形上的Sliced Wasserstein变换
    
    Parameters
    ----------
    pred : torch.Tensor
        预测的SPD矩阵,形状为 (batch_size, d, d) 或 (..., d, d)
    metric : str
        使用的度量类型: "ecm", "lecm", "olm", "lsm"
    power : float
        矩阵幂次
    lamda : float
        正则化参数lambda
    gamma : float
        正则化参数gamma
        
    Returns
    -------
    torch.Tensor
        变换后的矩阵
    """
    # 使用最后两个维度作为矩阵维度，更健壮
    d = pred.shape[-1]
    pred_power = power_matrix(pred, a=power)
    
    if metric == "ecm":
        manifold = CorEuclideanCholeskyMetric(d)
    elif metric == "lecm":
        manifold = CorLogEuclideanCholeskyMetric(d)
    elif metric == "olm":
        manifold = CorOffLogMetric(d)
    elif metric == "lsm":
        manifold = CorLogScaledMetric(d,max_iter=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    corr = manifold.covariance_to_correlation(pred_power)
    Lp = manifold.deformation(corr)
    
    # 返回 Lp（用于后续 spdsw(Lp) 组合计算）
    return Lp


class SPDSW:
    """
    Class for computing SPDSW distance and embedding

    Parameters
    ----------
    shape_X : int
        dim projections
    num_projections : int
        Number of projections
    device : str
        Device for computations, default None
    dtype : type
        Data type, default torch.float
    random_state : int
        Seed, default 123456
    sampling : str
        Sampling type
            - "lsm": Log-Scaled Metric
            - "olm": Off-Log Metric
            - "ecm": Euclidean Cholesky Metric
            - "lecm": Log-Euclidean Cholesky Metric
        Default "lsm"
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

        self.generate_projections(shape_X, num_projections, device, dtype, random_state)
        
        self.sampling = sampling
        self.device = device
        self.dtype = dtype
        self.shape_X = shape_X
        self.num_projections = num_projections
        self.random_state = random_state
        
        # 初始化 B 相关属性
        self.B = None
        self._B_batch_size = None
        # 初始化 C 相关属性
        self.C = None
        self._C_batch_size = None

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
        """
        rng = np.random.default_rng(random_state)

        self.A = torch.tensor(
            rng.normal(size=(num_projections, shape_X, shape_X)),
            dtype=dtype,
            device=device
        )

        self.A /= torch.norm(self.A, dim=(1, 2), keepdim=True)

    def guass_distr(self, shape_X, device, dtype, random_state, batch_size=None):
        """
        Sample L-2-norm row 0 in shape_X and ensure that they have a sum of zero
        Only valid for shape_X >= 2.

        Returns
        -------
        torch.Tensor
            B 的批次矩阵，形状为 (batch_size, shape_X, shape_X)
        """
        if batch_size is None:
            raise ValueError("'batch_size' must be provided for sampling B")
        n = int(shape_X)
        # 随机数发生器
        generator = torch.Generator(device=device).manual_seed(random_state)
        sampling = str(getattr(self, 'sampling', 'lsm')).lower()
        
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=dtype, device=device))
        sqrt6 = torch.sqrt(torch.tensor(6.0, dtype=dtype, device=device))
        
        if sampling == "ecm":
            # LT^1（单位对角）下三角，长度 d = n(n-1)/2
            off_len = n * (n - 1) // 2
            vec = torch.randn(
                batch_size,
                off_len,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            B = torch.zeros((batch_size, n, n), dtype=dtype, device=device)
            il = torch.tril_indices(n, n, offset=-1, device=device)
            B[:, il[0], il[1]] = vec
            # 设置单位对角
            idx = torch.arange(n, device=device)
            B[:, idx, idx] = 1.0
        elif sampling == "lecm":
            # LT^0（严格下三角，diag=0），长度 d = n(n-1)/2
            off_len = n * (n - 1) // 2
            vec = torch.randn(
                batch_size,
                off_len,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            B = torch.zeros((batch_size, n, n), dtype=dtype, device=device)
            il = torch.tril_indices(n, n, offset=-1, device=device)
            B[:, il[0], il[1]] = vec
            # 对角保持为 0
        elif sampling == "olm":
            # 对称空心（Hol），仅非对角，长度 d = n(n-1)/2，非对角按 1/√2 缩放
            off_len = n * (n - 1) // 2
            vec = torch.randn(
                batch_size,
                off_len,
                dtype=dtype,
                device=device,
                generator=generator,
            ) / sqrt2
            iu_off = torch.triu_indices(n, n, offset=1, device=device)
            B_half = torch.zeros((batch_size, n, n), dtype=dtype, device=device)
            B_half[:, iu_off[0], iu_off[1]] = vec
            B = B_half + B_half.transpose(1, 2)  # 对称
            # 对角保持为 0
        elif sampling == "lsm":
            # Row_0(n)：正交归一基（Eq.119）构造
            # 顶部 m×m（m=n-1）上三角（含对角）使用 d=m(m+1)/2 = n(n-1)/2 个系数
            m = n - 1
            vec = torch.randn(
                batch_size,
                m * (m + 1) // 2,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            B = torch.zeros((batch_size, n, n), dtype=dtype, device=device)
            iu_block = torch.triu_indices(m, m, device=device)
            # 构造缩放：对角 1/√3，非对角 1/√6
            block_scaling = torch.empty(iu_block.shape[1], dtype=dtype, device=device)
            diag_mask = (iu_block[0] == iu_block[1])
            block_scaling[diag_mask] = 1.0 / sqrt3
            block_scaling[~diag_mask] = 1.0 / sqrt6
            scaled_vec = vec * block_scaling
            # 填充顶部 m×m 上三角
            B[:, iu_block[0], iu_block[1]] = scaled_vec
            # 使顶部 m×m 块对称（保持原对角）
            top_left = B[:, :m, :m]
            B[:, :m, :m] = top_left + top_left.transpose(1, 2) - torch.diag_embed(
                torch.diagonal(top_left, dim1=-2, dim2=-1)
            )
            # 由行和为 0 约束计算最后一列/行
            row_sums = B[:, :m, :m].sum(dim=2)  # 每行（0..m-1）和
            last_col = -row_sums  # 令每行和为 0
            B[:, :m, n - 1] = last_col
            B[:, n - 1, :m] = last_col
            # 最后一行的对角由其非对角和确定
            B[:, n - 1, n - 1] = -last_col.sum(dim=1)
        else:
            raise ValueError(f"Unknown sampling metric: {sampling}")
        
        self.B = B
        self._B_batch_size = batch_size
        return self.B

    def sample_C_isometric(self, shape_X, device, dtype, random_state, batch_size=None):
        """
        根据等距映射从 R^{d(d+1)/2} 采样生成一批对称矩阵 self.C。
        Frobenius 等距：对角不缩放，非对角缩放 1/√2。

        参数
        ------
        shape_X : int
            对称矩阵维度 d。
        device : str
            设备。
        dtype : torch.dtype
            数据类型。
        random_state : int
            随机种子。
        batch_size : int, optional
            批量大小（必须提供）。
        返回
        ------
        torch.Tensor
            self.C，形状为 (batch_size, d, d)
        """
        if batch_size is None:
            raise ValueError("'batch_size' must be provided for sampling C")
        n = int(shape_X)
        expected_len = n * (n + 1) // 2
        generator = torch.Generator(device=device).manual_seed(random_state)
        vec = torch.randn(
            batch_size,
            expected_len,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        C = isometric_vector_to_symmetric_matrix_torch(vec, n)
        self.C = C
        self._C_batch_size = batch_size
        return self.C


    def emd1D(self, u_values, v_values, u_weights=None, v_weights=None, p=1):
        """
        计算一维Earth Mover's Distance
        
        Parameters
        ----------
        u_values : torch.Tensor
            第一组值
        v_values : torch.Tensor
            第二组值
        u_weights : torch.Tensor, optional
            第一组权重
        v_weights : torch.Tensor, optional
            第二组权重
        p : float
            距离的幂次
            
        Returns
        -------
        torch.Tensor
            EMD距离
        """
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

    def spdsw_from_X(self, X, u_weights=None, v_weights=None, p=2):
        d = X.shape[-1]
        n_proj, a_d, _ = self.A.shape
        if a_d != d:
            raise ValueError(f"Projection dim mismatch: A uses d={a_d}, but X has d={d}")
        n = int(np.prod(X.shape[:-2]))
        X = X.reshape(n, d, d)
        if self.B is None or self._B_batch_size != n:
            self.guass_distr(
                shape_X=self.shape_X,
                device=self.device,
                dtype=self.dtype,
                random_state=self.random_state,
                batch_size=n,
            )
            self._B_batch_size = n
        prod_Xs = (self.A[None] * X[:, None]).reshape(n, n_proj, -1)
        prod_Xt = (self.A[None] * self.B[:, None]).reshape(n, n_proj, -1)
        Xps = prod_Xs.sum(-1)
        Xpt = prod_Xt.sum(-1)
        return torch.mean(self.emd1D(Xps.T, Xpt.T, u_weights, v_weights, p))

    def spdsw2_from_X(self, X, u_weights=None, v_weights=None, p=2):
        d = X.shape[-1]
        n_proj, a_d, _ = self.A.shape
        if a_d != d:
            raise ValueError(f"Projection dim mismatch: A uses d={a_d}, but X has d={d}")
        n = int(np.prod(X.shape[:-2]))
        X = X.reshape(n, d, d)
        if self.C is None or self._C_batch_size != n:
            self.sample_C_isometric(
                shape_X=self.shape_X,
                device=self.device,
                dtype=self.dtype,
                random_state=self.random_state,
                batch_size=n,
            )
            self._C_batch_size = n
        prod_Xs = (self.A[None] * X[:, None]).reshape(n, n_proj, -1)
        prod_Xt = (self.A[None] * self.C[:, None]).reshape(n, n_proj, -1)
        Xps = prod_Xs.sum(-1)
        Xpt = prod_Xt.sum(-1)
        return torch.mean(self.emd1D(Xps.T, Xpt.T, u_weights, v_weights, p))

    def spdsw2_logm(self, pred, u_weights=None, v_weights=None, p=2):
        X = linalg.sym_logm(pred)
        return self.spdsw2_from_X(X, u_weights, v_weights, p)

    def get_quantiles(self, x, ts, weights=None):
        """
        计算分位数
        
        Parameters
        ----------
        x : torch.Tensor
            输入值,形状为 (n_projs, n_batch)
        ts : torch.Tensor
            分位点
        weights : torch.Tensor, optional
            权重
            
        Returns
        -------
        torch.Tensor
            分位数值
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

    def spdsw_total(self, pred, power=1, p=2):
        d = pred.shape[-1]
        n_proj, a_d, _ = self.A.shape
        if a_d != d:
            raise ValueError(f"Projection dim mismatch: A uses d={a_d}, but pred has d={d}")
        # Lp（由 get_spdsw 返回）
        Lp = get_spdsw(pred, metric=self.sampling, power=power)
        # 组合：spdsw(Lp) + spdsw2(logm(pred))
        return self.spdsw_from_X(Lp, p=p) + self.spdsw2_logm(pred, p=p)
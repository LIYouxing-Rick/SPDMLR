import torch
import torch.nn as nn
import torch.nn.functional as F
from .CorMatrix import Correlation
from .CorMatrix import (
    CorEuclideanCholeskyMetric, 
    CorLogEuclideanCholeskyMetric, 
    CorOffLogMetric, 
    CorLogScaledMetric
)
from .spdsw import SPDSW, get_spdsw
from geoopt import linalg


class SWDloss(nn.Module):
    """
    Sliced Wasserstein Distance Loss（双项权重，可通过 args 开关）

    总损失：
      L_total = CE + λ1 * spdsw(LP) + λ2 * spdsw2(sym_logm(SPD))
    其中 λ1、λ2 为正值（以 log 形式优化），LP = get_spdsw(SPD)。
    可通过 args 开关控制是否启用两项。
    """

    def __init__(self, power=1.0, d=None, n_proj=50,
                 device='cuda', dtype=torch.double, seed=42, metric="lsm",
                 loss_lambda1_init: float = 1.0, loss_lambda2_init: float = 1.0,
                 use_lp: bool = True, use_logm: bool = True,
                 lambda_reg_coef: float = 0.0):
        super().__init__()

        if d is None:
            raise ValueError("Parameter 'd' (SPD matrix dimension) must be specified!")

        self.metric = metric
        self.power = power
        self.d = d
        self.device = device
        self.dtype = dtype
        self.lambda_reg_coef = float(lambda_reg_coef)

        # 初始化 SPDSW 计算器
        self.spdsw_calculator = SPDSW(
            shape_X=d,
            num_projections=n_proj,
            device=device,
            dtype=dtype,
            random_state=seed,
            sampling=metric
        )

        self.B_initialized = False

        # 追踪最近一次前向中的各项损失（便于外部记录）
        self.last_sw_loss = torch.tensor(0.0, device=device, dtype=dtype)
        self.last_ce_loss = torch.tensor(0.0, device=device, dtype=dtype)
        self.last_total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        self.last_sw_error = None
        # 两个可学习的 λ（以 log 形式存储，保证正性）
        init_lambda1 = torch.tensor(loss_lambda1_init, dtype=dtype, device=device)
        init_lambda2 = torch.tensor(loss_lambda2_init, dtype=dtype, device=device)
        self.log_lambda1 = nn.Parameter(torch.log(torch.clamp(init_lambda1, min=torch.tensor(1e-8, dtype=dtype, device=device))))
        self.log_lambda2 = nn.Parameter(torch.log(torch.clamp(init_lambda2, min=torch.tensor(1e-8, dtype=dtype, device=device))))
        self.last_lambda_values = None

        # 开关：是否启用两项
        self.use_lp = use_lp
        self.use_logm = use_logm
        self.log_lambda1.requires_grad = bool(self.use_lp)
        self.log_lambda2.requires_grad = bool(self.use_logm)

    def _initialize_B(self, batch_size):
        """动态初始化参考分布 B（仅在使用 spdsw_from_X 时需要，且初始化一次）"""
        if not self.B_initialized:
            self.spdsw_calculator.guass_distr(
                shape_X=self.d,
                device=self.device,
                dtype=self.dtype,
                random_state=42,
                batch_size=batch_size
            )
            self.B_initialized = True

    def get_weight_parameters(self):
        params = []
        if self.use_lp:
            params.append(self.log_lambda1)
        if self.use_logm:
            params.append(self.log_lambda2)
        return params

    def forward(self, pred, target, spd_features=None):
        """
        计算总损失 = CE + λ1 * spdsw(LP) + λ2 * spdsw2(sym_logm(SPD))
        """
        # 交叉熵损失（主任务）
        ce_loss = F.cross_entropy(pred, target, reduction='mean').to(self.dtype)

        # 初始化两项的值（当相应开关为 False 时保持为 0）
        sw1 = torch.tensor(0.0, device=pred.device, dtype=self.dtype)
        sw2 = torch.tensor(0.0, device=pred.device, dtype=self.dtype)

        if spd_features is not None and (self.use_lp or self.use_logm):
            batch_size = spd_features.shape[0]
            # 先计算 LP = get_spdsw(SPD)
            LP = get_spdsw(spd_features, metric=self.metric, power=self.power)

            # 第一项：spdsw(LP)
            if self.use_lp:
                self._initialize_B(batch_size)
                sw1 = self.spdsw_calculator.spdsw_from_X(LP, p=2).to(self.dtype)

            # 第二项：spdsw2(sym_logm(SPD))
            if self.use_logm:
                sw2 = self.spdsw_calculator.spdsw2_from_X(linalg.sym_logm(spd_features), p=2).to(self.dtype)

        # 两个可学习权重（正值）
        lambda_w1 = torch.exp(self.log_lambda1)
        lambda_w2 = torch.exp(self.log_lambda2)

        # 总损失：CE + λ1*SW1 + λ2*SW2
        total_loss = ce_loss + lambda_w1 * sw1 + lambda_w2 * sw2
        # 可选：对 λ 权重加入二次正则，抑制其过大
        if self.lambda_reg_coef > 0.0:
            total_loss = total_loss + self.lambda_reg_coef * (lambda_w1.square() + lambda_w2.square())

        # 记录最近一次损失与 λ
        try:
            self.last_sw_loss = (sw1 + sw2).detach()
            self.last_ce_loss = ce_loss.detach()
            self.last_total_loss = total_loss.detach()
            self.last_lambda_values = (lambda_w1.detach(), lambda_w2.detach())
        except Exception:
            pass

        return total_loss
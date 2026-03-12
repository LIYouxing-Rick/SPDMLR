import math
from typing import Optional, Union
import torch

import spdnets.modules as modules
import spdnets.batchnorm as bn
from spdnets.SPDMLR import SPDRMLR
from .base import DomainAdaptFineTuneableModel, FineTuneableModel, PatternInterpretableModel
from loss_function.loss import SWDloss


class TSMNetMLR(DomainAdaptFineTuneableModel, FineTuneableModel, PatternInterpretableModel):
    def __init__(self, temporal_filters, spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bnorm : Optional[str] = 'spdbn', 
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 classifier='LogEigMLR',
                 metric='SPDEuclideanMetric', power=1.0, alpha=1.0, beta=0.,
                 # SWD 配置项（可通过 Hydra 覆盖）
                 n_proj: int = 50,
                 loss_lambda1_init: float = 1.0,
                 loss_lambda2_init: float = 1.0,
                 swd_metric: str = "lsm",
                 use_lp: bool = True,
                 use_logm: bool = True,
                 # 单独的 SWD 惩罚项幂指数；默认不传时沿用分类器的 power
                 swd_power: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.bnorm_ = bnorm
        self.spd_device_ = torch.device('cpu')
        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(subspacedims*(subspacedims+1)/2)
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.temporal_filters_, kernel_size=(1,temp_cnn_kernel),
                            padding='same', padding_mode='reflect'),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
            torch.nn.Flatten(start_dim=2),
        ).to(self.device_)

        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )

        if self.bnorm_ == 'spdbn':
            self.spdbnorm = bn.AdaMomSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                          dispersion=self.bnorm_dispersion_, 
                                          learn_mean=False,learn_std=True,
                                          eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_)
        elif self.bnorm_ == 'brooks':
            self.spdbnorm = modules.BatchNormSPDBrooks((1,subspacedims,subspacedims), batchdim=0, dtype=torch.double, device=self.spd_device_)
        elif self.bnorm_ == 'tsbn':
            self.tsbnorm = bn.AdaMomBatchNorm((1, tsdim), batchdim=0, dispersion=self.bnorm_dispersion_, 
                                        eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_).to(self.device_)
        elif self.bnorm_ == 'spddsbn':
            self.spddsbnorm = bn.AdaMomDomainSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=torch.double, device=self.spd_device_)
        elif self.bnorm_ == 'tsdsbn':
            self.tsdsbnorm = bn.AdaMomDomainBatchNorm((1, tsdim), batchdim=0, 
                                domains=self.domains_,
                                dispersion=self.bnorm_dispersion_,
                                eta=1., eta_test=.1, dtype=torch.double).to(self.device_)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')

        self.spdnet = torch.nn.Sequential(
            modules.BiMap((1,self.spatial_filters_,subspacedims), dtype=torch.double, device=self.spd_device_),
            modules.ReEig(threshold=1e-4),
        )
        self.construct_classifier(classifier,subspacedims,self.nclasses_,metric,power,alpha,beta)

        # 分布型正则损失（SWD）：CE + λ1 * spdsw(LP) + λ2 * spdsw2(sym_logm(SPD))
        # 说明：
        # - d 设置为子空间维度 subspacedims（SPD 矩阵的阶数）
        # - 设备与 dtype 与 SPD 分支保持一致（CPU + double），避免类型/设备不一致
        # - λ1/λ2 作为可学习参数，自动纳入优化器（作为子模块参数）
        # 若未指定 swd_power，则与分类器的 power 保持一致以保证向后兼容
        _swd_power = power if swd_power is None else swd_power
        self.swd_loss = SWDloss(
            power=_swd_power,
            d=subspacedims,
            n_proj=n_proj,
            device=str(self.spd_device_),
            dtype=torch.double,
            metric=swd_metric,
            loss_lambda1_init=loss_lambda1_init,
            loss_lambda2_init=loss_lambda2_init,
            use_lp=use_lp,
            use_logm=use_logm,
            lambda_reg_coef=0.0,
        )


    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        if device is not None:
            self.device_ = device
            self.cnn.to(self.device_)
        return super().to(device=None, dtype=dtype, non_blocking=non_blocking)

    def forward(self, x, d, return_latent=True, return_prebn=False, return_postbn=False):
        out = ()
        h = self.cnn(x.to(device=self.device_)[:,None,...])
        C = self.cov_pooling(h).to(device=self.spd_device_, dtype=torch.double)
        l = self.spdnet(C)
        out += (l,) if return_prebn else ()
        l = self.spdbnorm(l) if hasattr(self, 'spdbnorm') else l
        l = self.spddsbnorm(l,d.to(device=self.spd_device_)) if hasattr(self, 'spddsbnorm') else l
        out += (l,) if return_postbn else ()
        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out

    def domainadapt_finetune(self, x, y, d, target_domains):
        if hasattr(self, 'spddsbnorm'):
            self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsdsbnorm'):
            self.tsdsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            for du in d.unique():
                self.forward(x[d==du], d[d==du])

        if hasattr(self, 'spddsbnorm'):
            self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsdsbnorm'):
            self.tsdsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)  

    def finetune(self, x, y, d):
        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d)

        if hasattr(self, 'spdbnorm'):
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
        if hasattr(self, 'tsbnorm'):
            self.tsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

    def compute_patterns(self, x, y, d):
        pass

    def calculate_objective(self, model_pred, y_true, model_inp=None):
        """
        将损失改为：L_total = CE + λ1 * spdsw(LP) + λ2 * spdsw2(sym_logm(SPD))
        其中：
        - CE 为分类交叉熵；
        - 默认情况下，SWDloss 在 BN 之前的 SPD 矩阵上操作（pre-BN）。
          若无法从输入批次重建 pre-BN 特征，则回退到预测中携带的 SPD 特征。
        - λ1/λ2 为可学习正权重，来自 self.swd_loss 中的参数。
        """
        if isinstance(model_pred, (list, tuple)):
            y_class_hat = model_pred[0]
            spd_features = model_pred[1] if len(model_pred) > 1 else None
        else:
            y_class_hat = model_pred
            spd_features = None

        # 确保标签与预测在同一设备
        y_true = y_true.to(y_class_hat.device)

        # 优先使用 BN 前的 SPD 特征：从原始输入重建 pre-BN 矩阵
        prebn_features = None
        try:
            if isinstance(model_inp, dict) and ('x' in model_inp) and ('d' in model_inp):
                x_inp = model_inp['x']
                d_inp = model_inp['d']
                # 通过 CNN -> 协方差池化 -> BiMap + ReEig 获得 BN 前 SPD 特征
                h = self.cnn(x_inp.to(device=self.device_)[:, None, ...])
                C = self.cov_pooling(h).to(device=self.spd_device_, dtype=torch.double)
                prebn_features = self.spdnet(C)
        except Exception:
            # 保持容错：一旦失败，回退到预测中提供的特征
            prebn_features = None

        # 使用 SWDloss 组合总损失（优先 pre-BN，其次回退到预测携带的特征）
        total_loss = self.swd_loss(y_class_hat, y_true, prebn_features if prebn_features is not None else spd_features)
        return total_loss

    def construct_classifier(self,classifier,subspacedims,nclasses_,metric,power,alpha,beta):
        if classifier=='SPDMLR':
            self.classifier = torch.nn.Sequential(
                modules.UnsqueezeLayer(1),
                SPDRMLR(n=subspacedims,c=nclasses_,metric=metric,power=power,alpha=alpha,beta=beta).to(self.spd_device_).double()
                )
        elif classifier=='LogEigMLR':
            tsdim = int(subspacedims * (subspacedims + 1) / 2)
            self.classifier = torch.nn.Sequential(
                modules.LogEig(subspacedims,tril=True),
                torch.nn.Linear(tsdim, self.nclasses_).double(),
            ).to(self.spd_device_)
        else:
            raise Exception(f'wrong clssifier {classifier}')



class CNNNet(DomainAdaptFineTuneableModel, FineTuneableModel):
    def __init__(self, temporal_filters, spatial_filters = 40,
                 temp_cnn_kernel = 25,
                 bnorm : Optional[str] = 'bn', 
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 **kwargs):
        super().__init__(**kwargs)

        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.bnorm_ = bnorm

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.temporal_filters_, kernel_size=(1,temp_cnn_kernel),
                            padding='same', padding_mode='reflect'),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
            torch.nn.Flatten(start_dim=2),
        ).to(self.device_)

        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )

        if self.bnorm_ == 'bn':
            self.bnorm = bn.AdaMomBatchNorm((1, self.spatial_filters_), batchdim=0, dispersion=self.bnorm_dispersion_, 
                                        eta=1., eta_test=.1).to(self.device_)
        elif self.bnorm_ == 'dsbn':
            self.dsbnorm = bn.AdaMomDomainBatchNorm((1, self.spatial_filters_), batchdim=0, 
                                domains=self.domains_,
                                dispersion=self.bnorm_dispersion_,
                                eta=1., eta_test=.1).to(self.device_)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')

        self.logarithm = torch.nn.Sequential(
            modules.MyLog(),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.spatial_filters_,self.nclasses_),
        ).to(self.device_)

    def forward(self, x, d, return_latent=True):
        out = ()
        h = self.cnn(x.to(device=self.device_)[:,None,...])
        C = self.cov_pooling(h)
        l = torch.diagonal(C, dim1=-2, dim2=-1)
        l = self.logarithm(l)
        l = self.bnorm(l) if hasattr(self, 'bnorm') else l
        l = self.dsbnorm(l,d) if hasattr(self, 'dsbnorm') else l
        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out

    def domainadapt_finetune(self, x, y, d, target_domains):
        if hasattr(self, 'dsbnorm'):
            self.dsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            for du in d.unique():
                self.forward(x[d==du], d[d==du])

        if hasattr(self, 'dsbnorm'):
            self.dsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)  

    def finetune(self, x, y, d):
        if hasattr(self, 'bnorm'):
            self.bnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

        with torch.no_grad():
            self.forward(x, d)

        if hasattr(self, 'bnorm'):
            self.bnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

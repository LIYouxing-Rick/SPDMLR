import torch
import torch.nn as nn
import torch.nn.Functional as F
from CorMatrix import Correlation
from CorMatrix import CorEuclideanCholeskyMetric, CorLogEuclideanCholeskyMetric, CorOffLogMetric, CorLogScaledMetric
from spdsw import SPDSW,get_spdsw



class SWDloss(nn.Module):
    def __init__(self, lamda=0.05,gamma=0.05, d=None, n_proj=50, device='cuda', dtype=torch.double, seed=42, metric="lsm"):
        super().__init__() 
        self.metric = metric
        self.lamda = lamda
        self.gamma = gamma
        self.spdsw_caculator = SPDSW(
            d=d,
            n_proj=n_proj,
            device=device,
            dtype=dtype,
            random_state=seed,
            sampling=self.metric
        )
        

    def forward(self, pred, target):
        
        ce_loss = F.cross_entropy(pred, target, reduction='mean')
        sw_loss = self.spdsw_calculator(pred,p=2,alpha=self.lamda,beta=self.gamma) 
        
        loss = ce_loss +  sw_loss
    
        return loss
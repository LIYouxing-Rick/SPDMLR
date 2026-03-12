#!/bin/bash
# 多GPU并行训练脚本 - 使用3张GPU并行执行

# 确保安装了必要的依赖
# pip install -U hydra-core hydra-joblib-launcher

# 设置环境变量以显示完整错误信息（调试用）
export HYDRA_FULL_ERROR=1

# 实验1: lsm,ecm + use_logm=false
#[ $? -eq 0 ] && python TSMNet-MLR.py -m hydra/launcher=joblib hydra.launcher.n_jobs=20 hydra.sweeper.max_batch_size=20 \
 # fit.data_dir=/root/hinss2021 fit.seed=0,1,2,3,4 fit.device=GPU \
  #nnet.model.swd_metric=lsm,olm nnet.model.use_lp=true nnet.model.use_logm=true \
  #nnet.model.n_proj=150 nnet.model.swd_power=4.0 \
  #nnet.model.loss_lambda1_init=0.75 nnet.model.loss_lambda2_init=0.75 \
  #nnet.optimizer.lr=1e-3 nnet.optimizer.loss_lr=5e-4
  
[ $? -eq 0 ] && python TSMNet-MLR.py -m hydra/launcher=joblib hydra.launcher.n_jobs=20 hydra.sweeper.max_batch_size=20 \
  fit.data_dir=/root/hinss2021 fit.seed=0,1,2,3,4 fit.device=GPU \
  nnet.model.swd_metric=lecm nnet.model.use_lp=true nnet.model.use_logm=true \
  nnet.model.n_proj=150 nnet.model.swd_power=4.0 \
  nnet.model.loss_lambda1_init=0.75 nnet.model.loss_lambda2_init=0.75 \
  nnet.optimizer.lr=1e-3 nnet.optimizer.loss_lr=5e-4

[ $? -eq 0 ] && python TSMNet-MLR.py -m hydra/launcher=joblib hydra.launcher.n_jobs=20 hydra.sweeper.max_batch_size=20 \
  fit.data_dir=/root/hinss2021 fit.seed=0,1,2,3,4 fit.device=GPU \
  nnet.model.swd_metric=ecm nnet.model.use_lp=true nnet.model.use_logm=true \
  nnet.model.n_proj=150 nnet.model.swd_power=4.0 \
  nnet.model.loss_lambda1_init=0.75 nnet.model.loss_lambda2_init=0.75 \
  nnet.optimizer.lr=1e-3 nnet.optimizer.loss_lr=5e-4



# 实验3: ecm,lsm + use_logm=true
#[ $? -eq 0 ] && python TSMNet-MLR.py -m hydra/launcher=joblib hydra.launcher.n_jobs=18 hydra.sweeper.max_batch_size=18 \
 # fit.data_dir=/root/hinss2021 fit.seed=0,1,2,3,4 fit.device=GPU \
 # nnet.model.swd_metric=ecm,lsm,olm nnet.model.use_lp=true nnet.model.use_logm=true \
 # nnet.model.n_proj=150 nnet.model.swd_power=4.0 \
 # nnet.model.loss_lambda1_init=0.75 nnet.model.loss_lambda2_init=0.75 \
 # nnet.optimizer.lr=1e-3 nnet.optimizer.loss_lr=5e-4


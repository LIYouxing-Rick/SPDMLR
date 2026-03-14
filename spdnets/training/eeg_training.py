import time
import datetime
import fcntl

import os
from time import time
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from skorch.callbacks.scoring import EpochScoring
from skorch.dataset import ValidSplit
from skorch.callbacks import Checkpoint
import torch as th

import logging
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import json
import gc

import moabb
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from library.utils.moabb import CachedParadigm
from spdnets.models import DomainAdaptBaseModel, DomainAdaptJointTrainableModel, EEGNetv4
from spdnets.models import CPUModel
from library.utils.torch import BalancedDomainDataLoader, CombinedDomainDataset, DomainIndex, StratifiedDomainDataLoader
from spdnets.models.base import DomainAdaptFineTuneableModel, FineTuneableModel

from spdnets.utils.skorch import DomainAdaptNeuralNetClassifier
import mne

from spdnets.utils.common_utils import set_seed_thread

def ensure_valid_mne_config():
    """Ensure ~/.mne/mne-python.json exists and is valid JSON.

    If missing or corrupted, recreate it as an empty JSON object {}.
    Returns the path to the config file or None if it could not be ensured.
    """
    try:
        home = os.path.expanduser('~')
        cfg_dir = os.path.join(home, '.mne')
        cfg_path = os.path.join(cfg_dir, 'mne-python.json')
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir, exist_ok=True)
        if not os.path.exists(cfg_path):
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.write('{}')
            return cfg_path
        # Validate existing JSON
        with open(cfg_path, 'r', encoding='utf-8') as f:
            try:
                json.load(f)
            except Exception:
                with open(cfg_path, 'w', encoding='utf-8') as fw:
                    json.dump({}, fw)
        return cfg_path
    except Exception:
        return None

def training(cfg,args):
    # 优先使用 fit.data_dir，否则退回到顶层 data_dir
    try:
        data_dir = getattr(cfg.fit, 'data_dir')
    except Exception:
        data_dir = cfg.data_dir

    # Ensure MNE config file is valid to avoid RuntimeError on corrupted JSON
    ensure_valid_mne_config()
    # Also set environment variables as a fallback
    os.environ['MNE_DATA'] = str(data_dir)
    os.environ['MNEDATASET_TMP_DIR'] = str(data_dir)
    os.environ['_MNE_FAKE_HOME_DIR'] = str(data_dir)
    try:
        local_stieger_dir = getattr(getattr(cfg, 'data_paths', {}), 'moabb_stieger2021')
        if local_stieger_dir:
            os.environ['STIEGER2021_LOCAL_DIR'] = str(local_stieger_dir)
    except Exception:
        pass
    # Try to set MNE configs robustly
    try:
        mne.set_config("MNE_DATA", data_dir)
        mne.set_config("MNEDATASET_TMP_DIR", data_dir)
        mne.set_config("_MNE_FAKE_HOME_DIR", data_dir)
    except Exception:
        # Fall back quietly; environment variables above should suffice
        pass
    args.threadnum = cfg.threadnum
    args.is_debug = cfg.is_debug
    # 优先使用 fit.seed，否则退回到顶层 seed
    try:
        args.seed = int(getattr(cfg.fit, 'seed'))
    except Exception:
        args.seed = cfg.seed
    set_seed_thread(args.seed, args.threadnum)

    args.name = cfg.nnet.name
    args.classifier = cfg.nnet.model.classifier
    args.metric = cfg.nnet.model.metric
    args.power = cfg.nnet.model.power
    args.alpha = cfg.nnet.model.alpha
    args.beta = cfg.nnet.model.beta

    # SWD 相关特征用于命名（从模型配置中稳健读取）
    try:
        _model_cfg_dict = OmegaConf.to_container(cfg.nnet.model, resolve=True)
    except Exception:
        _model_cfg_dict = {}
    args.swd_metric = _model_cfg_dict.get('swd_metric', 'lsm')
    args.use_lp = bool(_model_cfg_dict.get('use_lp', True))
    args.use_logm = bool(_model_cfg_dict.get('use_logm', True))
    # 单独读取 SWD 惩罚项的 power（若未提供则回退到分类器的 power）
    try:
        args.swd_power = float(_model_cfg_dict.get('swd_power', args.power))
    except Exception:
        args.swd_power = args.power
    try:
        args.n_proj = int(_model_cfg_dict.get('n_proj', 50))
    except Exception:
        args.n_proj = 50
    try:
        args.loss_lambda1_init = float(_model_cfg_dict.get('loss_lambda1_init', 1.0))
    except Exception:
        args.loss_lambda1_init = 1.0
    try:
        args.loss_lambda2_init = float(_model_cfg_dict.get('loss_lambda2_init', 1.0))
    except Exception:
        args.loss_lambda2_init = 1.0

    args.optimiz = 'AMSGRAD' if cfg.nnet.optimizer.amsgrad else 'ADAM'
    # cfg.nnet.optimizer.amsgrad = True if args.optimiz == 'AMSGRAD' else False
    args.lr = cfg.nnet.optimizer.lr
    args.weight_decay = cfg.nnet.optimizer.weight_decay
    # 读取分组学习率：网络参数 vs SWD loss 权重（log_lambda1/2）
    try:
        args.net_lr = float(getattr(cfg.nnet.optimizer, 'net_lr'))
    except Exception:
        args.net_lr = args.lr
    try:
        args.loss_lr = float(getattr(cfg.nnet.optimizer, 'loss_lr'))
    except Exception:
        args.loss_lr = args.lr

    args.model_name = get_model_name(args)
    rng_seed = args.seed
    log = logging.getLogger(args.model_name)

    moabb.set_log_level("info")

    # setting device（优先使用 fit.device，否则退回顶层 device）
    try:
        device_cfg = getattr(cfg.fit, 'device')
    except Exception:
        device_cfg = cfg.device

    if device_cfg =='CPU':
        device = torch.device('cpu')
    elif device_cfg == 'GPU':
        gpuid = f"cuda:{HydraConfig.get().job.get('num', 0) % torch.cuda.device_count()}"
        # 首选 Hydra 分配的 GPU，但在显存不足时回退到空闲显存最多的 GPU
        try:
            device = torch.device(gpuid)
            if torch.cuda.is_available():
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                except Exception:
                    free_mem, total_mem = (0, 0)
                # 若该 GPU 空闲显存过低，则选择空闲显存最多的 GPU
                if free_mem < 256 * 1024 * 1024:
                    best_id, best_free = None, -1
                    for i in range(torch.cuda.device_count()):
                        try:
                            f, t = torch.cuda.mem_get_info(i)
                        except Exception:
                            f = 0
                        if f > best_free:
                            best_id, best_free = i, f
                    if best_id is not None:
                        device = torch.device(f"cuda:{best_id}")
                        log.info(f"Switched to cuda:{best_id} with free={best_free/1024**3:.2f} GiB")
        except Exception:
            device = torch.device(gpuid)
    elif 0 <= device_cfg and device_cfg<= th.cuda.device_count():
        device = torch.device(device_cfg)
    else:
        log.info('Wrong device or not available')
    log.info(f"device: {device}")
    cpu = torch.device('cpu')

    # 减少 CUDA allocator 碎片化，启用可扩展段
    if device.type == 'cuda':
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass

    with open_dict(cfg):
        if 'ft_pipeline' not in cfg.nnet:
            cfg.nnet.ft_pipeline = None
        if 'prep_pipeline' not in cfg.nnet:
            cfg.nnet.prep_pipeline = None

    # Diagnostics: log dataset and preprocessing configs
    try:
        if isinstance(cfg.dataset, DictConfig):
            log.info(f"dataset cfg: \n{OmegaConf.to_yaml(cfg.dataset)}")
        else:
            log.info(f"dataset cfg (raw): {cfg.dataset}")
        log.info(f"preprocessing cfg: \n{OmegaConf.to_yaml(cfg.preprocessing)}")
    except Exception:
        pass

    # Robust dataset instantiation that avoids strict reliance on cfg.dataset.type
    dataset = None
    if isinstance(cfg.dataset, DictConfig):
        ds_name = str(cfg.dataset.get('name', '')).lower() if ('name' in cfg.dataset) else ''
        ds_type_cfg = cfg.dataset.get('type', None)
        if ds_name == 'stieger2021' and isinstance(ds_type_cfg, DictConfig):
            if str(ds_type_cfg.get('_target_', '')) == 'moabb.datasets.Stieger2021':
                with open_dict(cfg):
                    cfg.dataset.type._target_ = 'datasets.eeg.moabb.Stieger2021Local'
        ds_type_cfg = cfg.dataset.get('type', None)
        if ds_type_cfg is not None:
            try:
                dataset = hydra.utils.instantiate(ds_type_cfg, _convert_='partial')
                log.info("Dataset instantiated via cfg.dataset.type")
            except Exception as e:
                log.warning(f"Failed to instantiate dataset via cfg.dataset.type: {e}")
        if dataset is None and ('name' in cfg.dataset):
            ds_name = str(cfg.dataset.name).lower()
            if ds_name == 'hinss2021':
                try:
                    from datasets.eeg.moabb.hinss2021 import Hinss2021
                    dataset = Hinss2021()
                    log.info("Dataset instantiated via direct import fallback (Hinss2021)")
                except Exception as e:
                    log.error(f"Fallback import for Hinss2021 failed: {e}")
                    raise
    else:
        # cfg.dataset might be a simple string alias; support common aliases
        try:
            ds_name = str(cfg.dataset).lower()
            if ds_name in ('hinss2021', 'hinss2021.yaml'):
                from datasets.eeg.moabb.hinss2021 import Hinss2021
                dataset = Hinss2021()
                log.info("Dataset instantiated from string alias (Hinss2021)")
        except Exception as e:
            log.warning(f"String alias dataset instantiation failed: {e}")

    # Final fallback to hydra class path if still not instantiated
    if dataset is None:
        try:
            dataset = hydra.utils.get_class('datasets.eeg.moabb.Hinss2021')()
            log.info("Dataset instantiated via hydra.get_class fallback (Hinss2021)")
        except Exception as e:
            log.error(f"Exhausted dataset instantiation fallbacks: {e}")
            raise

    # Instantiate preprocessing dict with guard
    try:
        ppreprocessing_dict = hydra.utils.instantiate(cfg.preprocessing, _convert_='partial')
    except Exception as e:
        log.error(f"Failed to instantiate preprocessing pipeline: {e}")
        raise
    assert (len(ppreprocessing_dict) == 1)  # only 1 paradigm is allowed per call
    prep_name, paradigm = next(iter(ppreprocessing_dict.items()))

    res_dir = os.path.join(cfg.evaluation.strategy, prep_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    results = pd.DataFrame( \
        columns=['dataset', 'subject', 'session', 'method', 'score_trn', 'score_tst',
                 'time', 'n_test', 'classes'])
    resix = 0
    results['score_trn'] = results['score_trn'].astype(np.double)
    results['score_tst'] = results['score_tst'].astype(np.double)
    results['time'] = results['time'].astype(np.double)
    results['n_test'] = results['n_test'].astype(int)
    results['classes'] = results['classes'].astype(int)

    results_fit = []

    scorefun = get_scorer(cfg.score)._score_func

    def masked_scorefun(y_true, y_pred, **kwargs):
        masked = y_true == -1
        if np.all(masked):
            log.warning('Nothing to score because all target values are masked (value = -1).')
            return np.nan
        return scorefun(y_true[~masked], y_pred[~masked], **kwargs)

    def scorer(net, X, y):
        y_true = np.asarray(y)
        y_pred = np.asarray(net.predict(X))
        return masked_scorefun(y_true, y_pred)

    dadapt = cfg.evaluation.adapt

    bacc_val_logger = EpochScoring(scoring=scorer,
                                   lower_is_better=False,
                                   on_train=False,
                                   name='score_val')
    bacc_trn_logger = EpochScoring(scoring=scorer,
                                   lower_is_better=False,
                                   on_train=True,
                                   name='score_trn')

    if 'inter-session' in cfg.evaluation.strategy:
        subset_iter = iter([[s] for s in dataset.subject_list])
        groupvarname = 'session'
    elif 'inter-subject' in cfg.evaluation.strategy:
        subset_iter = iter([None])
        groupvarname = 'subject'
    else:
        raise NotImplementedError()
    if args.is_debug:
        subset_iter = iter([[1]])
    number=0
    for subset in subset_iter:

        if groupvarname == 'session':
            domain_expression = "session"
        elif groupvarname == 'subject':
            domain_expression = "session + subject * 1000"

        # 兼容 OmegaConf：稳妥读取 sessions（不存在则为 None）
        selected_sessions = cfg.dataset.sessions if (isinstance(cfg.dataset, DictConfig) and ('sessions' in cfg.dataset)) else None

        ds = CombinedDomainDataset.from_moabb(paradigm, dataset, subjects=subset, domain_expression=domain_expression,
                                              dtype=cfg.nnet.inputtype, sessions=selected_sessions)

        if cfg.nnet.prep_pipeline is not None:
            ds = ds.cache()  # we need to load the entire dataset for preprocessing

        def _to_int_codes(series):
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().all():
                return numeric.astype(np.int64).values
            return pd.factorize(series.astype(str), sort=True)[0].astype(np.int64)

        sessions = _to_int_codes(ds.metadata.session)
        subjects = _to_int_codes(ds.metadata.subject)
        g = _to_int_codes(ds.metadata[groupvarname])
        groups = np.unique(g)

        domains = ds.domains.unique()

        n_classes = len(ds.labels.unique())

        if len(groups) < 2:
            log.warning(
                f"Insufficient number (n={len(groups)}) of groups ({groupvarname}) in the (sub-)dataset to run leave 1 group out CV!")
            continue

        mdl_kwargs = dict(nclasses=n_classes)

        mdl_kwargs['nchannels'] = ds.shape[1]
        mdl_kwargs['nsamples'] = ds.shape[2]
        mdl_kwargs['nbands'] = ds.shape[3] if ds.ndim == 4 else 1
        mdl_kwargs['input_shape'] = (1,) + ds.shape[1:]

        mdl_dict = OmegaConf.to_container(cfg.nnet.model, resolve=True)
        mdl_class = hydra.utils.get_class(mdl_dict.pop('_target_'))

        if issubclass(mdl_class, DomainAdaptBaseModel):
            mdl_kwargs['domains'] = domains
        if issubclass(mdl_class, EEGNetv4):
            if isinstance(paradigm, CachedParadigm):
                info = paradigm.get_info(dataset)
                mdl_kwargs['srate'] = int(info['sfreq'])
            else:
                raise NotImplementedError()
        if issubclass(mdl_class, FineTuneableModel) and isinstance(ds, CombinedDomainDataset):
            # we need to load the entire dataset
            ds = ds.cache()

        mdl_kwargs = {**mdl_kwargs, **mdl_dict}

        optim_kwargs = OmegaConf.to_container(cfg.nnet.optimizer, resolve=True)
        optim_class = hydra.utils.get_class(optim_kwargs.pop('_target_'))
        # These keys are used for naming or param_groups interpolation only; do not pass to optimizer ctor
        optim_kwargs.pop('net_lr', None)
        optim_kwargs.pop('loss_lr', None)

        metaddata = {
            'model_class': mdl_class,
            'model_kwargs': mdl_kwargs,
            'optim_class': optim_class,
            'optim_kwargs': optim_kwargs
        }
        if cfg.saving_model.is_save:
            mdl_metadata_dir = os.path.join(res_dir, 'metadata')
            if not os.path.exists(mdl_metadata_dir):
                os.makedirs(mdl_metadata_dir)
            _swd_tag = swd_tag_for_filename(args)
            torch.save(metaddata, f=os.path.join(mdl_metadata_dir, f'meta-{cfg.nnet.name}-{_swd_tag}.pth'))
            with open(os.path.join(mdl_metadata_dir, f'config-{cfg.nnet.name}-{_swd_tag}.yaml'), 'w+') as f:
                f.writelines(OmegaConf.to_yaml(cfg))

        if issubclass(mdl_class, CPUModel):
            device = cpu

        mdl_kwargs['device'] = device

        n_test_groups = int(np.clip(np.round(len(groups) * cfg.fit.test_size), 1, None))

        log.info(f"Performing leave {n_test_groups} (={cfg.fit.test_size * 100:.0f}%) {groupvarname}(s) out CV")
        cv = GroupKFold(n_splits=int(len(groups) / n_test_groups))
        number = number + len(ds.labels)
        print(f"number: {number}")
        ds.eval()  # unmask labels
        for train, test in cv.split(ds.labels, ds.labels, g):

            target_domains = ds.domains[test].unique().numpy()
            torch.manual_seed(rng_seed + target_domains[0])

            prep_pipeline = hydra.utils.instantiate(cfg.nnet.prep_pipeline, _convert_='partial')
            ft_pipeline = hydra.utils.instantiate(cfg.nnet.ft_pipeline, _convert_='partial')

            if dadapt is not None and dadapt.name != 'no':
                # extend training data with adaptation set
                if issubclass(mdl_class, DomainAdaptJointTrainableModel):
                    stratvar = ds.labels + ds.domains * n_classes
                    adapt_domain = test  # extract_adapt_idxs(dadapt.nadapt_domain, test, stratvar)
                else:
                    # some nets to not require target domain data during training
                    adapt_domain = np.array([], dtype=np.int64)
                    log.info("Model does not require adaptation. Using original training data.")

                train_source_doms = train
                train = np.concatenate((train, adapt_domain))

                if dadapt.name == 'uda':
                    ds.set_masked_labels(adapt_domain)
                elif dadapt.name == 'sda':
                    test = np.setdiff1d(test, adapt_domain)

                if len(test) == 0:
                    raise ValueError('No data left in the test set!')
            else:
                train_source_doms = train

            test_groups = np.unique(g[test])
            test_group_list = []
            for test_group in test_groups:
                test_dict = {}
                subject = np.unique(subjects[g == test_group])
                assert (len(subject) == 1)  # only one subject per group
                test_dict['subject'] = subject[0]
                if groupvarname == 'subject':
                    test_dict['session'] = -1
                else:
                    session = np.unique(sessions[g == test_group])
                    assert (len(session) == 1)  # only one session per group
                    test_dict['session'] = session[0]
                test_dict['idxs'] = np.intersect1d(test, np.nonzero(g == test_group))
                test_group_list.append(test_dict)

            t_start = time()

            ## preprocessing
            dsprep = ds.copy(deep=False)
            dsprep.train()  # mask labels

            if prep_pipeline is not None:
                prep_pipeline.fit(dsprep.features[train].numpy(), dsprep.labels[train])
                dsprep.set_features(prep_pipeline.transform(dsprep.features))

            # torch dataset generation
            batch_size_valid = cfg.fit.validation_size if type(cfg.fit.validation_size) == int else int(
                np.ceil(cfg.fit.validation_size * len(train)))

            dsprep.eval()  # unmask labels
            # extract stratified (classes and groups) validation data
            stratvar = dsprep.labels[train] + dsprep.domains[train] * n_classes
            valid_cv = ValidSplit(iter(StratifiedShuffleSplit(n_splits=1, test_size=cfg.fit.validation_size,
                                                              random_state=rng_seed + target_domains[0]).split(stratvar,
                                                                                                               stratvar)))

            netkwargs = {'module__' + k: v for k, v in mdl_kwargs.items()}
            netkwargs = {**netkwargs, **{'optimizer__' + k: v for k, v in optim_kwargs.items()}}
            if cfg.fit.stratified:

                n_train_domains = len(dsprep.domains[train].unique())
                domains_per_batch = min(cfg.fit.domains_per_batch, n_train_domains)
                batch_size_train = int(
                    max(np.round(cfg.fit.batch_size_train / domains_per_batch), 2) * domains_per_batch)

                netkwargs['iterator_train'] = StratifiedDomainDataLoader
                netkwargs['iterator_train__domains_per_batch'] = domains_per_batch
                netkwargs['iterator_train__shuffle'] = True
                netkwargs['iterator_train__batch_size'] = batch_size_train
                # 避免最后一个不完整 batch 导致峰值显存升高
                netkwargs['iterator_train__drop_last'] = True
            else:
                netkwargs['iterator_train'] = BalancedDomainDataLoader
                netkwargs['iterator_train__domains_per_batch'] = cfg.fit.domains_per_batch
                netkwargs['iterator_train__drop_last'] = True
                netkwargs['iterator_train__replacement'] = False
                netkwargs['iterator_train__batch_size'] = cfg.fit.batch_size_train
            netkwargs['iterator_valid__batch_size'] = batch_size_valid
            netkwargs['max_epochs'] = cfg.fit.epochs
            netkwargs['classes'] = np.arange(n_classes, dtype=np.int64)
            netkwargs[
                'callbacks__print_log__prefix'] = f'{dataset.code} {n_classes}cl | {test_groups} | {args.model_name} :'

            scheduler = hydra.utils.instantiate(cfg.nnet.scheduler, _convert_='partial')

            # save model
            if cfg.saving_model.is_save:
                _swd_tag = swd_tag_for_filename(args)
                mdl_path_tmp = os.path.join(res_dir, 'models', 'tmp', f'{test_groups}_{cfg.nnet.name}-{_swd_tag}.pth')
                if not os.path.exists(os.path.split(mdl_path_tmp)[0]):
                    os.makedirs(os.path.split(mdl_path_tmp)[0])
                checkpoint = Checkpoint(
                    f_params=mdl_path_tmp, f_criterion=None, f_optimizer=None, f_history=None,
                    monitor='valid_loss_best', load_best=True)
                net = DomainAdaptNeuralNetClassifier(
                    mdl_class,
                    train_split=valid_cv,
                    callbacks=[bacc_trn_logger, bacc_val_logger, scheduler, checkpoint],
                    optimizer=optim_class,
                    verbose=0,
                    device=device,
                    **netkwargs)
            else:
                net = DomainAdaptNeuralNetClassifier(
                    mdl_class,
                    train_split=valid_cv,
                    callbacks=[bacc_trn_logger, bacc_val_logger, scheduler],
                    optimizer=optim_class,
                    verbose=0,
                    device=device,
                    **netkwargs)
            if args.is_debug:
                print(net)
            dsprep.train()  # mask labels
            dstrn = torch.utils.data.Subset(dsprep, train)
            # 尝试训练，若出现显存不足则自动减小 batch 并重试
            cur_train_bs = netkwargs.get('iterator_train__batch_size', cfg.fit.batch_size_train)
            cur_valid_bs = batch_size_valid
            cur_domains_per_batch = netkwargs.get('iterator_train__domains_per_batch', 1)
            retries = 0
            max_retries = 4
            while True:
                try:
                    net.fit(dstrn, None)
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if ('CUDA out of memory' in msg) or ('CUDA error: out of memory' in msg):
                        log.warning(f"CUDA OOM at train_bs={cur_train_bs}, valid_bs={cur_valid_bs}. Retrying with smaller batches...")
                        retries += 1
                        if th.cuda.is_available():
                            try:
                                th.cuda.empty_cache()
                            except Exception:
                                pass
                        # 至少保证每个域有 2 个样本（若启用按域采样）
                        min_train_bs = max(2 * cur_domains_per_batch, 2)
                        if (retries <= max_retries) and (cur_train_bs > min_train_bs):
                            new_train_bs = max(min_train_bs, int(cur_train_bs // 2))
                            # 保持为 domains_per_batch 的整数倍
                            if cur_domains_per_batch > 1:
                                new_train_bs = max(min_train_bs, (new_train_bs // cur_domains_per_batch) * cur_domains_per_batch)
                            new_valid_bs = max(1, int(cur_valid_bs // 2))
                            net.set_params(iterator_train__batch_size=new_train_bs,
                                           iterator_valid__batch_size=new_valid_bs)
                            cur_train_bs, cur_valid_bs = new_train_bs, new_valid_bs
                            continue
                        else:
                            log.warning("Batch size reached minimum or retries exceeded; attempting CPU fallback.")
                            try:
                                net.set_params(device=cpu)
                                net.fit(dstrn, None)
                                log.info("Training continued on CPU after OOM.")
                                break
                            except Exception:
                                raise
                    else:
                        raise

            res = pd.DataFrame(net.history)
            res = res.drop(res.filter(regex='.*batches|_best|_count').columns, axis=1)
            res = res.drop(res.filter(regex='event.*').columns, axis=1)
            res = res.rename(columns=dict(train_loss="loss_trn", valid_loss="loss_val", dur="time"))
            res['domains'] = str(test_groups)
            res['method'] = cfg.nnet.name
            res['dataset'] = dataset.code
            results_fit.append(res)
            if cfg.is_timing:
                time_epochs = res.time;
                log.info('{} average time: {:.2f} and average of smallest 5 time: {:.2f} in total {} epoch'.format(
                    cfg.evaluation.strategy,\
                    np.mean(time_epochs[-5:]),np.mean(np.sort(time_epochs)[:5]),len(time_epochs)))
                return


            if cfg.evaluation.adapt.name == "uda":
                if isinstance(net.module_, DomainAdaptFineTuneableModel):
                    dsprep.train()  # mask target domain labels
                    for du in dsprep.domains.unique():
                        domain_data = dsprep[DomainIndex(du.item())]
                        net.module_.domainadapt_finetune(x=domain_data[0]['x'], y=domain_data[1], d=domain_data[0]['d'],
                                                         target_domains=target_domains)
            elif cfg.evaluation.adapt.name == "no":
                if isinstance(net.module_, FineTuneableModel):
                    dsprep.train()  # mask target domain labels
                    net.module_.finetune(x=dsprep.features[train], y=dsprep.labels[train], d=dsprep.domains[train])

            duration = time() - t_start

            # save the final model
            if cfg.saving_model.is_save:
                for test_group in test_group_list:
                    _swd_tag = swd_tag_for_filename(args)
                    mdl_path = os.path.join(res_dir, 'models', f'{test_group["subject"]}', f'{test_group["session"]}',
                                            f'{cfg.nnet.name}-{_swd_tag}.pth')
                    if not os.path.exists(os.path.split(mdl_path)[0]):
                        os.makedirs(os.path.split(mdl_path)[0])
                    net.save_params(f_params=mdl_path)

            ## evaluation
            dsprep.eval()  # unmask target domain labels

            y_hat = np.empty(dsprep.labels.shape)
            # find out latent space dimensionality
            _, l0 = net.forward(dsprep[DomainIndex(dsprep.domains[0])][0])
            l = np.empty((len(dsprep),) + l0.shape[1:])

            for du in dsprep.domains.unique():
                ixs = np.flatnonzero(dsprep.domains == du)
                domain_data = dsprep[DomainIndex(du)]

                y_hat_domain, l_domain, *_ = net.forward(domain_data[0])
                y_hat_domain, l_domain = y_hat_domain.numpy().argmax(axis=1), l_domain.to(device=cpu).numpy()
                y_hat[ixs] = y_hat_domain
                l[ixs] = l_domain

            score_trn = scorefun(dsprep.labels[train_source_doms], y_hat[train_source_doms])

            for test_group in test_group_list:
                score_tst = scorefun(dsprep.labels[test_group["idxs"]], y_hat[test_group["idxs"]])

                res = pd.DataFrame({'dataset': dataset.code,
                                    'subject': test_group["subject"],
                                    'session': test_group["session"],
                                    'method': cfg.nnet.name,
                                    'score_trn': score_trn,
                                    'score_tst': score_tst,
                                    'time': duration,
                                    'n_test': len(test),
                                    'classes': n_classes}, index=[resix])
                results = pd.concat([results, res])
                resix += 1
                r = res.iloc[0, :]
                log.info(
                    f'{r.dataset} {r.classes}cl | {r.subject} | {r.session} : trn={r.score_trn:.2f} tst={r.score_tst:.2f} time={duration:.2f}')

            ## fine tuning
            if ft_pipeline is not None:
                # fitting
                dsprep.train()  # mask target domain labels
                ft_pipeline.fit(l[train], dsprep.labels[train])
                y_hat_ft = ft_pipeline.predict(l)

                # evaluation
                dsprep.eval()  # unmask target domain labels
                ft_score_trn = scorefun(dsprep.labels[train_source_doms], y_hat_ft[train_source_doms])

                for test_group in test_group_list:
                    ft_score_tst = scorefun(dsprep.labels[test_group["idxs"]], y_hat_ft[test_group["idxs"]])

                    res = pd.DataFrame({'dataset': dataset.code,
                                        'subject': test_group["subject"],
                                        'session': test_group["session"],
                                        'method': f'{cfg.nnet.name}+FT',
                                        'score_trn': ft_score_trn,
                                        'score_tst': ft_score_tst,
                                        'time': duration,
                                        'n_test': len(test),
                                        'classes': n_classes}, index=[resix])
                    results = pd.concat([results, res])
                    resix += 1
                    r = res.iloc[0, :]
                    log.info(
                        f'{r.dataset} {r.classes}cl | {r.subject} | {r.session} | {r.method} :    trn={r.score_trn:.2f} tst={r.score_tst:.2f}')

            # 每个 fold 结束后尽量释放显存与对象
            try:
                del net
                del dstrn
                del dsprep
                del ft_pipeline
                del prep_pipeline
            except Exception:
                pass
            try:
                gc.collect()
                if device.type == 'cuda':
                    th.cuda.empty_cache()
            except Exception:
                pass

    if len(results_fit):
        results_fit = pd.concat(results_fit)

        results_fit['preprocessing'] = prep_name
        results_fit['evaluation'] = cfg.evaluation.strategy
        results_fit['adaptation'] = cfg.evaluation.adapt.name

        for method in results_fit['method'].unique():
            method_res = results[results['method'] == method]
            results_fit.to_csv(os.path.join(res_dir, f'nnfitscores_{method}.csv'), index=False)

    if len(results) > 0:

        results['preprocessing'] = prep_name
        results['evaluation'] = cfg.evaluation.strategy
        results['adaptation'] = cfg.evaluation.adapt.name
        if cfg.saving_model.is_save:
            for method in results['method'].unique():
                method_res = results[results['method'] == method]
                method_res.to_csv(os.path.join(res_dir, f'scores_{method}.csv'), index=False)
        tmp = results.groupby('method')[['score_trn', 'score_tst', 'time']].agg(['mean', 'std'])
        column_labels = [('score_trn', 'mean'), ('score_trn', 'std'), \
                         ('score_tst', 'mean'), ('score_tst', 'std')]
        time_lables = [('time', 'mean'), ('time', 'std')]
        row_label = tmp.index.tolist()[0]
        # print(tmp)
        # log.info(tmp.loc[row_label, column_labels] * 100)
        # log.info(tmp.loc[row_label, time_lables])
        final_results = "final results: score_trn: {:.2f}±{:.2f}, score_tst: {:.2f}±{:.2f}, time: {:.2f}±{:.2f}".format( \
            tmp.loc[row_label, column_labels[0]] * 100, tmp.loc[row_label, column_labels[1]] * 100, \
            tmp.loc[row_label, column_labels[2]] * 100, tmp.loc[row_label, column_labels[3]] * 100, \
            tmp.loc[row_label, time_lables[0]], tmp.loc[row_label, time_lables[1]]
        )
        log.info(final_results)

        log_filename = HydraConfig.get().job_logging.handlers.file.filename
        split_filename = log_filename.rsplit('.',1)
        final_filename = f"final_result_{split_filename[0]}.txt"
        final_file_path = os.path.join(os.getcwd(),final_filename)
        log.info("results file path: {}, and saving the results".format(final_file_path))
        write_final_results(final_file_path, args.model_name+'_'+final_results)
        # 保存聚合结果到 /root/SPDMLR-main/torch_results（如果不存在则创建）
        try:
            save_results(results, args, cfg, prep_name)
            log.info("torch results saved to /root/SPDMLR-main/torch_results")
        except Exception as e:
            log.warning(f"Failed to save torch results: {str(e)[:200]}")

def get_model_name(args):
    if args.classifier == 'SPDMLR':
        if args.metric == 'SPDLogEuclideanMetric':
            description = f'{args.metric}-[{args.alpha},{args.beta:.4f}]'
        elif args.metric == 'SPDLogCholeskyMetric':
            description = f'{args.metric}-[{args.power}]'

        description = '-' + description + '-'
    elif args.classifier == 'LogEigMLR':
        description=''
    else:
        raise NotImplementedError

    # 生成 SWD 特征标签并拼入模型名
    _swd_tag = swd_tag_for_filename(args)
    arch = getattr(args, 'architecture', 'tsm')
    # 将 net/loss LR 显示到模型名中，便于统计
    name = f'{args.lr}-{args.weight_decay}-nLR{getattr(args,"net_lr",args.lr)}-lLR{getattr(args,"loss_lr",args.lr)}-{args.name}{description}{args.classifier}-{arch}-{_swd_tag}-{datetime.datetime.now().strftime("%H_%M")}'
    return name

def swd_tag_for_filename(args):
    """构造用于文件命名的 SWD 特征标签。

    格式：P{n_proj}-{metric}-Pow{power:.2f}-L{lambda1:.2f}-G{lambda2:.2f}
    若某个正则项被关闭，则其权重记为 0.00。
    """
    try:
        n_proj = int(getattr(args, 'n_proj', 50))
    except Exception:
        n_proj = 50
    swd_metric = str(getattr(args, 'swd_metric', 'lsm'))
    use_lp = bool(getattr(args, 'use_lp', True))
    use_logm = bool(getattr(args, 'use_logm', True))
    # 记录 SWD 惩罚项的幂指数（若未显式提供则回退到分类器的 power）
    try:
        swd_power = float(getattr(args, 'swd_power', getattr(args, 'power', 1.0)))
    except Exception:
        swd_power = 1.0
    try:
        lam1 = float(getattr(args, 'loss_lambda1_init', 1.0)) if use_lp else 0.0
    except Exception:
        lam1 = 0.0 if not use_lp else 1.0
    try:
        lam2 = float(getattr(args, 'loss_lambda2_init', 1.0)) if use_logm else 0.0
    except Exception:
        lam2 = 0.0 if not use_logm else 1.0
    # 追加学习率信息到标签，确保最终保存的模型文件名也包含
    net_lr = getattr(args, 'net_lr', getattr(args, 'lr', 0.0))
    loss_lr = getattr(args, 'loss_lr', getattr(args, 'lr', 0.0))
    return f'P{n_proj}-{swd_metric}-Pow{swd_power:.2f}-L{lam1:.2f}-G{lam2:.2f}-nLR{net_lr}-lLR{loss_lr}'

def write_final_results(file_path,message):
    # Create a file lock
    with open(file_path, "a") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock

        # Write the message to the file
        file.write(message + "\n")

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Release the lock

def save_results(results, args, cfg, prep_name):
    """将最终聚合结果以 torch.save 的方式写入 /root/SPDMLR-main/torch_results

    保存字段仿照 RMLR-main 的保存逻辑，包含平均与方差等统计。
    """
    torch_results_dir = '/root/SPDMLR-main/torch_results'
    if not os.path.exists(torch_results_dir):
        os.makedirs(torch_results_dir)

    tmp = results.groupby('method').agg(['mean', 'std'])
    row_label = tmp.index.tolist()[0]

    payload = {
        'score_trn_mean': float(tmp.loc[row_label, ('score_trn', 'mean')]),
        'score_trn_std': float(tmp.loc[row_label, ('score_trn', 'std')]),
        'score_tst_mean': float(tmp.loc[row_label, ('score_tst', 'mean')]),
        'score_tst_std': float(tmp.loc[row_label, ('score_tst', 'std')]),
        'time_mean': float(tmp.loc[row_label, ('time', 'mean')]),
        'time_std': float(tmp.loc[row_label, ('time', 'std')]),
        'seed': int(getattr(args, 'seed', 0)),
        'modelname': str(getattr(args, 'model_name', '')),
        'preprocessing': str(prep_name),
        'evaluation': str(getattr(cfg.evaluation, 'strategy', '')),
        'adaptation': str(getattr(cfg.evaluation.adapt, 'name', '')),
        'lp': 'on' if bool(getattr(args, 'use_lp', True)) else 'off',
        'logm': 'on' if bool(getattr(args, 'use_logm', True)) else 'off',
        # 额外保存初始 λ 与 power，便于统计脚本识别分组
        'init_l1': str(getattr(args, 'loss_lambda1_init', 'unknown')),
        'init_l2': str(getattr(args, 'loss_lambda2_init', 'unknown')),
        # 保存 SWD 惩罚项的幂指数，并保持兼容：power 代表 swd_power
        'power': str(getattr(args, 'swd_power', getattr(args, 'power', 'unknown'))),
        'swd_power': str(getattr(args, 'swd_power', 'unknown')),
        'cls_power': str(getattr(args, 'power', 'unknown')),
    }

    # 文件名移除末尾时间戳，加入 seed 标识，便于后续统计
    base_name = str(getattr(args, 'model_name', '')).rsplit('-', 1)[0]
    save_path = os.path.join(torch_results_dir, f"{base_name}_seed{getattr(args, 'seed', 0)}")
    th.save(payload, save_path)

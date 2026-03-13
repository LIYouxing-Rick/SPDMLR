import os
import numpy as np
import torch as th
from torch.utils import data
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
import hydra

from library.utils.moabb import CachedMotorImagery


class DatasetSPDFromCov(data.Dataset):
    def __init__(self, covs, labels):
        self._covs = covs
        self._labels = labels

    def __len__(self):
        return self._covs.shape[0]

    def __getitem__(self, item):
        x = th.from_numpy(self._covs[item]).double()
        y = th.tensor(self._labels[item], dtype=th.long)
        return x, y


class DataLoaderMOABBCov:
    def __init__(self, data_root, dataset_key, batch_size, seed=1024, test_size=0.25):
        self.data_root = data_root
        self.dataset_key = str(dataset_key).lower()
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.test_size = float(test_size)

        os.environ['MNE_DATA'] = str(self.data_root)
        os.environ['MNEDATASET_TMP_DIR'] = str(self.data_root)
        os.environ['_MNE_FAKE_HOME_DIR'] = str(self.data_root)

        ds_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'conf',
            'TSMNet',
            'dataset',
            f'{self.dataset_key}.yaml'
        )
        if not os.path.isfile(ds_cfg_path):
            raise FileNotFoundError(f'Unknown EEG dataset key for SPDNet: {self.dataset_key}')

        ds_cfg = OmegaConf.load(ds_cfg_path)
        dataset_obj = hydra.utils.instantiate(ds_cfg.type, _convert_='partial')

        classes = list(ds_cfg.classes) if ('classes' in ds_cfg and ds_cfg.classes is not None) else None
        channels = None if ('channels' not in ds_cfg or ds_cfg.channels is None) else list(ds_cfg.channels)
        resample = None if ('resample' not in ds_cfg) else ds_cfg.resample
        tmin = float(ds_cfg.tmin) if ('tmin' in ds_cfg and ds_cfg.tmin is not None) else 0.0
        tmax = None if ('tmax' not in ds_cfg) else ds_cfg.tmax

        paradigm = CachedMotorImagery(
            fmin=4,
            fmax=36,
            events=classes,
            channels=channels,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
        )

        X, labels, _ = paradigm.get_data(dataset=dataset_obj)
        if X.ndim != 3:
            raise RuntimeError(f'Expected EEG epochs shape [N,C,T], got {X.shape}')

        if classes is None:
            unique_labels = sorted(set(labels.tolist()))
            label_to_idx = {k: i for i, k in enumerate(unique_labels)}
        else:
            label_to_idx = {str(k): i for i, k in enumerate(classes)}
            missing = sorted(set([str(lb) for lb in labels.tolist()]) - set(label_to_idx.keys()))
            if len(missing):
                start = len(label_to_idx)
                for i, k in enumerate(missing):
                    label_to_idx[k] = start + i
        y = np.array([label_to_idx[str(lb)] for lb in labels.tolist()], dtype=np.int64)

        covs = np.array([self._cov_epoch(ep) for ep in X], dtype=np.float64)
        covs = covs[:, None, :, :]

        idx = np.arange(len(y))
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y
        )

        x_tr, y_tr = covs[tr_idx], y[tr_idx]
        x_te, y_te = covs[te_idx], y[te_idx]

        train_set = DatasetSPDFromCov(x_tr, y_tr)
        test_set = DatasetSPDFromCov(x_te, y_te)

        self._train_generator = data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self._test_generator = data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        self.n_classes = int(len(np.unique(y)))
        self.spd_dim = int(covs.shape[-1])

    @staticmethod
    def _cov_epoch(epoch):
        c = np.cov(epoch)
        c = 0.5 * (c + c.T)
        c = c + np.eye(c.shape[0], dtype=c.dtype) * 1e-6
        return c

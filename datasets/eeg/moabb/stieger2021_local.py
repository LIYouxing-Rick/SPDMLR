import os
import glob
from pathlib import Path

from moabb.datasets import Stieger2021


class Stieger2021Local(Stieger2021):
    def __init__(self, interval=[0, 3], channels=None, srate=None):
        super().__init__(interval=interval)
        self.channels = channels
        self.srate = srate

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        candidates = []
        env_local = os.environ.get("STIEGER2021_LOCAL_DIR", "")
        if env_local:
            candidates.append(env_local)
        if path:
            candidates.append(path)
        mne_data = os.environ.get("MNE_DATA", "")
        if mne_data:
            candidates.extend(
                [
                    os.path.join(mne_data, "MNE-Stieger2021-data"),
                    mne_data,
                ]
            )

        checked = []
        for base in candidates:
            if not base:
                continue
            b = str(Path(base).expanduser().resolve())
            if b in checked:
                continue
            checked.append(b)
            if not os.path.isdir(b):
                continue
            subject_tokens = [
                f"{subject:03d}",
                f"{subject:02d}",
                str(subject),
            ]
            patterns = [
                "**/*.mat",
                "**/*.set",
                "**/*.edf",
                "**/*.bdf",
                "**/*.gdf",
                "**/*.vhdr",
                "**/*.fif",
            ]
            hits = []
            for p in patterns:
                for fn in glob.glob(os.path.join(b, p), recursive=True):
                    low = os.path.basename(fn).lower()
                    if any(tok in low for tok in subject_tokens):
                        hits.append(fn)
            if len(hits):
                return sorted(set(hits))

        raise FileNotFoundError(
            f"Stieger2021 local files not found for subject {subject}. "
            f"Checked roots: {checked}"
        )

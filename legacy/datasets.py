# datasets.py
# Generate supervised training pairs consistent with the paper's sampling/training protocol.
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from lattice import RotatedSurfaceCode
from errors import (
    depolarizing_error_on_data, circuit_noise_sample,
    extract_syndrome_once, extract_syndrome_window, detection_events_time_diff
)
from simple_decoder import SimpleDecoder

LOGICAL_IDX = {"I":0, "Xbar":1, "Zbar":2, "Ybar":3}

class HLDDatasetDepol(Dataset):
    """
    High-level decoder dataset for depolarizing model.
    One cycle only (perfect measurements). Label is which logical operator must be added
    to cancel the simple decoder's potential logical error. (Sec. II.C, Fig. 7) :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, code: RotatedSurfaceCode, p: float, n_samples: int, seed: int = 0):
        self.code = code
        self.p = p
        self.n = n_samples
        self.rng = np.random.default_rng(seed)
        self.simple = SimpleDecoder(code)
        self.features = []
        self.labels = []
        self._build()

    def _build(self):
        for _ in range(self.n):
            data_err = depolarizing_error_on_data(self.code.n_data(), self.p, self.rng)
            xs, zs = extract_syndrome_once(self.code, data_err)
            # Simple decoder proposes corrections from last syndrome (only 1 cycle)
            corrX, corrZ = self.simple.propose_corrections(xs, zs)
            logical = self.simple.logical_from_corrections(corrX, corrZ)
            # Input features are the syndrome bits; output is 1-hot probability target over {I,X,Z,Y}
            feat = np.concatenate([xs, zs]).astype(np.float32)[None, :]  # T=1
            lab = np.zeros(4, dtype=np.float32)
            lab[LOGICAL_IDX[logical]] = 1.0
            self.features.append(feat)
            self.labels.append(lab)
        self.X = torch.tensor(np.stack(self.features, axis=0))  # [N, 1, F]
        self.Y = torch.tensor(np.stack(self.labels, axis=0))    # [N, 4]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

class HLDDatasetCircuit(Dataset):
    """
    Circuit-noise dataset with imperfect measurements over window T=d cycles (paper practice).
    NN1 optional target: tag data vs measurement events.
    NN2 target: logical correction class to cancel simple decoder's logical error. :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, code: RotatedSurfaceCode, p: float, n_samples: int, seed: int = 0, T: int = None):
        self.code = code
        self.p = p
        self.n = n_samples
        self.T = T if T is not None else code.d  # common choice: T=d (window = d cycles) per paper
        self.rng = np.random.default_rng(seed)
        self.simple = SimpleDecoder(code)
        self.Xseq = []
        self.Ynn2 = []
        self.Ynn1 = []
        self._build()

    def _build(self):
        for _ in range(self.n):
            draw = circuit_noise_sample(self.code.n_data(), self.code.n_x_checks(), self.code.n_z_checks(), self.p, self.T, self.rng)
            Xs, Zs = extract_syndrome_window(self.code, draw["data"], draw["merr_x"], draw["merr_z"])
            # Detection events time differences are what matter to separate data vs measurement chains (Fig. 5). :contentReference[oaicite:5]{index=5}
            de_x = detection_events_time_diff(Xs)
            de_z = detection_events_time_diff(Zs)
            # For NN1, label "is_data_event" at last cycle by comparing consistency across time.
            # Heuristic target: a flip that alternates on the same check across consecutive times suggests measurement error.
            # We use last-cycle DE as target points.
            nn1_target = np.concatenate([de_x[-1], de_z[-1]]).astype(np.float32)  # crude signal; can be refined

            # NN2 input is the full T x F sequence; simple decoder uses last-cycle syndrome
            corrX, corrZ = self.simple.propose_corrections(Xs[-1], Zs[-1])
            logical = self.simple.logical_from_corrections(corrX, corrZ)
            y2 = np.zeros(4, dtype=np.float32)
            y2[LOGICAL_IDX[logical]] = 1.0

            feat_seq = np.concatenate([Xs, Zs], axis=1).astype(np.float32)  # [T, F]
            self.Xseq.append(feat_seq)
            self.Ynn2.append(y2)
            self.Ynn1.append(nn1_target)

        self.X = torch.tensor(np.stack(self.Xseq, axis=0))      # [N, T, F]
        self.Y2 = torch.tensor(np.stack(self.Ynn2, axis=0))     # [N, 4]
        self.Y1 = torch.tensor(np.stack(self.Ynn1, axis=0))     # [N, F]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], (self.Y2[i], self.Y1[i])

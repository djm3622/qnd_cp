# errors.py
# Depolarizing and circuit-noise models and syndrome extraction
from typing import Tuple, Dict
import numpy as np
from lattice import RotatedSurfaceCode

# Pauli encodings: I=0, X=1, Z=2, Y=3; this is purely internal
I, X, Z, Y = 0, 1, 2, 3

def depolarizing_error_on_data(n_data: int, p: float, rng: np.random.Generator) -> np.ndarray:
    # Apply iid Pauli with probs: I:1-p, X: p/3, Z: p/3, Y: p/3 to data only
    errs = np.zeros(n_data, dtype=np.int8)
    mask = rng.random(n_data) < p
    # for masked positions, sample X,Z,Y equiprobably
    k = mask.sum()
    if k > 0:
        draws = rng.integers(1, 4, size=k)  # 1..3
        errs[mask] = draws
    return errs

def circuit_noise_sample(n_data: int, n_x: int, n_z: int, p: float, T: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    # Simplified circuit noise: at each cycle, after each gate and measurement, apply depolarizing channels.
    # We approximate by sampling data and measurement errors per cycle with rates related to p.
    data_errs = np.zeros((T, n_data), dtype=np.int8)
    meas_errs_x = np.zeros((T, n_x), dtype=np.bool_)  # measurement bit flips for X checks
    meas_errs_z = np.zeros((T, n_z), dtype=np.bool_)  # measurement bit flips for Z checks
    for t in range(T):
        data_errs[t] = depolarizing_error_on_data(n_data, p, rng)
        # measurement error with prob p (paper assumes same order as data error; see Sec. I and model spec)
        meas_errs_x[t] = rng.random(n_x) < p
        meas_errs_z[t] = rng.random(n_z) < p
    return {"data": data_errs, "merr_x": meas_errs_x, "merr_z": meas_errs_z}

def extract_syndrome_once(code: RotatedSurfaceCode, data_paulis: np.ndarray, meas_flip_x: np.ndarray=None, meas_flip_z: np.ndarray=None, rng: np.random.Generator=None) -> Tuple[np.ndarray, np.ndarray]:
    # Compute X and Z check outcomes from data Paulis.
    # X-checks detect Z or Y on incident data; Z-checks detect X or Y on incident data.
    x_syn = np.zeros(code.n_x_checks(), dtype=np.int8)
    z_syn = np.zeros(code.n_z_checks(), dtype=np.int8)
    for i, cc in enumerate(code.x_check_coords):
        nbrs = code.data_neighbors_of_check(cc, "X")
        # parity of (Z or Y) on neighbors
        val = 0
        for q in nbrs:
            val ^= 1 if (data_paulis[q] in (Z, Y)) else 0
        x_syn[i] = val
    for i, cc in enumerate(code.z_check_coords):
        nbrs = code.data_neighbors_of_check(cc, "Z")
        # parity of (X or Y)
        val = 0
        for q in nbrs:
            val ^= 1 if (data_paulis[q] in (X, Y)) else 0
        z_syn[i] = val
    if meas_flip_x is not None:
        x_syn ^= meas_flip_x.astype(np.int8)
    if meas_flip_z is not None:
        z_syn ^= meas_flip_z.astype(np.int8)
    return x_syn, z_syn

def extract_syndrome_window(code: RotatedSurfaceCode, data_errs_T: np.ndarray, merr_x_T: np.ndarray=None, merr_z_T: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
    # Build time series of syndromes; returns arrays of shape (T, n_x) and (T, n_z)
    T = data_errs_T.shape[0]
    Xs = []
    Zs = []
    for t in range(T):
        mx = merr_x_T[t] if merr_x_T is not None else None
        mz = merr_z_T[t] if merr_z_T is not None else None
        xs, zs = extract_syndrome_once(code, data_errs_T[t], mx, mz)
        Xs.append(xs)
        Zs.append(zs)
    return np.stack(Xs, axis=0), np.stack(Zs, axis=0)

def detection_events_time_diff(syndrome_T: np.ndarray) -> np.ndarray:
    # Compute detection events as flips in time (0<->1) for a given check stream (Sec. I, detection events).
    # Returns array of same shape with 1 marking a flip wrt previous cycle.
    T = syndrome_T.shape[0]
    det = np.zeros_like(syndrome_T)
    for t in range(1, T):
        det[t] = syndrome_T[t] ^ syndrome_T[t-1]
    return det

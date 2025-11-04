# simple_decoder.py
# Fast "simple decoder" used in the HLD to propose physical corrections from the last-cycle syndrome
# Then NN2 predicts whether a logical correction is needed to cancel any induced logical error.
from typing import List, Tuple
import numpy as np
from lattice import RotatedSurfaceCode

class SimpleDecoder:
    def __init__(self, code: RotatedSurfaceCode):
        self.code = code
        self.ops = code.logical_operators()

    def propose_corrections(self, x_syn_last: np.ndarray, z_syn_last: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Naive strategy: connect each detection event to nearest boundary with strings of matching Pauli.
        # For X-type checks (detect Z-type errors), propose Z corrections; for Z-type checks, propose X corrections.
        # We return proposed Pauli corrections on data: arrays of same length as n_data with booleans for X and Z.
        n = self.code.n_data()
        corr_X = np.zeros(n, dtype=np.bool_)  # X corrections to apply
        corr_Z = np.zeros(n, dtype=np.bool_)  # Z corrections to apply

        # For simplicity, route vertical to top/bottom and horizontal to left/right by greedy shortest path
        # This produces valid-syndrome corrections quickly; quality is left to NN2 to correct at logical level.
        # Note: This aligns with the paper's "simple decoder" spirit: fast, always produces a valid matching. :contentReference[oaicite:1]{index=1}
        def draw_path_Z(check_idx: int):
            # Z correction chain for an X-type detection event
            cc = self.code.x_check_coords[check_idx]
            r, c = cc
            # choose nearest vertical boundary
            target_r = 0 if r < (self.code.d - 1) / 2 else (self.code.d - 2)
            rr = r
            # vertical segment
            while rr != target_r:
                # toggle Z on data along vertical corridor
                q = self.code._data_index((min(rr, rr+1), c))
                corr_Z[q] ^= True
                rr += -1 if rr > target_r else +1
            # small horizontal nudge to hit a boundary cell if needed (no-op for our simplified grid)

        def draw_path_X(check_idx: int):
            # X correction chain for a Z-type detection event
            cc = self.code.z_check_coords[check_idx]
            r, c = cc
            target_c = 0 if c < (self.code.d - 1) / 2 else (self.code.d - 2)
            ccurr = c
            while ccurr != target_c:
                q = self.code._data_index((r, min(ccurr, ccurr+1)))
                corr_X[q] ^= True
                ccurr += -1 if ccurr > target_c else +1

        for i, bit in enumerate(x_syn_last):
            if bit == 1:
                draw_path_Z(i)
        for i, bit in enumerate(z_syn_last):
            if bit == 1:
                draw_path_X(i)

        return corr_X, corr_Z

    def logical_from_corrections(self, corr_X: np.ndarray, corr_Z: np.ndarray) -> str:
        # Determine which logical operator (if any) is induced by the proposed corrections
        ops = self.ops
        xbar = ops["Xbar"]
        zbar = ops["Zbar"]
        # parity along logical supports
        px = int(np.sum(corr_X[zbar]) % 2 == 1)  # X anticommutes with Zbar
        pz = int(np.sum(corr_Z[xbar]) % 2 == 1)  # Z anticommutes with Xbar
        if px and pz:
            return "Ybar"
        elif px:
            return "Xbar"
        elif pz:
            return "Zbar"
        else:
            return "I"

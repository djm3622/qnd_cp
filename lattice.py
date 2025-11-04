# lattice.py
# Rotated surface code geometry and utilities
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

Coord = Tuple[int, int]

@dataclass
class RotatedSurfaceCode:
    # code distance d must be odd: 3,5,7,...
    d: int

    def __post_init__(self):
        assert self.d % 2 == 1 and self.d >= 3
        # number of data and ancilla checks (X and Z type) on rotated lattice
        # For distance d rotated code: data qubits = d^2, X-checks ~ (d^2-1)/2, Z-checks ~ (d^2-1)/2
        # We index checks separately for clarity.
        self.data_coords = self._build_data_coords()
        self.x_check_coords, self.z_check_coords = self._build_check_coords()

    def n_data(self) -> int:
        return self.d * self.d

    def n_x_checks(self) -> int:
        return len(self.x_check_coords)

    def n_z_checks(self) -> int:
        return len(self.z_check_coords)

    def n_checks_total(self) -> int:
        return self.n_x_checks() + self.n_z_checks()

    def _build_data_coords(self) -> List[Coord]:
        # data at integer lattice points inside d x d grid
        return [(r, c) for r in range(self.d) for c in range(self.d)]

    def _build_check_coords(self) -> Tuple[List[Coord], List[Coord]]:
        # In rotated code, checks are centered on plaquettes.
        # We place X-checks on "white" plaquettes and Z-checks on "gray" plaquettes, alternating.
        x_checks, z_checks = [], []
        for r in range(self.d - 1):
            for c in range(self.d - 1):
                # parity decides type (like Figure 3 in the paper)
                if (r + c) % 2 == 0:
                    x_checks.append((r, c))
                else:
                    z_checks.append((r, c))
        return x_checks, z_checks

    def data_neighbors_of_check(self, check_coord: Coord, check_type: str) -> List[int]:
        # Each plaquette check touches up to 4 surrounding data qubits (2 on edges)
        r, c = check_coord
        nbrs = []
        # corners of the plaquette
        for dr, dc in [(0,0),(0,1),(1,0),(1,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.d and 0 <= cc < self.d:
                nbrs.append(self._data_index((rr, cc)))
        return nbrs

    def _data_index(self, coord: Coord) -> int:
        r, c = coord
        return r * self.d + c

    def logical_operators(self) -> Dict[str, List[int]]:
        # Logical Xbar: any chain of X across Z-boundaries; choose a minimal rep (top to bottom column)
        # Logical Zbar: any chain of Z across X-boundaries; choose a minimal rep (left to right row)
        # We pick fixed minimal reps for detection of logical flips.
        zbar = [self._data_index((self.d // 2, c)) for c in range(self.d)]
        xbar = [self._data_index((r, self.d // 2)) for r in range(self.d)]
        return {"Xbar": xbar, "Zbar": zbar}

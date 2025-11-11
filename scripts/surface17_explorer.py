from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_qec.circuits import SurfaceCodeCircuit


def build_surface17_circuit(T: int = 3) -> SurfaceCodeCircuit:
    """
    Construct a distance-3 surface code circuit (17 qubits) with T cycles.

    Parameters
    ----------
    T : int
        Number of QEC cycles (rounds of stabilizer measurement).

    Returns
    -------
    SurfaceCodeCircuit
        qiskit_qec circuit object for the distance-3 surface code.
    """
    sc = SurfaceCodeCircuit(
        d=3,          # code distance => surface-17 layout
        T=T,          # number of measurement rounds
        basis="z",    # choose logical basis
        resets=True,  # reset ancillas between cycles
    )
    return sc


def extract_qiskit_circuit(sc: SurfaceCodeCircuit) -> QuantumCircuit:
    """
    Extract the underlying Qiskit QuantumCircuit from a SurfaceCodeCircuit.

    qiskit-qec has changed internally a few times; this function tries
    the most common patterns and fails loudly if none apply.
    """
    obj = getattr(sc, "circuit", sc)

    if isinstance(obj, QuantumCircuit):
        return obj

    # Many versions store a dict of circuits, keyed by string or int
    if isinstance(obj, dict):
        first_key = next(iter(obj.keys()))
        qc = obj[first_key]
        if isinstance(qc, QuantumCircuit):
            return qc

    # Some versions store a list of circuits
    if hasattr(obj, "__iter__"):
        for item in obj:
            if isinstance(item, QuantumCircuit):
                return item

    raise TypeError("Could not locate an embedded QuantumCircuit in SurfaceCodeCircuit")


def draw_surface17_circuit(
    sc: Optional[SurfaceCodeCircuit] = None,
    T: int = 3,
    scale: float = 1.5,
    width_in: float = 200.0,
    height_in: float = 15.0,
    dpi: int = 300,
):
    """
    Draw the surface-17 circuit as one long, very large row for readability.
    """
    if sc is None:
        sc = build_surface17_circuit(T=T)

    qc = extract_qiskit_circuit(sc)

    fig = qc.draw("mpl", fold=-1, scale=scale)

    fig.set_size_inches(width_in, height_in)
    fig.set_dpi(dpi)

    fig.suptitle(f"Surface-17 circuit (distance 3, T={T} cycles)", fontsize=20)
    return fig


@dataclass
class Surface17Layout:
    """
    Simple geometric layout for a distance-3 rotated surface code.

    We separate conceptual labels (d0..d8, x0..x3, z0..z3) from any
    specific qiskit-qec index ordering. This is for visualization.
    """

    data_coords: Dict[str, Tuple[float, float]]
    x_anc_coords: Dict[str, Tuple[float, float]]
    z_anc_coords: Dict[str, Tuple[float, float]]


def make_surface17_layout() -> Surface17Layout:
    """
    Create a simple planar drawing of the surface-17 lattice.

    Data qubits are arranged on a 3x3 grid.
    X and Z ancillas sit between data qubits in a checkerboard pattern.
    This matches the usual rotated surface code picture, but labels
    are purely conceptual.
    """
    data_coords = {}
    idx = 0
    for j in range(3):
        for i in range(3):
            data_coords[f"d{idx}"] = (float(i), float(j))
            idx += 1

    # Place Z-check ancillas roughly at "plaquette centers"
    z_anc_coords = {
        "z0": (0.5, 0.0),
        "z1": (1.5, 0.0),
        "z2": (0.5, 1.0),
        "z3": (1.5, 1.0),
    }

    # Place X-check ancillas slightly shifted (checkerboard)
    x_anc_coords = {
        "x0": (1.0, 0.5),
        "x1": (0.0, 0.5),
        "x2": (1.0, 1.5),
        "x3": (2.0, 1.5),
    }

    return Surface17Layout(
        data_coords=data_coords,
        x_anc_coords=x_anc_coords,
        z_anc_coords=z_anc_coords,
    )


def plot_surface17_layout(
    layout: Optional[Surface17Layout] = None,
    data_errors: Optional[Iterable[str]] = None,
    highlight_ancillas: Optional[Iterable[str]] = None,
):
    """
    Plot the conceptual surface-17 lattice.

    Parameters
    ----------
    layout : Surface17Layout, optional
        If None, a default layout is created.

    data_errors : iterable of str, optional
        Labels of data qubits to highlight as "errored" (e.g. {"d0","d4"}).

    highlight_ancillas : iterable of str, optional
        Labels of ancillas ("x0".."x3" or "z0".."z3") to highlight.
    """
    if layout is None:
        layout = make_surface17_layout()

    data_errors = set(data_errors or [])
    highlight_ancillas = set(highlight_ancillas or [])

    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot data qubits
    for name, (x, y) in layout.data_coords.items():
        if name in data_errors:
            facecolor = "red"
        else:
            facecolor = "white"
        ax.scatter(x, y, s=400, edgecolor="black", facecolor=facecolor, zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, zorder=4)

    # Plot Z ancillas
    for name, (x, y) in layout.z_anc_coords.items():
        if name in highlight_ancillas:
            fc = "orange"
        else:
            fc = "lightgray"
        ax.scatter(x, y, marker="s", s=250, edgecolor="black", facecolor=fc, zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=9, zorder=4)

    # Plot X ancillas
    for name, (x, y) in layout.x_anc_coords.items():
        if name in highlight_ancillas:
            fc = "yellowgreen"
        else:
            fc = "lightgray"
        ax.scatter(x, y, marker="D", s=250, edgecolor="black", facecolor=fc, zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=9, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Conceptual layout of surface-17 (data and ancilla qubits)")
    plt.tight_layout()


def sample_data_errors(p_phys: float, n_data: int = 9, rng: Optional[np.random.Generator] = None):
    """
    Sample a random X error pattern on data qubits with probability p_phys.

    This is NOT using qiskit-qec, it is a simple classical toy model.
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(n_data) < p_phys
    return mask 


def plot_random_error_configuration(p_phys: float = 0.1, rng: Optional[np.random.Generator] = None):
    """
    Draw a random error pattern on the surface-17 lattice.
    """
    layout = make_surface17_layout()
    mask = sample_data_errors(p_phys=p_phys, n_data=9, rng=rng)

    errored = [f"d{i}" for i, val in enumerate(mask) if val]
    plot_surface17_layout(layout, data_errors=errored)
    plt.title(f"Random X error pattern on data qubits (p_phys={p_phys})")
    plt.tight_layout()

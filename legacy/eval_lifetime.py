# eval_lifetime.py
# Lifetime simulations: sweep physical error prob and compute logical error rate
import argparse
import numpy as np
import torch
from lattice import RotatedSurfaceCode
from errors import depolarizing_error_on_data, circuit_noise_sample, extract_syndrome_once, extract_syndrome_window
from simple_decoder import SimpleDecoder
from models import HLD_NN2_LSTM, HLD_NN1_LSTM

def logical_error_rate_depol(model, code, p, trials=20000, seed=0):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    sd = SimpleDecoder(code)
    F = code.n_x_checks() + code.n_z_checks()
    n_log_err = 0
    for _ in range(trials):
        data_err = depolarizing_error_on_data(code.n_data(), p, rng)
        xs, zs = extract_syndrome_once(code, data_err)
        xlast, zlast = xs, zs
        # simple proposal
        corrX, corrZ = sd.propose_corrections(xlast, zlast)
        logical = sd.logical_from_corrections(corrX, corrZ)
        # NN2 suggestion to cancel logical
        feat = torch.tensor(np.concatenate([xs, zs])[None, None, :].astype(np.float32), device=device)
        probs = model(feat).detach().cpu().numpy()[0]
        add = np.argmax(probs)
        # combine
        if add == 1 and logical == "Xbar":  # cancel Xbar
            logical_final = "I"
        elif add == 2 and logical == "Zbar":
            logical_final = "I"
        elif add == 3 and logical == "Ybar":
            logical_final = "I"
        else:
            logical_final = logical
        n_log_err += int(logical_final != "I")
    return n_log_err / trials

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--mode", choices=["depol"], default="depol")
    ap.add_argument("--pmin", type=float, default=0.02)
    ap.add_argument("--pmax", type=float, default=0.18)
    ap.add_argument("--steps", type=int, default=9)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    code = RotatedSurfaceCode(args.d)
    F = code.n_x_checks() + code.n_z_checks()
    model = HLD_NN2_LSTM(input_dim=F).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ps = np.linspace(args.pmin, args.pmax, args.steps)
    for p in ps:
        rate = logical_error_rate_depol(model, code, p, trials=20000)
        print(f"p={p:.5f} LER={rate:.6e}")

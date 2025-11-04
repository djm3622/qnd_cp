# train_hld.py
# Train the high level decoder for either depolarizing or circuit noise models.
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lattice import RotatedSurfaceCode
from models import HLD_NN2_LSTM, HLD_NN1_LSTM
from datasets import HLDDatasetDepol, HLDDatasetCircuit

def stepwise_lr(optim, factor=0.5, min_lr=1e-4):
    for g in optim.param_groups:
        g["lr"] = max(min_lr, g["lr"] * factor)

def train_depol(d=3, p=0.10, n_train=200_000, n_val=20_000, batch=1000, epochs=5000, seed=0, hidden=(16,4), lr=1e-2):
    device = "mps"
    code = RotatedSurfaceCode(d=d)
    ds_tr = HLDDatasetDepol(code, p, n_train, seed)
    ds_va = HLDDatasetDepol(code, p, n_val, seed+1)

    F = code.n_x_checks() + code.n_z_checks()
    model = HLD_NN2_LSTM(input_dim=F, hidden_dims=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Paper used MSE with sigmoid outputs to approximate probabilities. :contentReference[oaicite:6]{index=6}
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False)

    best_va = math.inf
    patience_ctr = 0
    for ep in range(1, epochs+1):
        model.train()
        tot = 0.0
        for X, Y in tr_loader:
            X = X.to(device)  # [B, 1, F]
            Y = Y.to(device)  # [B, 4]
            opt.zero_grad()
            P = model(X)
            loss = loss_fn(P, Y)
            loss.backward()
            opt.step()
            tot += loss.item() * X.size(0)
        tr_loss = tot / len(ds_tr)

        # validation
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for X, Y in va_loader:
                X = X.to(device); Y = Y.to(device)
                P = model(X)
                loss = loss_fn(P, Y)
                tot += loss.item() * X.size(0)
            va_loss = tot / len(ds_va)

        # simple plateau LR schedule as in paper: step down when not improving after some iters. :contentReference[oaicite:7]{index=7}
        if va_loss + 1e-6 < best_va:
            best_va = va_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr % 50 == 0:
                stepwise_lr(opt, factor=0.5)

        if ep % 100 == 0 or ep == 1:
            print(f"[ep {ep}] train_mse={tr_loss:.6f} val_mse={va_loss:.6f} lr={opt.param_groups[0]['lr']:.2e}")

    return model

def train_circuit(d=5, p=1.2e-3, n_train=2_000_000, n_val=200_000, batch=10000, epochs=5000, seed=0, hidden=(16,4), lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code = RotatedSurfaceCode(d=d)
    ds_tr = HLDDatasetCircuit(code, p, n_train, seed, T=d)
    ds_va = HLDDatasetCircuit(code, p, n_val, seed+1, T=d)

    F = code.n_x_checks() + code.n_z_checks()
    nn2 = HLD_NN2_LSTM(input_dim=F, hidden_dims=hidden).to(device)
    nn1 = HLD_NN1_LSTM(input_dim=F, hidden_dims=hidden, out_dim=F).to(device)

    opt2 = torch.optim.Adam(nn2.parameters(), lr=lr)
    opt1 = torch.optim.Adam(nn1.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False)

    best_va = math.inf
    patience_ctr = 0
    for ep in range(1, epochs+1):
        nn1.train(); nn2.train()
        tot2 = 0.0; tot1 = 0.0
        for X, (Y2, Y1) in tr_loader:
            X = X.to(device)     # [B, T, F]
            Y2 = Y2.to(device)   # [B, 4]
            Y1 = Y1.to(device)   # [B, F]

            # train NN2
            opt2.zero_grad()
            P2 = nn2(X)
            loss2 = loss_fn(P2, Y2)
            loss2.backward()
            opt2.step()
            tot2 += loss2.item() * X.size(0)

            # train NN1
            opt1.zero_grad()
            P1 = nn1(X)
            loss1 = loss_fn(P1, Y1)
            loss1.backward()
            opt1.step()
            tot1 += loss1.item() * X.size(0)

        tr_loss2 = tot2 / len(ds_tr)
        tr_loss1 = tot1 / len(ds_tr)

        # validation on NN2 (primary metric)
        nn1.eval(); nn2.eval()
        with torch.no_grad():
            tot2 = 0.0
            for X, (Y2, _) in va_loader:
                X = X.to(device); Y2 = Y2.to(device)
                P2 = nn2(X)
                loss2 = loss_fn(P2, Y2)
                tot2 += loss2.item() * X.size(0)
            va_loss2 = tot2 / len(ds_va)

        if va_loss2 + 1e-6 < best_va:
            best_va = va_loss2
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr % 50 == 0:
                stepwise_lr(opt2, factor=0.5)
                stepwise_lr(opt1, factor=0.5)

        if ep % 100 == 0 or ep == 1:
            print(f"[ep {ep}] tr_mse_nn2={tr_loss2:.6f} val_mse_nn2={va_loss2:.6f} lr2={opt2.param_groups[0]['lr']:.2e} lr1={opt1.param_groups[0]['lr']:.2e}")

    return nn1, nn2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["depol","circuit"], required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--train", type=int, default=200_000)
    ap.add_argument("--val", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.model == "depol":
        train_depol(d=args.d, p=args.p, n_train=args.train, n_val=args.val, batch=args.batch, epochs=args.epochs, seed=args.seed)
    else:
        # circuit default hyperparams align with larger batch as in paper (1k or 10k) :contentReference[oaicite:8]{index=8}
        train_circuit(d=max(5,args.d), p=args.p, n_train=args.train, n_val=args.val, batch=args.batch, epochs=args.epochs, seed=args.seed)

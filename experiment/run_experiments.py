"""
run_experiments.py

Runs the full experiment grid and writes one .pt per configuration into
outputs/. Display is done separately, in the notebook, which only loads.

    python run_experiments.py            # everything not already done
    python run_experiments.py --force    # redo everything
    python run_experiments.py --only bimodal_theopoula_L20

Expect this to take hours. The dominant cost is the W2 evaluation, not
the training: exact optimal transport at N = 10^4 is roughly O(N^2.5).
"""

import argparse
import os
import time

import numpy as np
import torch

import explib as ex


# ============================================================
#  PARAMETERS TO CHANGE
# ============================================================

OUT_DIR = "outputs"

# --- what to run ------------------------------------------------------
TARGETS    = ["gaussian", "bimodal"]                      # drop "gaussian" to skip the control
DEPTHS     = [2, 20]                                      # number of AFFINE MAPS (L = depth - 1 hidden layers)
OPTIMISERS = ["adam", "sgld", "tusla", "theopoula"]
USE_SKIP   = False                                        # True adds a linear path z_0 -> output
SEEDS      = [0]                                          # add more for error bars across seeds

# --- target geometry --------------------------------------------------
D          = 2
MU         = np.array([1.3, 1.3])                         # mixture modes at +/- MU; Gaussian mean at MU
SIGMA_0    = 1.0                                          # component standard deviation
N_DATA     = 1_000_000                                    # size of the training pool

# --- forward / reverse process ---------------------------------------
T          = 2.0                                          # terminal time
T_0        = 1e-3                                         # epsilon, the early-stopping time
GAMMA      = 1e-3                                         # Euler-Maruyama step size
KAPPA_MODE = "sigma2"                                     # "sigma2" (recommended) or "one"

# --- training ---------------------------------------------------------
WIDTH      = 128
N_ITERS    = 10_000
BATCH_SIZE = 512
PRINT_EVERY = 2000

# --- evaluation -------------------------------------------------------
N_W2         = 10_000                                     # samples per W2 evaluation
N_REPEATS    = 3                                          # W2 repeats; cost scales linearly
NUM_ITER_MAX = 100_000_000                                # network-simplex ceiling; 10^8 is safe at N = 10^4
N_PLOT       = 20_000                                     # samples saved for histograms and scatter
FLOOR_NS     = [1_000, 2_000, 5_000, 10_000]              # floor is measured at each of these
FLOOR_REPEATS = 3
SCORE_GRID_TS = [0.05, 0.5, 2.0]                          # noise levels for the score-field figure

# --- gradient diagnostics --------------------------------------------
# Iterations at which the raw layerwise gradient and the applied update
# are recorded. Iteration 1 is effectively "at initialisation".
GRAD_CHECKPOINTS = (1, 100, 1000, 5000, N_ITERS)

# --- optimiser hyperparameters ---------------------------------------
# lam   : step size / learning rate
# beta  : inverse temperature; large beta means almost no injected noise
# eta   : regularisation strength (TUSLA and TheoPouLa only)
# r     : regularisation exponent; see r_min_theopoula for the admissible floor
# eps_b : boosting denominator offset (TheoPouLa only)
OPT_PARAMS = {
    "adam":      dict(lam=1e-3),
    "sgld":      dict(lam=1e-3, beta=1e10),
    "tusla":     dict(lam=0.02, beta=1e10, eta=5e-4 * np.sqrt(0.02), r=2.0),
    "theopoula": dict(lam=0.02, beta=1e10, eta=5e-4 * np.sqrt(0.02), r=2.0,
                      eps_b=0.2),
}

# ============================================================
#  END OF PARAMETERS
# ============================================================


def tag_of(target, optimiser, n_layers, seed, skip):
    s = f"{target}_{optimiser}_L{n_layers}"
    if skip:
        s += "_skip"
    if seed != 0:
        s += f"_s{seed}"
    return s


def run_one(target_name, optimiser, n_layers, seed, device):
    tag = tag_of(target_name, optimiser, n_layers, seed, USE_SKIP)
    print(f"\n=== {tag} ===")
    t_start = time.time()

    # deterministic per configuration, so arms are independent and reruns match
    ex.set_seed(abs(hash(tag)) % (2 ** 31) + seed)

    target = ex.make_target(target_name, D, MU, SIGMA_0)
    X_data = target["sample"](N_DATA)

    net = ex.build_net(D, n_layers, WIDTH, use_skip=USE_SKIP).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    run_cfg = dict(t_0=T_0, T=T, batch_size=BATCH_SIZE, n_iters=N_ITERS,
                   kappa_mode=KAPPA_MODE, print_every=PRINT_EVERY)
    opt_cfg = dict(name=optimiser, **OPT_PARAMS[optimiser])

    # raw gradient profile before any training
    g_init = ex.layerwise_grad_norms(net, X_data, T_0, T, D, BATCH_SIZE,
                                     device, KAPPA_MODE)

    diag = ex.train(net, X_data, target, opt_cfg, run_cfg, device,
                    grad_checkpoints=GRAD_CHECKPOINTS)

    # samples kept for plotting distributions
    samples_plot = ex.em_sample(net, GAMMA, T, D, N_PLOT, device) \
        if not diag["diverged"] else None

    # W2 against the target
    if diag["diverged"]:
        w2_mean = w2_std = float("nan")
        w2_all = []
    else:
        print(f"    computing W2 at N = {N_W2} ({N_REPEATS} repeats) ...")
        w2_mean, w2_std, w2_all = ex.estimate_w2(
            net, target, N_W2, N_REPEATS, GAMMA, T, D, device, NUM_ITER_MAX
        )
        print(f"    W2 = {w2_mean:.4f} +/- {w2_std:.4f}")

    grid = ex.score_field_grid(net, target, SCORE_GRID_TS, device) \
        if not diag["diverged"] else None

    result = {
        "config": dict(
            tag=tag, target=target_name, optimiser=optimiser,
            n_layers=n_layers, n_hidden=n_layers - 1, width=WIDTH,
            use_skip=USE_SKIP, seed=seed, n_params=n_params,
            T=T, t_0=T_0, gamma=GAMMA, kappa_mode=KAPPA_MODE,
            n_iters=N_ITERS, batch_size=BATCH_SIZE,
            N_w2=N_W2, n_repeats=N_REPEATS, N_plot=N_PLOT,
            mu=MU, sigma_0=SIGMA_0, opt_params=OPT_PARAMS[optimiser],
        ),
        "r_min_theopoula": ex.r_min_theopoula(n_layers),
        "r_used": OPT_PARAMS[optimiser].get("r"),
        "r_admissible": (OPT_PARAMS[optimiser].get("r") is not None and
                         OPT_PARAMS[optimiser]["r"] >= ex.r_min_theopoula(n_layers)),
        "grad_norms_init": g_init,
        "samples": samples_plot,
        "w2_mean": w2_mean, "w2_std": w2_std, "w2_all": w2_all,
        "score_grid": grid,
        "net_state": {k: v.cpu() for k, v in net.state_dict().items()},
        "runtime_s": time.time() - t_start,
        **diag,
    }
    ex.save_results(result, os.path.join(OUT_DIR, f"{tag}.pt"))
    return result


def run_floors():
    """
    Floor per target and per N. Saved once, separately, because every
    figure needs it and it must never be computed at a different N from
    the measurement it is drawn against.
    """
    path = os.path.join(OUT_DIR, "floors.pt")
    out = {}
    for tname in TARGETS:
        target = ex.make_target(tname, D, MU, SIGMA_0)
        out[tname] = {}
        for N in FLOOR_NS:
            print(f"  floor: {tname}, N = {N} ...", end="", flush=True)
            t0 = time.time()
            m, s, allv = ex.w2_floor(target, N, FLOOR_REPEATS, NUM_ITER_MAX)
            print(f" {m:.4f} +/- {s:.4f}   ({time.time() - t0:.0f}s)")
            out[tname][N] = dict(mean=m, std=s, all=allv)
        # convenience: the floor at the N actually used for reporting
        out[tname]["mean"] = out[tname][N_W2]["mean"]
        out[tname]["std"] = out[tname][N_W2]["std"]
    ex.save_results(out, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="recompute configurations whose output already exists")
    ap.add_argument("--only", type=str, default=None,
                    help="run a single tag, e.g. bimodal_theopoula_L20")
    ap.add_argument("--skip-floors", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = ex.get_device()
    print(f"device: {device}")

    if not args.skip_floors and (args.force or
                                 not os.path.exists(os.path.join(OUT_DIR, "floors.pt"))):
        print("\n=== finite-sample floors ===")
        run_floors()

    total = time.time()
    for seed in SEEDS:
        for tname in TARGETS:
            for n_layers in DEPTHS:
                for opt in OPTIMISERS:
                    tag = tag_of(tname, opt, n_layers, seed, USE_SKIP)
                    if args.only and tag != args.only:
                        continue
                    path = os.path.join(OUT_DIR, f"{tag}.pt")
                    if os.path.exists(path) and not args.force:
                        print(f"skip {tag} (exists)")
                        continue
                    run_one(tname, opt, n_layers, seed, device)

    print(f"\n=== complete in {(time.time() - total) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
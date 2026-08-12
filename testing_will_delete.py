"""
run_iteration_sweep.py

Trains ONE configuration continuously and evaluates W2 at a sequence of
iteration counts, to test whether the sampler keeps improving long after
the training loss has reached its floor.

The loss cannot answer this. With kappa(t) = sigma_t^2 the objective is
    E|z + sigma_t s|^2 = d e^{-2t}  +  sigma_t^2 E|s - grad log p_t|^2,
so the first term is irreducible and the second is weighted by sigma_t^2,
which is 0.002 at t = 1e-3. Score error at small t is therefore almost
invisible to the loss, while it is exactly what determines sample quality.

Training proceeds in segments and is NOT restarted at each checkpoint.
This is exact for SGLD, TUSLA and TheoPouLa, which carry no optimiser
state between steps. It is NOT exact for Adam, whose moment estimates
would reset at every segment boundary.

    python run_iteration_sweep.py
    python run_iteration_sweep.py --force
"""

import argparse
import os
import time

import numpy as np

import functions4 as fct


# ==========================================================================
#                       PARAMETERS TO CHANGE
# ==========================================================================

OUT_DIR = "outputs"
OUT_NAME = "itersweep"

# --- the configuration under test ---------------------------------------
TARGET    = "bimodal"        # "bimodal" or "gaussian"
ARCHS     = [20]             # add 2 to get the crossover against the shallow net
OPTIMISER = "theopoula"      # stateless, so segmented training is exact
WIDTH     = 128
USE_SKIP  = False
SEED      = 0

# --- iteration checkpoints ----------------------------------------------
# Log-spaced, because any remaining improvement will be slow and
# multiplicative rather than linear. Trim the tail if you want it shorter.
CHECKPOINTS = [500, 1_000, 2_000, 5_000, 10_000, 20_000,
               50_000, 100_000, 200_000, 500_000]

# --- optimiser hyperparameters ------------------------------------------
# Left explicit rather than read from best_params.pt so this script stands
# alone. Match them to your grid search result if you have one.
PARAMS = dict(lam=0.02, beta=1e10, r=2.0, eps_b=0.2,
              eta=5e-4 * np.sqrt(0.02))

# --- target geometry -----------------------------------------------------
D       = 2
MU      = np.array([1.3, 1.3])
SIGMA_0 = 1.0
N_DATA  = 1_000_000

# --- process and training ------------------------------------------------
T           = 2.0
T_0         = 1e-3
GAMMA       = 1e-3
KAPPA_MODE  = "sigma2"
BATCH_SIZE  = 512
PRINT_EVERY = 5_000

# --- evaluation at each checkpoint --------------------------------------
N_W2         = 10_000
N_REPEATS    = 2             # 2 gives a spread without doubling the cost
NUM_ITER_MAX = 100_000_000
EM_BATCH     = 20_000
SCORE_TS     = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]   # unweighted score error
N_MC_SCORE   = 20_000
KEEP_SAMPLES_AT = [10_000, 100_000, 500_000]   # checkpoints whose samples are stored

# --- floor, for context on the same axes --------------------------------
FLOOR_REPEATS = 3

# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = fct.get_device()
    print(f"device: {device}")

    target = fct.make_target(TARGET, D, MU, SIGMA_0)
    X_data = target["sample"](N_DATA)

    # floor at the reporting N, so the curve can be read against it
    floor_path = os.path.join(OUT_DIR, f"{OUT_NAME}_floor.pt")
    if args.force or not os.path.exists(floor_path):
        print(f"\nfloor at N={N_W2} ...")
        fm, fs, fall = fct.w2_floor(target, N_W2, FLOOR_REPEATS, NUM_ITER_MAX)
        fct.save_results(dict(mean=fm, std=fs, all=fall, N=N_W2), floor_path)
    else:
        f = fct.load_results(floor_path)
        fm, fs = f["mean"], f["std"]
    print(f"floor: {fm:.4f} +/- {fs:.4f}")

    for arch in ARCHS:
        tag = f"{OUT_NAME}_{TARGET}_{OPTIMISER}_{fct.q_of(arch):.0f}q_L{arch}"
        path = os.path.join(OUT_DIR, f"{tag}.pt")
        if os.path.exists(path) and not args.force:
            print(f"skip {tag} (exists)")
            continue

        print(f"\n================ {tag} ================")
        fct.set_seed(fct.seed_from_tag(tag, SEED))

        net = fct.build_net(D, arch, WIDTH, USE_SKIP).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"{n_params} parameters, q = {fct.q_of(arch):.0f}, "
              f"r_min = {fct.r_min(arch):.1f}, r used = {PARAMS['r']}")

        record = dict(
            config=dict(tag=tag, target=TARGET, optimiser=OPTIMISER, arch=arch,
                        n_hidden=arch - 1, width=WIDTH, use_skip=USE_SKIP,
                        seed=SEED, n_params=n_params, T=T, t_0=T_0,
                        gamma=GAMMA, kappa_mode=KAPPA_MODE,
                        batch_size=BATCH_SIZE, N_w2=N_W2,
                        n_repeats=N_REPEATS, params=PARAMS,
                        mu=MU, sigma_0=SIGMA_0),
            q=fct.q_of(arch), r_min=fct.r_min(arch), r_used=PARAMS["r"],
            r_admissible=bool(PARAMS["r"] >= fct.r_min(arch)),
            floor=dict(mean=fm, std=fs, N=N_W2),
            checkpoints=[], w2_mean={}, w2_std={}, w2_all={},
            score_error={}, theta_norm={}, loss_tail={},
            grad_norms={}, update_norms={}, samples={},
            loss_history=[], theta_norm_history=[],
            diverged=False, nan_iter=None,
        )

        done = 0
        t_start = time.time()

        for ckpt in CHECKPOINTS:
            n_new = ckpt - done
            if n_new <= 0:
                continue
            print(f"\n-- training {done} -> {ckpt} --")

            rc = dict(t_0=T_0, T=T, batch_size=BATCH_SIZE, n_iters=n_new,
                      kappa_mode=KAPPA_MODE, print_every=PRINT_EVERY)
            diag = fct.train(net, X_data, target,
                             dict(name=OPTIMISER, **PARAMS), rc, device,
                             grad_checkpoints=(n_new,))

            record["loss_history"].extend(diag["loss_history"])
            record["theta_norm_history"].extend(diag["theta_norm_history"])

            if diag["diverged"]:
                record["diverged"] = True
                record["nan_iter"] = done + diag["nan_iter"]
                print(f"DIVERGED at global iteration {record['nan_iter']}")
                fct.save_results(record, path)
                break

            done = ckpt

            # gradient profile at this checkpoint
            if n_new in diag["grad_norms"]:
                record["grad_norms"][ckpt] = diag["grad_norms"][n_new]
            if n_new in diag["update_norms"]:
                record["update_norms"][ckpt] = diag["update_norms"][n_new]

            # cheap, unweighted, resolved in t: this is the quantity the
            # loss cannot see
            se = fct.score_error_by_t(net, target, SCORE_TS, device,
                                      n_mc=N_MC_SCORE)
            net.train()

            # the expensive one
            m, s, allv = fct.estimate_w2(net, target, N_W2, N_REPEATS,
                                         GAMMA, T, D, device, NUM_ITER_MAX,
                                         sample_batch=EM_BATCH)

            record["checkpoints"].append(ckpt)
            record["w2_mean"][ckpt] = m
            record["w2_std"][ckpt] = s
            record["w2_all"][ckpt] = allv
            record["score_error"][ckpt] = se
            record["theta_norm"][ckpt] = record["theta_norm_history"][-1]
            record["loss_tail"][ckpt] = float(np.mean(diag["loss_history"][-200:]))

            if ckpt in KEEP_SAMPLES_AT:
                record["samples"][ckpt] = fct.em_sample(
                    net, GAMMA, T, D, N_W2, device, batch=EM_BATCH)

            print(f"   iter {ckpt:>7d}   W2 = {m:.4f} +/- {s:.4f}   "
                  f"(floor {fm:.4f})   loss {record['loss_tail'][ckpt]:.4f}   "
                  f"|theta| {record['theta_norm'][ckpt]:.2f}")
            print(f"   score error: " +
                  "  ".join(f"t={t:g}: {se[t]:.3f}" for t in SCORE_TS))

            record["net_state"] = {k: v.cpu() for k, v in net.state_dict().items()}
            record["runtime_s"] = time.time() - t_start
            fct.save_results(record, path)   # written after every checkpoint

        print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
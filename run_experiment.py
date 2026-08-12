"""
run_experiment.py

Runs the whole experiment and writes one .pt per configuration into
outputs/. Display is done separately in the notebook, which only loads.

    python run_experiment.py                 # every stage, skipping finished work
    python run_experiment.py --force         # redo everything
    python run_experiment.py --stages floors grid
    python run_experiment.py --only bimodal_theopoula_L20

Stages, in order:
    floors  finite-sample W2 floor per target and per N
    grid    small hyperparameter search per optimiser at a reference
            configuration, so that every arm of the main comparison is
            tuned on an equal footing
    main    the full grid of target x architecture x optimiser
    tsweep  optional: terminal time T, retrained at each value
    gsweep  optional: Euler-Maruyama step size gamma, at fixed T

Expect several hours. The dominant cost is the optimal transport, not the
training: exact W2 at N = 10^4 is roughly O(N^2.5).
"""

import argparse
import itertools
import os
import time

import numpy as np

import functions4 as fct


# ==========================================================================
#                       PARAMETERS TO CHANGE
# ==========================================================================

OUT_DIR = "outputs"

# --- what to run ---------------------------------------------------------
STAGES_DEFAULT = ["floors", "grid", "main"]     # add "tsweep", "gsweep" if wanted

TARGETS    = ["gaussian", "bimodal"]            # "gaussian" is the control: strongly
                                                # log-concave, so only the approximator
                                                # is at fault if anything goes wrong
ARCHS      = ["affine", 2, 20]                  # "affine" = Example 1 (gaussian only);
                                                # integers count AFFINE MAPS, so 20 means
                                                # 19 hidden layers
OPTIMISERS = ["adam", "sgld", "tusla", "theopoula"]
USE_SKIP   = False                              # True adds a linear path z_0 -> output
SEEDS      = [0]                                # add more for spread across seeds

# --- target geometry -----------------------------------------------------
D        = 2
MU       = np.array([1.3, 1.3])                 # modes at +/- MU (bimodal); mean MU (gaussian)
SIGMA_0  = 1.0                                  # component standard deviation
N_DATA   = 1_000_000                            # size of the training pool

# --- forward and reverse process ----------------------------------------
T          = 2.0                                # terminal time
T_0        = 1e-3                               # early-stopping time epsilon
GAMMA      = 1e-3                               # EM step size; note this scheme forces epsilon = gamma
KAPPA_MODE = "sigma2"                           # "sigma2" (recommended) or "one"

# --- training ------------------------------------------------------------
WIDTH       = 128
N_ITERS     = 10_000
BATCH_SIZE  = 512
PRINT_EVERY = 2_000

# --- evaluation ----------------------------------------------------------
N_W2          = 10_000                          # samples per W2 evaluation
N_REPEATS     = 3                               # W2 repeats; cost scales linearly
NUM_ITER_MAX  = 100_000_000                     # network-simplex ceiling; 10^8 is safe at N=10^4
N_PLOT        = 20_000                          # samples saved for histograms and scatter
EM_BATCH      = 20_000                          # sampling chunk size; lower it if memory is tight
SCORE_TS      = [0.05, 0.5, 2.0]                # noise levels for score-field figures
SCORE_ERR_TS  = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]   # noise levels for the score-error curve

FLOOR_REPEATS = 5

# Iterations at which the raw layerwise gradient and the applied update are
# recorded. Iteration 1 is effectively "at initialisation".
GRAD_CHECKPOINTS = (1, 100, 1_000, 5_000, N_ITERS)

# --- hyperparameter search ----------------------------------------------
# Scored on W2 at a deliberately smaller N: this ranks configurations, it
# does not report a number, and the ranking is stable far more cheaply.
GRID_REF_TARGET = "bimodal"
GRID_REF_ARCH   = 2
GRID_N_ITERS    = 5_000
GRID_N_W2       = 2_000
GRID_REPEATS    = 2

# The previous search selected boundary values on three axes, which means the
# optimum lay outside the box; these ranges extend it.
GRID = {
    "adam":      dict(lam=[1e-2, 3e-3, 1e-3, 3e-4]),
    "sgld":      dict(lam=[1e-2, 3e-3, 1e-3, 3e-4], beta=[1e8, 1e10]),
    "tusla":     dict(lam=[0.05, 0.02, 0.005], beta=[1e8, 1e10],
                      r=[2.0, 3.0], c=[5e-4]),
    "theopoula": dict(lam=[0.1, 0.05, 0.02], beta=[1e8, 1e10],
                      r=[2.0], eps_b=[0.5, 0.2], c=[5e-4]),
}

# Used only if stage "grid" is skipped and outputs/best_params.pt is absent.
FALLBACK_PARAMS = {
    "adam":      dict(lam=1e-3),
    "sgld":      dict(lam=1e-3, beta=1e10),
    "tusla":     dict(lam=0.02, beta=1e10, r=2.0, eta=5e-4 * np.sqrt(0.02)),
    "theopoula": dict(lam=0.02, beta=1e10, r=2.0, eps_b=0.2,
                      eta=5e-4 * np.sqrt(0.02)),
}

# --- optional sweeps -----------------------------------------------------
TSWEEP_TS      = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]   # each value retrained from scratch
GSWEEP_GAMMAS  = [0.05, 0.02, 0.01, 0.005, 0.001] # fixed T, one trained network
SWEEP_TARGET   = "bimodal"
SWEEP_ARCH     = 2
SWEEP_OPT      = "theopoula"

# ==========================================================================
#                      END OF PARAMETERS
# ==========================================================================


def arch_label(arch):
    return "affine" if arch == "affine" else f"L{arch}"


def tag_of(target, optimiser, arch, seed, skip):
    s = f"{target}_{optimiser}_{arch_label(arch)}"
    if skip and arch != "affine":
        s += "_skip"
    if seed != 0:
        s += f"_s{seed}"
    return s


def run_cfg_dict(n_iters=None):
    return dict(t_0=T_0, T=T, batch_size=BATCH_SIZE,
                n_iters=n_iters or N_ITERS, kappa_mode=KAPPA_MODE,
                print_every=PRINT_EVERY)


def opt_cfg_from(name, params):
    """Expand a parameter dict into an optimiser config, deriving eta = c sqrt(lam)."""
    p = dict(params)
    if "c" in p:
        p["eta"] = p.pop("c") * np.sqrt(p["lam"])
    return dict(name=name, **p)


# --------------------------------------------------------------------------
# Stage: floors
# --------------------------------------------------------------------------

def stage_floors():
    """
    Finite-sample floor at the single N used for reporting. Two
    independent N-samples of the same measure sit at strictly positive W2,
    so this is the resolution limit: no model can be distinguished from
    the target below it.
    """
    print("\n================ floor ================")
    out = {}
    for tname in TARGETS:
        target = fct.make_target(tname, D, MU, SIGMA_0)
        t0 = time.time()
        m, s, allv = fct.w2_floor(target, N_W2, FLOOR_REPEATS, NUM_ITER_MAX)
        out[tname] = dict(mean=m, std=s, all=allv, N=N_W2)
        print(f"  {tname:9s} N={N_W2}   {m:.4f} +/- {s:.4f}   "
              f"({time.time() - t0:.0f}s)")
    fct.save_results(out, os.path.join(OUT_DIR, "floors.pt"))
    return out


# --------------------------------------------------------------------------
# Stage: hyperparameter search
# --------------------------------------------------------------------------

def stage_grid(device):
    print("\n================ hyperparameter search ================")
    target = fct.make_target(GRID_REF_TARGET, D, MU, SIGMA_0)
    X_data = target["sample"](N_DATA)

    best, all_scores = {}, {}
    for opt_name in OPTIMISERS:
        space = GRID[opt_name]
        keys = list(space)
        combos = list(itertools.product(*(space[k] for k in keys)))
        print(f"\n  {opt_name}: {len(combos)} configurations")

        scored = []
        for combo in combos:
            params = dict(zip(keys, combo))
            label = "_".join(f"{k}{v:g}" for k, v in params.items())
            fct.set_seed(fct.seed_from_tag(f"grid_{opt_name}_{label}"))

            net = fct.build_net(D, GRID_REF_ARCH, WIDTH, USE_SKIP).to(device)
            diag = fct.train(net, X_data, target, opt_cfg_from(opt_name, params),
                             run_cfg_dict(GRID_N_ITERS), device)

            if diag["diverged"]:
                w2 = float("inf")
            else:
                w2, _, _ = fct.estimate_w2(net, target, GRID_N_W2, GRID_REPEATS,
                                           GAMMA, T, D, device, NUM_ITER_MAX,
                                           sample_batch=EM_BATCH)
            scored.append((w2, params))
            print(f"    {label:44s} W2={w2:.4f}"
                  f"{'  (diverged)' if diag['diverged'] else ''}")

        scored.sort(key=lambda x: x[0])
        best[opt_name] = scored[0][1]
        all_scores[opt_name] = scored
        print(f"  -> best {opt_name}: {scored[0][1]}  (W2={scored[0][0]:.4f})")

    payload = dict(best=best, all_scores=all_scores,
                   ref=dict(target=GRID_REF_TARGET, arch=GRID_REF_ARCH,
                            n_iters=GRID_N_ITERS, N_w2=GRID_N_W2))
    fct.save_results(payload, os.path.join(OUT_DIR, "best_params.pt"))
    return payload


def load_best_params():
    path = os.path.join(OUT_DIR, "best_params.pt")
    if os.path.exists(path):
        return fct.load_results(path)["best"]
    print("  no best_params.pt found: using FALLBACK_PARAMS")
    return {k: dict(v) for k, v in FALLBACK_PARAMS.items()}


# --------------------------------------------------------------------------
# Stage: main grid
# --------------------------------------------------------------------------

def run_one(target_name, optimiser, arch, seed, params, device,
            tag_override=None, run_cfg=None, gamma=None, T_override=None):
    tag = tag_override or tag_of(target_name, optimiser, arch, seed, USE_SKIP)
    print(f"\n--- {tag} ---")
    t_start = time.time()

    T_use = T_override if T_override is not None else T
    gamma_use = gamma if gamma is not None else GAMMA
    rc = run_cfg or run_cfg_dict()
    rc = dict(rc, T=T_use)

    fct.set_seed(fct.seed_from_tag(tag, seed))
    target = fct.make_target(target_name, D, MU, SIGMA_0)
    X_data = target["sample"](N_DATA)

    net = fct.build_net(D, arch, WIDTH, USE_SKIP).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"    {n_params} parameters")

    g_init = fct.layerwise_grad_norms(net, X_data, T_0, T_use, D, BATCH_SIZE,
                                      device, KAPPA_MODE)

    diag = fct.train(net, X_data, target, opt_cfg_from(optimiser, params),
                     rc, device, grad_checkpoints=GRAD_CHECKPOINTS)

    if diag["diverged"]:
        samples, w2_mean, w2_std, w2_all, grid, serr = \
            None, float("nan"), float("nan"), [], None, None
    else:
        samples = fct.em_sample(net, gamma_use, T_use, D, N_PLOT, device,
                               batch=EM_BATCH)
        print(f"    computing W2 at N={N_W2}, {N_REPEATS} repeats")
        w2_mean, w2_std, w2_all = fct.estimate_w2(
            net, target, N_W2, N_REPEATS, gamma_use, T_use, D, device,
            NUM_ITER_MAX, sample_batch=EM_BATCH)
        grid = fct.score_field_grid(net, target, SCORE_TS, device)
        serr = fct.score_error_by_t(net, target, SCORE_ERR_TS, device)

    r_used = params.get("r")
    result = dict(
        config=dict(tag=tag, target=target_name, optimiser=optimiser,
                    arch=arch_label(arch), arch_raw=arch,
                    n_hidden=(0 if arch == "affine" else int(arch) - 1),
                    width=WIDTH, use_skip=USE_SKIP, seed=seed,
                    n_params=n_params, T=T_use, t_0=T_0, gamma=gamma_use,
                    kappa_mode=KAPPA_MODE, n_iters=rc["n_iters"],
                    batch_size=BATCH_SIZE, N_w2=N_W2, n_repeats=N_REPEATS,
                    N_plot=N_PLOT, mu=MU, sigma_0=SIGMA_0, params=params),
        q=fct.q_of(arch),
        r_min_theopoula=fct.r_min(arch, "theopoula"),
        r_min_tusla=fct.r_min(arch, "tusla"),
        r_used=r_used,
        r_admissible=(None if r_used is None else
                      bool(r_used >= fct.r_min(arch,
                           "tusla" if optimiser == "tusla" else "theopoula"))),
        grad_norms_init=g_init,
        samples=samples,
        w2_mean=w2_mean, w2_std=w2_std, w2_all=w2_all,
        score_grid=grid, score_error=serr,
        net_state={k: v.cpu() for k, v in net.state_dict().items()},
        runtime_s=time.time() - t_start,
        **diag,
    )
    fct.save_results(result, os.path.join(OUT_DIR, f"{tag}.pt"))
    if not diag["diverged"]:
        print(f"    W2 = {w2_mean:.4f} +/- {w2_std:.4f}   "
              f"({result['runtime_s'] / 60:.1f} min)")
    return result


def stage_main(device, best, force, only):
    print("\n================ main grid ================")
    for seed in SEEDS:
        for tname in TARGETS:
            for arch in ARCHS:
                # the affine family contains the exact score only for the
                # Gaussian target, so it is not run on the mixture
                if arch == "affine" and tname != "gaussian":
                    continue
                for opt in OPTIMISERS:
                    tag = tag_of(tname, opt, arch, seed, USE_SKIP)
                    if only and tag != only:
                        continue
                    path = os.path.join(OUT_DIR, f"{tag}.pt")
                    if os.path.exists(path) and not force:
                        print(f"skip {tag} (exists)")
                        continue
                    run_one(tname, opt, arch, seed, best[opt], device)


# --------------------------------------------------------------------------
# Optional sweeps
# --------------------------------------------------------------------------

def stage_tsweep(device, best, force):
    """
    Terminal time. Each value is trained from scratch at its own T, so the
    network is never evaluated at noise levels it did not see in training.
    """
    print("\n================ T sweep ================")
    for T_val in TSWEEP_TS:
        tag = f"tsweep_{SWEEP_TARGET}_{SWEEP_OPT}_{arch_label(SWEEP_ARCH)}_T{T_val:g}"
        path = os.path.join(OUT_DIR, f"{tag}.pt")
        if os.path.exists(path) and not force:
            print(f"skip {tag} (exists)")
            continue
        run_one(SWEEP_TARGET, SWEEP_OPT, SWEEP_ARCH, 0, best[SWEEP_OPT],
                device, tag_override=tag, T_override=T_val)


def stage_gsweep(device, best, force):
    """
    Discretisation step size, at fixed T, reusing one trained network. This
    isolates gamma from training effects.
    """
    print("\n================ gamma sweep ================")
    base_tag = f"gsweep_base_{SWEEP_TARGET}_{SWEEP_OPT}_{arch_label(SWEEP_ARCH)}"
    base_path = os.path.join(OUT_DIR, f"{base_tag}.pt")
    if not os.path.exists(base_path) or force:
        run_one(SWEEP_TARGET, SWEEP_OPT, SWEEP_ARCH, 0, best[SWEEP_OPT],
                device, tag_override=base_tag)
    base = fct.load_results(base_path)

    target = fct.make_target(SWEEP_TARGET, D, MU, SIGMA_0)
    net = fct.build_net(D, SWEEP_ARCH, WIDTH, USE_SKIP).to(device)
    net.load_state_dict(base["net_state"])

    out = {}
    for g in GSWEEP_GAMMAS:
        print(f"  gamma = {g}")
        m, s, allv = fct.estimate_w2(net, target, N_W2, N_REPEATS, g, T, D,
                                     device, NUM_ITER_MAX, sample_batch=EM_BATCH)
        out[g] = dict(mean=m, std=s, all=allv,
                      n_steps=int(round(T / g)))
        print(f"    W2 = {m:.4f} +/- {s:.4f}")
    fct.save_results(dict(base_tag=base_tag, sweep=out),
                     os.path.join(OUT_DIR, "gsweep.pt"))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES_DEFAULT,
                    choices=["floors", "grid", "main", "tsweep", "gsweep"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = fct.get_device()
    print(f"device: {device}")
    t0 = time.time()

    if "floors" in args.stages and FLOOR_REPEATS > 0:
        if args.force or not os.path.exists(os.path.join(OUT_DIR, "floors.pt")):
            stage_floors()
        else:
            print("floor already computed")

    if "grid" in args.stages:
        if args.force or not os.path.exists(os.path.join(OUT_DIR, "best_params.pt")):
            stage_grid(device)
        else:
            print("hyperparameter search already done")

    best = load_best_params()
    print("\nparameters in use:")
    for k, v in best.items():
        print(f"  {k:10s} {v}")

    if "main" in args.stages:
        stage_main(device, best, args.force, args.only)
    if "tsweep" in args.stages:
        stage_tsweep(device, best, args.force)
    if "gsweep" in args.stages:
        stage_gsweep(device, best, args.force)

    print(f"\n=== finished in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
"""
run_experiment5.py

The full experimental programme of Chapter "Experimentation and Results",
one stage per experiment label in tab:experiment_summary. Everything is
saved incrementally into OUT_DIR, so any stage can be interrupted and
re-run; existing files are skipped unless --force is given.

    python run_experiment5.py                            # all stages
    python run_experiment5.py --stages floors,grid
    python run_experiment5.py --stages main --force

Stages:
  floors  tab:w2_floor. Floor at several N against the N^{-1/4} rate of
          Fournier & Guillin (2015) in d = 2, plus the floor at N_W2
          used as comparator everywhere else.
  grid    tab:gridsearch_space / tab:gridsearch_results. EQUAL tuning
          budget: the same lambda grid for all four optimisers, taken
          from Lim & Sabanis (2024) Section 4.2, who search
          lambda in {10, 1, 0.1, 0.01, 0.001} for every comparator.
          TheoPouLa additionally searches eps_b over the values studied
          in their Section 4.3.
  main    E1 + E3. For each example (1, 2, 3) and each optimiser at its
          grid optimum: one continuous 50,000-iteration run per seed,
          with W2, score error and mode balance evaluated at checkpoints
          via the in-loop callback (E1 curves), the final W2 at N = 10,000
          with common random numbers (E3 table), and N_PLOT = 1,000,000
          generated samples stored for seed 0 (histogram figures).
          For Example 1 the observable eps_AL = |theta - mu*|^2 is
          recorded, since theta* is known there and nowhere else.
  tsweep  E2. Terminal time sweep on the bimodal target, trained afresh
          at each T, against the t* prediction of Prop. beta_integral.
  depth   E4. Depth ladder L in {1, 3, 7, 11, 15, 19} hidden layers, the
          q = 2L + 1 admissibility story, with the layerwise gradient
          ratio at initialisation.

Hyperparameter provenance (all from the cited papers, see comments at
each constant):
  lambda grid         Lim & Sabanis (2024) Sec 4.2
  eps_b grid          Lim & Sabanis (2024) Sec 4.3 (0.1 best; 0.001
                      reported unstable and excluded)
  beta = 1e10         Lim & Sabanis (2024) Sec 4.3/4.4 (fixed there;
                      their beta study spans {1e4,...,1e12})
  eta = 5e-4 sqrt(lam) Lim & Sabanis (2024) Sec 4.4
  r = 10              Lim & Sabanis (2024) Sec 4.4 (their admissible-r
                      experiments). At L = 1 hidden layer q = 3, so
                      r = 10 >= q/2 = 1.5 (TheoPouLa) and
                      r = 10 >= q/2 + 1 = 2.5 (TUSLA): the theory holds
                      for both tamed algorithms in every main run.
  Adam beta1, beta2   PyTorch defaults (Kingma & Ba 2015)
  SGLD                lambda from the shared grid, same beta; its step
                      restriction for Example 1 (eq:ex1_lambda) is
                      computed and recorded with each Example 1 run.
"""

import argparse
import os
import time

import numpy as np

import functions5 as fct


# ==========================================================================
#                       PARAMETERS TO CHANGE
# ==========================================================================

OUT_DIR = "outputs5"

# --- target geometry (eq:bimodal_target; Example 1/2 use the Gaussian) ---
D        = 2
MU       = np.array([1.3, 1.3])       # mixture mean / unknown mean mu*
SIGMA_0  = 1.0
N_DATA   = 1_000_000                  # training samples from pi_D

# --- process and training -------------------------------------------------
T           = 2.0                     # terminal time (tsweep varies this)
T_0         = 1e-3                    # early stopping; epsilon = gamma
GAMMA       = 1e-3                    # EM step size
KAPPA_MODE  = "sigma2"                # kappa = sigma_t^2
BATCH_SIZE  = 512
N_ITERS     = 50_000                  # training iterations, all main runs
PRINT_EVERY = 5_000

# --- evaluation ------------------------------------------------------------
N_W2            = 10_000              # samples per Wasserstein evaluation
N_REPEATS_FINAL = 3                   # repeats for reported W2
N_REPEATS_CKPT  = 1                   # repeats at intermediate checkpoints
NUM_ITER_MAX    = 100_000_000         # POT network-simplex ceiling
EM_BATCH        = 20_000
EVAL_SEED       = 9_000               # common random numbers across configs
N_PLOT          = 1_000_000           # stored samples for histogram figures
SCORE_TS        = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
N_MC_SCORE      = 20_000

# --- E1 checkpoints (in-loop; Adam state preserved) -------------------------
W2_CHECKPOINTS  = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
GRAD_CHECKPOINTS = (1, 1_000, 10_000, N_ITERS)

# --- seeds ------------------------------------------------------------------
SEEDS = [0, 1, 2]                     # per-run training seeds (main stage)

# --- floors (tab:w2_floor) ---------------------------------------------------
FLOOR_NS      = [1_000, 2_000, 5_000, 10_000]
FLOOR_REPEATS = 3

# --- grid search (tab:gridsearch_space), provenance in the docstring --------
LAM_GRID   = [10.0, 1.0, 0.1, 0.01, 0.001]   # Lim & Sabanis Sec 4.2, all arms
EPSB_GRID  = [0.1, 0.01]                     # Lim & Sabanis Sec 4.3
BETA       = 1e10                            # Lim & Sabanis Sec 4.3/4.4
R_USED     = 10.0                            # Lim & Sabanis Sec 4.4
ETA_OF     = lambda lam: 5e-4 * np.sqrt(lam) # Lim & Sabanis Sec 4.4
GRID_TARGET = "bimodal"                      # tune on the headline example
GRID_ARCH   = 2                              # two-layer network of the paper
GRID_N_ITERS = 20_000   # selection horizon; final runs use N_ITERS. Set to
                        # N_ITERS for full parity at ~3x the grid cost.
GRID_REPEATS = 2
WIDTH     = 128
USE_SKIP  = False

# --- main stage (E1 + E3) -----------------------------------------------------
EXAMPLES = [
    # (example label, target, arch)
    ("ex1", "gaussian", "affine"),
    ("ex2", "gaussian", GRID_ARCH),
    ("ex3", "bimodal",  GRID_ARCH),
]
OPTIMISERS = ["adam", "sgld", "tusla", "theopoula"]

# --- E2: terminal time sweep ---------------------------------------------------
TSWEEP_TS         = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
TSWEEP_OPTIMISERS = ["adam", "theopoula"]
TSWEEP_TARGET     = "bimodal"
TSWEEP_ARCH       = GRID_ARCH
TSTAR_R_VALUES    = [5.0, 10.0, 20.0, 50.0, 100.0]  # radius grid for tab:t_star

# --- E4: depth ladder -----------------------------------------------------------
DEPTH_ARCHS      = [2, 4, 8, 12, 16, 20]      # affine maps; L = arch - 1
DEPTH_OPTIMISERS = ["adam", "theopoula"]
DEPTH_TARGET     = "bimodal"

# ==========================================================================


def opt_params(name, lam, eps_b=0.1):
    """Optimiser config with the paper-sourced defaults."""
    return dict(name=name, lam=lam, beta=BETA, r=R_USED, eps_b=eps_b,
                eta=ETA_OF(lam))


def make_eval_fn(target, device, n_repeats):
    """Checkpoint evaluation: W2 (CRN), score error by t, mode balance."""
    def eval_fn(net, it):
        se = fct.score_error_by_t(net, target, SCORE_TS, device,
                                  n_mc=N_MC_SCORE)
        m, s, vals, bal = fct.estimate_w2(
            net, target, N_W2, n_repeats, GAMMA, target.get("T", T), D,
            device, NUM_ITER_MAX, sample_batch=EM_BATCH,
            eval_seed=EVAL_SEED, verbose=False)
        print(f"      [ckpt {it}] W2 = {m:.4f} +/- {s:.4f}   "
              f"Ebar = {fct.score_error_aggregate(se):.4f}   "
              f"balance = {bal:.4f}")
        return dict(w2_mean=m, w2_std=s, w2_all=vals, mode_balance=bal,
                    score_error=se,
                    score_error_agg=fct.score_error_aggregate(se))
    return eval_fn


def sgld_lambda_max_ex1(T_val):
    """Step restriction eq:ex1_lambda for Example 1, by quadrature."""
    ts = np.linspace(1e-8, T_val, 200_001)
    s2m2 = (1 - np.exp(-2 * ts)) * np.exp(-2 * ts)
    e1 = np.trapz(s2m2, ts) / T_val
    e2 = np.trapz(s2m2 ** 2, ts) / T_val
    return float(min(e1 / (4 * e2), 1 / (2 * e1)))


def run_one(tag, example, target_name, arch, opt_cfg, seed, device,
            n_iters, T_val, keep_plot_samples, extra_config=None):
    """One training run with in-loop checkpoint evaluation and final W2."""
    path = os.path.join(OUT_DIR, f"{tag}.pt")
    target = fct.make_target(target_name, D, MU, SIGMA_0)
    target["T"] = T_val
    X_data = target["sample"](N_DATA)

    fct.set_seed(fct.seed_from_tag(tag, seed))
    net = fct.build_net(D, arch, WIDTH, USE_SKIP).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    # layerwise gradient profile at initialisation (grad. ratio column)
    g0 = fct.layerwise_grad_norms(net, X_data, T_0, T_val, D, BATCH_SIZE,
                                  device, KAPPA_MODE)

    print(f"\n================ {tag} ================")
    print(f"{n_params} parameters | q = {fct.q_of(arch):.0f} | "
          f"r_min(Theo) = {fct.r_min(arch):.1f} | "
          f"r_min(TUSLA) = {fct.r_min(arch, 'tusla'):.1f} | "
          f"r used = {opt_cfg.get('r')} | lam = {opt_cfg['lam']:g}")

    checkpoints = tuple(c for c in W2_CHECKPOINTS if c < n_iters)
    eval_fn = make_eval_fn(target, device, N_REPEATS_CKPT)

    rc = dict(t_0=T_0, T=T_val, batch_size=BATCH_SIZE, n_iters=n_iters,
              kappa_mode=KAPPA_MODE, print_every=PRINT_EVERY)
    t0 = time.time()
    diag = fct.train(net, X_data, target, opt_cfg, rc, device,
                     grad_checkpoints=GRAD_CHECKPOINTS,
                     checkpoints=checkpoints, eval_fn=eval_fn)

    record = dict(
        config=dict(tag=tag, example=example, target=target_name, arch=arch,
                    optimiser=opt_cfg["name"], params=dict(opt_cfg),
                    seed=seed, n_params=n_params, T=T_val, t_0=T_0,
                    gamma=GAMMA, kappa_mode=KAPPA_MODE,
                    batch_size=BATCH_SIZE, n_iters=n_iters, N_w2=N_W2,
                    n_repeats=N_REPEATS_FINAL, eval_seed=EVAL_SEED,
                    width=WIDTH, use_skip=USE_SKIP, mu=MU, sigma_0=SIGMA_0,
                    **(extra_config or {})),
        q=fct.q_of(arch),
        r_min_theopoula=fct.r_min(arch, "theopoula"),
        r_min_tusla=fct.r_min(arch, "tusla"),
        r_used=opt_cfg.get("r"),
        r_admissible=fct.r_admissible(arch, opt_cfg.get("r", 0.0),
                                      opt_cfg["name"]),
        grad_norms_init=g0, grad_ratio=fct.grad_ratio(g0),
        loss_history=diag["loss_history"],
        theta_norm_history=diag["theta_norm_history"],
        grad_norms=diag["grad_norms"], update_norms=diag["update_norms"],
        checkpoint_evals=diag["checkpoint_evals"],
        diverged=diag["diverged"], nan_iter=diag["nan_iter"],
    )

    if not diag["diverged"]:
        # final reported W2, common random numbers, N_REPEATS_FINAL repeats
        print(f"    final W2 at N={N_W2}, {N_REPEATS_FINAL} repeats")
        m, s, vals, bal = fct.estimate_w2(
            net, target, N_W2, N_REPEATS_FINAL, GAMMA, T_val, D, device,
            NUM_ITER_MAX, sample_batch=EM_BATCH, eval_seed=EVAL_SEED)
        se = fct.score_error_by_t(net, target, SCORE_TS, device,
                                  n_mc=N_MC_SCORE)
        record.update(w2_mean=m, w2_std=s, w2_all=vals, mode_balance=bal,
                      score_error=se,
                      score_error_agg=fct.score_error_aggregate(se))

        # Example 1: theta* = mu* is known, so eps_AL is observable
        if arch == "affine":
            theta_hat = net.theta.detach().cpu().numpy()
            record["theta_hat"] = theta_hat
            record["eps_AL_observed"] = float(np.sum((theta_hat - MU) ** 2))
            record["sgld_lambda_max"] = sgld_lambda_max_ex1(T_val)

        # 1,000,000 samples for the smooth histogram figures (seed 0 only)
        if keep_plot_samples:
            print(f"    generating {N_PLOT} plot samples")
            fct.set_seed(EVAL_SEED + 777)
            record["plot_samples"] = fct.em_sample(
                net, GAMMA, T_val, D, N_PLOT, device,
                batch=50_000).astype(np.float32)
            record["plot_mode_balance"] = fct.mode_balance(
                record["plot_samples"], MU)

    record["net_state"] = {k: v.cpu() for k, v in net.state_dict().items()}
    record["runtime_s"] = time.time() - t0
    fct.save_results(record, path)
    return record


# ==========================================================================
# Stages
# ==========================================================================

def stage_floors(device, force):
    for tname in ["gaussian", "bimodal"]:
        path = os.path.join(OUT_DIR, f"floor_{tname}.pt")
        if os.path.exists(path) and not force:
            print(f"skip floor_{tname} (exists)")
            continue
        target = fct.make_target(tname, D, MU, SIGMA_0)
        by_N = {}
        for N in FLOOR_NS:
            print(f"floor {tname} N={N}")
            m, s, vals = fct.w2_floor(target, N, FLOOR_REPEATS,
                                      NUM_ITER_MAX, eval_seed=EVAL_SEED)
            by_N[N] = dict(mean=m, std=s, all=vals)
            print(f"   {m:.4f} +/- {s:.4f}")
        # top-level mean/std/N = the floor at the reporting N_W2
        fct.save_results(dict(by_N=by_N, mean=by_N[N_W2]["mean"],
                              std=by_N[N_W2]["std"], N=N_W2), path)


def stage_grid(device, force):
    """Equal tuning budget: same lambda grid for every optimiser
    (Lim & Sabanis 2024 Sec 4.2); TheoPouLa additionally sweeps eps_b at
    its best lambda (their Sec 4.3 two-stage design). Scored by W2 with
    common random numbers. Selection horizon GRID_N_ITERS."""
    summary_path = os.path.join(OUT_DIR, "grid_summary.pt")
    rows = [] if force or not os.path.exists(summary_path) else \
        fct.load_results(summary_path)

    def done(o, lam, eb):
        return any(r_["optimiser"] == o and r_["lam"] == lam
                   and r_.get("eps_b") == eb for r_ in rows)

    target = fct.make_target(GRID_TARGET, D, MU, SIGMA_0)
    X_data = target["sample"](N_DATA)

    def score(opt_cfg, tag):
        fct.set_seed(fct.seed_from_tag(tag, 0))
        net = fct.build_net(D, GRID_ARCH, WIDTH, USE_SKIP).to(device)
        rc = dict(t_0=T_0, T=T, batch_size=BATCH_SIZE,
                  n_iters=GRID_N_ITERS, kappa_mode=KAPPA_MODE,
                  print_every=10_000)
        diag = fct.train(net, X_data, target, opt_cfg, rc, device)
        if diag["diverged"]:
            return None, None, True
        m, s, _, _ = fct.estimate_w2(net, target, N_W2, GRID_REPEATS,
                                     GAMMA, T, D, device, NUM_ITER_MAX,
                                     sample_batch=EM_BATCH,
                                     eval_seed=EVAL_SEED, verbose=False)
        return m, s, False

    # stage 1: lambda, all optimisers, eps_b fixed at 0.1 for theopoula
    for opt in OPTIMISERS:
        for lam in LAM_GRID:
            if done(opt, lam, 0.1 if opt == "theopoula" else None):
                continue
            tag = f"grid_{opt}_lam{lam:g}"
            print(f"\n--- {tag} ---")
            cfg = opt_params(opt, lam)
            m, s, div = score(cfg, tag)
            rows.append(dict(optimiser=opt, lam=lam,
                             eps_b=0.1 if opt == "theopoula" else None,
                             beta=BETA, r=R_USED, w2_mean=m, w2_std=s,
                             diverged=div))
            print(f"   W2 = {m}" if not div else "   diverged")
            fct.save_results(rows, summary_path)

    # stage 2: eps_b at TheoPouLa's best lambda (Sec 4.3)
    theo = [r_ for r_ in rows if r_["optimiser"] == "theopoula"
            and r_["w2_mean"] is not None]
    if theo:
        best_lam = min(theo, key=lambda r_: r_["w2_mean"])["lam"]
        for eb in EPSB_GRID:
            if eb == 0.1 or done("theopoula", best_lam, eb):
                continue
            tag = f"grid_theopoula_lam{best_lam:g}_eps{eb:g}"
            print(f"\n--- {tag} ---")
            m, s, div = score(opt_params("theopoula", best_lam, eps_b=eb),
                              tag)
            rows.append(dict(optimiser="theopoula", lam=best_lam, eps_b=eb,
                             beta=BETA, r=R_USED, w2_mean=m, w2_std=s,
                             diverged=div))
            fct.save_results(rows, summary_path)

    # best per optimiser
    best = {}
    for opt in OPTIMISERS:
        cand = [r_ for r_ in rows if r_["optimiser"] == opt
                and r_["w2_mean"] is not None]
        if cand:
            b = min(cand, key=lambda r_: r_["w2_mean"])
            best[opt] = dict(lam=b["lam"],
                             eps_b=b.get("eps_b") or 0.1)
            print(f"best {opt}: lam={b['lam']:g} "
                  f"eps_b={best[opt]['eps_b']:g} W2={b['w2_mean']:.4f}")
    fct.save_results(best, os.path.join(OUT_DIR, "best_params5.pt"))


def load_best():
    path = os.path.join(OUT_DIR, "best_params5.pt")
    if os.path.exists(path):
        return fct.load_results(path)
    print("WARNING: no grid results found; using lam=0.1, eps_b=0.1 "
          "(Lim & Sabanis Sec 4.2/4.3 defaults) for every optimiser")
    return {o: dict(lam=0.1, eps_b=0.1) for o in OPTIMISERS}


def stage_main(device, force):
    best = load_best()
    for example, target_name, arch in EXAMPLES:
        for opt in OPTIMISERS:
            for seed in SEEDS:
                tag = f"{example}_{target_name}_{opt}_L{arch}_s{seed}"
                path = os.path.join(OUT_DIR, f"{tag}.pt")
                if os.path.exists(path) and not force:
                    print(f"skip {tag} (exists)")
                    continue
                b = best.get(opt, dict(lam=0.1, eps_b=0.1))
                cfg = opt_params(opt, b["lam"], eps_b=b["eps_b"])
                run_one(tag, example, target_name, arch, cfg, seed,
                        device, N_ITERS, T,
                        keep_plot_samples=(seed == SEEDS[0]))


def stage_tsweep(device, force):
    best = load_best()
    for T_val in TSWEEP_TS:
        for opt in TSWEEP_OPTIMISERS:
            tag = f"tsweep_T{T_val:g}_{opt}"
            path = os.path.join(OUT_DIR, f"{tag}.pt")
            if os.path.exists(path) and not force:
                print(f"skip {tag} (exists)")
                continue
            b = best.get(opt, dict(lam=0.1, eps_b=0.1))
            cfg = opt_params(opt, b["lam"], eps_b=b["eps_b"])
            run_one(tag, "ex3", TSWEEP_TARGET, TSWEEP_ARCH, cfg, 0,
                    device, N_ITERS, T_val, keep_plot_samples=False,
                    extra_config=dict(sweep="T"))


def stage_depth(device, force):
    best = load_best()
    for arch in DEPTH_ARCHS:
        for opt in DEPTH_OPTIMISERS:
            tag = f"depth_L{arch}_{opt}"
            path = os.path.join(OUT_DIR, f"{tag}.pt")
            if os.path.exists(path) and not force:
                print(f"skip {tag} (exists)")
                continue
            b = best.get(opt, dict(lam=0.1, eps_b=0.1))
            cfg = opt_params(opt, b["lam"], eps_b=b["eps_b"])
            run_one(tag, "ex3", DEPTH_TARGET, arch, cfg, 0, device,
                    N_ITERS, T, keep_plot_samples=False,
                    extra_config=dict(sweep="depth"))


# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="floors,grid,main,tsweep,depth")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",")]

    os.makedirs(OUT_DIR, exist_ok=True)
    device = fct.get_device()
    print(f"device: {device}")
    print(f"stages: {stages}")

    t0 = time.time()
    if "floors" in stages:
        stage_floors(device, args.force)
    if "grid" in stages:
        stage_grid(device, args.force)
    if "main" in stages:
        stage_main(device, args.force)
    if "tsweep" in stages:
        stage_tsweep(device, args.force)
    if "depth" in stages:
        stage_depth(device, args.force)
    print(f"\nall done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()

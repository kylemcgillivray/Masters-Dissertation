"""
gridsearch3.py

Refined grid search over TheoPouLa hyperparameters, narrowed based on
the first sweep's results (lambda is the dominant lever, smaller is
better within the tested range; eps=0.1 beats eps=0.01; beta=1e10
edges out 1e8; r tested at 0, 1.5, 2). For each config: train, sample
2000 points, compute W2 against the true distribution, and save
everything incrementally so a crash mid-run doesn't lose completed configs.
"""

import time
import itertools
import numpy as np
import functions3 as fct


def final_loss(loss_history, tail_frac=0.1):
    tail_n = max(1, int(len(loss_history) * tail_frac))
    return float(np.mean(loss_history[-tail_n:]))


def main():
    fct.set_seed(0)
    device = fct.get_device()
    print("Using device:", device)

    # ============================================================
    # Setup - matches previous sweep exactly
    # ============================================================
    d = 2
    T = 2
    mu1 = np.array([1.3, 1.3])
    mu2 = np.array([-1.3, -1.3])
    sigma_mix = 1.0
    weight = 0.5

    X_data_bimodal = fct.sample_bimodal_data(
        N=1_000_000, d=d, mu1=mu1, mu2=mu2, sigma=sigma_mix, weight=weight
    )
    samples_true = X_data_bimodal[:2000]   # fixed reference set for all W2 comparisons

    N_ITERS = 10_000
    BATCH_SIZE = 512
    T_0 = 1e-3
    N_W2 = 2000

    kappa_sigma2 = lambda tau: 1 - np.exp(-2 * tau)

    all_results = {}
    save_path = "outputs/gridsearch3_results.pt"
    sweep_start = time.time()

    # ============================================================
    # Refined grid
    # ============================================================
    lam_grid = [0.02, 0.01, 0.005]
    eps_grid = [0.2, 0.1]
    beta_grid = [1e10, 1e12]
    r_grid = [0, 1.5, 2]

    n_configs = len(lam_grid) * len(eps_grid) * len(beta_grid) * len(r_grid)
    print(f"\n=== TheoPouLa refined grid: {n_configs} configs, {N_ITERS} iters each ===")

    for lam, eps, beta, r in itertools.product(lam_grid, eps_grid, beta_grid, r_grid):
        eta = 5e-4 if r == 0 else 5e-4 * np.sqrt(lam)
        tag = f"theopoula_lam{lam}_eps{eps}_beta{beta:.0e}_r{r}"
        print(f"\n--- {tag} ---")

        # --- Train ---
        net = fct.ScoreNetDeep(d=d, hidden_dims=[128] * 4).to(device)
        t0 = time.time()
        net, loss_hist = fct.train_network_theopoula(
            net, X_data_bimodal, t_0=T_0, T=T, d=d,
            n_iters=N_ITERS, batch_size=BATCH_SIZE,
            lam=lam, eta=eta, r=r, eps=eps, beta=beta,
            device=device, kappa_fn=kappa_sigma2, print_every=100,
        )
        train_elapsed = time.time() - t0

        # --- Sample (for later plotting) ---
        samples = fct.euler_maruyama_sample_nn_batch(
            net, gamma=0.001, T=T, d=d, N_samples=N_W2, device=device
        )

        # --- W2 against the true distribution ---
        w2_mean, w2_std = fct.estimate_W2(
            net, samples_true, N_w2=N_W2, n_repeats=5, gamma=0.001, T=T, d=d, device=device
        )

        total_elapsed = time.time() - t0

        all_results[tag] = {
            "config": dict(optimizer="theopoula", lam=lam, eta=eta, r=r, eps=eps, beta=beta),
            "loss_history": loss_hist,
            "final_loss": final_loss(loss_hist),
            "net_state": net.state_dict(),
            "samples": samples,
            "w2_mean": w2_mean,
            "w2_std": w2_std,
            "train_runtime_seconds": train_elapsed,
            "total_runtime_seconds": total_elapsed,
        }

        print(f"  [{tag}] final_loss={all_results[tag]['final_loss']:.4f}  "
              f"W2={w2_mean:.4f}±{w2_std:.4f}  ({total_elapsed:.1f}s)")

        fct.save_results(all_results, save_path)   # save after every config, not just at the end

    total_sweep_elapsed = time.time() - sweep_start
    print(f"\n=== Grid search complete: {total_sweep_elapsed:.1f}s ({total_sweep_elapsed/60:.1f} min) total ===")
    print("\nRanked by W2 (best first):")
    for tag, r in sorted(all_results.items(), key=lambda kv: kv[1]["w2_mean"]):
        print(f"  {tag:50s}  W2={r['w2_mean']:.4f}±{r['w2_std']:.4f}  final_loss={r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
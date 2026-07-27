"""
gridsearch3.py

Grid search over TheoPouLa hyperparameters (lam, eps, beta, r) on the
5-layer ScoreNetDeep, trained on the bimodal mixture target. Training
only - no sampling / W2 here, since that's expensive and better done
as a separate pass once you know which configs are worth sampling from.
Every run's final network state is saved, so sampling can be done later
without retraining.
"""

import time
import itertools
import numpy as np
import functions3 as fct


def final_loss(loss_history, tail_frac=0.1):
    """Average loss over the final tail_frac of training, to reduce
    noise from a single noisy last-iteration value."""
    tail_n = max(1, int(len(loss_history) * tail_frac))
    return float(np.mean(loss_history[-tail_n:]))


def main():
    fct.set_seed(0)
    device = fct.get_device()
    print("Using device:", device)

    # ============================================================
    # Setup - matches the notebook exactly
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

    N_ITERS = 10_000
    BATCH_SIZE = 512
    T_0 = 1e-3

    kappa_sigma2 = lambda tau: 1 - np.exp(-2 * tau)

    all_results = {}
    save_path = "outputs/gridsearch3_results.pt"
    sweep_start = time.time()

    def run_and_record(tag, net, loss_history, config, elapsed):
        all_results[tag] = {
            "config": config,
            "loss_history": loss_history,
            "final_loss": final_loss(loss_history),
            "net_state": net.state_dict(),
            "runtime_seconds": elapsed,
        }
        print(f"  [{tag}] final_loss={all_results[tag]['final_loss']:.4f}  ({elapsed:.1f}s)")
        fct.save_results(all_results, save_path)   # save after every run, not just at the end

    # ============================================================
    # Grid search: TheoPouLa
    # ============================================================
    # eta is tied to lam via the paper's own rescaling rule (Section 4.4)
    # whenever r > 0, so the regularizer's saturated strength stays
    # comparable across the grid rather than varying arbitrarily with r.
    lam_grid = [0.1, 0.05, 0.01]
    eps_grid = [0.1, 0.01]
    beta_grid = [1e8, 1e10]
    r_grid = [0, 3]

    n_configs = len(lam_grid) * len(eps_grid) * len(beta_grid) * len(r_grid)
    print(f"\n=== TheoPouLa grid: {n_configs} configs, {N_ITERS} iters each ===")

    for lam, eps, beta, r in itertools.product(lam_grid, eps_grid, beta_grid, r_grid):
        eta = 5e-4 if r == 0 else 5e-4 * np.sqrt(lam)
        tag = f"theopoula_lam{lam}_eps{eps}_beta{beta:.0e}_r{r}"
        print(f"\n--- {tag} ---")

        net = fct.ScoreNetDeep(d=d, hidden_dims=[128] * 4).to(device)
        t0 = time.time()
        net, loss_hist = fct.train_network_theopoula(
            net, X_data_bimodal, t_0=T_0, T=T, d=d,
            n_iters=N_ITERS, batch_size=BATCH_SIZE,
            lam=lam, eta=eta, r=r, eps=eps, beta=beta,
            device=device, kappa_fn=kappa_sigma2, print_every=2000,
        )
        run_and_record(tag, net, loss_hist,
                        dict(optimizer="theopoula", lam=lam, eta=eta, r=r, eps=eps, beta=beta),
                        time.time() - t0)

    total_elapsed = time.time() - sweep_start
    print(f"\n=== Grid search complete: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) total ===")
    print("\nFinal loss summary (best first):")
    for tag, r in sorted(all_results.items(), key=lambda kv: kv[1]["final_loss"]):
        print(f"  {tag:50s}  final_loss={r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
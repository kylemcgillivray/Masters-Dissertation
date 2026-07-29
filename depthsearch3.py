"""
depth_sweep3.py

Train the best TheoPouLa config found in the refined grid search
(lam=0.02, eps=0.2, beta=1e10, r=2) across network depths 2 to 15,
on the bimodal mixture target. For each depth: check layerwise gradient
norms at initialisation (to track how conditioning changes with depth),
train, sample, and compute W2 against the true distribution. Results
saved incrementally so a crash mid-run doesn't lose completed depths.
"""

import time
import numpy as np
import functions3 as fct


def main():
    fct.set_seed(0)
    device = fct.get_device()
    print("Using device:", device)

    # ============================================================
    # Setup - matches gridsearch3.py exactly
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
    samples_true = X_data_bimodal[:5000]

    T_0 = 1e-3
    BATCH_SIZE = 512
    N_ITERS = 10_000
    N_W2 = 5000

    kappa_sigma2 = lambda tau: 1 - np.exp(-2 * tau)

    # ============================================================
    # Best config from the refined grid search - fixed, no further search.
    # theopoula_lam0.02_eps0.2_beta1e+10_r2, W2=0.1905 at 5 layers
    # ============================================================
    BEST_LAM = 0.02
    BEST_EPS = 0.2
    BEST_BETA = 1e10
    BEST_R = 2
    BEST_ETA = 5e-4 * np.sqrt(BEST_LAM)

    all_results = {}
    save_path = "outputs/depth_sweep_results.pt"
    sweep_start = time.time()

    # ============================================================
    # Train depths 2 to 15 (inclusive). "N layers" matches the
    # ScoreNet/ScoreNet3/ScoreNetDeep convention: the count includes
    # the output layer, so hidden_dims has (N-1) entries.
    # ============================================================
    for n_layers in range(2, 16):
        hidden_dims = [128] * (n_layers - 1)
        tag = f"depth{n_layers}"
        print(f"\n=== {tag} (hidden_dims={hidden_dims}) ===")

        # --- Gradient-norm check at initialisation ---
        net_init = fct.ScoreNetDeep(d=d, hidden_dims=hidden_dims).to(device)
        grad_norms_init = fct.compute_layerwise_grad_norms(
            net_init, X_data_bimodal, t_0=T_0, T=T, d=d, batch_size=BATCH_SIZE, device=device
        )
        grad_values = [v for v in grad_norms_init.values() if v is not None]
        grad_ratio = max(grad_values) / min(grad_values) if grad_values else None
        print(f"  Grad norm range at init: min={min(grad_values):.4e}, max={max(grad_values):.4e}, "
              f"ratio={grad_ratio:.1f}x")

        # --- Train ---
        net = fct.ScoreNetDeep(d=d, hidden_dims=hidden_dims).to(device)
        t0 = time.time()
        net, loss_hist = fct.train_network_theopoula(
            net, X_data_bimodal, t_0=T_0, T=T, d=d,
            n_iters=N_ITERS, batch_size=BATCH_SIZE,
            lam=BEST_LAM, eta=BEST_ETA, r=BEST_R, eps=BEST_EPS, beta=BEST_BETA,
            device=device, kappa_fn=kappa_sigma2, print_every=2000,
        )
        train_elapsed = time.time() - t0

        # --- Sample ---
        samples = fct.euler_maruyama_sample_nn_batch(
            net, gamma=0.001, T=T, d=d, N_samples=N_W2, device=device
        )

        # --- W2 against true distribution ---
        w2_mean, w2_std = fct.estimate_W2(
            net, samples_true, N_w2=N_W2, n_repeats=5, gamma=0.001, T=T, d=d, device=device
        )

        total_elapsed = time.time() - t0

        all_results[tag] = {
            "n_layers": n_layers,
            "hidden_dims": hidden_dims,
            "grad_norms_init": grad_norms_init,
            "grad_ratio_init": grad_ratio,
            "loss_history": loss_hist,
            "final_loss": float(np.mean(loss_hist[-max(1, len(loss_hist) // 10):])),
            "net_state": net.state_dict(),
            "samples": samples,
            "w2_mean": w2_mean,
            "w2_std": w2_std,
            "train_runtime_seconds": train_elapsed,
            "total_runtime_seconds": total_elapsed,
        }

        print(f"  [{tag}] final_loss={all_results[tag]['final_loss']:.4f}  "
              f"W2={w2_mean:.4f}±{w2_std:.4f}  ({total_elapsed:.1f}s)")

        fct.save_results(all_results, save_path)   # save after every depth, not just at the end

    total_sweep_elapsed = time.time() - sweep_start
    print(f"\n=== Depth sweep complete: {total_sweep_elapsed:.1f}s ({total_sweep_elapsed/60:.1f} min) total ===")
    print(f"\n{'depth':>6s} {'grad_ratio':>12s} {'W2':>18s} {'final_loss':>12s}")
    for tag, r in sorted(all_results.items(), key=lambda kv: kv[1]["n_layers"]):
        print(f"{r['n_layers']:6d} {r['grad_ratio_init']:12.1f} "
              f"{r['w2_mean']:.4f} ± {r['w2_std']:.4f}  {r['final_loss']:12.4f}")


if __name__ == "__main__":
    main()
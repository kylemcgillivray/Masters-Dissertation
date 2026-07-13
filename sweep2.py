import time
import numpy as np
import functions2 as fct


def main():
    fct.set_seed(0)
    device = fct.get_device()
    print("Using device:", device)

    d = 2
    mu1, mu2 = np.array([2.0, 2.0]), np.array([-2.0, -2.0])
    X_data_bimodal = fct.sample_bimodal_data(
        N=1_000_000, d=d, mu1=mu1, mu2=mu2, sigma=3.0, weight=0.5
    )

    checkpoints_standard = [20, 50, 100, 250, 500, 1000, 2000, 3000, 5000]

    runs = [
            # --- Group 1: T sweep (initialisation error term, C2 e^{-2L(T-eps)}) ---
            dict(tag="T1",              T=1,    N_w2=2000, n_repeats=5),
            dict(tag="T2",              T=2,    N_w2=2000, n_repeats=5),
            dict(tag="T3",              T=3,    N_w2=2000, n_repeats=5),
            dict(tag="T5",              T=5,    N_w2=2000, n_repeats=5),

            # --- Group 2: gamma sweep (discretisation error term, C4 gamma^alpha) ---
            # Fixed at T=2 (your best-performing T from Group 1).
            dict(tag="gamma_0.01",      T=2, gamma=0.01,   N_w2=2000, n_repeats=5),
            dict(tag="gamma_0.005",     T=2, gamma=0.005,  N_w2=2000, n_repeats=5),
            dict(tag="gamma_0.001",     T=2, gamma=0.001,  N_w2=2000, n_repeats=5),
            dict(tag="gamma_0.0005",    T=2, gamma=0.0005, N_w2=2000, n_repeats=5),

            # --- Group 3: t_0 (early stopping) sweep (C1 sqrt(eps) term) ---
            dict(tag="t0_1e-1",         T=2, t_0=1e-1, N_w2=2000, n_repeats=5),
            dict(tag="t0_1e-2",         T=2, t_0=1e-2, N_w2=2000, n_repeats=5),
            dict(tag="t0_1e-3",         T=2, t_0=1e-3, N_w2=2000, n_repeats=5),
            dict(tag="t0_1e-4",         T=2, t_0=1e-4, N_w2=2000, n_repeats=5),

            # --- Group 4: network width sweep (score estimation error, C3 sqrt(eps_SN)) ---
            dict(tag="D1_32",           T=2, D_1=32,  D_2=32,  N_w2=2000, n_repeats=5),
            dict(tag="D1_64",           T=2, D_1=64,  D_2=64,  N_w2=2000, n_repeats=5),
            dict(tag="D1_128",          T=2, D_1=128, D_2=128, N_w2=2000, n_repeats=5),
            dict(tag="D1_256",          T=2, D_1=256, D_2=256, N_w2=2000, n_repeats=5),
        ]

    sweep_start = time.time()
    run_times = {}

    for run in runs:
        tag = run.pop("tag")
        save_path = f"outputs/results_{tag}.pt"
        print(f"\n=== Running {tag}: {run} ===")

        run_start = time.time()
        results = fct.run_W2_experiment(
            X_data_bimodal, checkpoints_standard, d, device,
            save_path=save_path, **run,
        )
        elapsed = time.time() - run_start
        run_times[tag] = elapsed

        # Store the timing inside the saved file too, so it's available
        # later without needing to keep this log around.
        results["runtime_seconds"] = elapsed
        fct.save_results(results, save_path)

        print(f"=== Finished {tag} in {elapsed:.1f}s ({elapsed/60:.1f} min) ===")

    total_elapsed = time.time() - sweep_start
    print(f"\n=== Sweep complete: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) total ===")
    for tag, t in run_times.items():
        print(f"  {tag:20s}: {t:8.1f}s  ({t/60:5.1f} min)")


if __name__ == "__main__":
    main()
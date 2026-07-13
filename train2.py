import argparse
import numpy as np
import functions2 as fct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--N_w2", type=int, default=2000)
    parser.add_argument("--D_1", type=int, default=128)
    parser.add_argument("--D_2", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--t_0", type=float, default=1e-3)
    parser.add_argument("--checkpoints", type=int, nargs="+",
                         default=[20, 50, 100, 250, 500, 1000, 2000, 3000, 5000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None,
                         help="Output filename tag; defaults to T{T}_Nw2{N_w2}_seed{seed}")
    args = parser.parse_args()

    fct.set_seed(args.seed)
    device = fct.get_device()
    print("Using device:", device)

    d = 2
    mu1, mu2 = np.array([2.0, 2.0]), np.array([-2.0, -2.0])
    X_data_bimodal = fct.sample_bimodal_data(
        N=1_000_000, d=d, mu1=mu1, mu2=mu2, sigma=3.0, weight=0.5
    )

    tag = args.tag or f"T{args.T}_Nw2{args.N_w2}_seed{args.seed}"
    save_path = f"outputs/results_{tag}.pt"

    fct.run_W2_experiment(
        X_data_bimodal, args.checkpoints, d, device,
        D_1=args.D_1, D_2=args.D_2, batch_size=args.batch_size, lr=args.lr,
        N_w2=args.N_w2, n_repeats=args.n_repeats, gamma=args.gamma,
        T=args.T, t_0=args.t_0, save_path=save_path,
    )
    print(f"Done. Results saved to {save_path}")


if __name__ == "__main__":
    main()
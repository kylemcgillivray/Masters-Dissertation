"""
tusla_r25_check.py

One-off check: TUSLA at its admissible minimum r = q/2 + 1 = 2.5 (rather than
the r = 10 of Lim & Sabanis) on the bimodal target, L = 1 hidden layer of
width 128. Nothing is written to disk; the only output is the marginals plot
and a few printed diagnostics. No Wasserstein evaluation.

    python tusla_r25_check.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

import functions5 as fct

# ---- config: identical to run_experiment5.py except r and lam ----------
D, MU, SIGMA_0 = 2, np.array([1.3, 1.3]), 1.0
N_DATA      = 1_000_000
T, T_0      = 2.0, 1e-3
GAMMA       = 1e-3
KAPPA_MODE  = "sigma2"
BATCH_SIZE  = 512
N_ITERS     = 50_000
WIDTH, ARCH = 128, 2          # ARCH = 2 affine maps -> L = 1 hidden layer
USE_SKIP    = False
BETA        = 1e10

R_USED = 2.5                  # <-- TUSLA's r_min at q = 3;  was 10.0
LAM    = 1e-3                 # <-- most favourable admissible step size
ETA    = 5e-4 * np.sqrt(LAM)

N_PLOT   = 200_000            # enough for smooth 240-bin histograms
EM_BATCH = 50_000
LIM      = 6.0
BINS     = np.linspace(-LIM, LIM, 241)
SEED     = 0

device = fct.get_device()
print(f"device: {device}")

# ---- data and network --------------------------------------------------
target = fct.make_target("bimodal", D, MU, SIGMA_0)
X_data = target["sample"](N_DATA)

fct.set_seed(fct.seed_from_tag(f"tusla_r{R_USED}_lam{LAM}", SEED))
net = fct.build_net(D, ARCH, WIDTH, USE_SKIP).to(device)
n_params = sum(p.numel() for p in net.parameters())

theta0 = float(torch.sqrt(sum((p ** 2).sum() for p in net.parameters())))
print(f"\n{n_params} parameters | q = {fct.q_of(ARCH):.0f} | "
      f"r_min(TUSLA) = {fct.r_min(ARCH, 'tusla'):.1f} | r used = {R_USED} | "
      f"lam = {LAM:g}")
print(f"||theta|| at init = {theta0:.3f}")
print(f"  taming denominator 1 + sqrt(lam)*||theta||^(2r) = "
      f"{1 + np.sqrt(LAM) * theta0 ** (2 * R_USED):.3e}")
print(f"  effective data step lam_eff = "
      f"{LAM / (1 + np.sqrt(LAM) * theta0 ** (2 * R_USED)):.3e}   "
      f"(SGLD comparator: 1.0e-02)")

# ---- train -------------------------------------------------------------
opt_cfg = dict(name="tusla", lam=LAM, beta=BETA, eta=ETA, r=R_USED, eps_b=0.1)
run_cfg = dict(t_0=T_0, T=T, batch_size=BATCH_SIZE, n_iters=N_ITERS,
               kappa_mode=KAPPA_MODE, print_every=10_000)

diag = fct.train(net, X_data, target, opt_cfg, run_cfg, device)

tn = diag["theta_norm_history"]
print(f"\n||theta||: init {tn[0]:.3f} -> final {tn[-1]:.3f}   "
      f"(min {min(tn):.3f}, max {max(tn):.3f})")
print(f"  final taming denominator = "
      f"{1 + np.sqrt(LAM) * tn[-1] ** (2 * R_USED):.3e}")
print(f"  diverged: {diag['diverged']}")

# ---- sample ------------------------------------------------------------
if diag["diverged"]:
    raise SystemExit("run diverged; nothing to plot")

print(f"\ngenerating {N_PLOT:,} samples")
fct.set_seed(9_000 + 777)          # same CRN seed as the main-stage figures
gen = fct.em_sample(net, GAMMA, T, D, N_PLOT, device,
                    batch=EM_BATCH).astype(np.float32)

frac = float(np.mean(np.all(np.abs(gen) <= LIM, axis=1)))
print(f"  in window [-6,6]^2 : {frac:.1%}")
print(f"  mode balance       : {fct.mode_balance(gen, MU):.4f}  (0.5 is perfect)")
print(f"  sample sd per coord: {gen.std(0)}   (target: "
      f"{np.sqrt(1 + MU[0] ** 2):.3f})")

# ---- plot --------------------------------------------------------------
np.random.seed(0)
true = target["sample"](1_000_000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for k in range(2):
    axes[k].hist(true[:, k], bins=BINS, density=True, alpha=0.3, color="k",
                 label="target")
    axes[k].hist(gen[:, k], bins=BINS, density=True, histtype="step",
                 lw=1.6, color="C2",
                 label=f"tusla $r={R_USED}$, $\\lambda={LAM:g}$ "
                       f"({frac:.0%} in window)")
    axes[k].set_xlim(-LIM, LIM)
    axes[k].set_xlabel(f"coordinate {k + 1}")
    axes[k].legend(fontsize=8)
plt.suptitle(f"TUSLA at admissible $r={R_USED}$, $L=1$, width {WIDTH}")
plt.tight_layout()
plt.show()
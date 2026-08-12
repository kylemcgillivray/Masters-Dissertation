# %% [markdown]
# # display5 — figures and tables for Chapter "Experimentation and Results"
# Run each cell in order (VSCode renders `# %%` blocks as notebook cells).
# Every figure is saved as a PDF into `figures5/` at paper quality, and
# every table is printed as LaTeX ready to paste into Overleaf.

# %% Cell 1: load everything
import os

import matplotlib.pyplot as plt
import numpy as np

import functions5 as fct

OUT_DIR = "outputs5"
FIG_DIR = "figures5"
os.makedirs(FIG_DIR, exist_ok=True)

MU = np.array([1.3, 1.3])
SCORE_TS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

runs, floors_raw, other = fct.load_all(OUT_DIR)

floors = {name.replace("floor_", ""): rec for name, rec in floors_raw.items()}
grid_rows = other.get("grid_summary", [])
best_params = other.get("best_params5", {})

main_runs  = {k: v for k, v in runs.items() if k.startswith("ex")}
tsweep_runs = {k: v for k, v in runs.items() if k.startswith("tsweep")}
depth_runs = {k: v for k, v in runs.items() if k.startswith("depth")}
print(f"main {len(main_runs)} | tsweep {len(tsweep_runs)} | "
      f"depth {len(depth_runs)}")


def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    plt.show()


# %% Cell 2: tab:w2_floor — floor against N with the N^{-1/4} prediction
for tname, rec in floors.items():
    by_N = rec["by_N"]
    Ns = sorted(by_N)
    plt.errorbar(Ns, [by_N[N]["mean"] for N in Ns],
                 yerr=[by_N[N]["std"] for N in Ns], marker="o", label=tname)
    ref = by_N[Ns[-1]]["mean"]
    plt.plot(Ns, [ref * (Ns[-1] / N) ** 0.25 for N in Ns], "k--", lw=0.8,
             label=f"{tname} $N^{{-1/4}}$" if tname == "bimodal" else None)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("$N$"); plt.ylabel(r"$\widehat W_2^{\,\mathrm{floor}}(N)$")
plt.legend(); plt.tight_layout()
savefig("w2_floor")

print(fct.latex_floor_table(floors["bimodal"]["by_N"]))

# %% Cell 3: tab:gridsearch_results
if grid_rows:
    print(f"{'optimiser':12s} {'lam':>8s} {'eps_b':>6s} {'W2':>9s} {'sd':>8s}")
    for r_ in sorted(grid_rows, key=lambda x: (x["optimiser"],
                                               x.get("w2_mean") or 1e9)):
        w2 = "DIVERGED" if r_.get("diverged") else f"{r_['w2_mean']:.4f}"
        eb = f"{r_['eps_b']:g}" if r_.get("eps_b") else "-"
        sd = f"{r_['w2_std']:.4f}" if r_.get("w2_std") is not None else "-"
        print(f"{r_['optimiser']:12s} {r_['lam']:>8g} {eb:>6s} {w2:>9s} {sd:>8s}")
    print("\nbest:", best_params)
    print("\n" + fct.latex_grid_table(
        [r_ for r_ in grid_rows if not r_.get("diverged")]))

# %% Cell 4: E3 main table — one row per (example, optimiser), seeds pooled
from collections import defaultdict

pooled = {}
by_group = defaultdict(list)
for tag, rec in main_runs.items():
    c = rec["config"]
    by_group[(c["example"], c["optimiser"])].append(rec)

print(f"{'example':8s} {'optimiser':11s} {'W2 (seed mean)':>15s} "
      f"{'seed sd':>8s} {'Ebar':>8s} {'balance':>8s} {'div':>4s}")
for (ex, opt), recs in sorted(by_group.items()):
    ok = [r_ for r_ in recs if not r_["diverged"] and "w2_mean" in r_]
    n_div = sum(r_["diverged"] for r_ in recs)
    if ok:
        w2s = [r_["w2_mean"] for r_ in ok]
        ebars = [r_["score_error_agg"] for r_ in ok]
        bals = [r_["mode_balance"] for r_ in ok]
        rep = dict(ok[0])   # representative record for the LaTeX builder
        rep["w2_mean"], rep["w2_std"] = float(np.mean(w2s)), float(np.std(w2s))
        rep["score_error_agg"] = float(np.mean(ebars))
        rep["mode_balance"] = float(np.mean(bals))
        rep["diverged"] = n_div > 0
        pooled[f"{ex}_{opt}"] = rep
        print(f"{ex:8s} {opt:11s} {np.mean(w2s):>15.4f} {np.std(w2s):>8.4f} "
              f"{np.mean(ebars):>8.4f} {np.mean(bals):>8.4f} "
              f"{n_div:>3d}/{len(recs)}")
    else:
        print(f"{ex:8s} {opt:11s} {'all diverged':>15s} ({len(recs)} seeds)")

fl = {t: dict(mean=floors[t]["mean"]) for t in floors}
print("\n" + fct.latex_main_table(pooled, fl))

# %% Cell 5: E1 — W2 and score error against iterations (fig:e1_iterations)
EXAMPLE = "ex3"          # <-- change to ex1/ex2/ex3
SEED = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for opt in ["adam", "sgld", "tusla", "theopoula"]:
    recs = [r_ for k, r_ in main_runs.items()
            if r_["config"]["example"] == EXAMPLE
            and r_["config"]["optimiser"] == opt
            and r_["config"]["seed"] == SEED]
    if not recs:
        continue
    ce = recs[0]["checkpoint_evals"]
    its = sorted(ce)
    axes[0].errorbar(its, [ce[i]["w2_mean"] for i in its],
                     yerr=[ce[i]["w2_std"] for i in its],
                     marker="o", ms=3, label=opt)
    axes[1].plot(its, [ce[i]["score_error_agg"] for i in its],
                 marker="o", ms=3, label=opt)

tname = {"ex1": "gaussian", "ex2": "gaussian", "ex3": "bimodal"}[EXAMPLE]
axes[0].axhline(floors[tname]["mean"], color="grey", ls="--",
                label=f"floor {floors[tname]['mean']:.3f}")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("training iterations")
axes[0].set_ylabel(r"$\widehat W_2$")
axes[0].legend(fontsize=8)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("training iterations")
axes[1].set_ylabel(r"$\bar{\mathcal E}$")
axes[1].legend(fontsize=8)
plt.suptitle(f"E1, {EXAMPLE}")
plt.tight_layout()
savefig(f"e1_iterations_{EXAMPLE}")

# %% Cell 6: E1 — score error resolved in t (fig:e1_score_error_by_t)
OPT = "theopoula"        # <-- change

recs = [r_ for k, r_ in main_runs.items()
        if r_["config"]["example"] == EXAMPLE
        and r_["config"]["optimiser"] == OPT
        and r_["config"]["seed"] == SEED]
if recs:
    ce = recs[0]["checkpoint_evals"]
    for it in sorted(ce):
        se = ce[it]["score_error"]
        ts = sorted(se)
        plt.plot(ts, [se[t] for t in ts], marker="o", ms=3,
                 label=f"iter {it:,}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("$t$"); plt.ylabel(r"$\mathcal E(t)$")
    plt.legend(fontsize=8); plt.title(f"{EXAMPLE}, {OPT}")
    plt.tight_layout()
    savefig(f"e1_score_error_by_t_{EXAMPLE}_{OPT}")

# %% Cell 7: E3 — smooth histograms from the 1,000,000 stored samples
EXAMPLE_H = "ex3"
TARGET_H = "bimodal"
LIM = 6.0
BINS = np.linspace(-LIM, LIM, 241)   # 1M samples support fine bins

target = fct.make_target(TARGET_H, 2, MU, 1.0)
np.random.seed(0)
true = target["sample"](1_000_000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for k in range(2):
    axes[k].hist(true[:, k], bins=BINS, density=True, alpha=0.3,
                 color="k", label="target")
    for opt in ["adam", "sgld", "tusla", "theopoula"]:
        recs = [r_ for _, r_ in main_runs.items()
                if r_["config"]["example"] == EXAMPLE_H
                and r_["config"]["optimiser"] == opt
                and "plot_samples" in r_]
        if recs:
            s = recs[0]["plot_samples"]
            frac = float(np.mean(np.all(np.abs(s) <= LIM, axis=1)))
            axes[k].hist(s[:, k], bins=BINS, density=True, histtype="step",
                         lw=1.4, label=f"{opt} ({frac:.0%} in window)")
    axes[k].set_xlim(-LIM, LIM)
    axes[k].set_xlabel(f"coordinate {k + 1}")
    axes[k].legend(fontsize=7)
plt.tight_layout()
savefig(f"e3_marginals_{EXAMPLE_H}")

# scatter (subsampled from the 1M for legibility)
opts_avail = [o for o in ["adam", "sgld", "tusla", "theopoula"]
              if any(r_["config"]["example"] == EXAMPLE_H
                     and r_["config"]["optimiser"] == o
                     and "plot_samples" in r_ for r_ in main_runs.values())]
fig, axes = plt.subplots(1, len(opts_avail) + 1,
                         figsize=(3.6 * (len(opts_avail) + 1), 3.6))
axes[0].scatter(true[:5000, 0], true[:5000, 1], s=2, alpha=0.3)
axes[0].set_title("target", fontsize=9)
for ax, opt in zip(axes[1:], opts_avail):
    rec = next(r_ for r_ in main_runs.values()
               if r_["config"]["example"] == EXAMPLE_H
               and r_["config"]["optimiser"] == opt
               and "plot_samples" in r_)
    s = rec["plot_samples"]
    ax.scatter(s[:5000, 0], s[:5000, 1], s=2, alpha=0.3, color="C1")
    ax.set_title(f"{opt}  balance {rec['plot_mode_balance']:.3f}", fontsize=9)
for ax in axes:
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
plt.tight_layout()
savefig(f"e3_scatter_{EXAMPLE_H}")

# %% Cell 8: E2 — terminal time sweep against the t* prediction
K = fct.K_bimodal(MU)
ts_lo = fct.t_star(K, fct.mu_of_R(MU, 100.0))   # mu -> 1: lower end
ts_hi = fct.t_star(K, fct.mu_of_R(MU, 5.0))     # small R: upper end

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for opt in ["adam", "theopoula"]:
    recs = sorted([r_ for r_ in tsweep_runs.values()
                   if r_["config"]["optimiser"] == opt
                   and "w2_mean" in r_],
                  key=lambda r_: r_["config"]["T"])
    Ts = [r_["config"]["T"] for r_ in recs]
    axes[0].errorbar(Ts, [r_["w2_mean"] for r_ in recs],
                     yerr=[r_["w2_std"] for r_ in recs], marker="o",
                     label=opt)
    axes[1].plot(Ts, [r_["score_error_agg"] for r_ in recs], marker="o",
                 label=opt)
axes[0].axhline(floors["bimodal"]["mean"], color="grey", ls="--",
                label="floor")
axes[0].axvspan(ts_lo, ts_hi, color="C2", alpha=0.15,
                label=f"$t^\\star \\in [{ts_lo:.2f}, {ts_hi:.2f}]$")
axes[0].set_xlabel("$T$"); axes[0].set_ylabel(r"$\widehat W_2$")
axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("$T$"); axes[1].set_ylabel(r"$\bar{\mathcal E}$")
axes[1].legend(fontsize=8)
plt.tight_layout()
savefig("e2_terminal_time")

print(fct.latex_tstar_table(MU, [5.0, 10.0, 20.0, 50.0, 100.0]))

# %% Cell 9: E4 — depth table and figure
theo_depth = {k: v for k, v in depth_runs.items()
              if v["config"]["optimiser"] == "theopoula"}
print(fct.latex_depth_table(theo_depth))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for opt in ["adam", "theopoula"]:
    recs = sorted([r_ for r_ in depth_runs.values()
                   if r_["config"]["optimiser"] == opt
                   and "w2_mean" in r_],
                  key=lambda r_: r_["config"]["arch"])
    Ls = [r_["config"]["arch"] - 1 for r_ in recs]
    axes[0].errorbar(Ls, [r_["w2_mean"] for r_ in recs],
                     yerr=[r_["w2_std"] for r_ in recs], marker="o",
                     label=opt)
    axes[1].plot(Ls, [r_["grad_ratio"] for r_ in recs], marker="o",
                 label=opt)
axes[0].axhline(floors["bimodal"]["mean"], color="grey", ls="--",
                label="floor")
axes[0].set_xlabel("hidden layers $L$"); axes[0].set_ylabel(r"$\widehat W_2$")
axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("hidden layers $L$")
axes[1].set_ylabel("layerwise gradient ratio at init")
axes[1].set_yscale("log"); axes[1].legend(fontsize=8)
plt.tight_layout()
savefig("e4_depth")

# %% Cell 10: Example 1 — the observable eps_AL and the SGLD step bound
print(f"{'optimiser':11s} {'seed':>4s} {'|theta-mu*|^2':>14s} "
      f"{'W2':>8s} {'lam used':>9s} {'lam_max (SGLD)':>14s}")
for tag, rec in sorted(main_runs.items()):
    c = rec["config"]
    if c["example"] != "ex1" or "eps_AL_observed" not in rec:
        continue
    print(f"{c['optimiser']:11s} {c['seed']:>4d} "
          f"{rec['eps_AL_observed']:>14.6f} {rec['w2_mean']:>8.4f} "
          f"{c['params']['lam']:>9g} {rec['sgld_lambda_max']:>14.4f}")

# %% Cell 11: loss and |theta| trajectories (any run)
TAG = "ex3_bimodal_theopoula_L2_s0"    # <-- change
if TAG in main_runs:
    rec = main_runs[TAG]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rec["loss_history"], lw=0.5)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss")
    axes[1].plot(rec["theta_norm_history"], lw=0.8)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel(r"$|\theta|$")
    plt.suptitle(TAG)
    plt.tight_layout()
    savefig(f"trajectories_{TAG}")

# %% Cell 12: raw vs applied layerwise gradient norms (mechanism figure)
TAG_G = "depth_L20_theopoula"          # <-- change
rec = depth_runs.get(TAG_G) or main_runs.get(TAG_G)
if rec and rec["grad_norms"]:
    it = max(rec["grad_norms"])
    names = [n for n in rec["grad_norms"][it] if "weight" in n]
    raw = [rec["grad_norms"][it][n] for n in names]
    upd = [rec["update_norms"][it][n] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(x, raw, "o-", label="raw gradient")
    ax.semilogy(x, upd, "s-", label="applied update")
    ax.set_xlabel("layer (input to output)")
    ax.set_ylabel("norm")
    ax.set_title(f"{TAG_G}, iteration {it}")
    ax.legend()
    plt.tight_layout()
    savefig(f"grad_profile_{TAG_G}")

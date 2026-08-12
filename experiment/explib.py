"""
explib.py

Extensions and corrections on top of functions3.py for the dissertation
experiments. Imports the network definitions and data samplers from
functions3, and replaces the training loops, the W2 computation and the
diagnostics so that every optimiser is measured on an identical footing.

Key differences from functions3.py:
  * one train() for all four optimisers, so the loss is identical by
    construction (functions3.train_network used nn.MSELoss, which averages
    over output dimensions, while the others summed: a factor of d)
  * the loss is computed as |z + sigma_t * s|^2 when kappa = sigma_t^2,
    which is algebraically identical but bounded as t -> 0
  * W2 uses a numItermax large enough for N = 10^4 and RAISES if the
    solver stops before optimality, instead of silently returning a
    non-optimal (too large) cost
  * both raw gradient norms and effective update norms are recorded
"""

import gc
import time

import numpy as np
import ot
import torch
import torch.nn as nn

from functions3 import (
    ScoreNetDeep,
    get_device,
    set_seed,
    save_results,
    load_results,
    sample_bimodal_data,
)


# ============================================================
# Architectures
# ============================================================

class ScoreNetDeepSkip(nn.Module):
    """
    ScoreNetDeep plus a linear path from the input z_0 straight to the
    output. Remark on representability: the target score contains an
    affine field in x that a bounded activation cannot represent without
    driving |theta| large. This variant represents it exactly.
    """

    def __init__(self, d, hidden_dims):
        super().__init__()
        D_0 = d + 1
        dims = [D_0] + list(hidden_dims) + [d]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(D_0, d, bias=False)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        h = z
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h) + self.skip(z)


def build_net(d, n_layers, width, use_skip=False):
    """
    n_layers counts affine maps, matching the ScoreNet / ScoreNet3
    convention, so hidden_dims has (n_layers - 1) entries and the number
    of hidden layers is L = n_layers - 1.
    """
    hidden_dims = [width] * (n_layers - 1)
    cls = ScoreNetDeepSkip if use_skip else ScoreNetDeep
    return cls(d=d, hidden_dims=hidden_dims)


# ============================================================
# Targets: data, exact marginal samples, exact marginal score
# ============================================================

def sample_gaussian_data(N, d, mu, sigma=1.0):
    """N(mu, sigma^2 I_d). Strongly log-concave, so every data-side
    hypothesis holds in its strongest form."""
    return np.random.normal(loc=mu, scale=sigma, size=(N, d))


def make_target(name, d, mu_vec, sigma_0=1.0):
    """
    Returns a dict describing a target distribution:
      sample(N)       -> i.i.d. draws from pi_D
      sample_pt(N, t) -> i.i.d. draws from the marginal p_t
      score(x, t)     -> exact grad_x log p_t(x), numpy (n,d)
    """
    mu_vec = np.asarray(mu_vec, dtype=np.float64)

    if name == "gaussian":
        def sample(N):
            return sample_gaussian_data(N, d, mu_vec, sigma_0)

        def v_t(t):
            return 1.0 + np.exp(-2 * t) * (sigma_0 ** 2 - 1.0)

        def sample_pt(N, t):
            m_t = np.exp(-t)
            return np.random.normal(
                loc=m_t * mu_vec, scale=np.sqrt(v_t(t)), size=(N, d)
            )

        def score(x, t):
            m_t = np.exp(-t)
            return -(x - m_t * mu_vec) / v_t(t)

    elif name == "bimodal":
        def sample(N):
            return sample_bimodal_data(N, d, mu_vec, -mu_vec, sigma_0, 0.5)

        def v_t(t):
            return 1.0 + np.exp(-2 * t) * (sigma_0 ** 2 - 1.0)

        def sample_pt(N, t):
            m_t = np.exp(-t)
            sgn = np.where(np.random.uniform(size=(N, 1)) < 0.5, 1.0, -1.0)
            centres = sgn * (m_t * mu_vec)
            return centres + np.sqrt(v_t(t)) * np.random.normal(size=(N, d))

        def score(x, t):
            m_t, v = np.exp(-t), v_t(t)
            u = (x @ mu_vec) * m_t / v                      # (n,)
            return -x / v + (m_t / v) * np.tanh(u)[:, None] * mu_vec[None, :]

    else:
        raise ValueError(f"unknown target {name!r}")

    return {"name": name, "d": d, "mu": mu_vec, "sigma_0": sigma_0,
            "sample": sample, "sample_pt": sample_pt, "score": score}


# ============================================================
# Training data
# ============================================================

def sample_training_batch(X_data, t_0, T, d, batch_size):
    """
    One minibatch of training triples xi = (tau, x_0, z).
    Returns tau, x_t, z, sigma_tau as float32 numpy arrays. Note that z
    is returned rather than the target -z/sigma, so that the loss can be
    formed in the numerically bounded way; see compute_loss.
    """
    N = X_data.shape[0]
    tau = np.random.uniform(t_0, T, size=(batch_size, 1))
    idx = np.random.randint(0, N, size=batch_size)
    x0 = X_data[idx]
    z = np.random.normal(0.0, 1.0, size=(batch_size, d))

    m_tau = np.exp(-tau)
    sigma_tau = np.sqrt(1.0 - np.exp(-2.0 * tau))
    x_t = m_tau * x0 + sigma_tau * z
    return (tau.astype(np.float32), x_t.astype(np.float32),
            z.astype(np.float32), sigma_tau.astype(np.float32))


def compute_loss(net, tau_t, x_t_t, z_t, sigma_t, kappa_mode):
    """
    Single-sample loss, averaged over the batch and SUMMED over output
    coordinates. Identical for every optimiser.

    kappa_mode == "sigma2":  kappa(t) |z/sigma_t + s|^2 = |z + sigma_t s|^2
    kappa_mode == "one":     |z/sigma_t + s|^2
    """
    s = net(x_t_t, tau_t)
    if kappa_mode == "sigma2":
        resid = z_t + sigma_t * s
    elif kappa_mode == "one":
        resid = z_t / sigma_t + s
    else:
        raise ValueError(f"unknown kappa_mode {kappa_mode!r}")
    return (resid ** 2).sum(dim=1).mean()


# ============================================================
# Diagnostics
# ============================================================

def layerwise_grad_norms(net, X_data, t_0, T, d, batch_size, device,
                         kappa_mode="sigma2"):
    """
    One fresh forward/backward pass; returns {parameter_name: ||grad||}.
    This is the RAW gradient, before any taming, boosting or
    preconditioning, so it is the quantity that vanishes with depth.
    """
    net.to(device)
    net.train()
    net.zero_grad()

    tau, x_t, z, sig = sample_training_batch(X_data, t_0, T, d, batch_size)
    loss = compute_loss(
        net,
        torch.tensor(tau, device=device),
        torch.tensor(x_t, device=device),
        torch.tensor(z, device=device),
        torch.tensor(sig, device=device),
        kappa_mode,
    )
    loss.backward()
    out = {n: (p.grad.norm().item() if p.grad is not None else None)
           for n, p in net.named_parameters()}
    net.zero_grad()
    return out


def score_field_grid(net, target, t_values, device, lim=4.0, n_side=25):
    """
    Learned and exact score fields on a fixed square grid, at several
    noise levels. Saved so the figure is reproducible from the artefact
    without reloading the network.
    """
    xs = np.linspace(-lim, lim, n_side)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    out = {}
    net.eval()
    with torch.no_grad():
        for t in t_values:
            x_t = torch.tensor(pts, dtype=torch.float32, device=device)
            t_t = torch.full((pts.shape[0], 1), float(t),
                             dtype=torch.float32, device=device)
            s_pred = net(x_t, t_t).cpu().numpy()
            out[float(t)] = {
                "X": pts.copy(),
                "S_pred": s_pred,
                "S_true": target["score"](pts, float(t)),
            }
    return out


# ============================================================
# Training: one loop, four optimisers
# ============================================================

def train(net, X_data, target_cfg, opt_cfg, run_cfg, device,
          grad_checkpoints=()):
    """
    Train `net` and return a diagnostics dict.

    opt_cfg["name"] in {"adam", "sgld", "tusla", "theopoula"}.

    Records, at every iteration: the loss and |theta|.
    Records, at each iteration in grad_checkpoints: the raw layerwise
    gradient norms and the layerwise norms of the update actually
    applied. The second is what shows how Adam and TheoPouLa counteract
    a vanishing raw gradient, since both rescale it before applying.
    """
    net.to(device)
    d = target_cfg["d"]
    t_0, T = run_cfg["t_0"], run_cfg["T"]
    B, n_iters = run_cfg["batch_size"], run_cfg["n_iters"]
    kappa_mode = run_cfg["kappa_mode"]

    name = opt_cfg["name"]
    lam = opt_cfg.get("lam", 1e-3)
    beta = opt_cfg.get("beta", 1e10)
    eta = opt_cfg.get("eta", 0.0)
    r = opt_cfg.get("r", 2.0)
    eps_b = opt_cfg.get("eps_b", 0.2)
    sqrt_lam = float(np.sqrt(lam))

    torch_opt = torch.optim.Adam(net.parameters(), lr=lam) if name == "adam" else None

    loss_history, theta_norm_history = [], []
    grad_norms, update_norms = {}, {}
    diverged, nan_iter = False, None

    net.train()
    for it in range(1, n_iters + 1):
        tau, x_t, z, sig = sample_training_batch(X_data, t_0, T, d, B)
        tau_t = torch.tensor(tau, device=device)
        x_t_t = torch.tensor(x_t, device=device)
        z_t = torch.tensor(z, device=device)
        sig_t = torch.tensor(sig, device=device)

        net.zero_grad()
        loss = compute_loss(net, tau_t, x_t_t, z_t, sig_t, kappa_mode)
        loss.backward()

        capture = (it in grad_checkpoints)
        if capture:
            grad_norms[it] = {n: (p.grad.norm().item() if p.grad is not None else None)
                              for n, p in net.named_parameters()}
            before = {n: p.detach().clone() for n, p in net.named_parameters()}

        if name == "adam":
            torch_opt.step()

        else:
            with torch.no_grad():
                theta_sq = sum((p ** 2).sum() for p in net.parameters())
                theta_norm = torch.sqrt(theta_sq)
                theta_2r = theta_norm ** (2 * r)

                for p in net.parameters():
                    G = p.grad

                    if name == "sgld":
                        step = G

                    elif name == "tusla":
                        # H^reg = G + eta * theta |theta|^{2r}, then a single
                        # global taming factor on the whole parameter vector
                        step = (G + eta * p * theta_2r) / (1 + sqrt_lam * theta_2r)

                    elif name == "theopoula":
                        # elementwise taming * boosting, plus the tamed
                        # regularisation term
                        Ga = G.abs()
                        tamed = G / (1 + sqrt_lam * Ga)
                        boosted = tamed * (1 + sqrt_lam / (eps_b + Ga))
                        reg = eta * p * theta_2r / (1 + sqrt_lam * theta_2r)
                        step = boosted + reg

                    else:
                        raise ValueError(f"unknown optimiser {name!r}")

                    noise = torch.randn_like(p) * np.sqrt(2 * lam / beta)
                    p.add_(-lam * step + noise)

        with torch.no_grad():
            tn = torch.sqrt(sum((p ** 2).sum() for p in net.parameters())).item()

        if capture:
            with torch.no_grad():
                update_norms[it] = {
                    n: (p.detach() - before[n]).norm().item()
                    for n, p in net.named_parameters()
                }

        l = loss.item()
        loss_history.append(l)
        theta_norm_history.append(tn)

        if not np.isfinite(l) or not np.isfinite(tn):
            diverged, nan_iter = True, it
            print(f"    [{name}] diverged at iteration {it}")
            break

        if it % run_cfg.get("print_every", 2000) == 0:
            print(f"    [{name}] iter {it}/{n_iters}  loss={l:.4f}  |theta|={tn:.3f}")

    return {
        "loss_history": loss_history,
        "theta_norm_history": theta_norm_history,
        "grad_norms": grad_norms,
        "update_norms": update_norms,
        "diverged": diverged,
        "nan_iter": nan_iter,
    }


# ============================================================
# Sampling and W2
# ============================================================

def em_sample(net, gamma, T, d, N_samples, device):
    """Euler-Maruyama on the practical reverse SDE, started at N(0, I_d)."""
    n_steps = int(round(T / gamma))
    net.eval()
    Y = np.random.normal(0.0, 1.0, size=(N_samples, d))
    with torch.no_grad():
        for k in range(n_steps):
            t_k = T - k * gamma
            Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
            t_t = torch.full((N_samples, 1), float(t_k),
                             dtype=torch.float32, device=device)
            s = net(Y_t, t_t).cpu().numpy()
            Y = Y + gamma * (Y + 2.0 * s) + np.sqrt(2.0 * gamma) * \
                np.random.normal(size=(N_samples, d))
    return Y


def w2_exact(samples_a, samples_b, num_iter_max):
    """
    Exact W2 via POT's network simplex. Raises if the solver stops before
    optimality: that failure mode is silent and returns a cost that is too
    large, which would look like a worse model.
    """
    n = samples_a.shape[0]
    assert samples_b.shape[0] == n
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = ot.dist(np.asarray(samples_a, dtype=np.float64),
                np.asarray(samples_b, dtype=np.float64),
                metric="sqeuclidean")
    cost, log = ot.emd2(a, b, M, numItermax=num_iter_max, log=True)
    warn = log.get("warning")
    del M
    gc.collect()
    if warn is not None:
        raise RuntimeError(f"ot.emd2 did not reach optimality: {warn}")
    return float(np.sqrt(cost))


def estimate_w2(net, target, N_w2, n_repeats, gamma, T, d, device,
                num_iter_max):
    """
    Mean and per-repeat W2 between generated and reference samples.
    BOTH sets are redrawn on every repeat, so the reported spread
    includes reference-draw variability and is directly comparable with
    the floor below.
    """
    vals = []
    for _ in range(n_repeats):
        gen = em_sample(net, gamma, T, d, N_w2, device)
        ref = target["sample"](N_w2)
        vals.append(w2_exact(gen, ref, num_iter_max))
    return float(np.mean(vals)), float(np.std(vals)), vals


def w2_floor(target, N, n_repeats, num_iter_max):
    """
    Finite-sample floor: W2 between two INDEPENDENT N-samples of the same
    measure. No model can be distinguished from the target below this.
    """
    vals = []
    for _ in range(n_repeats):
        vals.append(w2_exact(target["sample"](N), target["sample"](N),
                             num_iter_max))
    return float(np.mean(vals)), float(np.std(vals)), vals


# ============================================================
# Reporting
# ============================================================

def r_min_theopoula(n_layers):
    """q = 2L + 1 with L = n_layers - 1 hidden layers, so r_min = L + 1/2."""
    L = n_layers - 1
    return L + 0.5


def latex_table(results, floors, caption="", label="tab:w2_results"):
    """
    booktabs table of W2 against the floor, ready to paste into Overleaf.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \begin{tabular}{llrrrrc}",
        r"    \toprule",
        r"    Target & Optimiser & Layers & $\widehat W_2$ & s.d. & floor & diverged \\",
        r"    \midrule",
    ]
    for tag in sorted(results):
        r_ = results[tag]
        c = r_["config"]
        fl = floors.get(c["target"], {}).get("mean", float("nan"))
        w2 = r_.get("w2_mean")
        sd = r_.get("w2_std")
        w2s = "---" if w2 is None or not np.isfinite(w2) else f"{w2:.4f}"
        sds = "---" if sd is None or not np.isfinite(sd) else f"{sd:.4f}"
        lines.append(
            f"    {c['target']} & {c['optimiser']} & {c['n_layers']} & "
            f"{w2s} & {sds} & {fl:.4f} & "
            f"{'yes' if r_['diverged'] else 'no'} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)
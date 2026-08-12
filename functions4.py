"""
functions4.py

Self-contained library for the dissertation experiments. Nothing is
imported from functions3.py.

What changed relative to functions3.py, and why:

  * ONE training loop serves all four optimisers, so the loss is
    identical across arms by construction. functions3.train_network used
    nn.MSELoss, which averages over output dimensions, while the SGLD,
    TUSLA and TheoPouLa loops summed over them: a silent factor of d.

  * The loss is formed as |z + sigma_t * s|^2 when kappa(t) = sigma_t^2.
    This is algebraically identical to kappa |z/sigma_t + s|^2 but stays
    bounded as t -> 0, where the old form produced targets of order 700
    at t = 1e-6 and divided by zero at t = 0.

  * W2 uses a network-simplex ceiling large enough for N = 10^4 and
    RAISES if the solver stops before optimality. That failure is silent
    in POT and returns a cost which is too large, i.e. it looks like a
    worse model.

  * estimate_w2 redraws BOTH the generated and the reference sample on
    every repeat, so its spread is comparable with the floor, which does
    the same. functions3.estimate_W2 held the reference fixed.

  * Diagnostics record the raw layerwise gradient AND the layerwise norm
    of the update actually applied. The pair is what shows the mechanism:
    at depth the raw gradient vanishes toward the input side, and Adam's
    preconditioner and TheoPouLa's boosting function each flatten the
    applied update, while SGLD and TUSLA do not.
"""

import gc
import os
import time

import numpy as np
import ot
import torch
import torch.nn as nn


# ============================================================
# Device, reproducibility, IO
# ============================================================

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    seed = int(seed) % (2 ** 31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_from_tag(tag, offset=0):
    """Deterministic per-configuration seed, stable across sessions."""
    h = 0
    for ch in tag:
        h = (h * 131 + ord(ch)) % (2 ** 31 - 1)
    return (h + offset) % (2 ** 31 - 1)


def save_results(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)
    print(f"    saved {path}")


def load_results(path):
    return torch.load(path, weights_only=False)


# ============================================================
# Architectures
#
# n_layers counts AFFINE MAPS, so a network with n_layers = N has
# L = N - 1 hidden layers. This matches the ScoreNet / ScoreNet3
# convention used throughout the dissertation.
# ============================================================

class ScoreNetDeep(nn.Module):
    """tanh hidden layers, identity output. Input z_0 = (x, t)."""

    def __init__(self, d, hidden_dims):
        super().__init__()
        dims = [d + 1] + list(hidden_dims) + [d]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.activation = nn.Tanh()

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        return self.layers[-1](z)


class ScoreNetDeepSkip(nn.Module):
    """
    As above, plus a linear path from z_0 straight to the output. The
    target score contains an affine field in x which a bounded activation
    cannot represent without driving |theta| large; this variant
    represents it exactly.
    """

    def __init__(self, d, hidden_dims):
        super().__init__()
        dims = [d + 1] + list(hidden_dims) + [d]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(d + 1, d, bias=False)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        h = z
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h) + self.skip(z)


class MeanNet(nn.Module):
    """
    The closed-form approximator of Example 1: s(t, theta, x) = -x + m_t theta.
    Contains the exact score of N(mu, I) at theta = mu, so the
    approximation error at the optimum is zero and theta* is known.
    Same forward(x, t) signature as the networks, so every routine below
    accepts it unchanged.
    """

    def __init__(self, d):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(d))

    def forward(self, x, t):
        return -x + torch.exp(-t) * self.theta


def build_net(d, arch, width, use_skip=False):
    """arch is either the string "affine" or an integer number of affine maps."""
    if arch == "affine":
        return MeanNet(d)
    hidden_dims = [width] * (int(arch) - 1)
    cls = ScoreNetDeepSkip if use_skip else ScoreNetDeep
    return cls(d=d, hidden_dims=hidden_dims)


# ============================================================
# Targets
# ============================================================

def make_target(name, d, mu_vec, sigma_0=1.0):
    """
    Returns a dict with
        sample(N)        i.i.d. draws from pi_D
        sample_pt(N, t)  i.i.d. draws from the marginal p_t
        score(x, t)      exact grad_x log p_t(x), numpy (n, d)

    "gaussian" : N(mu, sigma_0^2 I). Strongly log-concave, so every
                 data-side hypothesis holds in its strongest form.
    "bimodal"  : equally weighted mixture of N(+-mu, sigma_0^2 I).
                 Semiconvex only.
    """
    mu_vec = np.asarray(mu_vec, dtype=np.float64)

    def v_t(t):
        return 1.0 + np.exp(-2.0 * t) * (sigma_0 ** 2 - 1.0)

    if name == "gaussian":
        def sample(N):
            return np.random.normal(mu_vec, sigma_0, size=(N, d))

        def sample_pt(N, t):
            return np.random.normal(np.exp(-t) * mu_vec, np.sqrt(v_t(t)),
                                    size=(N, d))

        def score(x, t):
            return -(x - np.exp(-t) * mu_vec) / v_t(t)

    elif name == "bimodal":
        def sample(N):
            sgn = np.where(np.random.uniform(size=(N, 1)) < 0.5, 1.0, -1.0)
            return sgn * mu_vec + sigma_0 * np.random.normal(size=(N, d))

        def sample_pt(N, t):
            m_t, v = np.exp(-t), v_t(t)
            sgn = np.where(np.random.uniform(size=(N, 1)) < 0.5, 1.0, -1.0)
            return sgn * (m_t * mu_vec) + np.sqrt(v) * np.random.normal(size=(N, d))

        def score(x, t):
            m_t, v = np.exp(-t), v_t(t)
            u = (x @ mu_vec) * m_t / v
            return -x / v + (m_t / v) * np.tanh(u)[:, None] * mu_vec[None, :]

    else:
        raise ValueError(f"unknown target {name!r}")

    return dict(name=name, d=d, mu=mu_vec, sigma_0=sigma_0,
                sample=sample, sample_pt=sample_pt, score=score, v_t=v_t)


# ============================================================
# Training data and loss
# ============================================================

def sample_training_batch(X_data, t_0, T, d, batch_size):
    """
    One minibatch of training triples xi = (tau, x_0, z).

    Returns tau, x_t, z, sigma_tau. Note that z is returned rather than
    the target -z/sigma_tau, so the loss can be formed in the bounded way.
    """
    N = X_data.shape[0]
    tau = np.random.uniform(t_0, T, size=(batch_size, 1))
    x0 = X_data[np.random.randint(0, N, size=batch_size)]
    z = np.random.normal(0.0, 1.0, size=(batch_size, d))

    m_tau = np.exp(-tau)
    sigma_tau = np.sqrt(1.0 - np.exp(-2.0 * tau))
    x_t = m_tau * x0 + sigma_tau * z

    return (tau.astype(np.float32), x_t.astype(np.float32),
            z.astype(np.float32), sigma_tau.astype(np.float32))


def to_device_batch(batch, device):
    return [torch.tensor(a, device=device) for a in batch]


def compute_loss(net, tau_t, x_t_t, z_t, sig_t, kappa_mode):
    """
    Single-sample loss, averaged over the batch and SUMMED over output
    coordinates. Identical for every optimiser.

    "sigma2" : kappa(t) = sigma_t^2, so kappa |z/sigma + s|^2 = |z + sigma s|^2
    "one"    : kappa(t) = 1
    """
    s = net(x_t_t, tau_t)
    if kappa_mode == "sigma2":
        resid = z_t + sig_t * s
    elif kappa_mode == "one":
        resid = z_t / sig_t + s
    else:
        raise ValueError(f"unknown kappa_mode {kappa_mode!r}")
    return (resid ** 2).sum(dim=1).mean()


# ============================================================
# Diagnostics
# ============================================================

def layerwise_grad_norms(net, X_data, t_0, T, d, batch_size, device,
                         kappa_mode="sigma2"):
    """
    Fresh forward/backward pass. Returns {parameter_name: ||grad||}, the
    RAW gradient before any taming, boosting or preconditioning.
    """
    net.to(device)
    net.train()
    net.zero_grad()
    b = to_device_batch(sample_training_batch(X_data, t_0, T, d, batch_size), device)
    compute_loss(net, b[0], b[1], b[2], b[3], kappa_mode).backward()
    out = {n: (p.grad.norm().item() if p.grad is not None else None)
           for n, p in net.named_parameters()}
    net.zero_grad()
    return out


def score_field_grid(net, target, t_values, device, lim=4.0, n_side=25):
    """Learned and exact score fields on a fixed grid, at several noise levels."""
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
            out[float(t)] = dict(
                X=pts.copy(),
                S_pred=net(x_t, t_t).cpu().numpy(),
                S_true=target["score"](pts, float(t)),
            )
    return out


def score_error_by_t(net, target, t_values, device, n_mc=20000):
    """
    E_{X ~ p_t} |s(theta,(X,t)) - grad log p_t(X)|^2, by Monte Carlo over
    exact draws from the marginal. Reported as a diagnostic, not as a
    metric: it carries no confidence interval.
    """
    out = {}
    net.eval()
    with torch.no_grad():
        for t in t_values:
            X = target["sample_pt"](n_mc, float(t))
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            t_t = torch.full((n_mc, 1), float(t), dtype=torch.float32, device=device)
            S = net(x_t, t_t).cpu().numpy()
            out[float(t)] = float(np.mean(np.sum((S - target["score"](X, float(t))) ** 2,
                                                 axis=1)))
    return out


# ============================================================
# Training: one loop, four optimisers
# ============================================================

VALID_OPTIMISERS = ("adam", "sgld", "tusla", "theopoula")


def train(net, X_data, target, opt_cfg, run_cfg, device, grad_checkpoints=()):
    """
    Train and return a diagnostics dict.

    opt_cfg keys:
        name   one of VALID_OPTIMISERS
        lam    step size / learning rate
        beta   inverse temperature (Langevin arms); large beta means little noise
        eta    regularisation strength (TUSLA, TheoPouLa)
        r      regularisation exponent
        eps_b  boosting denominator offset (TheoPouLa)
    """
    net.to(device)
    d = target["d"]
    t_0, T = run_cfg["t_0"], run_cfg["T"]
    B, n_iters = run_cfg["batch_size"], run_cfg["n_iters"]
    kappa_mode = run_cfg["kappa_mode"]
    print_every = run_cfg.get("print_every", 2000)

    name = opt_cfg["name"]
    if name not in VALID_OPTIMISERS:
        raise ValueError(f"unknown optimiser {name!r}")
    lam = float(opt_cfg.get("lam", 1e-3))
    beta = float(opt_cfg.get("beta", 1e10))
    eta = float(opt_cfg.get("eta", 0.0))
    r = float(opt_cfg.get("r", 2.0))
    eps_b = float(opt_cfg.get("eps_b", 0.2))
    sqrt_lam = float(np.sqrt(lam))
    noise_scale = float(np.sqrt(2.0 * lam / beta))

    torch_opt = torch.optim.Adam(net.parameters(), lr=lam) if name == "adam" else None

    loss_history, theta_norm_history = [], []
    grad_norms, update_norms = {}, {}
    diverged, nan_iter = False, None

    net.train()
    for it in range(1, n_iters + 1):
        b = to_device_batch(sample_training_batch(X_data, t_0, T, d, B), device)

        net.zero_grad()
        loss = compute_loss(net, b[0], b[1], b[2], b[3], kappa_mode)
        loss.backward()

        capture = it in grad_checkpoints
        if capture:
            grad_norms[it] = {n: (p.grad.norm().item() if p.grad is not None else None)
                              for n, p in net.named_parameters()}
            before = {n: p.detach().clone() for n, p in net.named_parameters()}

        if name == "adam":
            torch_opt.step()
        else:
            with torch.no_grad():
                theta_norm_t = torch.sqrt(sum((p ** 2).sum() for p in net.parameters()))
                theta_2r = theta_norm_t ** (2.0 * r)

                for p in net.parameters():
                    G = p.grad

                    if name == "sgld":
                        step = G

                    elif name == "tusla":
                        # regularised gradient, then ONE global taming factor
                        step = (G + eta * p * theta_2r) / (1.0 + sqrt_lam * theta_2r)

                    else:  # theopoula
                        Ga = G.abs()
                        tamed = G / (1.0 + sqrt_lam * Ga)
                        boosted = tamed * (1.0 + sqrt_lam / (eps_b + Ga))
                        reg = eta * p * theta_2r / (1.0 + sqrt_lam * theta_2r)
                        step = boosted + reg

                    p.add_(-lam * step + noise_scale * torch.randn_like(p))

        with torch.no_grad():
            tn = torch.sqrt(sum((p ** 2).sum() for p in net.parameters())).item()
            if capture:
                update_norms[it] = {n: (p.detach() - before[n]).norm().item()
                                    for n, p in net.named_parameters()}

        l = loss.item()
        loss_history.append(l)
        theta_norm_history.append(tn)

        if not (np.isfinite(l) and np.isfinite(tn)):
            diverged, nan_iter = True, it
            print(f"    [{name}] DIVERGED at iteration {it}")
            break

        if it % print_every == 0:
            print(f"    [{name}] iter {it}/{n_iters}  loss={l:.4f}  |theta|={tn:.3f}")

    return dict(loss_history=loss_history, theta_norm_history=theta_norm_history,
                grad_norms=grad_norms, update_norms=update_norms,
                diverged=diverged, nan_iter=nan_iter)


# ============================================================
# Reverse-time sampling
# ============================================================

def em_sample(net, gamma, T, d, N_samples, device, batch=None):
    """
    Euler-Maruyama on the practical reverse SDE, started at N(0, I_d).
    The smallest time argument reached is gamma, so this scheme realises
    epsilon = gamma; the two are not independently controllable here.
    """
    n_steps = int(round(T / gamma))
    net.eval()
    Y = np.random.normal(0.0, 1.0, size=(N_samples, d))
    bs = batch or N_samples
    with torch.no_grad():
        for k in range(n_steps):
            t_k = T - k * gamma
            S = np.empty_like(Y)
            for i in range(0, N_samples, bs):
                j = min(i + bs, N_samples)
                y = torch.tensor(Y[i:j], dtype=torch.float32, device=device)
                tt = torch.full((j - i, 1), float(t_k), dtype=torch.float32,
                                device=device)
                S[i:j] = net(y, tt).cpu().numpy()
            Y = Y + gamma * (Y + 2.0 * S) + \
                np.sqrt(2.0 * gamma) * np.random.normal(size=(N_samples, d))
    return Y


# ============================================================
# Wasserstein-2
# ============================================================

def w2_exact(samples_a, samples_b, num_iter_max):
    """
    Exact W2 by network simplex (POT). Raises if the solver stops before
    optimality, since that failure is silent and returns a cost which is
    too large.
    """
    n = samples_a.shape[0]
    assert samples_b.shape[0] == n, "sample sets must be the same size"
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = ot.dist(np.ascontiguousarray(samples_a, dtype=np.float64),
                np.ascontiguousarray(samples_b, dtype=np.float64),
                metric="sqeuclidean")
    cost, log = ot.emd2(a, b, M, numItermax=num_iter_max, log=True)
    warn = log.get("warning")
    del M
    gc.collect()
    if warn is not None:
        raise RuntimeError(f"ot.emd2 did not reach optimality: {warn}")
    return float(np.sqrt(cost))


def estimate_w2(net, target, N_w2, n_repeats, gamma, T, d, device, num_iter_max,
                sample_batch=None):
    """Both generated and reference samples redrawn on every repeat."""
    vals = []
    for i in range(n_repeats):
        gen = em_sample(net, gamma, T, d, N_w2, device, batch=sample_batch)
        ref = target["sample"](N_w2)
        v = w2_exact(gen, ref, num_iter_max)
        vals.append(v)
        print(f"      W2 repeat {i + 1}/{n_repeats}: {v:.4f}")
    return float(np.mean(vals)), float(np.std(vals)), vals


def w2_floor(target, N, n_repeats, num_iter_max):
    """
    Finite-sample floor: W2 between two INDEPENDENT N-samples of the same
    measure. Nothing below this can be distinguished from the target.
    """
    vals = [w2_exact(target["sample"](N), target["sample"](N), num_iter_max)
            for _ in range(n_repeats)]
    return float(np.mean(vals)), float(np.std(vals)), vals


# ============================================================
# Admissibility of the regularisation exponent
# ============================================================

def q_of(arch):
    """
    Polynomial Lipschitz exponent of the stochastic gradient.
    q = 2L + 1 for L hidden layers; the affine family has an affine H, so
    no polynomial growth in theta.
    """
    if arch == "affine":
        return 1.0
    return 2.0 * (int(arch) - 1) + 1.0


def r_min(arch, which="theopoula"):
    q = q_of(arch)
    return q / 2.0 + (1.0 if which == "tusla" else 0.0)


# ============================================================
# Reporting
# ============================================================

def latex_table(results, floors, caption="", label="tab:w2_results",
                order=None):
    """booktabs table of W2 against the floor, for pasting into Overleaf."""
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{llrrrrrc}", r"    \toprule",
             r"    Target & Optimiser & Layers & $\widehat W_2$ & s.d.\ & "
             r"floor & $|\theta|$ & diverged \\",
             r"    \midrule"]
    keys = order or sorted(results)
    for tag in keys:
        r_ = results[tag]
        c = r_["config"]
        fl = floors.get(c["target"], {}).get("mean", float("nan"))
        tn = r_["theta_norm_history"][-1] if r_["theta_norm_history"] else float("nan")
        w2, sd = r_.get("w2_mean"), r_.get("w2_std")
        fmt = lambda v: "---" if v is None or not np.isfinite(v) else f"{v:.4f}"
        lines.append(
            f"    {c['target']} & {c['optimiser']} & {c['arch']} & "
            f"{fmt(w2)} & {fmt(sd)} & {fl:.4f} & {tn:.2f} & "
            f"{'yes' if r_['diverged'] else 'no'} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)
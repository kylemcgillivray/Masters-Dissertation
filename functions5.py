"""
functions5.py

Fifth-generation library for the dissertation experiments. Self-contained;
nothing is imported from functions3/functions4.

Changes relative to functions4.py, and why:

  * score_field_grid is REMOVED. Vector-field plots are not reported.

  * train() now takes `checkpoints` and an `eval_fn` callback, so W2 and
    score-error curves against iterations (experiment E1) come from ONE
    continuous run. This matters for Adam: segmenting training into
    separate train() calls resets its moment estimates at every segment
    boundary, whereas a callback leaves the optimiser state intact.

  * estimate_w2 accepts `eval_seed`. When set, every repeat re-seeds the
    generator, so all configurations are evaluated under IDENTICAL
    Brownian increments and reference draws (common random numbers).
    Differences between rows of a table are then attributable to the
    trained network alone.

  * mode_balance() is recorded at every evaluation. On the symmetric
    bimodal target, W2 is dominated by the mixture weight: an imbalance
    delta costs approximately W2^2 = delta * |2 mu|^2, so a 0.5% skew
    moves W2 from 0.13 to 0.28 while every score error is unchanged.
    Without this diagnostic the W2 curve is uninterpretable.

  * score_error_aggregate() gives the scalar E-bar reported in the tables,
    the average of E(t) over the fixed grid SCORE_TS.

  * t_star helpers implement Proposition 16 of Bruno & Sabanis (2025)
    (arXiv:2505.03432): beta_os, its closed-form integral B, and the
    threshold t* by bisection, for the tab:t_star prediction.

  * r_min covers both constraints: TheoPouLa needs r >= q/2 (Lim &
    Sabanis 2024, arXiv:2105.13937), TUSLA needs r >= q/2 + 1 (Lovas et
    al. 2023, Definition 1).

  * LaTeX table builders for each experiment table in the paper:
    tab:w2_floor, tab:gridsearch_results, the main results table,
    tab:depth_sweep_results, and tab:t_star.
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


def load_all(out_dir):
    """
    Load every .pt in out_dir, routed by shape:
        runs   {name: record}   anything with a "config" key
        floors {name: record}   anything with "mean" and "N" keys
        other  {name: record}   the rest (grid summaries etc.)
    """
    import glob
    runs, floors, other = {}, {}, {}
    for p in sorted(glob.glob(os.path.join(out_dir, "*.pt"))):
        rec = load_results(p)
        name = os.path.splitext(os.path.basename(p))[0]
        if isinstance(rec, dict) and "config" in rec:
            runs[name] = rec
        elif isinstance(rec, dict) and "mean" in rec and "N" in rec:
            floors[name] = rec
        else:
            other[name] = rec
    print(f"loaded {len(runs)} runs, {len(floors)} floors, {len(other)} other")
    return runs, floors, other


# ============================================================
# Architectures
#
# `arch` counts AFFINE MAPS: arch = N means L = N - 1 hidden layers.
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
    """As above, plus a linear path from z_0 straight to the output
    (Remark rem:representability: the affine field -x/v_t has no exact
    tanh representation without large |theta|)."""

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
    """Example 1 approximator: s(t, theta, x) = -x + m_t theta.
    Contains the exact score of N(mu, I) at theta = mu, so theta* is
    KNOWN and eps_AL = E|theta - mu|^2 is directly observable."""

    def __init__(self, d):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(d))

    def forward(self, x, t):
        return -x + torch.exp(-t) * self.theta


def build_net(d, arch, width, use_skip=False):
    """arch is the string "affine" or an integer number of affine maps."""
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

    "gaussian" : N(mu, sigma_0^2 I). Strongly log-concave (Ass. sab23_data).
    "bimodal"  : equally weighted mixture of N(+-mu, sigma_0^2 I).
                 Semiconvex only (Ass. sab25_data), score in closed form
                 via eq:marginal_score_tanh.
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
# Semiconvexity constants and the t* threshold
# (Prop. beta_integral / Remark K_mu_bimodal of the paper)
# ============================================================

def K_bimodal(mu_vec):
    """K = |mu|^2 - 1 for the sigma_0 = 1 symmetric mixture."""
    return float(np.dot(mu_vec, mu_vec) - 1.0)


def mu_of_R(mu_vec, R):
    """mu(R) = 1 - 2|mu|/R, positive for R > 2|mu|."""
    return float(1.0 - 2.0 * np.linalg.norm(mu_vec) / R)


def beta_os(t, K, mu):
    """One-sided Lipschitz rate beta_t^{OS,K,mu} (eq:beta_OS_Kmu)."""
    e = np.exp(-2.0 * t)
    den = mu + (1.0 - mu) * e
    return mu / den - e * (K + mu) / den ** 2


def B_integral(t, K, mu):
    """Closed-form integral of beta_os from 0 to t (eq:beta_integral)."""
    g = mu * (np.exp(2.0 * t) - 1.0) + 1.0
    return 0.5 * (np.log(g) + (K / mu + 1.0) * (1.0 / g - 1.0))


def t_star(K, mu, hi=20.0, tol=1e-10):
    """Threshold t* = inf{t > 0 : B(t) > 0} by bisection."""
    lo = 1e-12
    if B_integral(hi, K, mu) <= 0:
        return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if B_integral(mid, K, mu) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ============================================================
# Training data and loss
# ============================================================

def sample_training_batch(X_data, t_0, T, d, batch_size):
    """One minibatch of training triples xi = (tau, x_0, z).
    Returns tau, x_t, z, sigma_tau (z rather than -z/sigma, so the loss
    can be formed in the bounded way)."""
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
    """Single-sample loss, averaged over the batch and SUMMED over output
    coordinates. Identical for every optimiser.
        "sigma2" : kappa = sigma_t^2, formed as |z + sigma s|^2 (bounded)
        "one"    : kappa = 1
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
    """Fresh forward/backward pass. {parameter_name: ||grad||}, RAW."""
    net.to(device)
    net.train()
    net.zero_grad()
    b = to_device_batch(sample_training_batch(X_data, t_0, T, d, batch_size),
                        device)
    compute_loss(net, b[0], b[1], b[2], b[3], kappa_mode).backward()
    out = {n: (p.grad.norm().item() if p.grad is not None else None)
           for n, p in net.named_parameters()}
    net.zero_grad()
    return out


def grad_ratio(grad_norm_dict):
    """Ratio of largest to smallest layerwise WEIGHT gradient norm
    ('grad. ratio' column of tab:depth_sweep_results)."""
    vals = [v for k, v in grad_norm_dict.items()
            if v is not None and "weight" in k and v > 0]
    if len(vals) < 2:
        return float("nan")
    return float(max(vals) / min(vals))


def score_error_by_t(net, target, t_values, device, n_mc=20000):
    """E_{X~p_t} |s(theta,(X,t)) - grad log p_t(X)|^2 by Monte Carlo
    over exact draws from the marginal. Diagnostic for eps_SN,
    unweighted, resolved in t (eq:score_error)."""
    out = {}
    net.eval()
    with torch.no_grad():
        for t in t_values:
            X = target["sample_pt"](n_mc, float(t))
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            t_t = torch.full((n_mc, 1), float(t), dtype=torch.float32,
                             device=device)
            S = net(x_t, t_t).cpu().numpy()
            out[float(t)] = float(np.mean(
                np.sum((S - target["score"](X, float(t))) ** 2, axis=1)))
    return out


def score_error_aggregate(se_dict):
    """Scalar E-bar: average of E(t) over the t grid."""
    return float(np.mean(list(se_dict.values())))


def mode_balance(samples, mu_vec):
    """Fraction of samples assigned to the +mu mode by the sign of
    <x, mu>. 0.5 is perfect. Only meaningful for the bimodal target."""
    return float(np.mean(samples @ np.asarray(mu_vec) > 0))


# ============================================================
# Training: one loop, four optimisers, in-loop evaluation callback
# ============================================================

VALID_OPTIMISERS = ("adam", "sgld", "tusla", "theopoula")


def train(net, X_data, target, opt_cfg, run_cfg, device,
          grad_checkpoints=(), checkpoints=(), eval_fn=None):
    """
    Train and return a diagnostics dict.

    opt_cfg keys:
        name   one of VALID_OPTIMISERS
        lam    step size / learning rate
        beta   inverse temperature (Langevin arms)
        eta    regularisation strength (TUSLA, TheoPouLa)
        r      regularisation exponent
        eps_b  boosting denominator offset (TheoPouLa)

    checkpoints / eval_fn:
        at each iteration in `checkpoints`, eval_fn(net, it) is called
        (net switched to eval mode and back), and its return value stored
        in diag["checkpoint_evals"][it]. Because this happens INSIDE the
        loop, Adam's moment estimates are never reset.
    """
    net.to(device)
    d = target["d"]
    t_0, T = run_cfg["t_0"], run_cfg["T"]
    B, n_iters = run_cfg["batch_size"], run_cfg["n_iters"]
    kappa_mode = run_cfg["kappa_mode"]
    print_every = run_cfg.get("print_every", 5000)

    name = opt_cfg["name"]
    if name not in VALID_OPTIMISERS:
        raise ValueError(f"unknown optimiser {name!r}")
    lam = float(opt_cfg.get("lam", 1e-3))
    beta = float(opt_cfg.get("beta", 1e10))
    eta = float(opt_cfg.get("eta", 0.0))
    r = float(opt_cfg.get("r", 2.0))
    eps_b = float(opt_cfg.get("eps_b", 0.1))
    sqrt_lam = float(np.sqrt(lam))
    noise_scale = float(np.sqrt(2.0 * lam / beta))

    torch_opt = torch.optim.Adam(net.parameters(), lr=lam) if name == "adam" else None

    loss_history, theta_norm_history = [], []
    grad_norms, update_norms, checkpoint_evals = {}, {}, {}
    diverged, nan_iter = False, None

    net.train()
    for it in range(1, n_iters + 1):
        b = to_device_batch(sample_training_batch(X_data, t_0, T, d, B), device)

        net.zero_grad()
        loss = compute_loss(net, b[0], b[1], b[2], b[3], kappa_mode)
        loss.backward()

        capture = it in grad_checkpoints
        if capture:
            grad_norms[it] = {n: (p.grad.norm().item() if p.grad is not None
                                  else None)
                              for n, p in net.named_parameters()}
            before = {n: p.detach().clone() for n, p in net.named_parameters()}

        if name == "adam":
            torch_opt.step()
        else:
            with torch.no_grad():
                theta_norm_t = torch.sqrt(sum((p ** 2).sum()
                                              for p in net.parameters()))
                theta_2r = theta_norm_t ** (2.0 * r)

                for p in net.parameters():
                    G = p.grad

                    if name == "sgld":
                        step = G

                    elif name == "tusla":
                        step = (G + eta * p * theta_2r) / \
                               (1.0 + sqrt_lam * theta_2r)

                    else:  # theopoula, eq:theopoula_gradient
                        Ga = G.abs()
                        tamed = G / (1.0 + sqrt_lam * Ga)
                        boosted = tamed * (1.0 + sqrt_lam / (eps_b + Ga))
                        reg = eta * p * theta_2r / (1.0 + sqrt_lam * theta_2r)
                        step = boosted + reg

                    p.add_(-lam * step + noise_scale * torch.randn_like(p))

        with torch.no_grad():
            tn = torch.sqrt(sum((p ** 2).sum()
                                for p in net.parameters())).item()
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
            print(f"    [{name}] iter {it}/{n_iters}  loss={l:.4f}  "
                  f"|theta|={tn:.3f}")

        if eval_fn is not None and it in checkpoints:
            net.eval()
            checkpoint_evals[it] = eval_fn(net, it)
            net.train()

    return dict(loss_history=loss_history,
                theta_norm_history=theta_norm_history,
                grad_norms=grad_norms, update_norms=update_norms,
                checkpoint_evals=checkpoint_evals,
                diverged=diverged, nan_iter=nan_iter)


# ============================================================
# Reverse-time sampling
# ============================================================

def em_sample(net, gamma, T, d, N_samples, device, batch=None):
    """Euler-Maruyama on the practical reverse SDE (eq:em_scheme),
    started at N(0, I_d). The scheme realises epsilon = gamma
    (Remark eps_gamma_coupling)."""
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
    """Exact W2 by network simplex (POT). Raises if the solver stops
    before optimality (that failure is silent in POT and inflates the
    reported cost)."""
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


def estimate_w2(net, target, N_w2, n_repeats, gamma, T, d, device,
                num_iter_max, sample_batch=None, eval_seed=None,
                verbose=True):
    """
    Both generated and reference samples redrawn on every repeat.

    eval_seed: when set, repeat i re-seeds with eval_seed + i, so every
    configuration is evaluated under identical Brownian increments and
    identical reference draws (common random numbers). Training noise is
    unaffected because this runs after training.

    Returns mean, std, values, and the mode balance of the LAST
    generated set.
    """
    vals, bal = [], float("nan")
    for i in range(n_repeats):
        if eval_seed is not None:
            set_seed(eval_seed + i)
        gen = em_sample(net, gamma, T, d, N_w2, device, batch=sample_batch)
        ref = target["sample"](N_w2)
        v = w2_exact(gen, ref, num_iter_max)
        vals.append(v)
        bal = mode_balance(gen, target["mu"])
        if verbose:
            print(f"      W2 repeat {i + 1}/{n_repeats}: {v:.4f}"
                  f"   balance {bal:.4f}")
    return float(np.mean(vals)), float(np.std(vals)), vals, bal


def w2_floor(target, N, n_repeats, num_iter_max, eval_seed=None):
    """Finite-sample floor (eq:w2_floor): W2 between two INDEPENDENT
    N-samples of the same measure."""
    vals = []
    for i in range(n_repeats):
        if eval_seed is not None:
            set_seed(eval_seed + i)
        vals.append(w2_exact(target["sample"](N), target["sample"](N),
                             num_iter_max))
    return float(np.mean(vals)), float(np.std(vals)), vals


# ============================================================
# Admissibility of the regularisation exponent
# ============================================================

def q_of(arch):
    """Polynomial Lipschitz exponent, q = 2L + 1 for L hidden layers
    (eq:q_depth); the affine family of Example 1 has affine H, q = 1."""
    if arch == "affine":
        return 1.0
    return 2.0 * (int(arch) - 1) + 1.0


def r_min(arch, which="theopoula"):
    """TheoPouLa: r >= q/2 (Lim & Sabanis 2024).
    TUSLA: r >= q/2 + 1 (Lovas et al. 2023, Definition 1)."""
    q = q_of(arch)
    return q / 2.0 + (1.0 if which == "tusla" else 0.0)


def r_admissible(arch, r, optimiser):
    if optimiser in ("adam", "sgld"):
        return None
    return bool(r >= r_min(arch, which=optimiser))


# ============================================================
# LaTeX table builders (one per paper table)
# ============================================================

def _fmt(v, nd=4):
    if v is None:
        return "---"
    try:
        if not np.isfinite(v):
            return "---"
    except TypeError:
        return str(v)
    return f"{v:.{nd}f}"


def latex_floor_table(floor_by_N, caption=None, label="tab:w2_floor"):
    """floor_by_N: {N: dict(mean=, std=)}. Fits tab:w2_floor with the
    N^{-1/4} prediction column (Fournier & Guillin, d=2 => E[W2] ~ N^{-1/4})."""
    Ns = sorted(floor_by_N)
    ref_N, ref_m = Ns[-1], floor_by_N[Ns[-1]]["mean"]
    caption = caption or (r"Finite-sample floor $\widehat W_2^{\,\mathrm{floor}}(N)$, "
                          r"against the $N^{-1/4}$ prediction of "
                          r"\cite{fournier2015rate} anchored at the largest $N$.")
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{rccc}", r"    \toprule",
             r"    $N$ & $\widehat W_2^{\,\mathrm{floor}}(N)$ & s.d. & "
             r"$N^{-1/4}$ prediction \\", r"    \midrule"]
    for N in Ns:
        pred = ref_m * (ref_N / N) ** 0.25
        d_ = floor_by_N[N]
        lines.append(f"    {N} & {_fmt(d_['mean'])} & {_fmt(d_['std'])} & "
                     f"{_fmt(pred)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_grid_table(grid_rows, caption=None, label="tab:gridsearch_results"):
    """grid_rows: list of dicts with keys optimiser, lam, eps_b, beta, r,
    w2_mean, w2_std. Ranked by w2 within optimiser."""
    caption = caption or r"Grid search outcome, ranked by $\widehat W_2$."
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{lcccccc}", r"    \toprule",
             r"    Optimiser & $\lambda$ & $\varepsilon_b$ & $\beta$ & $r$ & "
             r"$\widehat W_2$ & s.d. \\", r"    \midrule"]
    for row in sorted(grid_rows, key=lambda x: (x["optimiser"],
                                                x.get("w2_mean") or 1e9)):
        eb = _fmt(row.get("eps_b"), 3) if row["optimiser"] == "theopoula" else "---"
        bt = (f"$10^{{{int(np.log10(row['beta']))}}}$"
              if row["optimiser"] != "adam" else "---")
        rr = _fmt(row.get("r"), 1) if row["optimiser"] in ("tusla", "theopoula") else "---"
        lines.append(f"    {row['optimiser']} & {row['lam']:g} & {eb} & {bt} & "
                     f"{rr} & {_fmt(row.get('w2_mean'))} & "
                     f"{_fmt(row.get('w2_std'))} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_main_table(records, floors, caption=None, label="tab:w2_results"):
    """records: {tag: record}. One row per run: example, optimiser, W2,
    sd, E-bar, balance, |theta|, diverged, against the floor."""
    caption = caption or (
        r"Empirical Wasserstein-2 distance at $N=10{,}000$, mean over "
        r"repeats with both samples redrawn, against the finite-sample "
        r"floor at the same $N$. $\bar{\mathcal E}$ is the aggregate "
        r"score error; balance is the fraction of generated mass in the "
        r"$+\bar\mu$ mode.")
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{llcccccc}", r"    \toprule",
             r"    Example & Optimiser & $\widehat W_2$ & s.d. & floor & "
             r"$\bar{\mathcal E}$ & balance & div. \\", r"    \midrule"]
    for tag in sorted(records):
        rec = records[tag]
        c = rec["config"]
        fl = floors.get(c["target"], {}).get("mean", float("nan"))
        bal = rec.get("mode_balance")
        bal_s = _fmt(bal, 3) if c["target"] == "bimodal" else "---"
        lines.append(
            f"    {c.get('example', c['target'])} & {c['optimiser']} & "
            f"{_fmt(rec.get('w2_mean'))} & {_fmt(rec.get('w2_std'))} & "
            f"{_fmt(fl)} & {_fmt(rec.get('score_error_agg'))} & {bal_s} & "
            f"{'yes' if rec.get('diverged') else 'no'} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_depth_table(records, caption=None, label="tab:depth_sweep_results"):
    """records: {tag: record} from the depth stage."""
    caption = caption or (
        r"Depth sweep. $L$ hidden layers gives $q=2L+1$ "
        r"(eq:q\_depth) and $r_{\min}$ per algorithm; `adm.' records "
        r"whether the $r$ used satisfies the binding constraint; grad.\ "
        r"ratio is the largest-to-smallest layerwise gradient norm at "
        r"initialisation.")
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{ccccccccc}", r"    \toprule",
             r"    $L$ & $q$ & $r_{\min}^{\mathrm{Theo}}$ & $r$ used & adm. & "
             r"grad.\ ratio & $\widehat W_2$ & s.d. & $\bar{\mathcal E}$ \\",
             r"    \midrule"]
    def sort_key(t):
        return records[t]["config"]["arch"] if isinstance(
            records[t]["config"]["arch"], int) else 0
    for tag in sorted(records, key=sort_key):
        rec = records[tag]
        c = rec["config"]
        L = c["arch"] - 1 if isinstance(c["arch"], int) else 0
        adm = rec.get("r_admissible")
        lines.append(
            f"    {L} & {q_of(c['arch']):.0f} & "
            f"{r_min(c['arch']):.1f} & {_fmt(rec.get('r_used'), 1)} & "
            f"{'y' if adm else ('n' if adm is not None else '---')} & "
            f"{_fmt(rec.get('grad_ratio'), 1)} & {_fmt(rec.get('w2_mean'))} & "
            f"{_fmt(rec.get('w2_std'))} & "
            f"{_fmt(rec.get('score_error_agg'))} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_tstar_table(mu_vec, R_values, caption=None, label="tab:t_star"):
    """tab:t_star from Remark K_mu_bimodal and Prop. beta_integral."""
    K = K_bimodal(mu_vec)
    caption = caption or (
        r"Constants of the semiconvexity assumption for the bimodal "
        r"target, and the resulting threshold $t^\star$.")
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{ccccc}", r"    \toprule",
             r"    $K$ & $R$ & $\mu(R)$ & $\ln\sqrt{1+K/\mu^2}$ & $t^\star$ \\",
             r"    \midrule"]
    for R in R_values:
        m = mu_of_R(mu_vec, R)
        if m <= 0:
            continue
        lb = float(np.log(np.sqrt(1.0 + K / m ** 2)))
        ts = t_star(K, m)
        lines.append(f"    {K:.2f} & {R:g} & {m:.3f} & {lb:.3f} & "
                     f"{ts:.3f} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)

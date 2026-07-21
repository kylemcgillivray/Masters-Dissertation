"""
functions3.py

for training and sampling from neural-network score
estimators, following the framework in Sabanis (2023, 2025)

this file is just the reusable machinery: networks, data generation, training loops
(Adam and SGLD variants), the reverse-SDE sampler, and the W2 metric.
"""

import os
import numpy as np

import torch
import torch.nn as nn

import ot


# ============================================================
# Device / reproducibility / save-load helpers
# ============================================================

def get_device():
    """Return MPS device if available (Apple Silicon GPU), else CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    """Seed both numpy and torch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_results(results, path):
    """Save a results dict (or any torch-serialisable object) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(results, path)
    print(f"Saved results to {path}")


def load_results(path):
    """Load a results dict previously written by save_results."""
    return torch.load(path, weights_only=False)


# ============================================================
# Score network architectures
# ============================================================
#
# Both networks approximate s(t, theta, x) ~ grad_x log p_t(x), per
# equation \eqref{eq:scorenet_forward} in the dissertation. Input is
# z_0 = (x, t) in R^{D_0}, D_0 = d + 1. Output is in R^{D_2}, D_2 = d.
# Hidden activation is tanh (no ReLU, per supervisor requirement);
# output activation is identity, since the score is unbounded and a
# saturating output activation would cap the representable score values.

class ScoreNet(nn.Module):
    """2-layer (one hidden layer) score estimator s(t, theta, x)."""

    def __init__(self, d, D_1):
        super().__init__()
        D_0 = d + 1   # input dim: d spatial coordinates + 1 time coordinate
        D_2 = d       # output dim: matches data dimension (the score vector)

        self.layer1 = nn.Linear(D_0, D_1)
        self.layer2 = nn.Linear(D_1, D_2)
        self.phi_1 = nn.Tanh()

    def forward(self, x, t):
        """
        x: tensor of shape (batch_size, d)
        t: tensor of shape (batch_size, 1)
        returns: tensor of shape (batch_size, d), the score estimate
        """
        z_0 = torch.cat([x, t], dim=1)   # concatenate spatial input with time, per z_0 = (x, t)
        a_1 = self.layer1(z_0)
        z_1 = self.phi_1(a_1)            # tanh hidden activation
        a_2 = self.layer2(z_1)           # identity output activation (a_2 IS the output)
        return a_2


class ScoreNet3(nn.Module):
    """3-layer (two hidden layer) score estimator s(t, theta, x)."""

    def __init__(self, d, D_1, D_2):
        super().__init__()
        D_0 = d + 1
        D_3 = d       # output dim: matches data dimension

        self.layer1 = nn.Linear(D_0, D_1)
        self.layer2 = nn.Linear(D_1, D_2)
        self.layer3 = nn.Linear(D_2, D_3)

        self.phi_1 = nn.Tanh()
        self.phi_2 = nn.Tanh()

    def forward(self, x, t):
        z_0 = torch.cat([x, t], dim=1)
        a_1 = self.layer1(z_0)
        z_1 = self.phi_1(a_1)
        a_2 = self.layer2(z_1)
        z_2 = self.phi_2(a_2)
        a_3 = self.layer3(z_2)           # identity output activation
        return a_3



class ScoreNetDeep(nn.Module):
    """
    General L-hidden-layer score estimator s(t, theta, x), generalising
    ScoreNet (1 hidden layer) and ScoreNet3 (2 hidden layers) to arbitrary depth.

    hidden_dims: list of hidden layer widths, e.g. [128]*19 gives 19 hidden
    layers + 1 output layer = 20 linear layers total (matching the
    "N-layer" naming convention used for ScoreNet/ScoreNet3, where the
    count includes the output layer).
    """

    def __init__(self, d, hidden_dims):
        super().__init__()
        D_0 = d + 1   # input dim: d spatial + 1 time
        D_out = d     # output dim: matches data dimension

        dims = [D_0] + list(hidden_dims) + [D_out]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.activation = nn.Tanh()

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:      # tanh on every hidden layer
            z = self.activation(layer(z))
        z = self.layers[-1](z)              # identity output activation
        return z
    
    

# ============================================================
# Training data generation
# ============================================================

def sample_bimodal_data(N, d, mu1, mu2, sigma=1.0, weight=0.5):
    """
    Draw N i.i.d. samples from the equally-(or unequally-)weighted
    Gaussian mixture pi_D = weight * N(mu1, sigma^2 I_d)
                          + (1-weight) * N(mu2, sigma^2 I_d).

    Used as the primary bimodal test case (Sabanis 2025, Remark 18).
    """
    component = np.random.uniform(size=N) < weight   # True -> mode 1, False -> mode 2
    X = np.zeros((N, d))
    X[component] = np.random.normal(mu1, sigma, size=(component.sum(), d))
    X[~component] = np.random.normal(mu2, sigma, size=((~component).sum(), d))
    return X


def sample_training_batch(X_data, t_0, T, d, batch_size):
    """
    Draw one minibatch of training triples (tau, x_t, g) for denoising
    score matching, per the tractable objective

        tau  ~ Uniform([t_0, T])
        x_0  ~ empirical distribution of X_data
        z    ~ N(0, I_d)
        x_t  = m_tau * x_0 + sigma_tau * z            (OU transition, eq. \eqref{eq:score_closed_form})
        g    = -z / sigma_tau                          (closed-form training target)

    Returns (tau, x_t, g), each a numpy array of shape (batch_size, ...).
    """

    N = X_data.shape[0]

    tau = np.random.uniform(t_0, T, size=(batch_size, 1))          # (B, 1), broadcasts over d
    idx = np.random.randint(0, N, size=batch_size)
    x0_n = X_data[idx]                                              # (B, d)
    z_n = np.random.normal(0, 1, size=(batch_size, d))              # (B, d)

    m_tau = np.exp(-tau)                                            # m_t = e^{-t}
    sigma_tau = np.sqrt(1 - np.exp(-2 * tau))                       # sigma_t^2 = 1 - e^{-2t}

    x_t = m_tau * x0_n + sigma_tau * z_n                            # OU forward transition
    g = -z_n / sigma_tau                                            # target score: -sigma_t^{-1} z

    return tau, x_t, g


# ============================================================
# Training: Adam-based 
# ============================================================

def train_network(net, X_data, t_0, T, d, n_iters, batch_size, lr, device, print_every=500):
    """
    Train a single 2 or 3 layer NN using denoising score
    matching, using the Adam optimiser.

    This corresponds to minimising the tractable objective U~(theta) in
    \eqref{eq:U_tilde}, using the stochastic gradient H(theta) 

    Per Assumption 1 of Sabanis 2025, Adam is a valid choice of optimiser:
    the theory only requires the output theta_hat to satisfy an L^2
    accuracy bound relative to the population minimiser theta*, and is
    agnostic to which optimiser achieves it.

    Returns (net, loss_history) - loss_history is a plain list of floats,
    one per iteration, for inspecting convergence in the notebook.
    """
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()   # default reduction='mean': averages over batch AND output dims

    loss_history = []
    net.train()
    for it in range(1, n_iters + 1):
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)

        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        loss = loss_fn(net(x_t_t, tau_t), g_t)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if it % print_every == 0:
            print(f"[Adam] iter {it}/{n_iters}, loss = {loss.item():.4f}")

    return net, loss_history


# ============================================================
# Training: SGLD-based computed via backprop through the network
# rather than in closed form)
# ============================================================

def train_network_sgld(net, X_data, t_0, T, d, n_iters, batch_size,
                        lam, beta, device, kappa_fn=None, print_every=500):
    """
    Train net via Stochastic Gradient Langevin Dynamics:

    theta_{k+1} = theta_k - lam * H(theta_k) + sqrt(2*lam/beta) * xi_k,
    xi_k ~ N(0, I_{d_theta})

    matching Sabanis (2023) eq. (15) / theta_nplusone_func in the toy
    unknown-mean example, but with H(theta) now the gradient of the
    neural network loss (computed via backprop), not the closed-form
    Gaussian-mean gradient.


    kappa_fn: optional callable tau -> weight, applied to the loss per
    Sabanis's kappa(t) weighting in the objective. If None, kappa=1
    (current default behaviour). Sabanis 2023's own numerical experiment
    uses kappa(t) = sigma_t^2 = 1 - exp(-2t), which cancels the 1/sigma_tau
    blow-up in the target g = -z/sigma_tau as tau -> 0 - pass
    kappa_fn=lambda tau: 1 - np.exp(-2 * tau) to reproduce that choice.

        Returns (net, loss_history).

    """
    net.to(device)
    loss_history = []

    net.train()
    for it in range(1, n_iters + 1):
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)

        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)

        pred = net(x_t_t, tau_t)
        per_sample_sq_error = ((pred - g_t) ** 2).sum(dim=1)

        if kappa_fn is not None:
            kappa_t = torch.tensor(kappa_fn(tau_b), dtype=torch.float32, device=device).squeeze(-1)
            per_sample_sq_error = kappa_t * per_sample_sq_error

        loss = per_sample_sq_error.mean()

        net.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in net.parameters():
                noise = torch.randn_like(p) * np.sqrt(2 * lam / beta)
                p.add_(-lam * p.grad + noise)

        loss_history.append(loss.item())

        if it % print_every == 0:
            print(f"[SGLD] iter {it}/{n_iters}, loss = {loss.item():.4f}")

    return net, loss_history

# ============================================================
# Sampling from a trained network (reverse SDE, Euler-Maruyama)
# ============================================================

def euler_maruyama_sample_nn_batch(net, gamma, T, d, N_samples, device="cpu"):
    """
    Simulate the practical reverse-time SDE

        dY_t = (Y_t + 2 s_theta(Y_t, T-t)) dt + sqrt(2) dB_bar_t,   Y_0 ~ N(0, I_d)

    via Euler-Maruyama with fixed step size gamma, for N_samples
    trajectories simultaneously (one network forward pass per timestep,
    not per sample - this is what makes it fast).

    Returns final samples only, shape (N_samples, d).
    """
    K_plus_1 = int(round(T / gamma))

    net.eval()
    Y_k = np.random.normal(0, 1, size=(N_samples, d))   # Y_0 ~ N(0, I_d), the invariant measure

    with torch.no_grad():
        for k in range(K_plus_1):
            t_k = T - (k * gamma)   # current time argument to s_theta is T - t, per the reverse SDE

            Y_k_t = torch.tensor(Y_k, dtype=torch.float32, device=device)
            t_k_t = torch.full((N_samples, 1), t_k, dtype=torch.float32, device=device)

            s = net(Y_k_t, t_k_t).cpu().numpy()        # (N_samples, d)

            increment = gamma * (Y_k + 2 * s)
            noise = np.sqrt(2 * gamma) * np.random.normal(0, 1, size=(N_samples, d))

            Y_k = Y_k + increment + noise

    return Y_k


# ============================================================
# Wasserstein-2 distance
# ============================================================

def wasserstein2_distance(samples_a, samples_b):
    """
    Compute the empirical W_2 distance between two equally-sized sample
    sets via exact discrete optimal transport (ot.emd2), with uniform
    weights on each set of points.

    Note: this is a biased estimator of the true W_2 at finite N - see
    the noise-floor discussion in the dissertation (comparing two
    independent samples of pi_D gives a nonzero W_2 due to finite-sample
    bias, which only vanishes as N -> infinity).
    """
    N = samples_a.shape[0]
    assert samples_b.shape[0] == N, "samples_a and samples_b must have equal size"

    a = np.ones(N) / N
    b = np.ones(N) / N
    M = ot.dist(samples_a, samples_b, metric='sqeuclidean')
    return np.sqrt(ot.emd2(a, b, M, numItermax=1_000_000))


def estimate_W2(net, samples_true, N_w2, n_repeats, gamma, T, d, device):
    """
    Estimate W_2(generated samples, samples_true), averaged over
    n_repeats independent sampling runs to reduce Monte Carlo variance
    in the estimate (each run draws a fresh batch via Euler-Maruyama).

    Returns (mean, std) of the n_repeats W_2 estimates.
    """
    W2_vals = []
    net.eval()
    for _ in range(n_repeats):
        samples = euler_maruyama_sample_nn_batch(
            net, gamma=gamma, T=T, d=d, N_samples=N_w2, device=device
        )
        W2_vals.append(wasserstein2_distance(samples, samples_true))
    return np.mean(W2_vals), np.std(W2_vals)


def W2_self_floor(X_data, N, n_repeats=5):
    """
    Estimate the W_2 noise floor at sample size N: the W_2 distance
    between two INDEPENDENT samples both drawn from pi_D (via X_data).
    Not zero at finite N - this is the finite-sample estimator bias,
    and serves as a baseline below which improvements in a trained
    network cannot be distinguished from sampling noise.
    """
    W2_vals = []
    for _ in range(n_repeats):
        idx = np.random.permutation(len(X_data))
        s1 = X_data[idx[:N]]
        s2 = X_data[idx[N:2 * N]]
        W2_vals.append(wasserstein2_distance(s1, s2))
    return np.mean(W2_vals), np.std(W2_vals)


# ============================================================
# Sweep machinery: train 2-layer and 3-layer nets together,
# tracking W2 at specified checkpoints (used by train2.py / CLI sweeps)
# ============================================================

def run_W2_experiment(X_data, checkpoints, d, device,
                       D_1=128, D_2=128, batch_size=512, lr=1e-3,
                       N_w2=2000, n_repeats=5, gamma=0.001, T=1, t_0=1e-3,
                       save_path=None):
    """
    Train a 2-layer and 3-layer ScoreNet continuously on X_data (via Adam),
    evaluating W_2(generated samples, X_data[:N_w2]) at each iteration
    count in `checkpoints`.

    If save_path is given, results are written to disk after every
    checkpoint, so a crash mid-run does not lose completed checkpoints.

    Returns a results dict containing checkpoint values, W_2 mean/std
    curves for both architectures, final network state_dicts, and config.
    """
    samples_true = X_data[:N_w2]

    net = ScoreNet(d=d, D_1=D_1).to(device)
    net3 = ScoreNet3(d=d, D_1=D_1, D_2=D_2).to(device)

    optimizer_2 = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer_3 = torch.optim.Adam(net3.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    W2_2layer_curve, W2_2layer_std = [], []
    W2_3layer_curve, W2_3layer_std = [], []

    max_iters = checkpoints[-1]
    checkpoint_set = set(checkpoints)

    net.train()
    net3.train()

    for it in range(1, max_iters + 1):

        # 2-layer update
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)
        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)
        optimizer_2.zero_grad()
        loss_fn(net(x_t_t, tau_t), g_t).backward()
        optimizer_2.step()

        # 3-layer update (independent minibatch draw)
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)
        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)
        optimizer_3.zero_grad()
        loss_fn(net3(x_t_t, tau_t), g_t).backward()
        optimizer_3.step()

        if it in checkpoint_set:
            print(f"Checkpoint {it}...")

            w2_mean, w2_std = estimate_W2(net, samples_true, N_w2=N_w2,
                                           n_repeats=n_repeats, gamma=gamma, T=T,
                                           d=d, device=device)
            W2_2layer_curve.append(w2_mean)
            W2_2layer_std.append(w2_std)
            net.train()

            w2_mean3, w2_std3 = estimate_W2(net3, samples_true, N_w2=N_w2,
                                             n_repeats=n_repeats, gamma=gamma, T=T,
                                             d=d, device=device)
            W2_3layer_curve.append(w2_mean3)
            W2_3layer_std.append(w2_std3)
            net3.train()

            print(f"  W2 2-layer: {w2_mean:.4f} +/- {w2_std:.4f}"
                  f" | W2 3-layer: {w2_mean3:.4f} +/- {w2_std3:.4f}")

            results = {
                "checkpoints": [c for c in checkpoints if c <= it],
                "W2_2layer_curve": W2_2layer_curve,
                "W2_2layer_std": W2_2layer_std,
                "W2_3layer_curve": W2_3layer_curve,
                "W2_3layer_std": W2_3layer_std,
                "net_state": net.state_dict(),
                "net3_state": net3.state_dict(),
                "config": dict(D_1=D_1, D_2=D_2, batch_size=batch_size, lr=lr,
                               N_w2=N_w2, n_repeats=n_repeats, gamma=gamma,
                               T=T, t_0=t_0),
            }
            if save_path is not None:
                save_results(results, save_path)

    return results



########################################################################################################################
############################################## TUSLA. #########################################################
########################################################################################################################

def train_network_tusla(net, X_data, t_0, T, d, n_iters, batch_size,
                         lam, beta, eta, r, device, kappa_fn=None, print_every=500):
    """
    Train via TUSLA (Lovas, Lytras, Rasonyi, Sabanis 2020), designed specifically
    for the case where the network's stochastic gradient H(theta,x) is only
    LOCALLY Lipschitz in theta, with a "constant" growing polynomially in
    ||theta|| - exactly the gap flagged relative to Assumption 3.a, which
    requires a theta-independent Lipschitz constant.

    Two ingredients added on top of plain SGLD:
      1. Regularisation: add eta * theta * |theta|^(2r) to the raw gradient,
         creating a restoring force for large ||theta||.
      2. Taming: divide the regularised gradient by (1 + sqrt(lam)*|theta|^(2r)),
         where |theta| is the norm of the FULL flattened parameter vector
         (not per-tensor). This caps the effective step size regardless of
         how large the raw gradient becomes, preventing destabilising
         overshoot events - directly targeting the spikes seen with plain SGLD.

    r should satisfy r >= q/2 + 1 where q-1 = 2n+1 for an n-hidden-layer
    network with a bounded, Lipschitz-derivative activation (tanh qualifies).
    For a 1-hidden-layer ScoreNet, this gives q=4, so r=3 is a reasonable
    starting point - still treat r, eta, lam as tunable hyperparameters.

    Returns (net, loss_history, param_norm_history) - the parameter norm
    trace lets you directly check whether taming is keeping ||theta||
    bounded, the mechanism TUSLA relies on for stability.
    """
    net.to(device)
    loss_history = []
    param_norm_history = []

    net.train()
    for it in range(1, n_iters + 1):
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)

        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)

        pred = net(x_t_t, tau_t)
        per_sample_sq_error = ((pred - g_t) ** 2).sum(dim=1)

        if kappa_fn is not None:
            kappa_t = torch.tensor(kappa_fn(tau_b), dtype=torch.float32, device=device).squeeze(-1)
            per_sample_sq_error = kappa_t * per_sample_sq_error

        loss = per_sample_sq_error.mean()

        net.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Full parameter-vector norm across ALL tensors - taming is defined
            # relative to the FULL theta, not per-tensor.
            theta_norm_sq = sum((p ** 2).sum() for p in net.parameters())
            theta_norm = torch.sqrt(theta_norm_sq)

            taming_denom = 1 + np.sqrt(lam) * (theta_norm ** (2 * r))

            for p in net.parameters():
                # H(theta,x) = G(theta,x) + eta * theta * |theta|^(2r)
                regularised_grad = p.grad + eta * p * (theta_norm ** (2 * r))
                # H_lambda(theta,x) = H(theta,x) / (1 + sqrt(lambda)|theta|^(2r))
                tamed_grad = regularised_grad / taming_denom

                noise = torch.randn_like(p) * np.sqrt(2 * lam / beta)
                p.add_(-lam * tamed_grad + noise)

            param_norm_history.append(theta_norm.item())

        loss_history.append(loss.item())

        if it % print_every == 0:
            print(f"[TUSLA] iter {it}/{n_iters}, loss = {loss.item():.4f}, "
                  f"||theta|| = {theta_norm.item():.4f}")

    return net, loss_history, param_norm_history
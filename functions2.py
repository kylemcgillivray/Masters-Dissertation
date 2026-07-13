import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import ot


# ------------------------------------------------------------
# Device / reproducibility helpers
# ------------------------------------------------------------

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(results, path)
    print(f"Saved results to {path}")

def load_results(path):
    return torch.load(path)


# ------------------------------------------------------------
# SGLD (motivating example)
# ------------------------------------------------------------

def H_func(theta, X):
    '''
    Computes the stochastic gradient. takes in current theta and unpacks X.
    equation (14)
    '''
    tau, X_0, z = X

    m_tau = np.exp(-tau)
    sigma2_tau = 1 - np.exp(-2 * tau)
    sigma_tau = np.sqrt(sigma2_tau)

    return 2 * sigma2_tau * m_tau * ((1/sigma_tau) * z - m_tau * X_0 - sigma_tau * z + m_tau * theta)


def theta_nplusone_func(lambda_, beta, theta_n, X):
    '''
    Computes the new theta value using the recurrence relation given by (15)
    '''
    return theta_n - lambda_ * H_func(theta=theta_n, X=X) + (np.sqrt(2 * lambda_ / beta) * np.random.normal(0, 1, len(theta_n)))


def thetahat_func(X_data, beta, lambda_, theta_0, N_sgld, t_0, T):
    '''
    SGLD for theta_hat which is a d dimensional vector approximating the means.
    '''
    theta = theta_0
    theta_tracker = [theta_0]

    N = X_data.shape[0]
    d = X_data.shape[1]

    for i in range(N_sgld):
        tau = np.random.uniform(t_0, T)
        x0_n = X_data[np.random.randint(N)]
        z_n = np.random.normal(0, 1, d)
        X_n = (tau, x0_n, z_n)

        theta = theta_nplusone_func(lambda_=lambda_, beta=beta, theta_n=theta, X=X_n)
        theta_tracker.append(theta)

    return theta, theta_tracker


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------

class ScoreNet(nn.Module):
    '''2 layer (one hidden) neural network approximating the score s(t, theta, x).'''

    def __init__(self, d, D_1):
        super().__init__()
        D_0 = d + 1
        D_2 = d

        self.layer1 = nn.Linear(D_0, D_1)
        self.layer2 = nn.Linear(D_1, D_2)
        self.phi_1 = nn.Tanh()

    def forward(self, x, t):
        z_0 = torch.cat([x, t], dim=1)
        a_1 = self.layer1(z_0)
        z_1 = self.phi_1(a_1)
        a_2 = self.layer2(z_1)
        z_2 = a_2
        return z_2


class ScoreNet3(nn.Module):
    '''3 layer (two hidden) neural network approximating the score s(t, theta, x).'''

    def __init__(self, d, D_1, D_2):
        super().__init__()
        D_0 = d + 1
        D_3 = d

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
        a_3 = self.layer3(z_2)
        z_3 = a_3
        return z_3


# ------------------------------------------------------------
# Analytic-score reverse SDE (motivating example)
# ------------------------------------------------------------

def euler_maruyama_approximation(Y_k, theta_hat, k, gamma, T):
    d = Y_k.shape[0]
    s = -Y_k + np.exp(-(T - (k * gamma))) * theta_hat
    increment = gamma * (Y_k + 2 * s)
    noise = np.sqrt(2 * gamma) * np.random.normal(0, 1, d)
    k += 1
    return Y_k + increment + noise, k


def euler_maruyama_sample(theta_hat, gamma, T, d):
    K_plus_1 = int(round(T / gamma))
    Y_k = np.random.normal(0, 1, d)
    trajectory = [Y_k]

    for k in range(K_plus_1):
        Y_k, _ = euler_maruyama_approximation(Y_k, theta_hat, k, gamma, T)
        trajectory.append(Y_k)

    return Y_k, trajectory


# ------------------------------------------------------------
# Training data + batch sampling
# ------------------------------------------------------------

def sample_training_batch(X_data, t_0, T, d, batch_size):
    N = X_data.shape[0]

    tau = np.random.uniform(t_0, T, size=(batch_size, 1))
    idx = np.random.randint(0, N, size=batch_size)
    x0_n = X_data[idx]
    z_n = np.random.normal(0, 1, size=(batch_size, d))

    m_tau = np.exp(-tau)
    sigma_tau = np.sqrt(1 - np.exp(-2 * tau))

    x_t = m_tau * x0_n + sigma_tau * z_n
    g = -z_n / sigma_tau

    return tau, x_t, g


def sample_bimodal_data(N, d, mu1, mu2, sigma=1.0, weight=0.5):
    component = np.random.uniform(size=N) < weight
    X = np.zeros((N, d))
    X[component] = np.random.normal(mu1, sigma, size=(component.sum(), d))
    X[~component] = np.random.normal(mu2, sigma, size=((~component).sum(), d))
    return X


# ------------------------------------------------------------
# Training loop (single network)
# ------------------------------------------------------------

def train_score_net(net, X_data, t_0, T, d, n_iters=5000, batch_size=256,
                     lr=1e-3, print_every=500, device="cpu"):

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    net.train()
    for it in range(n_iters):
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)

        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)

        s_pred = net(x_t_t, tau_t)
        loss = loss_fn(s_pred, g_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % print_every == 0:
            print(f"iter {it}, loss = {loss.item():.4f}")

    return net


def euler_maruyama_sample_nn_batch(net, gamma, T, d, N_samples, device="cpu"):
    K_plus_1 = int(round(T / gamma))

    net.eval()
    Y_k = np.random.normal(0, 1, size=(N_samples, d))

    with torch.no_grad():
        for k in range(K_plus_1):
            t_k = T - (k * gamma)

            Y_k_t = torch.tensor(Y_k, dtype=torch.float32, device=device)
            t_k_t = torch.full((N_samples, 1), t_k, dtype=torch.float32, device=device)

            s = net(Y_k_t, t_k_t).cpu().numpy()

            increment = gamma * (Y_k + 2 * s)
            noise = np.sqrt(2 * gamma) * np.random.normal(0, 1, size=(N_samples, d))

            Y_k = Y_k + increment + noise

    return Y_k


# ------------------------------------------------------------
# W2 estimation
# ------------------------------------------------------------

def estimate_W2(net, samples_true, N_w2, n_repeats, gamma, T, d, device):
    a = np.ones(N_w2) / N_w2
    b = np.ones(N_w2) / N_w2
    W2_vals = []
    net.eval()
    for _ in range(n_repeats):
        samples = euler_maruyama_sample_nn_batch(
            net, gamma=gamma, T=T, d=d, N_samples=N_w2, device=device
        )
        M = ot.dist(samples, samples_true, metric='sqeuclidean')
        W2_vals.append(np.sqrt(ot.emd2(a, b, M, numItermax=1000000)))
    return np.mean(W2_vals), np.std(W2_vals)


def W2_self_floor(X_data, N, n_repeats=5):
    W2_vals = []
    a = np.ones(N) / N
    b = np.ones(N) / N
    for _ in range(n_repeats):
        idx = np.random.permutation(len(X_data))
        s1 = X_data[idx[:N]]
        s2 = X_data[idx[N:2 * N]]
        M = ot.dist(s1, s2, metric='sqeuclidean')
        W2_vals.append(np.sqrt(ot.emd2(a, b, M, numItermax=10_000_000)))
    return np.mean(W2_vals), np.std(W2_vals)


# ------------------------------------------------------------
# Main experiment: train 2-layer and 3-layer nets, track W2 at checkpoints
# ------------------------------------------------------------

def run_W2_experiment(X_data, checkpoints, d, device,
                       D_1=128, D_2=128, batch_size=512, lr=1e-3,
                       N_w2=2000, n_repeats=5, gamma=0.001, T=1, t_0=1e-3,
                       save_path=None):
    """
    Train a 2-layer and 3-layer ScoreNet continuously on X_data, evaluating
    W_2(generated samples, X_data[:N_w2]) at each iteration count in `checkpoints`.

    If save_path is given, results are written to disk after every checkpoint,
    so a crash mid-run does not lose completed checkpoints.
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

        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)
        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_b, dtype=torch.float32, device=device)
        optimizer_2.zero_grad()
        loss_fn(net(x_t_t, tau_t), g_t).backward()
        optimizer_2.step()

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

            print(f"  W2 2-layer: {w2_mean:.4f} \u00b1 {w2_std:.4f}"
                  f" | W2 3-layer: {w2_mean3:.4f} \u00b1 {w2_std3:.4f}")

            results = {
                "checkpoints_done": [c for c in checkpoints if c <= it],
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

    return {
        "checkpoints": checkpoints,
        "W2_2layer_curve": W2_2layer_curve,
        "W2_2layer_std": W2_2layer_std,
        "W2_3layer_curve": W2_3layer_curve,
        "W2_3layer_std": W2_3layer_std,
        "net": net,
        "net3": net3,
    }


def plot_W2_curve(results, title_suffix=""):
    """Plot the W_2 curves from run_W2_experiment with error bars."""
    plt.figure(figsize=(8, 5))
    plt.errorbar(results["checkpoints"], results["W2_2layer_curve"],
                 yerr=results["W2_2layer_std"],
                 marker='o', label='2-layer ScoreNet', capsize=4)
    plt.errorbar(results["checkpoints"], results["W2_3layer_curve"],
                 yerr=results["W2_3layer_std"],
                 marker='o', label='3-layer ScoreNet', capsize=4)
    plt.xlabel('Training iterations')
    plt.ylabel(r'$\hat{W}_2$')
    plt.title(r'Empirical $\hat{W}_2$ vs training iterations ' + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.show()
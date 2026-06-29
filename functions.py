
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import ot





# H(theta, x) expands to H(theta, t, X_0, z). z is of dimension d. 
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


def theta_nplusone_func(lambda_, beta,  theta_n, X):

    '''
    Computes the new theta value using the recurrence relation given byt (15)
    '''


    return theta_n - lambda_ * H_func(theta = theta_n, X= X) + (np.sqrt(2* lambda_ / beta) * np.random.normal(0, 1, len(theta_n)))



def thetahat_func(X_data,  beta, lambda_, theta_0, N_sgld, t_0, T):

    ''' 
    SGLD for theta_hat which is a d dimensional vector approximating the means . Initialise theta. 
    '''
    # initialise theta
    theta = theta_0
    theta_tracker = [theta_0]

    N = X_data.shape[0]
    d = X_data.shape[1]
    
    for i in range(N_sgld):
    # draw fresh sample
        tau = np.random.uniform(t_0, T)
        x0_n = X_data[np.random.randint(N)]

        z_n = np.random.normal(0, 1, d)
        X_n = (tau, x0_n, z_n)

        # update theta
        theta = theta_nplusone_func(lambda_ = lambda_, beta = beta, theta_n = theta, X = X_n)

        theta_tracker.append(theta)



    return theta, theta_tracker








class ScoreNet(nn.Module):
    '''2 layer (one hidden) neural network approximating the score
    s(t, theta, x).

    Inputs: x concatenated with t -> dimension D_0 = d + 1
    Hidden layer: D_1 units, with activation function phi_1.
    Output : dimension D_2 = d ( the score vector)
    '''

    def __init__(self, d, D_1):
        super().__init__()

        # dimension of d with t 
        D_0 = d + 1
        # output score matches input
        D_2 = d

        # In and out features 
        self.layer1 = nn.Linear(D_0, D_1)

        self.layer2 = nn.Linear(D_1, D_2)

        # chose the tanh activation function.
        self.phi_1 = nn.Tanh()
    

    def forward(self, x, t):
        '''
        x: tensor of shape (batch_size, d)
        t: tensor of shape (batch_size, 1)
        returns: tensor of shape (batch_size, d) which is the score estimate
        '''

        # Join together x and t as the input tensor
        z_0 = torch.cat([x, t], dim = 1)

        
        a_1 = self.layer1(z_0)          # compute the fist later of the neural net

        z_1 = self.phi_1(a_1)           # apply the activation function

        a_2 = self.layer2(z_1)          # compute the second layer of the neural net

        # Maybe I need a linear activation function here for the output?
        z_2 = a_2

        return z_2





class ScoreNet3(nn.Module):
    '''3 layer (two hidden) neural network approximating the score
    s(t, theta, x).

    Inputs: x concatenated with t -> dimension D_0 = d + 1
    Hidden layer 1: D_1 units, tanh activation
    Hidden layer 2: D_2 units, tanh activation  
    Output: dimension D_3 = d (the score vector), identity activation
    '''

    def __init__(self, d, D_1, D_2):
        super().__init__()

        D_0 = d + 1   # input dimension: d spatial + 1 time
        D_3 = d       # output dimension matches data dimension

        self.layer1 = nn.Linear(D_0, D_1)
        self.layer2 = nn.Linear(D_1, D_2)
        self.layer3 = nn.Linear(D_2, D_3)

        # phi_1 = phi_2 = tanh, phi_3 = identity
        self.phi_1 = nn.Tanh()
        self.phi_2 = nn.Tanh()

    def forward(self, x, t):
        '''
        x: tensor of shape (batch_size, d)
        t: tensor of shape (batch_size, 1)
        returns: tensor of shape (batch_size, d), the score estimate
        '''

        # Layer 0: input
        z_0 = torch.cat([x, t], dim=1)          # (batch_size, D_0)

        # Layer 1: pre-activation and activation
        a_1 = self.layer1(z_0)                   # (batch_size, D_1)
        z_1 = self.phi_1(a_1)                    # (batch_size, D_1)

        # Layer 2: pre-activation and activation
        a_2 = self.layer2(z_1)                   # (batch_size, D_2)
        z_2 = self.phi_2(a_2)                    # (batch_size, D_2)

        # Layer 3: output, phi_3 = id
        a_3 = self.layer3(z_2)                   # (batch_size, D_3)
        z_3 = a_3                                # identity output activation

        return z_3
    



# Functions from the understanding notebook.


def euler_maruyama_approximation(Y_k, theta_hat, k, gamma, T):
    '''
    For constant gamma ,t_k = k*gamma.

    Then T - t_k = T - (k* gamma)



    '''

    d = Y_k.shape[0]

    s = -Y_k + np.exp( - (T - (k * gamma))) * theta_hat
    increment = gamma * (Y_k  + 2 * s)
    noise = np.sqrt(2 * gamma) * np.random.normal(0, 1, d)

    k += 1 

    return Y_k + increment + noise, k


def euler_maruyama_sample(theta_hat, gamma, T, d):
    K_plus_1 = int(round(T/gamma))

    Y_k = np.random.normal(0, 1, d)
    trajectory = [Y_k]

    for k in range(K_plus_1):
        Y_k, _ = euler_maruyama_approximation(Y_k, theta_hat, k, gamma, T)
        trajectory.append(Y_k)

    return Y_k, trajectory




# Functions from the understanding notebook. These train a network


def sample_training_batch(X_data, t_0, T, d, batch_size):
    '''
    Vectorized version: generates a whole batch of (tau, x_t, g) at once,
    instead of looping sample_training_point batch_size times.
    '''
    N = X_data.shape[0]

    tau = np.random.uniform(t_0, T, size=(batch_size, 1))         # (B, 1)
    idx = np.random.randint(0, N, size=batch_size)
    x0_n = X_data[idx]                                             # (B, d)
    z_n = np.random.normal(0, 1, size=(batch_size, d))             # (B, d)

    m_tau = np.exp(-tau)                                           # (B, 1), broadcasts
    sigma_tau = np.sqrt(1 - np.exp(-2 * tau))                       # (B, 1)

    x_t = m_tau * x0_n + sigma_tau * z_n                            # (B, d)
    g = -z_n / sigma_tau                                            # (B, d)

    return tau, x_t, g


#  loss + training loop

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
        g_t   = torch.tensor(g_b,   dtype=torch.float32, device=device)

        s_pred = net(x_t_t, tau_t)
        loss = loss_fn(s_pred, g_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % print_every == 0:
            print(f"iter {it}, loss = {loss.item():.4f}")

    return net


# ---------------------------------------------------------------
# Step 5: vectorized Euler-Maruyama sampler (all samples at once)
# ---------------------------------------------------------------

def euler_maruyama_sample_nn_batch(net, gamma, T, d, N_samples, device="cpu"):
    '''
    Runs N_samples reverse-time trajectories simultaneously.
    Y_k has shape (N_samples, d) instead of (d,) -- this is what
    makes it fast: one network call per timestep, not per sample.
    '''
    K_plus_1 = int(round(T / gamma))

    net.eval()
    Y_k = np.random.normal(0, 1, size=(N_samples, d))

    with torch.no_grad():
        for k in range(K_plus_1):
            t_k = T - (k * gamma)

            Y_k_t = torch.tensor(Y_k, dtype=torch.float32, device=device)
            t_k_t = torch.full((N_samples, 1), t_k, dtype=torch.float32, device=device)

            s = net(Y_k_t, t_k_t).cpu().numpy()        # (N_samples, d)

            increment = gamma * (Y_k + 2 * s)
            noise = np.sqrt(2 * gamma) * np.random.normal(0, 1, size=(N_samples, d))

            Y_k = Y_k + increment + noise

    return Y_k   # shape (N_samples, d), final samples only




def sample_bimodal_data(N, d, mu1, mu2, sigma=1.0, weight=0.5):
    '''
    Mixture of two Gaussians: weight*N(mu1, sigma^2 I) + (1-weight)*N(mu2, sigma^2 I)
    '''
    component = np.random.uniform(size=N) < weight   # True -> mode 1, False -> mode 2
    X = np.zeros((N, d))
    X[component] = np.random.normal(mu1, sigma, size=(component.sum(), d))
    X[~component] = np.random.normal(mu2, sigma, size=((~component).sum(), d))
    return X








# Running experiments


def estimate_W2(net, samples_true, N_w2, n_repeats, gamma, T, d, device):
    """
    Estimate the Wasserstein-2 distance between samples generated by `net` and
    samples_true, averaged over n_repeats sampling runs to reduce Monte Carlo
    variance.

    Returns (mean, std) of the n_repeats W_2 estimates.
    """
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


def run_W2_experiment(X_data, checkpoints, d, device,
                       D_1=128, D_2=128, batch_size=512, lr=1e-3,
                       N_w2=2000, n_repeats=5, gamma=0.001, T=1, t_0=1e-3):
    """
    Train a 2-layer and 3-layer ScoreNet continuously on X_data, evaluating
    W_2(generated samples, X_data[:N_w2]) at each iteration count in `checkpoints`.

    Returns a dict containing the checkpoint values, W_2 mean and std curves
    for both architectures, and the final trained networks.
    """
    samples_true = X_data[:N_w2]

    net  = ScoreNet(d=d, D_1=D_1).to(device)
    net3 = ScoreNet3(d=d, D_1=D_1, D_2=D_2).to(device)

    optimizer_2 = torch.optim.Adam(net.parameters(),  lr=lr)
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
        g_t   = torch.tensor(g_b,   dtype=torch.float32, device=device)
        optimizer_2.zero_grad()
        loss_fn(net(x_t_t, tau_t), g_t).backward()
        optimizer_2.step()

        # 3-layer update
        tau_b, x_t_b, g_b = sample_training_batch(X_data, t_0, T, d, batch_size)
        tau_t = torch.tensor(tau_b, dtype=torch.float32, device=device)
        x_t_t = torch.tensor(x_t_b, dtype=torch.float32, device=device)
        g_t   = torch.tensor(g_b,   dtype=torch.float32, device=device)
        optimizer_3.zero_grad()
        loss_fn(net3(x_t_t, tau_t), g_t).backward()
        optimizer_3.step()

        if it in checkpoint_set:
            print(f"Checkpoint {it}...")

            w2_mean, w2_std = estimate_W2(net,  samples_true, N_w2=N_w2,
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

            print(f"  W2 2-layer: {w2_mean:.4f} ± {w2_std:.4f}"
                  f" | W2 3-layer: {w2_mean3:.4f} ± {w2_std3:.4f}")

    return {
        "checkpoints":     checkpoints,
        "W2_2layer_curve": W2_2layer_curve,
        "W2_2layer_std":   W2_2layer_std,
        "W2_3layer_curve": W2_3layer_curve,
        "W2_3layer_std":   W2_3layer_std,
        "net":             net,
        "net3":            net3,
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
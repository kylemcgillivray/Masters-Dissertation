# sample_gridsearch3.py — run after gridsearch3.py, once you've picked winners
import functions3 as fct
import numpy as np

fct.set_seed(1)
device = fct.get_device()
results = fct.load_results("outputs/gridsearch3_results.pt")

d, T = 2, 2
N_SAMPLES = 10_000

# Edit to whichever tags looked best in the gridsearch3.py summary printout
tags_to_sample = [
    "theopoula_lam0.1_eps0.1_beta1e+08_r0",
]

for tag in tags_to_sample:
    net = fct.ScoreNetDeep(d=d, hidden_dims=[128] * 4).to(device)
    net.load_state_dict(results[tag]["net_state"])
    samples = fct.euler_maruyama_sample_nn_batch(
        net, gamma=0.001, T=T, d=d, N_samples=N_SAMPLES, device=device
    )
    results[tag]["samples"] = samples
    print(f"{tag}: sampled {N_SAMPLES}")

fct.save_results(results, "outputs/gridsearch3_results.pt")
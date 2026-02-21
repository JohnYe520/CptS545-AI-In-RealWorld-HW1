import torch
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
torch.manual_seed(0)

# Load UCI Airfoil Self-Noise dataset
def load_airfoil():
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00291/airfoil_self_noise.dat"
    )
    local_file = "airfoil_self_noise.dat"

    if not os.path.exists(local_file):
        print("Downloading Airfoil dataset from UCI...")
        urllib.request.urlretrieve(url, local_file)

    df = pd.read_csv(local_file, sep=r"\s+", header=None)

    X_np = df.iloc[:, :5].values.astype(np.float64)
    y_np = df.iloc[:,  5].values.astype(np.float64)

    X_min, X_max = X_np.min(axis=0), X_np.max(axis=0)
    X_np = (X_np - X_min) / (X_max - X_min + 1e-8)

    X = torch.tensor(X_np, device=device, dtype=dtype)
    Y_min = torch.tensor(y_np, device=device, dtype=dtype).unsqueeze(-1)

    return X, Y_min

print("Loading UCI Airfoil Self-Noise dataset ...")
X_pool, Y_pool = load_airfoil()
N = X_pool.shape[0]
print(f"  {N} rows x {X_pool.shape[1]} features  |  "
      f"dB range [{Y_pool.min():.1f}, {Y_pool.max():.1f}]")

# Return noisy objective to MINIMIZE
def observe(indices):
    return Y_pool[indices]

# Random baseline
def run_random(n_iters=100, n_init=10):
    idx  = torch.randperm(N, device=device)[:n_iters + n_init].tolist()
    Yobs = observe(idx[:n_init])
    best_track = []
    for i in range(n_iters):
        Yobs = torch.cat([Yobs, observe([idx[n_init + i]])], dim=0)
        best_track.append(Yobs.min().item())
    return best_track

# Bayesian Optimisation loop
def run_bo(acq_type="qEI", q=1, n_iters=100, n_init=10):
    # Initial random design
    init_idx   = torch.randperm(N, device=device)[:n_init].tolist()
    observed   = set(init_idx)
    train_X    = X_pool[init_idx]
    train_Ymin = observe(init_idx)

    best_track = []

    n_steps = n_iters // q
    for _ in range(n_steps):
        Y = -train_Ymin

        # Fit GP
        model = SingleTaskGP(train_X, Y, outcome_transform=Standardize(m=1))
        mll   = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Acquisition function qEI or qNEI
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
        if acq_type == "qEI":
            acqf = qExpectedImprovement(model=model, best_f=Y.max(), sampler=sampler)
        else:
            acqf = qNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)

        # Evaluate acquisition over discrete candidate pool
        remaining = [i for i in range(N) if i not in observed]
        choices   = X_pool[remaining].unsqueeze(1)

        with torch.no_grad():
            acq_values = acqf(choices).squeeze(-1)

        # Select top-q candidates
        q_eff = min(q, len(remaining))
        topk  = torch.topk(acq_values, k=q_eff).indices.cpu().tolist()
        chosen_idx = [remaining[i] for i in topk]

        # Observe and update training data
        new_X      = X_pool[chosen_idx]
        new_Ymin   = Y_pool[chosen_idx]
        train_X    = torch.cat([train_X,    new_X],    dim=0)
        train_Ymin = torch.cat([train_Ymin, new_Ymin], dim=0)
        for ci in chosen_idx:
            observed.add(ci)

        current_best = train_Ymin.min().item()
        for _ in range(len(chosen_idx)):
            best_track.append(current_best)

    return best_track[:n_iters]

results = {
    "Random":     run_random(100),
    "qEI (q=1)":  run_bo("qEI",  q=1),
    "qNEI (q=1)": run_bo("qNEI", q=1),
    "qEI (q=4)":  run_bo("qEI",  q=4),
    "qNEI (q=4)": run_bo("qNEI", q=4),
}


# Plot
plt.figure(figsize=(10, 6))
for label, data in results.items():
    plt.plot(data, label=label)
plt.xlabel("Iterations")
plt.ylabel("Best observed noise ")
plt.title("BO on UCI Airfoil Self-Noise")
plt.legend()
plt.grid(True)
plt.show()


print("\nAblation Table:")
for label, data in results.items():
    print(f"{label:<12} : {data[-1]:.6f}")
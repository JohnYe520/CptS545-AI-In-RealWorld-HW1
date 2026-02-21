import torch
import matplotlib.pyplot as plt

from botorch.test_functions import Hartmann
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
torch.manual_seed(0)

noise_std = 0.1
f = Hartmann(dim=6, negate=False).to(device=device, dtype=dtype)
true_bounds = f.bounds.to(device=device, dtype=dtype)

# Return noisy objective to MINIMIZE
def observe_ymin(X):
    return f(X).unsqueeze(-1) + noise_std * torch.randn(X.shape[0], 1, device=device, dtype=dtype)

# Random baseline
def run_random(n_iters=100):
    X = torch.rand(n_iters, 6, device=device, dtype=dtype)
    Ymin = observe_ymin(X)
    best = torch.cummin(Ymin.squeeze(-1), dim=0).values
    return best.cpu().numpy()

# Bayesian Optimisation loop
def run_bo(acq_type="qEI", q=1, n_iters=100, n_init=10):
    # Initial random design
    X = torch.rand(n_init, 6, device=device, dtype=dtype)
    Ymin = observe_ymin(X)

    best_track = []

    n_steps = n_iters // q
    for _ in range(n_steps):
        Xn = normalize(X, true_bounds)

        Y = -Ymin

        # Fit GP
        model = SingleTaskGP(Xn, Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        if acq_type == "qEI":
            acqf = qExpectedImprovement(model=model, best_f=Y.max(), sampler=sampler)
        else:
            acqf = qNoisyExpectedImprovement(model=model, X_baseline=Xn, sampler=sampler)

        # Optimise over candidate pool
        bounds01 = torch.stack([torch.zeros(6, device=device, dtype=dtype),
                                torch.ones(6, device=device, dtype=dtype)])
        Xcand_n, _ = optimize_acqf(acq_function=acqf, bounds=bounds01, q=q, num_restarts=10, raw_samples=512)

        # Map candidate back to original bounds
        Xcand = unnormalize(Xcand_n, true_bounds)

        # Observe and update training data
        Ycand_min = observe_ymin(Xcand)

        X = torch.cat([X, Xcand], dim=0)
        Ymin = torch.cat([Ymin, Ycand_min], dim=0)

        # Track best-so-far per iteration
        current_best = Ymin.min().item()
        for _ in range(q):
            best_track.append(current_best)

    return best_track[:n_iters]

results = {
    "Random": run_random(100),
    "qEI (q=1)": run_bo("qEI", q=1),
    "qNEI (q=1)": run_bo("qNEI", q=1),
    "qEI (q=4)": run_bo("qEI", q=4),
    "qNEI (q=4)": run_bo("qNEI", q=4),
}

# Plot
plt.figure(figsize=(10, 6))
for label, data in results.items():
    plt.plot(data, label=label)
plt.xlabel("Iterations")
plt.ylabel("Best observed value")
plt.title("BO on Noisy 6D Hartmann")
plt.legend()
plt.grid(True)
plt.show()

print("\nAblation Table:")
for label, data in results.items():
    print(f"{label:<12} : {data[-1]:.6f}")
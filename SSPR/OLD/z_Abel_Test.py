import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

import scico

from scico.linop.abel import AbelTransform  # For my computer for some reason?
# from scico.linop.xray.abel import AbelTransform # For the supercomputer for some reason?
from scico import functional, linop, loss, metric
from scico.examples import create_circular_phantom
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

print("scico version: ", scico.__version__)

N = 256
# Ground truth
x_gt = create_circular_phantom((N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])

idx = N//2
value = 0
# x_gt = x_gt.at[:, idx:].set(value)

A = AbelTransform(x_gt.shape)

# Measurement
y = A @ x_gt
np.random.seed(12345)
y = y + 5 * np.random.normal(size=y.shape).astype(np.float32)

x_inv = A.inverse(y)

f = loss.SquaredL2Loss(y=y, A=A)


# # Existing (edge-preserving): L1 on finite differences
# lam1 = 50.5
# g1 = lam1 * functional.L1Norm()
# C1 = linop.FiniteDifference(input_shape=x_gt.shape)
# rho = 103
# maxiter = 100
# cg_tol = 1e-4
# cg_maxiter = 25
#
# # Added (smoothing): L2 on finite differences
# lam2 = 5.0   # tune this (start small)
# g2 = lam2 * functional.SquaredL2Norm()
# C2 = linop.FiniteDifference(input_shape=x_gt.shape)
#
# solver = ADMM(
#     f=f,
#     g_list=[g1, g2],
#     C_list=[C1, C2],
#     rho_list=[rho, rho],   # can tune separately too
#     x0=x_inv,
#     maxiter=maxiter,
#     subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
#     itstat_options={"display": True, "period": 10},
# )


lamm = 50.5  # L1 norm regularization parameter
g = lamm * functional.L1Norm()
C = linop.FiniteDifference(input_shape=x_gt.shape)

rho = 103  # ADMM penalty parameter -- best is 20
maxiter = 100  # number of ADMM iterations -- best is 1000
cg_tol = 1e-4  # CG relative tolerance -- best is 1e-6
cg_maxiter = 25  # maximum CG iterations per ADMM iteration -- best is 200

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[rho],
    x0=x_inv,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 10},
)

print(f"Solving on {device_info()}\n")
solver.solve()
x_tv = solver.x

norm = Normalize(vmin=-0.1, vmax=1.2)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
im0 = ax[0, 0].imshow(x_gt, cmap="Blues", norm=norm)
ax[0, 0].set_title("Ground Truth")
im1 = ax[0, 1].imshow(y, cmap="Blues")
ax[0, 1].set_title("Measurement")
im2 = ax[1, 0].imshow(
    x_inv,
    cmap="Blues",
    norm=norm,
)
ax[1, 0].set_title("Inverse Abel: %.2f (dB)" % metric.psnr(x_gt, x_inv))
im3 = ax[1, 1].imshow(
    x_tv,
    cmap="Blues",
    norm=norm,
)
ax[1, 1].set_title("TV-Regularized Inversion: %.2f (dB)" % metric.psnr(x_gt, x_tv))
for a in ax.flat:
    a.axis("off")
plt.tight_layout()
plt.show()

import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import abel

import scico
from scico import functional, linop, loss, metric
from scico.examples import create_circular_phantom
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

print("scico version: ", scico.__version__)




class PyAbelForward(linop.LinearOperator):
    def __init__(self, input_shape, method, origin, symmetry_axis, use_quadrants, reg, dr, recast_as_float64):
        self.method = method
        self.origin = origin
        self.symmetry_axis = symmetry_axis
        self.use_quadrants = use_quadrants
        self.reg = reg
        self.dr = dr
        self.recast_as_float64 = recast_as_float64
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, x):
        x = np.asarray(x, dtype=np.float32)
        proj = abel.Transform(
            x,
            direction="forward",
            method=self.method,
            origin=origin,
            symmetry_axis=self.symmetry_axis,
            use_quadrants=use_quadrants,
            transform_options=dict(reg=self.reg, dr=self.dr),
            recast_as_float64=self.recast_as_float64
        ).transform
        return proj


class PyAbelInverse(linop.LinearOperator):
    def __init__(self, input_shape, method, origin, symmetry_axis, use_quadrants, reg, dr, recast_as_float64):
        self.method = method
        self.origin = origin
        self.symmetry_axis = symmetry_axis
        self.use_quadrants = use_quadrants
        self.reg = reg
        self.dr = dr
        self.recast_as_float64 = recast_as_float64
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, y):
        y = np.asarray(y, dtype=np.float32)
        recon = abel.Transform(
            y,
            direction="inverse",
            method=self.method,
            origin=origin,
            symmetry_axis=self.symmetry_axis,
            use_quadrants=use_quadrants,
            transform_options=dict(reg=self.reg, dr=self.dr),
            recast_as_float64=self.recast_as_float64
        ).transform
        return recon


class PyAbelOperator(linop.LinearOperator):
    def __init__(self, input_shape, method, origin, symmetry_axis, use_quadrants, reg, dr, recast_as_float64):
        self.forward = PyAbelForward(input_shape, method, origin, symmetry_axis, use_quadrants,
                                     reg, dr, recast_as_float64)
        self.inverse = PyAbelInverse(input_shape, method, origin, symmetry_axis, use_quadrants,
                                    reg, dr, recast_as_float64)
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, x):
        return self.forward(x)

    def adj(self, y):
        return self.inverse(y)


method = "daun"
origin = "none"  # Default
use_quadrants = (True, True, True, True)  # Default
symmetry_axis = None
dr = 1  # Pixel size for the transform to use -- default is 1
reg = 20  # Smoothing regularization parameter in PyAbel -- best is 20
recast_as_float64 = False

N = 256
# Ground truth
x_gt = create_circular_phantom((N, N), [0.4 * N, 0.2 * N, 0.1 * N], [1, 0, 0.5])

A = PyAbelOperator(input_shape=x_gt.shape,
                   method=method,
                   origin=origin,
                   symmetry_axis=symmetry_axis,
                   use_quadrants=use_quadrants,
                   reg=reg,
                   dr=dr,
                   recast_as_float64=recast_as_float64)

# Measurement
y = A @ x_gt
np.random.seed(12345)
y = y + np.random.normal(size=y.shape).astype(np.float32)

x_inv = A.inverse(y)

f = loss.SquaredL2Loss(y=y, A=A)
lamm = 0.2  # L1 norm regularization parameter -- best is 0.2
g = lamm * functional.L1Norm()
C = linop.FiniteDifference(input_shape=x_gt.shape)

rho = 20  # ADMM penalty parameter -- best is 20
maxiter = 1000  # number of ADMM iterations -- best is 1000
cg_tol = 1e-6  # CG relative tolerance -- best is 1e-6
cg_maxiter = 200  # maximum CG iterations per ADMM iteration -- best is 200

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

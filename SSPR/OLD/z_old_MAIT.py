"""
MAIT implementation in Python.

Based on:
Chris Sparling and Jolijn Onvlee,
"Revisiting the inverse Abel integral for reconstructing velocity-map images",
Phys. Chem. Chem. Phys. 2025, 27, 18694–18709.

Core equation used by MAIT:
    I(y, r) = -(1 / (2*pi)) * A{ (1/z) * dP(y,z)/dz }

Interpretation:
- P(y,z) is the measured 2D projection image.
- The usual inverse Abel integral is numerically noisy because:
    1) the derivative amplifies noise
    2) the kernel blows up toward the centerline
- MAIT avoids evaluating that unstable inverse integral directly by rewriting
  it as a forward Abel transform of an intermediate image.

This script uses PyAbel for the forward Abel transform.

Install:
    pip install pyabel numpy scipy matplotlib scico
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift
import abel
from tifffile import imwrite

import scico.numpy as snp
from scico import functional, linop, loss
from scico.optimize.admm import ADMM
from scico.optimize.admm import LinearSubproblemSolver


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
def _as_float_image(img: np.ndarray) -> np.ndarray:
    """Force the input into a 2D float array."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")
    return img


def _recenter_image(
    img: np.ndarray,
    center: tuple[float, float] | None = None,
    shift_order: int = 1,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Shift the user-supplied center onto the geometric center of the array.
    """
    img = _as_float_image(img)
    ny, nx = img.shape
    target_center = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    if center is None:
        return img.copy(), target_center

    dy = target_center[0] - center[0]
    dx = target_center[1] - center[1]

    shifted = ndi_shift(img, shift=(dy, dx), order=shift_order, mode="nearest")
    return shifted, target_center


def coordinate_grids(
    shape: tuple[int, int],
    center: tuple[float, float] | None = None,
    dy: float = 1.0,
    dz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build coordinate grids centered on the assumed symmetry axis.

    Returns
    -------
    Y : vertical coordinate, positive upward
    Z : horizontal coordinate, positive to the right
    R : radius = sqrt(Y^2 + Z^2)
    cos_theta_y : Y / R
    cos_theta_z : Z / R
    """
    ny, nx = shape
    if center is None:
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    else:
        cy, cx = center

    rows = np.arange(ny)
    cols = np.arange(nx)

    y = -(rows - cy) * dy
    z = (cols - cx) * dz

    Y, Z = np.meshgrid(y, z, indexing="ij")
    R = np.hypot(Y, Z)

    cos_theta_y = np.zeros_like(R)
    cos_theta_z = np.zeros_like(R)

    mask = R > 0
    cos_theta_y[mask] = Y[mask] / R[mask]
    cos_theta_z[mask] = Z[mask] / R[mask]

    cos_theta_y[~mask] = 1.0
    cos_theta_z[~mask] = 0.0

    return Y, Z, R, cos_theta_y, cos_theta_z


# -----------------------------------------------------------------------------
# Optional SCICO TV denoising for the derivative image
# -----------------------------------------------------------------------------
def scico_anisotropic_tv_denoise(
    y_np: np.ndarray,
    lam: float = 20.0,
    rho: float = 100.0,
    maxiter: int = 75,
) -> np.ndarray:
    """
    Denoise a 2D image using anisotropic TV in SCICO.

    Solves approximately:
        min_x  0.5 * ||x - y||_2^2 + lam * TV_aniso(x)
    """
    y = snp.array(y_np.astype(np.float32))

    f = loss.SquaredL2Loss(y=y)
    g = lam * functional.AnisotropicTVNorm()

    solver = ADMM(
        f=f,
        g_list=[g],
        C_list=[linop.Identity(y.shape)],
        rho_list=[rho],
        x0=y,
        subproblem_solver=LinearSubproblemSolver(),
        maxiter=maxiter,
    )
    x = solver.solve()
    return np.array(x)


# -----------------------------------------------------------------------------
# Symmetry helpers
# -----------------------------------------------------------------------------
def enforce_antisymmetry_z(img: np.ndarray) -> np.ndarray:
    """
    Enforce D(y,z) = -D(y,-z).
    """
    return 0.5 * (img - np.fliplr(img))


def enforce_symmetry_z(img: np.ndarray) -> np.ndarray:
    """
    Enforce F(y,z) = F(y,-z).
    """
    return 0.5 * (img + np.fliplr(img))


# -----------------------------------------------------------------------------
# Safe divide by z
# -----------------------------------------------------------------------------
def safe_divide_by_z(
    deriv_img: np.ndarray,
    Z: np.ndarray,
    dz: float = 1.0,
    zero_tol: float | None = None,
) -> np.ndarray:
    """
    Compute deriv_img / Z safely.

    Away from the centerline, compute directly.
    On the centerline, estimate the finite z->0 limit numerically.
    """
    deriv_img = _as_float_image(deriv_img)

    if zero_tol is None:
        zero_tol = 1e-12 * max(1.0, dz)

    out = np.empty_like(deriv_img)
    nonzero = np.abs(Z) > zero_tol
    out[nonzero] = deriv_img[nonzero] / Z[nonzero]

    d_dz = np.gradient(deriv_img, dz, axis=1)
    out[~nonzero] = d_dz[~nonzero]

    return out


# -----------------------------------------------------------------------------
# Forward Abel transform
# -----------------------------------------------------------------------------
def forward_abel(img: np.ndarray, method: str = "hansenlaw") -> np.ndarray:
    """
    Forward Abel transform using PyAbel.
    """
    img = _as_float_image(img)
    return abel.Transform(
        img,
        method=method,
        direction="forward",
        origin="none",
        verbose=False,
    ).transform


# -----------------------------------------------------------------------------
# MAIT
# -----------------------------------------------------------------------------
def mait(
    projection_img: np.ndarray,
    center: tuple[float, float] | None = None,
    dy: float = 1.0,
    dz: float = 1.0,
    forward_method: str = "hansenlaw",
    clip_negative: bool = False,
    recenter: bool = True,
    tv_denoise: bool = True,
    tv_lam: float = 50.0,
    tv_rho: float = 500.0,
    tv_maxiter: int = 75,
) -> np.ndarray:
    """
    Reconstruct using MAIT.

    Steps:
      1) Start from measured projection image P(y,z)
      2) Differentiate with respect to z
      3) Enforce anti-symmetry of the derivative image
      4) Optionally denoise derivative image with anisotropic TV
      5) Divide by z to form the intermediate image
      6) Enforce symmetry of that intermediate image
      7) Apply the forward Abel transform
      8) Multiply by -(1 / 2pi)
    """
    projection_img = _as_float_image(projection_img)

    if recenter:
        img, ctr = _recenter_image(projection_img, center=center)
    else:
        img = projection_img.copy()
        if center is None:
            ctr = ((img.shape[0] - 1) / 2.0, (img.shape[1] - 1) / 2.0)
        else:
            ctr = center

    # Step 1: numerical derivative dP/dz
    deriv = np.gradient(img, dz, axis=1)

    # Step 2: enforce expected odd symmetry
    deriv = enforce_antisymmetry_z(deriv)

    # Step 3: optional TV denoising
    if tv_denoise:
        deriv = scico_anisotropic_tv_denoise(
            deriv,
            lam=tv_lam,
            rho=tv_rho,
            maxiter=tv_maxiter,
        )
        deriv = enforce_antisymmetry_z(deriv)

    # Step 4: build Z coordinates for the (1/z) step
    _, Z, _, _, _ = coordinate_grids(img.shape, center=ctr, dy=dy, dz=dz)

    # Step 5: intermediate image = (1/z) * dP/dz
    intermediate = safe_divide_by_z(deriv, Z, dz=dz)

    # Step 6: intermediate image should be even in z
    intermediate = enforce_symmetry_z(intermediate)

    # Step 7: apply forward Abel transform with MAIT prefactor
    recon = -(1.0 / (2.0 * np.pi)) * forward_abel(intermediate, method=forward_method)

    if clip_negative:
        recon = np.maximum(recon, 0.0)

    return recon


# -----------------------------------------------------------------------------
# Example / test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ny, nx = 400, 400
    cy, cx = ny // 2.0, nx // 2.0
    method = 'daun'

    # Build coordinates for a synthetic test object
    Y, Z, R, cos_theta_y, cos_theta_z = coordinate_grids(
        (ny, nx), center=(cy, cx)
    )

    # Example 1: angularly structured synthetic object
    P2y = 0.5 * (3.0 * cos_theta_y**2 - 1.0)
    I_true = (
        1.2 * np.exp(-0.5 * ((R - 70.0) / 5.0) ** 2) * (1.0 + 1.0 * P2y)
        + 0.7 * np.exp(-0.5 * ((R - 120.0) / 8.0) ** 2) * (1.0 - 0.7 * P2y)
    )
    I_true = np.maximum(I_true, 0.0)

    # # Example 2: uncomment for a shock/void-style object
    # void = np.exp(-((R - 80) ** 2) / (2 * 6 ** 2))
    # shock = 1 / (1 + np.exp(-(Y + 20) / 2))
    # compression = np.exp(-((Y + 30) ** 2) / (2 * 8 ** 2)) * np.exp(-(Z ** 2) / (2 * 25 ** 2))
    # I_true = 1.2 * shock - 0.8 * void + 0.6 * compression
    # I_true = np.maximum(I_true, 0.0)

    # Forward project the true object
    P = forward_abel(I_true, method=method)

    # Add a small amount of Gaussian noise
    rng = np.random.default_rng(1)
    P_noisy = P + 0.01 * np.max(P) * rng.standard_normal(P.shape)

    # MAIT reconstruction
    I_mait = mait(
        P_noisy,
        center=(cy, cx),
        forward_method=method,
        clip_negative=True,
        recenter=True,
        tv_denoise=True,
        tv_lam=50.0,
        tv_rho=1000.0,
        tv_maxiter=50,
    )

    imwrite("/Users/danielhodge/Desktop/I_mait.tiff", I_mait)

    # Display the true object, projection, and MAIT reconstruction
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    axs[0].imshow(I_true, origin="lower")
    axs[0].set_title("True I(y,z)")

    axs[1].imshow(P_noisy, origin="lower")
    axs[1].set_title("Projection P(y,z)")

    axs[2].imshow(I_mait, origin="lower")
    axs[2].set_title("MAIT")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
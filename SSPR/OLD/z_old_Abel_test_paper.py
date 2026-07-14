"""
MAIT and fMAIT implementation in Python.

Based on:
Chris Sparling and Jolijn Onvlee,
"Revisiting the inverse Abel integral for reconstructing velocity-map images",
Phys. Chem. Chem. Phys. 2025, 27, 18694–18709.

Core equation used by MAIT:
    I(y, r) = -(1 / (2*pi)) * A{ (1/z) * dP(y,z)/dz }

Interpretation from the paper:
- P(y,z) is the measured 2D projection image.
- The usual inverse Abel integral (their eq. 3) is numerically noisy because:
    1) the derivative amplifies noise
    2) the kernel blows up toward the centerline
- MAIT avoids evaluating that unstable inverse integral directly by rewriting
  it as a forward Abel transform of an intermediate image (their eqs. 5 and 6).

This script uses PyAbel for the forward Abel transform.

Install:
    pip install pyabel numpy scipy matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift
from scipy.special import eval_legendre
import abel
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_tv_chambolle

import scico.numpy as snp
from scico import functional, linop, loss
from scico.optimize.admm import ADMM
from scico.optimize.admm import LinearSubproblemSolver


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
def _as_float_image(img: np.ndarray) -> np.ndarray:
    """
    Force the input into a 2D float array.

    This is just a housekeeping step so all later operations
    (gradient, shifting, fitting, Abel transform) use a consistent dtype.
    """
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

    Why this matters for MAIT:
    - The paper's derivation assumes the vertical symmetry axis is correctly placed.
    - If the image is off-center, then the derivative will not be properly odd
      about the centerline, and the MAIT/fMAIT steps will show artifacts.

    Parameters
    ----------
    center : (row_center, col_center)
        Your estimate of the symmetry center in the original image.
    shift_order : int
        Interpolation order used by scipy.ndimage.shift.
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

    Relation to the paper:
    - The manuscript uses y as the symmetry axis and z as the transverse coordinate.
    - The derivative in MAIT is taken with respect to z.
    - The derivative image becomes odd in z, so z and z/r are the natural variables
      for the later symmetry and fMAIT filtering steps.
    """
    ny, nx = shape
    if center is None:
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    else:
        cy, cx = center

    rows = np.arange(ny)
    cols = np.arange(nx)

    # By convention here:
    # - Y increases upward
    # - Z increases to the right
    y = -(rows - cy) * dy
    z = (cols - cx) * dz

    Y, Z = np.meshgrid(y, z, indexing="ij")
    R = np.hypot(Y, Z)

    cos_theta_y = np.zeros_like(R)
    cos_theta_z = np.zeros_like(R)

    mask = R > 0
    cos_theta_y[mask] = Y[mask] / R[mask]
    cos_theta_z[mask] = Z[mask] / R[mask]

    # Values at the center are arbitrary because the angle is undefined there.
    cos_theta_y[~mask] = 1.0
    cos_theta_z[~mask] = 0.0

    return Y, Z, R, cos_theta_y, cos_theta_z


def scico_anisotropic_tv_denoise(y_np, lam=20, rho=100.0, maxiter=75):
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

def enforce_antisymmetry_z(img: np.ndarray) -> np.ndarray:
    """
    Enforce D(y,z) = -D(y,-z).

    Why:
    - The paper states that after differentiating the projection image with
      respect to z, the derivative image is no longer symmetric about the y-axis;
      it becomes anti-symmetric instead.
    - Enforcing this numerically removes small asymmetry errors caused by noise,
      centering error, and discretization.
    """
    return 0.5 * (img - np.fliplr(img))


def enforce_symmetry_z(img: np.ndarray) -> np.ndarray:
    """
    Enforce F(y,z) = F(y,-z).

    Why:
    - The derivative image is odd in z.
    - After dividing by z, the intermediate image should become even in z again.
    - Symmetrizing here reduces numerical mismatch before the forward Abel step.
    """
    return 0.5 * (img + np.fliplr(img))


def safe_divide_by_z(
    deriv_img: np.ndarray,
    Z: np.ndarray,
    dz: float = 1.0,
    zero_tol: float | None = None,
) -> np.ndarray:
    """
    Compute deriv_img / Z safely.

    Paper connection:
    - The paper explains that because the derivative image is already an odd
      function in z, it already contains a factor of z.
    - Therefore the 1/z step does NOT create a true singularity on the centerline.
      Instead, z=0 gives a 0/0 indeterminate form with a finite limit.

    Implementation:
    - Away from the centerline, compute deriv_img / Z directly.
    - On the centerline, estimate the z->0 limit numerically from d(deriv_img)/dz.
    """
    deriv_img = _as_float_image(deriv_img)

    if zero_tol is None:
        zero_tol = 1e-12 * max(1.0, dz)

    out = np.empty_like(deriv_img)
    nonzero = np.abs(Z) > zero_tol
    out[nonzero] = deriv_img[nonzero] / Z[nonzero]

    # Numerical estimate of the finite centerline limit.
    d_dz = np.gradient(deriv_img, dz, axis=1)
    out[~nonzero] = d_dz[~nonzero]

    return out


def forward_abel(img: np.ndarray, method: str = "hansenlaw") -> np.ndarray:
    """
    Forward Abel transform using PyAbel.

    Paper connection:
    - The key MAIT idea is to replace the unstable inverse integral with a
      numerically stable forward Abel transform of the intermediate image.
    - The paper notes that in principle any stable forward Abel implementation
      could be used; they used a rapid matrix approach in MATLAB.
    - Here we use PyAbel as a practical Python substitute.
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
# fMAIT filtering
# -----------------------------------------------------------------------------
def odd_orders_up_to(max_order: int) -> list[int]:
    """
    Return odd Legendre orders 1, 3, 5, ..., max_order.

    Paper connection:
    - The derivative image is anti-symmetric, so its angular structure is
      described by odd-degree terms only.
    """
    return [l for l in range(1, max_order + 1, 2)]


def legendre_filter_derivative_image(
    deriv_img: np.ndarray,
    center: tuple[float, float] | None = None,
    lmax_deriv: int = 3,
    dy: float = 1.0,
    dz: float = 1.0,
    rbin_width: float = 1.0,
    min_points_per_bin: int = 12,
) -> np.ndarray:
    """
    Filter the derivative image using odd Legendre terms.

    Paper connection:
    - In fMAIT, the derivative image is expanded using only the allowed odd
      angular terms.
    - The paper explains that if the final reconstructed image has even content
      up to lmax = 2N, then the derivative image requires odd content up to
      2N + 1.
    - This filtering acts as a noise suppression step, similar in spirit to pBASEX.

    Why use u = z/r here:
    - The derivative image is odd in z and vanishes on the vertical symmetry axis.
    - So z/r is the natural angular variable for fitting the derivative image.
    """
    deriv_img = _as_float_image(deriv_img)

    _, _, R, _, cos_theta_z = coordinate_grids(
        deriv_img.shape, center=center, dy=dy, dz=dz
    )

    orders = odd_orders_up_to(lmax_deriv)
    filtered = np.zeros_like(deriv_img)

    # Bin pixels by radius so the Legendre coefficients can vary with r.
    # This matches the paper's idea that the angular coefficients depend on radius.
    rmax = np.max(R)
    nbins = int(np.floor(rmax / rbin_width)) + 1
    rbin = np.floor(R / rbin_width).astype(int)

    valid = np.isfinite(deriv_img)

    for rb in range(nbins):
        mask = (rbin == rb) & valid
        npts = np.count_nonzero(mask)

        # If there are too few points in the shell, skip the fit and keep raw values.
        if npts < min_points_per_bin:
            filtered[mask] = deriv_img[mask]
            continue

        # Angular coordinate for this shell
        u = cos_theta_z[mask]

        # Raw derivative-image values in this radial shell
        vals = deriv_img[mask]

        # Build matrix of odd Legendre basis functions:
        # [P1(u), P3(u), P5(u), ...]
        A = np.column_stack([eval_legendre(l, u) for l in orders])

        # Least-squares solve for the radial coefficients b_l(r)
        coef, *_ = np.linalg.lstsq(A, vals, rcond=None)

        # Reconstruct the filtered derivative image on this shell
        filtered[mask] = A @ coef

    # Re-enforce exact odd symmetry after fitting
    filtered = enforce_antisymmetry_z(filtered)
    return filtered


# -----------------------------------------------------------------------------
# MAIT and fMAIT
# -----------------------------------------------------------------------------
def mait(
    projection_img: np.ndarray,
    center: tuple[float, float] | None = None,
    dy: float = 1.0,
    dz: float = 1.0,
    forward_method: str = "hansenlaw",
    clip_negative: bool = False,
    recenter: bool = True,
) -> np.ndarray:
    """
    Reconstruct using MAIT.

    This directly follows the paper's MAIT sequence:

      1) Start from measured projection image P(y,z)
      2) Differentiate with respect to z
      3) Enforce anti-symmetry of the derivative image
      4) Divide by z to form the intermediate image
      5) Enforce symmetry of that intermediate image
      6) Apply the forward Abel transform
      7) Multiply by -(1 / 2pi)

    This is the coded version of the manuscript's eqs. (5) and (6).
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
    # The paper says they compute this with MATLAB's built-in gradient function.
    deriv = np.gradient(img, dz, axis=1)

    # Step 2: enforce the expected odd symmetry of the derivative image
    deriv = enforce_antisymmetry_z(deriv)

    # deriv = gaussian_filter(deriv, sigma=4 / 2.35, truncate=2)
    # deriv = denoise_tv_chambolle(deriv, weight=5)
    deriv = scico_anisotropic_tv_denoise(deriv, lam=50, rho=500)

    # Build Z coordinates for the (1/z) step
    _, Z, _, _, _ = coordinate_grids(img.shape, center=ctr, dy=dy, dz=dz)

    # Step 3: intermediate image = (1/z) * dP/dz
    intermediate = safe_divide_by_z(deriv, Z, dz=dz)

    # Step 4: after dividing by z, the image should be even in z
    intermediate = enforce_symmetry_z(intermediate)

    # Step 5: apply the forward Abel transform and scale by -(1 / 2pi)
    recon = -(1.0 / (2.0 * np.pi)) * forward_abel(
        intermediate, method=forward_method
    )

    if clip_negative:
        recon = np.maximum(recon, 0.0)

    return recon


def fmait(
    projection_img: np.ndarray,
    center: tuple[float, float] | None = None,
    dy: float = 1.0,
    dz: float = 1.0,
    final_lmax_even: int = 2,
    forward_method: str = "hansenlaw",
    clip_negative: bool = False,
    recenter: bool = True,
    rbin_width: float = 1.0,
    min_points_per_bin: int = 12,
) -> np.ndarray:
    """
    Reconstruct using filtered MAIT (fMAIT).

    Paper connection:
    - fMAIT inserts an angular filtering step on the derivative image before
      dividing by z.
    - If the final reconstructed image is expected to contain even Legendre
      terms up to final_lmax_even, then the derivative image must contain odd
      terms up to final_lmax_even + 1.

    This is useful when the image's angular content is well described by a
    low-order Legendre expansion. For more complicated angular structure,
    the paper suggests plain MAIT may be the safer option.
    """
    if final_lmax_even < 0:
        raise ValueError("final_lmax_even must be >= 0")
    if final_lmax_even % 2 != 0:
        raise ValueError("final_lmax_even should be even")

    projection_img = _as_float_image(projection_img)

    if recenter:
        img, ctr = _recenter_image(projection_img, center=center)
    else:
        img = projection_img.copy()
        if center is None:
            ctr = ((img.shape[0] - 1) / 2.0, (img.shape[1] - 1) / 2.0)
        else:
            ctr = center

    # Step 1: derivative of the measured projection
    deriv = np.gradient(img, dz, axis=1)

    # Step 2: enforce the paper's expected odd symmetry
    deriv = enforce_antisymmetry_z(deriv)

    # Step 3: Legendre-filter the derivative image using only odd orders
    # up to lmax+1, as described in the paper
    lmax_deriv = final_lmax_even + 1
    deriv_filtered = legendre_filter_derivative_image(
        deriv,
        center=ctr,
        lmax_deriv=lmax_deriv,
        dy=dy,
        dz=dz,
        rbin_width=rbin_width,
        min_points_per_bin=min_points_per_bin,
    )

    # Step 4: divide the filtered derivative image by z
    _, Z, _, _, _ = coordinate_grids(img.shape, center=ctr, dy=dy, dz=dz)
    intermediate = safe_divide_by_z(deriv_filtered, Z, dz=dz)

    # Step 5: restore even symmetry before the forward Abel transform
    intermediate = enforce_symmetry_z(intermediate)

    # Step 6: forward Abel transform with the MAIT prefactor
    recon = -(1.0 / (2.0 * np.pi)) * forward_abel(
        intermediate, method=forward_method
    )

    if clip_negative:
        recon = np.maximum(recon, 0.0)

    return recon


# -----------------------------------------------------------------------------
# Example / test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ny, nx = 401, 401
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Build coordinates for a synthetic test object
    Y, Z, R, cos_theta_y, cos_theta_z = coordinate_grids(
        (ny, nx), center=(cy, cx)
    )

    # Construct a synthetic cylindrically symmetric object I(y,z)
    # using a simple P2 angular dependence relative to the symmetry axis.
    # This mirrors the kind of Legendre-structured test images discussed in the paper.
    P2y = 0.5 * (3.0 * cos_theta_y**2 - 1.0)

    I_true = (
        1.2 * np.exp(-0.5 * ((R - 70.0) / 5.0) ** 2) * (1.0 + 1.0 * P2y)
        + 0.7 * np.exp(-0.5 * ((R - 120.0) / 8.0) ** 2) * (1.0 - 0.7 * P2y)
    )
    I_true = np.maximum(I_true, 0.0)

    # # spherical void
    # void = np.exp(-((R - 80) ** 2) / (2 * 6 ** 2))
    # # planar shock front approaching from below
    # shock = 1 / (1 + np.exp(-(Y + 20) / 2))
    # # localized compression region
    # compression = np.exp(-((Y + 30) ** 2) / (2 * 8 ** 2)) * np.exp(-(Z ** 2) / (2 * 25 ** 2))
    # I_true = 1.2 * shock - 0.8 * void + 0.6 * compression
    # I_true = np.maximum(I_true, 0)

    # Forward project the true object to create a synthetic measured image P(y,z)
    P = forward_abel(I_true, method="daun")

    # Add a small amount of Gaussian noise to emulate imperfect measured data
    rng = np.random.default_rng(1)
    P_noisy = P + 0.01 * np.max(P) * rng.standard_normal(P.shape)

    # MAIT reconstruction
    I_mait = mait(
        P_noisy,
        center=(cy, cx),
        forward_method="daun",
        clip_negative=True,
        recenter=True,
    )

    # I_mait = gaussian_filter(I_mait, sigma=4 / 2.35, truncate=2)

    # fMAIT reconstruction using final even Legendre content up to l=2
    I_fmait = fmait(
        P_noisy,
        center=(cy, cx),
        final_lmax_even=2,
        forward_method="daun",
        clip_negative=True,
        recenter=True,
        rbin_width=1.0,
    )

    # Display the true object, projection, and reconstructions
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    ims = [
        axs[0].imshow(I_true, origin="lower"),
        axs[1].imshow(P_noisy, origin="lower"),
        axs[2].imshow(I_mait, origin="lower"),
        axs[3].imshow(I_fmait, origin="lower"),
    ]

    axs[0].set_title("True I(y,z)")
    axs[1].set_title("Projection P(y,z)")
    axs[2].set_title("MAIT")
    axs[3].set_title("fMAIT")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
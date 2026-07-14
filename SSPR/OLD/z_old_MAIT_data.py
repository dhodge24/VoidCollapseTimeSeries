from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift
from tifffile import imread, imwrite
import abel

import scico.numpy as snp
from scico import functional, linop, loss
from scico.optimize.admm import ADMM
from scico.optimize.admm import LinearSubproblemSolver

from scipy.ndimage import shift
from utilities import cropToCenter, padToSize


# =============================================================================
# Basic utilities
# =============================================================================
def _as_float_image(img: np.ndarray) -> np.ndarray:
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
    Shift the supplied center to the geometric center of the array.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return Y, Z, R


# =============================================================================
# SCICO TV denoising
# =============================================================================
def scico_anisotropic_tv_denoise(
    y_np: np.ndarray,
    lam: float = 20.0,
    rho: float = 100.0,
    maxiter: int = 75,
) -> np.ndarray:
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


# =============================================================================
# Symmetry helpers
# =============================================================================
def enforce_antisymmetry_z(img: np.ndarray) -> np.ndarray:
    return 0.5 * (img - np.fliplr(img))


def enforce_symmetry_z(img: np.ndarray) -> np.ndarray:
    return 0.5 * (img + np.fliplr(img))


# =============================================================================
# Safer divide by z
# =============================================================================
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


# =============================================================================
# Forward Abel transform
# =============================================================================
def forward_abel(img: np.ndarray, method: str = "hansenlaw") -> np.ndarray:
    img = _as_float_image(img)
    return abel.Transform(
        img,
        method=method,
        direction="forward",
        origin="none",
        verbose=False,
    ).transform


# =============================================================================
# MAIT
# =============================================================================
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
    return_intermediate: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MAIT reconstruction for arbitrary 2D image size.

    Parameters
    ----------
    projection_img : 2D array
        Projected quantity P(y,z).
    center : tuple or None
        Symmetry center in the CURRENT array coordinates.
    dy, dz : float
        Pixel spacing in y and z.
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

    # 1) derivative along z
    deriv = np.gradient(img, dz, axis=1)

    # 2) enforce odd symmetry
    deriv = enforce_antisymmetry_z(deriv)

    # 3) optional TV denoise
    if tv_denoise:
        deriv = scico_anisotropic_tv_denoise(
            deriv,
            lam=tv_lam,
            rho=tv_rho,
            maxiter=tv_maxiter,
        )
        deriv = enforce_antisymmetry_z(deriv)

    # 4) build z grid
    _, Z, _ = coordinate_grids(img.shape, center=ctr, dy=dy, dz=dz)

    # 5) intermediate = (1/z) dP/dz with center blending
    intermediate = safe_divide_by_z(
        deriv,
        Z,
        dz=dz,
    )

    # 6) enforce even symmetry
    intermediate = enforce_symmetry_z(intermediate)

    # 7) forward Abel transform with prefactor
    recon = -(1.0 / (2.0 * np.pi)) * forward_abel(intermediate, method=forward_method)

    if clip_negative:
        recon = np.maximum(recon, 0.0)

    if return_intermediate:
        return recon, deriv, intermediate
    return recon


# =============================================================================
# Your experimental pipeline
# =============================================================================
if __name__ == "__main__":
    run_holo = "572"
    dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped/"
    sim = True
    run_type = "sim"

    scale_factor = 2.5
    dx = 25.000069739608307

    # Experimental parameters
    E = 18000
    lam = (1240 / E) * 1e-9
    c = 2.9979e8
    m_e = 9.1094e-31
    eps0 = 8.852e-12
    e = 1.6022e-19
    r_e = 2.82e-15
    N_A = 6.022e23

    m_to_nm = 1e-9
    num_elec = 10e6
    cmPerPx = 9.998679161071777e-06

    # --------------------------------------------------
    # Imaging geometry
    # --------------------------------------------------

    z01 = 120.41e-3  # source -> sample distance (m)
    z12 = 4.668995  # sample -> detector distance (m)
    z02 = z01 + z12  # source -> detector distance

    M = z02 / z01  # geometric magnification
    z_eff = z12 / M  # effective propagation distance

    scale_fac = 4  # scintillator / optical magnification
    det_pixel_size = 6.5e-6  # detector pixel size (m)

    # Effective pixel size at the object
    dx_eff = det_pixel_size / M / scale_fac
    dy_eff = det_pixel_size / M / scale_fac

    # -------------------------------------------------------------------------
    # Load phase image
    # -------------------------------------------------------------------------
    if sim:
        dir_use = f"run{run_holo}_sim/"
    else:
        dir_use = f"run{run_holo}_exp/"

    tiff_ph = f"run{run_holo}_{run_type}_ph_final.tiff"
    ph = np.array(imread(dir_main + dir_use + tiff_ph), dtype=np.float32)

    end_size = [ph.shape[0], ph.shape[1]]
    out_size = [2 * end_size[0], 2 * end_size[1]]

    # -------------------------------------------------------------------------
    # Convert phase -> projected electron density
    # -------------------------------------------------------------------------
    n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e ** 2
    projected_electron_density_total = -ph * lam * n_c / np.pi * m_to_nm**2 / num_elec  # units: 10^6 e-/nm^2

    # -------------------------------------------------------------------------
    # Apply horizontal centering shift before padding
    # -------------------------------------------------------------------------
    projected_electron_density_total = shift(
        projected_electron_density_total,
        shift=(0, dx),
        order=3,
        mode="nearest",
    )

    projected_electron_density_total = cropToCenter(
        img=projected_electron_density_total,
        newSize=end_size,
    )

    projected_electron_density_total = padToSize(
        img=projected_electron_density_total,
        outputSize=out_size,
        padMethod="replicate",
        padType="both",
        padValue=None,
    )

    # -------------------------------------------------------------------------
    # MAIT on padded image
    # IMPORTANT:
    # Use the center of the padded array unless you have a better fitted center.
    # -------------------------------------------------------------------------
    ny_pad, nx_pad = projected_electron_density_total.shape
    center_pad = ((ny_pad - 1) / 2.0, (nx_pad - 1) / 2.0)

    x_result, deriv_img, intermediate = mait(
        projected_electron_density_total,
        center=center_pad,
        dy=1.0,
        dz=1.0,
        forward_method="hansenlaw",   # or "hansenlaw"
        clip_negative=False,
        recenter=True,          # already centered by your shift/pad workflow
        tv_denoise=False,
        tv_lam=50.0,
        tv_rho=1000.0,
        tv_maxiter=50,
        return_intermediate=True,
    )

    # -------------------------------------------------------------------------
    # Convert units
    # -------------------------------------------------------------------------
    x_result = x_result * num_elec * (1e7 ** 2) * cmPerPx / 10e20 * scale_factor


    # -------------------------------------------------------------------------
    # Crop back to original size and clip negatives
    # -------------------------------------------------------------------------
    x_result = cropToCenter(
        img=x_result,
        newSize=end_size,
    )
    x_result[x_result < 0] = 0

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    imwrite(f"/Users/danielhodge/Desktop/run{run_holo}_{run_type}_mait_result.tiff", x_result.astype(np.float32))

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    axs[0].imshow(projected_electron_density_total, origin="lower", cmap="viridis")
    axs[0].set_title("Projected e-density (padded)")

    axs[1].imshow(deriv_img, origin="lower", cmap="viridis")
    axs[1].set_title("dP/dz")

    axs[2].imshow(intermediate, origin="lower", cmap="viridis")
    axs[2].set_title("(1/z) dP/dz")

    axs[3].imshow(x_result, origin="lower", cmap="viridis")
    axs[3].set_title("MAIT result (cropped)")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
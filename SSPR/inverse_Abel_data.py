
# Import python modules
import numpy as np
import abel
from matplotlib import pyplot as plt
from tifffile import imread, imwrite
from scipy.ndimage import shift, gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from scico import linop

# Import custom modules
from SSPR.utilities import cropToCenter, padToSize, removeOutliers, padFadeOut


def best_symmetry_center(img, search_halfwidth=300, mask=None, subpixel=True):
    """
    Find the best left-right symmetry center column by minimizing MSE between
    left side and mirrored right side. APPLY TO THE SIMULATED NOISELESS DATA

    Parameters
    ----------
    img : 2D array (ny, nx)
    search_halfwidth : int
        Search columns in [nx//2 - search_halfwidth, nx//2 + search_halfwidth]
    mask : 2D bool array or None
        If provided, only masked pixels contribute to the score.
    subpixel : bool
        If True, do a simple 3-point quadratic refinement around the best integer.

    Returns
    -------
    best_c : float
        Best center column (float if subpixel=True, else int).
    dx : float
        Shift in x that would move best_c to the image center (nx-1)/2.
        Use with scipy.ndimage.shift(img, shift=(0, dx), ...)
    best_score : float
        Best MSE score (lower is better).
    """
    I = np.asarray(img, dtype=np.float64)
    ny, nx = I.shape

    if mask is None:
        M = np.ones_like(I, dtype=bool)
    else:
        M = np.asarray(mask, dtype=bool)

    # image "true" center coordinate in x (pixel coordinate system)
    x_center = (nx - 1) / 2.0

    c0 = nx // 2
    cmin = max(1, c0 - int(search_halfwidth))
    cmax = min(nx - 2, c0 + int(search_halfwidth))

    def score_at(c_int):
        # split at candidate center column c_int (exclude the center column itself)
        L = I[:, :c_int]
        R = I[:, c_int+1:]

        m = min(L.shape[1], R.shape[1])
        if m < 4:
            return np.inf

        # take equal-width strips near the center
        Lm = L[:, -m:]                 # left strip (closest to center)
        Rm = R[:, :m][:, ::-1]         # right strip mirrored

        Mm = M[:, :c_int][:, -m:] & M[:, c_int+1:][:, :m]  # corresponding mask

        if not np.any(Mm):
            return np.inf

        d = Lm - Rm
        return np.mean((d[Mm])**2)

    # brute-force integer scan
    cols = np.arange(cmin, cmax + 1)
    scores = np.array([score_at(c) for c in cols], dtype=np.float64)

    i_best = int(np.nanargmin(scores))
    c_best_int = int(cols[i_best])
    best_score = float(scores[i_best])
    best_c = float(c_best_int)

    # optional sub-pixel refinement via quadratic fit to (c-1,c,c+1)
    if subpixel and (i_best > 0) and (i_best < len(cols) - 1):
        y1, y2, y3 = scores[i_best - 1], scores[i_best], scores[i_best + 1]
        denom = (y1 - 2*y2 + y3)
        if np.isfinite(denom) and abs(denom) > 1e-30:
            # vertex offset in [-0.5, 0.5]ish if well-behaved
            delta = 0.5 * (y1 - y3) / denom
            best_c = c_best_int + float(delta)

    # dx to shift best_c onto x_center (positive dx shifts content to the right)
    dx = x_center - best_c
    return best_c, dx, best_score


# ----------------
# MAKE LEFT/RIGHT COMPOSITES
# left half = reconstruction
# right half = ground truth
# ----------------
def make_half_composite1(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp


def make_half_composite2(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2

    # left half = ground truth
    comp[:, :mid] = gt[:, :mid]

    # right half = reconstruction
    comp[:, mid:] = img[:, mid:]

    return comp


class PyAbelForward(linop.LinearOperator):
    def __init__(self, input_shape, method, symmetry_axis, reg, dr, recast_as_float64):
        self.method = method
        self.symmetry_axis = symmetry_axis
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
            symmetry_axis=self.symmetry_axis,
            transform_options=dict(reg=self.reg, dr=self.dr),
            recast_as_float64=self.recast_as_float64).transform
        return proj


class PyAbelInverse(linop.LinearOperator):
    def __init__(self, input_shape, method, symmetry_axis, symmetrize_method, use_quadrants, origin, reg, dr,
                 degree, recast_as_float64):
        self.method = method
        self.symmetry_axis = symmetry_axis
        self.symmetrize_method = symmetrize_method
        self.use_quadrants = use_quadrants
        self.origin = origin
        self.reg = reg
        self.dr = dr
        self.degree = degree
        self.recast_as_float64 = recast_as_float64
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, y):
        y = np.asarray(y, dtype=np.float32)
        recon = abel.Transform(
            y,
            direction="inverse",
            method=self.method,
            symmetry_axis=self.symmetry_axis,
            symmetrize_method=self.symmetrize_method,
            use_quadrants=self.use_quadrants,
            origin=self.origin,
            transform_options=dict(reg=self.reg, dr=self.dr, degree=self.degree),
            recast_as_float64=self.recast_as_float64).transform
        return recon


class PyAbelOperator(linop.LinearOperator):
    def __init__(self, input_shape, method, symmetry_axis, symmetrize_method, use_quadrants, origin, reg, dr,
                 degree, recast_as_float64):
        self.forward = PyAbelForward(input_shape, method, symmetry_axis, reg, dr, recast_as_float64)
        self.inverse = PyAbelInverse(input_shape, method, symmetry_axis, symmetrize_method, use_quadrants,
                                     origin, reg, dr, degree, recast_as_float64)
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, x):
        return self.forward(x)

    def adj(self, y):
        return self.inverse(y)


# def build_from_left_half(img):
#     ny, nx = img.shape
#     mid = nx // 2
#
#     out = np.zeros_like(img)
#
#     # keep left half
#     out[:, :mid] = img[:, :mid]
#
#     # fill right half by mirroring the left half
#     out[:, mid:] = np.fliplr(img[:, :nx-mid])
#
#     return out
#
# def build_from_right_half(img):
#     ny, nx = img.shape
#     mid = nx // 2
#
#     out = np.zeros_like(img)
#
#     # keep right half
#     out[:, mid:] = img[:, mid:]
#
#     # fill left half by mirroring the right half
#     out[:, :mid] = np.fliplr(img[:, mid:nx])[:, :mid]
#
#     return out


def average_left_with_flipped_right(img):
    ny, nx = img.shape
    mid = nx // 2

    left = img[:, :mid]
    right = img[:, mid:]

    # flip right half so it maps onto the left side
    right_flipped_to_left = np.fliplr(right)

    # if odd/even sizes ever differ by 1 pixel, trim to common width
    m = min(left.shape[1], right_flipped_to_left.shape[1])
    left = left[:, :m]
    right_flipped_to_left = right_flipped_to_left[:, :m]

    left_avg = 0.5 * (left + right_flipped_to_left)
    return left_avg


def make_leftavg_rightgt_composite(img, gt):
    ny, nx = img.shape
    mid = nx // 2

    comp = np.zeros_like(img)

    # averaged left-half image from reconstruction
    left_avg = average_left_with_flipped_right(img)

    # put averaged image on left
    comp[:, :left_avg.shape[1]] = left_avg

    # put ground truth on right
    comp[:, mid:] = gt[:, mid:]

    return comp


run_holo = "582"
sim = False
type = "exp"

# ------------------------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------------------------

dir_main = "/Users/danielhodge/Desktop/"
dir_recons = "time_series_recons_cropped/"
# dir_recons = "time_series_new_recons/"
dir_runs = "run" + run_holo + "_" + type + "/"
dir_gt_maps = "GT_adjusted_maps/"
dir_run_gt_maps = "run" + run_holo + "_GT_maps/"

# ------------------------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------------------------

E = 18000  # Initial energy of the beam in eV
lam = (1240 / E) * 1e-9  # Wavelength
c = 2.9979e8  # Speed of light in m/s
m_e = 9.1094e-31  # Electron mass in kg
eps0 = 8.852e-12  # Permittivity of free space in units C^2 / (N * m^2)
e = 1.6022e-19  # Charge of an electron in C
r_e = 2.82e-15  # Classical electron radius in meters
N_A = 6.022e23  # Avogadro's number in mol^-1

m_to_nm = 1e-9  # To put the scaling in # of electrons per nm^2
num_elec = 10e6  # Scaling the electrons for a more reasonable looking plot
cmPerPx = 9.998679161071777e-06  # Computed in the sim_XRAGE_maps.py file
cmPerPx = 4.086e-6
# scale_factor = 2.5  # We must scale the pixel size in the transform for the values to be accurate
dx = 50.4996321295979
dy = -45

method = "daun"  # Seems to work better than hansenlaw on sim + exp data
symmetry_axis = 0  # Across the y-axis
symmetrize_method = "average"  # Default is 'average'
origin = "none"
degree = 0  # Default is 0
use_quadrants = (True, True, True, True)  # Default is (True, True, True, True)
dr = cmPerPx #/ scale_factor  # Pixel size for the transform to use -- default is 1
reg = 1e7  # Smoothing regularization parameter in PyAbel
recast_as_float64 = True

# ------------------------------------------------------------------------------------------------
# IMPORT DATA
# ------------------------------------------------------------------------------------------------

electron_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "electron_density_total_GT_adjusted.tiff"
mass_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "density_total_GT_adjusted.tiff"
electron_density_gt = np.array(imread(electron_density_gt_path), dtype=np.float32)
mass_density_gt = np.array(imread(mass_density_gt_path), dtype=np.float32)

if sim:
    dir_sim = "run" + run_holo + "_sim/"
    tiff_ph = "run" + run_holo + "_" + type + "_ph_final.tiff"
    ph = np.array(imread(dir_main + dir_recons + dir_sim + tiff_ph), dtype=np.float32)
    # dir_sim = "run" + run_holo + "_sim_sampling_test/"
    # tiff_ph = "ph_final_run576sim_test.tiff"
    # ph = np.array(imread("/Users/danielhodge/Desktop/time_series_new_recons/run576_sim_sampling_test/run576_sim_phase.tiff"), dtype=np.float32)
else:
    dir_exp = "run" + run_holo + "_exp/"
    tiff_ph = "run" + run_holo + "_" + type + "_ph_final.tiff"
    ph = np.array(imread(dir_main + dir_recons + dir_exp + tiff_ph), dtype=np.float32)

# ------------------------------------------------------------------------------------------------
# COLOR LIMITS
# adjust these as needed
# ------------------------------------------------------------------------------------------------

elec_clim = (np.min(electron_density_gt), 1.25 * np.max(electron_density_gt))
mass_clim = (np.min(mass_density_gt), 1.25 * np.max(mass_density_gt))

# ------------------------------------------------------------------------------------------------
# CALCULATE INVERSE ABEL
# ------------------------------------------------------------------------------------------------

# Seems that the best is when the padding size is twice the cropped size
end_size = [ph.shape[0], ph.shape[1]]  # (y, x)
out_size = [2 * end_size[0], 2 * end_size[1]]

n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e ** 2
projected_electron_density_total = -ph * lam * n_c / np.pi * m_to_nm ** 2 / num_elec  # Units 10^6 e-/nm^2
projected_electron_density_total = gaussian_filter(projected_electron_density_total, sigma=15)

# best_c, dx, score = best_symmetry_center(projected_electron_density_total, search_halfwidth=400, mask=None, subpixel=True)
# print("best center col:", best_c, "dx:", dx, "score:", score)

projected_electron_density_total = shift(projected_electron_density_total, shift=(dy, dx), order=3, mode='nearest')

projected_electron_density_total = cropToCenter(img=projected_electron_density_total,
                                                newSize=end_size)

projected_electron_density_total = padToSize(img=projected_electron_density_total,
                                             outputSize=out_size,
                                             padMethod='replicate',
                                             padType='both',
                                             padValue=None)

A = PyAbelOperator(input_shape=projected_electron_density_total.shape,
                   method=method,
                   symmetry_axis=symmetry_axis,
                   symmetrize_method=symmetrize_method,
                   use_quadrants=use_quadrants,
                   origin=origin,
                   reg=reg,
                   dr=dr,
                   degree=degree,
                   recast_as_float64=recast_as_float64)

y = projected_electron_density_total
electron_density_recon = A.inverse(y)
electron_density_recon -= 60  # Subtract background
electron_density_recon[electron_density_recon < 0] = 0
electron_density_recon = cropToCenter(img=electron_density_recon, newSize=end_size)
electron_density_recon = removeOutliers(electron_density_recon, 2)

imwrite("/Users/danielhodge/Desktop/elec_density_recon.tiff", electron_density_recon)

# ------------------------------------------------------------------------------------------------
# COMPUTE THE RECONSTRUCTED MASS DENSITY FROM THE RECONSTRUCTED ELECTRON DENSITY
# ------------------------------------------------------------------------------------------------

elec_fac = 10e20  # This is what I scaled the electron density by -- 10e-20 * e-/cm^3 (for better plotting)
# We assume only SU8 with chemical composition: C87 H118 O16, the same used for XPCI forward modeling
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16  # In g/mol
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8  # Unitless
mass_density_recon = electron_density_recon * elec_fac * A_SU8 / (N_A * Z_SU8)

imwrite("/Users/danielhodge/Desktop/mass_density_recon.tiff", mass_density_recon)

electron_density_recon_comp = make_half_composite1(electron_density_recon, electron_density_gt)
mass_density_recon_comp = make_half_composite1(mass_density_recon, mass_density_gt)

# # average
# electron_density_recon_avg_comp = make_leftavg_rightgt_composite(electron_density_recon, electron_density_gt)
# mass_density_recon_avg_comp = make_leftavg_rightgt_composite(mass_density_recon, mass_density_gt)

# ----------------
# SPATIAL CALIBRATION FOR PLOTS
# ----------------
E = 18000  # eV
lam = (1240 / E) * 1e-9
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac  # meters per pixel

# ----------------
# FIGURE
# ----------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs = axs.ravel()

images = [electron_density_recon_comp, mass_density_recon_comp]
titles = [
    "Electron Density\nLeft: Reconstruction | Right: Ground Truth",
    "Mass Density\nLeft: Reconstruction | Right: Ground Truth"
]
cmaps = ["inferno", "inferno"]
clims = [elec_clim, mass_clim]

ims = []
for ax, img, title, cmap, clims in zip(axs, images, titles, cmaps, clims):
    im = ax.imshow(img, cmap=cmap, clim=clims)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ims.append(im)

    scalebar = ScaleBar(
        dx=dx_eff * 1e6,   # microns per pixel
        units='µm',
        fixed_value=25,
        location='upper right',
        height_fraction=0.025,
        box_alpha=1,
        color='black',
        font_properties={"size": 10}
    )
    ax.add_artist(scalebar)


# # ----------------
# # SUBPLOT LABELS
# # ----------------
# labels = ["(a)", "(b)"]
# for ax, lab in zip(axs, labels):
#     ax.text(
#         0.03, 0.96, lab,
#         transform=ax.transAxes,
#         fontsize=13,
#         fontweight="bold",
#         color="black",
#         va="top",
#         ha="left"
#     )

# ----------------
# COLORBARS
# ----------------
for i, (ax, im) in enumerate(zip(axs, ims)):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)

    if i == 0:
        cbar.set_label(
            r"Electron Density ($\mathbf{10^{20}}$ e$^-$/cm$\mathbf{^3}$)",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 1:
        cbar.set_label(
            r"Mass Density (g/cm$\mathbf{^3}$)",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 2:
        cbar.set_label(
            r"|$\Delta$| Electron Density",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 3:
        cbar.set_label(
            r"|$\Delta$| Mass Density",
            fontsize=10,
            fontweight="bold"
        )

fig.tight_layout()
plt.savefig(f"/Users/danielhodge/Desktop/run{run_holo}_{type}_3D_density_plots.pdf",
            dpi=300,
            transparent=False)
plt.show()


# ----------------
# SECOND FIGURE: averaged-left composite
# left = average(actual left, flipped right)
# right = ground truth
# ----------------
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
axs2 = axs2.ravel()

images2 = [electron_density_recon_comp, mass_density_recon_comp]
titles2 = [
    "Electron Density\nLeft: Recon| Right: Ground Truth",
    "Mass Density\nLeft: Recon | Right: Ground Truth"
]
cmaps2 = ["inferno", "inferno"]
clims2 = [elec_clim, mass_clim]

ims2 = []
for ax, img, title, cmap, clim in zip(axs2, images2, titles2, cmaps2, clims2):
    im = ax.imshow(img, cmap=cmap, clim=clim)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ims2.append(im)

    scalebar = ScaleBar(
        dx=dx_eff * 1e6,
        units='µm',
        fixed_value=25,
        location='upper right',
        height_fraction=0.025,
        box_alpha=1,
        color='black',
        font_properties={"size": 10}
    )
    ax.add_artist(scalebar)

for i, (ax, im) in enumerate(zip(axs2, ims2)):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig2.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)

    if i == 0:
        cbar.set_label(
            r"Electron Density ($\mathbf{10^{20}}$ e$^-$/cm$\mathbf{^3}$)",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 1:
        cbar.set_label(
            r"Mass Density (g/cm$\mathbf{^3}$)",
            fontsize=10,
            fontweight="bold"
        )

fig2.tight_layout()
plt.savefig(
    f"/Users/danielhodge/Desktop/run{run_holo}_{type}_3D_density_plots_avgleft_rightgt.pdf",
    dpi=300,
    transparent=False
)
plt.show()

# imwrite("/Users/danielhodge/Desktop/elec_dens_comp.tiff", electron_density_recon_comp)

# # ------------------------------------------------------------------------------------------------
# # PLOT
# # ------------------------------------------------------------------------------------------------
#
# electron_density_recon_comp = make_half_composite1(electron_density_recon, electron_density_gt)
# mass_density_recon_comp = make_half_composite1(mass_density_recon, mass_density_gt)
#
# plt.figure()
# plt.imshow(electron_density_recon_comp, clim=(0, 1.25 * np.max(electron_density_gt)), cmap='inferno')
# plt.savefig(
#     f"/Users/danielhodge/Desktop/run{run_holo}_{type}_electron_density_composite.pdf",
#     dpi=300,
#     transparent=False
# )
# plt.show()











############################################ Need supercomputer for the remainder ###########################

# f = loss.SquaredL2Loss(y=y, A=A)
# lamm = 1.0  # L1 norm regularization parameter
# g = lamm * functional.L1Norm()
# C = linop.FiniteDifference(input_shape=y.shape)

# rho = 20  # ADMM penalty parameter
# maxiter = 10  # number of ADMM iterations
# cg_tol = 1e-6  # CG relative tolerance
# cg_maxiter = 25  # maximum CG iterations per ADMM iteration
#
# solver = ADMM(
#     f=f,
#     g_list=[g],
#     C_list=[C],
#     rho_list=[rho],
#     x0=x_inv,
#     maxiter=maxiter,
#     subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
#     itstat_options={"display": True, "period": 10},
# )
#
# print(f"Solving on {device_info()}\n")
# solver.solve()
# x_tv = solver.x
#
# x_tv = cropToCenter(img=x_tv,
#                     newSize=end_size)
#
# plt.figure()
# plt.imshow(x_tv, clim=(0, 1000))
# plt.show()

# imwrite("/Users/danielhodge/Desktop/inverse_Abel_data_regularized.tiff", x_tv)






























# electron_density_total_recon = abel.Transform(projected_electron_density_total,
#                                               direction='inverse',
#                                               method='daun',
#                                               symmetry_axis=0,
#                                               symmetrize_method='fourier',
#                                               transform_options=dict(reg=100, dr=cmPerPx / scale_factor),
#                                               recast_as_float64=True).transform
# # OR if we make dr=1, then we use the below
# electron_density_total_recon = electron_density_total_recon #* num_elec * (1e7 ** 2) / cmPerPx / scale_factor / 10e20


# # This must be manually tuned after output. The Abel transform does not know the absolute 0 so we adjust the
# # image globally based off a known value that is unperturbed by the shockwave. Additionally, we know from the
# # ground truth that inside the void should be ~0 e-/cm^3 and/or the SU8 should be ~390 x 10^20 e-/cm^3.
# # electron_density_total_recon -= 67
# # electron_density_total_recon[electron_density_total_recon < 0] = 0
# electron_density_total_recon = cropToCenter(img=electron_density_total_recon, newSize=end_size)
# plt.figure()
# plt.imshow(electron_density_total_recon, clim=(0, 1000))
# plt.show()





import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift

run_holo = "572"
type = "sim"
# for run 574, we shifted +25 pixels in y, for run 576 we shifted -54.99983132112288 in x,
shift_y = 0
shift_x = 25.000069739608307

# ----------------
# COLOR LIMITS
# adjust these as needed
# ----------------
elec_clim = (0, 2000)
mass_clim = (0, 6.5)

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_abel = "Inverse_Abel/"
dir_electron_density = "run" + run_holo + "_" + type + "/"
dir_gt_maps = "GT_adjusted_maps/"
dir_run_gt_maps = "run" + run_holo + "_GT_maps/"

# Reconstructed electron density (e-/cm^3) from which we will calculate the mass density (assuming SU8 only)
# electron_density_path = (dir_main + dir_abel + dir_electron_density + "run_" + run_holo + "_" + type +
#                          "_inverse_Abel_data_regularized_Gaussian_blurred8.tiff")
electron_density_path = "/Users/danielhodge/Desktop/run_572_sim_inverse_Abel_data_regularized_nodeconvolve_Gaussian_blurred8.tiff"
# Ground truths for the electron density and mass density from xRAGE
electron_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "electron_density_total_GT_adjusted.tiff"
mass_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "density_total_GT_adjusted.tiff"


# ----------------
# LOAD IMAGES
# ----------------
electron_density_recon = np.array(imread(electron_density_path), dtype=np.float32)
electron_density_recon -= 60
electron_density_recon[electron_density_recon < 0] = 0
electron_density_gt = np.array(imread(electron_density_gt_path), dtype=np.float32)
mass_density_gt = np.array(imread(mass_density_gt_path), dtype=np.float32)

# print(electron_density_recon.shape)
# print(electron_density_gt.shape)
# print(mass_density_gt.shape)

# ----------------
# COMPUTE THE RECONSTRUCTED MASS DENSITY FROM THE RECONSTRUCTED ELECTRON DENSITY
# ----------------
N_A = 6.022e23  # Avogadro's number in mol^-1
elec_fac = 10e20  # This is what I scaled the electron density by -- 10e-20 * e-/cm^3 (for better plotting)
print("Calculating and plotting the areal mass density map assuming purely SU-8 material...")
# We assume only SU8 with chemical composition: C87 H118 O16, the same used for XPCI forward modeling
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16  # In g/mol
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8  # Unitless
mass_density_recon = electron_density_recon * elec_fac * A_SU8 / (N_A * Z_SU8)

# ----------------
# MAKE LEFT/RIGHT COMPOSITES
# left half = reconstruction
# right half = ground truth
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp


electron_density_gt = shift(electron_density_gt, shift=(shift_y, shift_x), order=3, mode='nearest')
mass_density_gt = shift(mass_density_gt, shift=(shift_y, shift_x), order=3, mode='nearest')

elec_comp = make_half_composite(electron_density_recon, electron_density_gt)
mass_comp = make_half_composite(mass_density_recon, mass_density_gt)


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

images = [elec_comp, mass_comp]
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
        box_alpha=0,
        color='cyan',
        font_properties={"size": 10}
    )
    ax.add_artist(scalebar)


# ----------------
# SUBPLOT LABELS
# ----------------
labels = ["(a)", "(b)"]
for ax, lab in zip(axs, labels):
    ax.text(
        0.03, 0.96, lab,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color="cyan",
        va="top",
        ha="left"
    )

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
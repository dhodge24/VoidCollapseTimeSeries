import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift

run_holo = "590"
type = "exp"
# for run 574, we shifted +25 pixels in y and x, for run 576 we shifted -54.99983132112288 in x,
shift_y = 0
shift_x = -49.99998557735444

# ----------------
# COLOR LIMITS
# adjust these as needed
# ----------------
proj_elec_clim = (5, 30)
areal_clim = (0.01, 0.1)

color_letters = "black"
color_scale_bar = "white"

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_recons = "time_series_recons_cropped/"
dir_runs = "run" + run_holo + "_" + type + "/"
dir_gt_maps = "GT_adjusted_maps/"
dir_run_gt_maps = "run" + run_holo + "_GT_maps/"


phase_path = dir_main + dir_recons + dir_runs + "run" + run_holo + "_" + type + "_ph_final.tiff"
# Ground truths for the electron density and mass density from xRAGE
projected_electron_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "projected_electron_density_total_adjusted.tiff"
areal_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "areal_density_total_adjusted.tiff"


# ----------------
# LOAD IMAGES
# ----------------
phase = np.array(imread(phase_path), dtype=np.float32)
projected_electron_density_gt = np.array(imread(projected_electron_density_gt_path), dtype=np.float32)
areal_density_gt = np.array(imread(areal_density_gt_path), dtype=np.float32)

# ----------------
# COMPUTE THE RECONSTRUCTED MASS DENSITY FROM THE RECONSTRUCTED ELECTRON DENSITY
# ----------------
Ny, Nx = phase.shape
E = 18000  # Initial energy of the beam in eV
lam = (1240 / E) * 1e-9  # Wavelength
z01 = 120.41e-3  # Distance from source to sample
z12 = 4.668995  # Distance from sample to detector
z02 = z01 + z12  # Distance from source to detector
M = z02 / z01  # Magnification
z_eff = z12 / M  # Effective propagation distance
scale_fac = 4  # Lens magnification factor at scintillator
det_pixel_size = 6.5e-6  # Detector pixel size
dx_eff = det_pixel_size / M / scale_fac  # Effective pixel size in x
dy_eff = det_pixel_size / M / scale_fac  # Effective pixel size in y
extent_x = Nx * dx_eff  # Object domain length in x
extent_y = Ny * dy_eff  # Object domain length in y


E = 18000  # Energy of the x-ray beam in eV
c = 2.9979e8  # Speed of light in m/s
m_e = 9.1094e-31  # Electron mass in kg
eps0 = 8.852e-12  # Permittivity of free space in units C^2 / (N * m^2)
e = 1.6022e-19  # Charge of an electron in C
r_e = 2.82e-15  # Classical electron radius in meters
N_A = 6.022e23  # Avogadro's number in mol^-1
m_to_nm = 1e-9  # To scale our values in nm^2
num_elec = 10e6  # Arbitrary value to make our plots scale nicely
print("Calculating and plotting the projected electron density map...")
n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e**2
projected_electron_density_recon = -phase * lam * n_c / np.pi * m_to_nm**2 / num_elec

m_to_cm = 1e-2
projected_electron_density_recon_cm = -phase * lam * n_c / np.pi * m_to_cm**2
print("Calculating and plotting the areal mass density map assuming purely SU-8 material...")
# We assume only SU8 with chemical composition: C87 H118 O16, the same used for XPCI forward modeling
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16  # In g/mol
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8  # Unitless
areal_density_recon = projected_electron_density_recon_cm * A_SU8 / (N_A * Z_SU8)

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

projected_electron_density_gt = shift(projected_electron_density_gt, shift=(shift_y, shift_x), order=3, mode='nearest')
areal_density_gt = shift(areal_density_gt, shift=(shift_y, shift_x), order=3, mode='nearest')

proj_elec_comp = make_half_composite(projected_electron_density_recon, projected_electron_density_gt)
areal_comp = make_half_composite(areal_density_recon, areal_density_gt)


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

images = [proj_elec_comp, areal_comp]
titles = [
    "Projected Electron Density\nLeft: Reconstruction | Right: Ground Truth",
    "Areal Density\nLeft: Reconstruction | Right: Ground Truth"
]
cmaps = ["jet", "jet"]
clims = [proj_elec_clim, areal_clim]

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
        color=color_scale_bar,
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
        color=color_letters,
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
            r"Projected Electron Density ($\mathbf{10^{6}}$ e$^-$/nm$\mathbf{^2}$)",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 1:
        cbar.set_label(
            r"Areal Density (g/cm$\mathbf{^2}$)",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 2:
        cbar.set_label(
            r"|$\Delta$| Projected Electron Density",
            fontsize=10,
            fontweight="bold"
        )
    elif i == 3:
        cbar.set_label(
            r"|$\Delta$| Areal Density",
            fontsize=10,
            fontweight="bold"
        )

fig.tight_layout()
plt.savefig(f"/Users/danielhodge/Desktop/run{run_holo}_{type}_2D_density_plots.pdf",
            dpi=300,
            transparent=False)
plt.show()
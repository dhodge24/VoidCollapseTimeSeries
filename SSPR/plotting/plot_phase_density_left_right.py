import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift

# ----------------
# RUNS / DELAYS
# ----------------
runs = ["572", "576", "580", "582", "590"]
delay_times = [6.82, 8.12, 9.67, 10.52, 13.42]

# ----------------
# OPTIONAL SHIFTS FOR GT ALIGNMENT
# ----------------
gt_shifts = {
    "572": (0, 25.000069739608307),
    "576": (0, -54.99983132112288),
    "580": (0, 75.49991558517377),
    "582": (0, 50.4996321295979),
    "590": (0, -49.99998557735444),
}

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_data = "time_series_recons_cropped/"

# ----------------
# CONSTANTS
# ----------------
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac

# ----------------
# PHASE -> PROJECTED ELECTRON DENSITY CONVERSION
# ----------------
E = 18000  # eV
lam = (1240 / E) * 1e-9
c = 2.9979e8
m_e = 9.1094e-31
eps0 = 8.852e-12
e = 1.6022e-19
m_to_nm = 1e-9
num_elec = 10e6

n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e**2
ped_scale = lam * n_c / np.pi * m_to_nm**2 / num_elec

def phase_to_ped(phase_val):
    return -phase_val * ped_scale

def ped_to_phase(ped_val):
    return -ped_val / ped_scale

# ----------------
# HELPER: MAKE HALF COMPOSITE
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp

# ----------------
# HELPER: DUAL COLORBAR
# left side = projected electron density
# right side = phase
# ----------------
def add_dual_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)

    # tighter colorbar block so columns can sit closer together
    cax = divider.append_axes("right", size="3.2%", pad=0.48)

    # Right side: phase
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(axis="y", labelsize=6.5, pad=1)

    cbar.set_label(
        "Phase (rad)",
        fontsize=8.0,
        fontweight="bold",
        labelpad=2
    )

    # Left side: projected electron density
    secax = cbar.ax.secondary_yaxis(
        "left",
        functions=(phase_to_ped, ped_to_phase)
    )
    secax.yaxis.set_ticks_position("left")
    secax.yaxis.set_label_position("left")
    secax.tick_params(axis="y", labelsize=6.5, pad=1)

    # Negative labelpad moves the label closer to the colorbar
    secax.set_ylabel(
        r"Projected Electron Density ($\mathbf{10^{6}}$ e$^-$/nm$\mathbf{^2}$)",
        fontsize=8.0,
        fontweight="bold",
        labelpad=-1
    )

    return cbar, secax

# ----------------
# LOAD DATA AND BUILD COMPOSITES
# ----------------
sim_composites = []
exp_composites = []

for run in runs:
    dir_sim = dir_main + dir_data + f"run{run}_sim/"
    dir_exp = dir_main + dir_data + f"run{run}_exp/"

    gt_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_gt_final.tiff"), dtype=np.float32)
    sim_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_final.tiff"), dtype=np.float32)
    exp_phase = np.array(imread(dir_exp + f"run{run}_exp_ph_final.tiff"), dtype=np.float32)

    shift_y, shift_x = gt_shifts.get(run, (0, 0))
    gt_phase_shifted = shift(gt_phase, shift=(shift_y, shift_x), order=3, mode="nearest")

    sim_composites.append(make_half_composite(sim_phase, gt_phase_shifted))
    exp_composites.append(make_half_composite(exp_phase, gt_phase_shifted))

# ----------------
# FIGURE CONTENT
# ----------------
images_grid = [[sim_composites[i], exp_composites[i]] for i in range(len(runs))]
phase_clim = (-56, -15)
col_titles = ["Sim vs xRAGE", "Exp vs xRAGE"]

# ----------------
# CREATE FIGURE
# ----------------
nrows, ncols = 5, 2
fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(10.0, 15),
    constrained_layout=False
)
axs = np.atleast_2d(axs)

for r in range(nrows):
    for c in range(ncols):
        ax = axs[r, c]

        im = ax.imshow(
            images_grid[r][c],
            cmap="RdBu_r",
            vmin=phase_clim[0],
            vmax=phase_clim[1],
            interpolation="none",
            aspect="equal"
        )

        ax.set_axis_off()

        if r == 0:
            ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=18)

        ax.text(
            0.25, 1.01, "Recon",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

        ax.text(
            0.75, 1.01, "xRAGE GT",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

        ax.text(
            0.50, 1.01, "|",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

        if c == 0:
            ax.text(
                -0.10, 0.5,
                f"X-Ray Delay\n{delay_times[r]} ns",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold"
            )

            scalebar = ScaleBar(
                dx=dx_eff * 1e6,
                units="µm",
                fixed_value=25,
                location="upper right",
                height_fraction=0.012,
                width_fraction=0.04,
                box_alpha=1.0,
                pad=0.02,
                border_pad=0.02,
                sep=1.0,
                color="black",
                font_properties={"size": 10}
            )
            ax.add_artist(scalebar)

        add_dual_colorbar(fig, ax, im)

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.17,
    right=0.965,
    top=0.95,
    bottom=0.04,
    wspace=0.10,
    hspace=0.08
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x2_phase_sim_exp_composite_dual_colorbar.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()
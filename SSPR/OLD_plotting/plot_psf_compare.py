import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import signal

# My own imports
from utilities import cropToCenter, create_circular_mask


Ny, Nx = 2500, 2500
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


run_wfs = "561"
run_holo = "572"
type = "exp"

dir_main = "/Users/danielhodge/Desktop/"
dir_folder1 = "TimeSeriesDataAndParams/"
dir_folder2 = "571_void_params_output/"
dir_holo_preprocessed = "run" + run_holo + "_" + type + "_preprocessed/"
dir_wfs_preprocessed = "run" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo + "_exp_preprocessed/"


run_sim = np.array(imread(dir_main + dir_folder1 + dir_folder2 + "testsim_571.tiff"))
run_exp = np.array(imread(dir_main + dir_folder1 + dir_folder2 + "testexp_571.tiff"))

# Get center indices
center_x, center_y = run_sim.shape[1] // 2, run_sim.shape[0] // 2

start_y = 200
end_y = 2310

start_x = 400
end_x = 2100

midpoint_y = (start_y + end_y) // 2  # Compute the center of the extracted range
midpoint_x = (start_y + end_y) // 2  # Compute the center of the extracted range

# Compute centered x and y positions in microns (µm)
y_positions = (np.arange(start_y, end_y) - midpoint_y) * dy_eff * 1e6  # Center at 0
x_positions = (np.arange(start_x, end_x) - midpoint_x) * dx_eff * 1e6  # Center at 0

specific_row = 1800  # was center_y
# Extract horizontal and vertical lineouts
horizontal_lineout_exp = run_exp[specific_row, start_x:end_x]  # Middle row (horizontal slice)
vertical_lineout_exp = run_exp[start_y:end_y, center_x]  # Middle column (vertical slice)
horizontal_lineout_sim = run_sim[specific_row, start_x:end_x]  # Middle row (horizontal slice)
vertical_lineout_sim = run_sim[start_y:end_y, center_x]  # Middle column (vertical slice)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle("Experiment vs. Simulation with Gaussian PSF $\sigma$ = 462.88 nm", fontsize=16, fontweight='bold')


axs[0, 0].imshow(run_exp, cmap="Greys_r", clim=(0, 2))
axs[0, 0].set_title("FFC Static Exp XPC Image", size=14, fontweight='bold')
axs[0, 0].plot([start_x, end_x], [specific_row, specific_row], color='blue', linestyle='--', linewidth=3)
axs[0, 0].plot([center_x, center_x], [start_y, end_y], color='black', linestyle='--', linewidth=3)
axs[0, 0].axis('off')

axs[0, 1].imshow(run_sim, cmap="Greys_r", clim=(0, 2))
axs[0, 1].set_title("FFC Static Sim XPC Image", size=14, fontweight='bold')
axs[0, 1].plot([start_x, end_x], [specific_row, specific_row], color='magenta', linestyle='--', linewidth=3)
axs[0, 1].plot([center_x, center_x], [start_y, end_y], color='red', linestyle='--', linewidth=3)
axs[0, 1].axis('off')

# Plot Vertical Lineout (along y-axis)
axs[1, 0].plot(y_positions, vertical_lineout_exp, color='black', label="Experiment")
axs[1, 0].plot(y_positions, vertical_lineout_sim, color='red', label="Simulation")
axs[1, 0].set_title("Vertical Lineout", size=14, fontweight='bold')
axs[1, 0].set_ylabel("Intensity", size=12, fontweight='bold')
axs[1, 0].set_xlabel("Y Position [$\mathbf{\mu}$m]", size=12, fontweight='bold')
axs[1, 0].legend(loc="upper right")  # loc was "best"

# Plot Horizontal Lineout (along x-axis)
axs[1, 1].plot(x_positions, horizontal_lineout_exp, color='blue', label="Experiment")
axs[1, 1].plot(x_positions, horizontal_lineout_sim, color='magenta', label="Simulation")
axs[1, 1].set_title("Horizontal Lineout", size=14, fontweight='bold')
axs[1, 1].set_ylabel("Intensity", size=12, fontweight='bold')
axs[1, 1].set_xlabel("X Position [$\mathbf{\mu}$m]", size=12, fontweight='bold')
axs[1, 1].legend(loc="upper right")  # loc was "best"


# Define subplot labels
labels = ['(a)', '(b)', '(c)', '(d)']
x_pos, y_pos = 0.025, 0.95
axs[0, 0].text(x_pos,
               y_pos,
               labels[0],
               transform=axs[0, 0].transAxes,
               fontsize=16,
               fontweight="bold",
               va="top",
               ha="left",
               color="white")
axs[0, 1].text(x_pos,
               y_pos,
               labels[1],
               transform=axs[0, 1].transAxes,
               fontsize=16,
               fontweight="bold",
               va="top",
               ha="left",
               color="white")
axs[1, 0].text(x_pos,
               y_pos,
               labels[2],
               transform=axs[1, 0].transAxes,
               fontsize=16,
               fontweight="bold",
               va="top",
               ha="left")
axs[1, 1].text(x_pos,
               y_pos,
               labels[3],
               transform=axs[1, 1].transAxes,
               fontsize=16,
               fontweight="bold",
               va="top",
               ha="left")

# ----------------
# SCALE BARS
# ----------------
for ax in [axs[0, 0], axs[0, 1]]:
    scalebar = ScaleBar(dx=dx_eff,
                        units='m',
                        location='upper right',
                        length_fraction=0.3,
                        height_fraction=0.025,
                        box_alpha=0,
                        font_properties={"size": 16},
                        color='white')
    ax.add_artist(scalebar)


plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/psfCompare.pdf', dpi=300, transparent=False)
plt.show()

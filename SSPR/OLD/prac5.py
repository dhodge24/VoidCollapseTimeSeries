from SSPR.utilities import rotateImage, shiftRotateMagnifyImage, cropToCenter, create_circular_mask
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Experimental parameters
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

run_holo = "576"
type = "sim"

dir_main = "/Users/danielhodge/Desktop/"
dir_recons = "time_series_new_recons/"
dir_sim = "run" + run_holo + "_" + type + "/"

img = np.array(imread(dir_main + dir_recons + dir_sim + "ph_final_run" + run_holo + type + "2.tiff"), dtype=np.float32)
ph_gt = np.array(imread(dir_main + dir_recons + "run" + run_holo + "_" + type + "_phase.tiff")[0], dtype=np.float32)


# Get center indices
center_x, center_y = ph_gt.shape[1] // 2, ph_gt.shape[0] // 2

start = 1100
end = 1800

midpoint = (start + end) // 2  # Compute the center of the extracted range
# Compute centered x and y positions in microns (um)
y_positions = (np.arange(start, end) - midpoint) * dy_eff * 1e6  # Center at 0

vertical_lineout_gt0 = ph_gt[start:end, center_x]  # Middle column (vertical slice)


thickness = 5
half = thickness // 2
# Non-overlapping, adjacent 5-col windows
offsets = np.array([-20, -10, 0, 10, 20]) * thickness  # windows are [start, start+5)
starts = (center_x - half) + offsets
lineouts_1d = []
lineouts_gt_1d = []
for s in starts:
    cols = slice(s, s + thickness)          # exactly 5 columns, disjoint by construction
    block = img[start:end, cols]            # (end-start, 5)
    block_gt = ph_gt[start:end, cols]
    lineouts_gt_1d.append(block_gt.mean(axis=1))
    lineouts_1d.append(block.mean(axis=1))  # (end-start,)
lineouts_gt_5 = np.stack(lineouts_gt_1d, axis=0)
lineout_gt_avg = np.mean(lineouts_gt_5, axis=0)
lineouts_5 = np.stack(lineouts_1d, axis=0)  # shape: (5, end-start)
lineout_avg = np.mean(lineouts_5, axis=0)


vertical_lineout_sim0 = img[start:end, center_x]  # Middle column (vertical slice)

colors = ["tab:purple", "tab:blue", "tab:red", "tab:green", "tab:orange"]
labels = ["Left 2", "Left 1", "Center", "Right 1", "Right 2"]

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(img, cmap="Greys_r")

for s, c, lbl in zip(starts, colors, labels):
    # rectangle
    rect = Rectangle(
        (s, start),
        thickness,
        end - start,
        fill=False,
        linewidth=2,
        edgecolor='black'
    )
    ax.add_patch(rect)

    # colored center line
    x_center = s + thickness / 2
    ax.vlines(
        x_center,
        ymin=start,
        ymax=end,
        color=c,
        linewidth=3,
        label=lbl
    )

ax.set_title("2D Image Lineouts")
ax.set_axis_off()
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/tempPlot1', bbox_inches='tight', dpi=300, transparent=False)
plt.show()

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # 1 row, 2 columns

# plot single line down middle
axs[0].plot(y_positions, vertical_lineout_gt0, color='black', label="Ground Truth")
# axs[0].plot(y_positions, lineout_gt_avg, color='black', label="Ground Truth")
axs[0].plot(y_positions, vertical_lineout_sim0, color='blue', label="Simulation - Single Column")
axs[0].set_title("Single Vertical Lineout -- Simulation", size=14, fontweight='bold')
axs[0].set_ylabel("Phase (rad)", size=12, fontweight='bold')
axs[0].set_xlabel("y Position [um]", size=12, fontweight='bold')
axs[0].legend(loc="best")

# # Plot Vertical Lineout Average of 5 columns
# axs[1].plot(y_positions, vertical_lineout_gt0, color='black', label="Ground Truth")
# axs[1].plot(y_positions, lineouts_1d, color='red', label="Simulation - Averaged Columns")
# axs[1].set_title("Multiple Vertical Lineouts Averaged -- Simulation", size=14, fontweight='bold')
# axs[1].set_ylabel("Phase (rad)", size=12, fontweight='bold')
# axs[1].set_xlabel("y Position [um]", size=12, fontweight='bold')
# axs[1].legend(loc="best")


# Plot Vertical Lineout Average + the 5 individual lineouts
axs[1].plot(y_positions, lineout_gt_avg, color='black', label="Ground Truth")

# plot each of the 5 lineouts
labels = ["Left2", "Left1", "Center", "Right1", "Right2"]
for i in range(lineouts_5.shape[0]):
    axs[1].plot(y_positions, lineouts_5[i], linestyle='--', linewidth=1.5, alpha=0.8,
                label=f"Sim {labels[i]}")

# plot their average on top
axs[1].plot(y_positions, lineout_avg, color='red', linewidth=3,
            label="Simulation - Mean of 5 lineouts")

axs[1].set_title("Multiple Vertical Lineouts + Mean -- Simulation", size=14, fontweight='bold')
axs[1].set_ylabel("Phase (rad)", size=12, fontweight='bold')
axs[1].set_xlabel("y Position [um]", size=12, fontweight='bold')
axs[1].legend(loc="best")
for i, s in enumerate(starts):
    axs[1].plot(y_positions, lineouts_5[i], linestyle='--', linewidth=1.5, alpha=0.8,
                label=f"Sim cols {s}:{s+thickness-1}")



# Define subplot labels
labels = ['(a)', '(b)']
x_pos, y_pos = -0.1, 1.05
axs[0].text(x_pos,
               y_pos,
               labels[0],
               transform=axs[0].transAxes,
               fontsize=16,
               fontweight="bold",
               va="top",
               ha="left")
axs[1].text(x_pos,
               y_pos,
               labels[1],
               transform=axs[1].transAxes,
               fontsize=16,
               fontweight="bold",
               va="top",
               ha="left")
# plt.tight_layout()
# plt.show()

# # Improve layout
plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/tempPlot2', bbox_inches='tight', dpi=300, transparent=False)
plt.show()













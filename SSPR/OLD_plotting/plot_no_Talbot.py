import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def connect_inset_to_roi(ax, axins, roi, color="white", lw=1.5):
    x0, x1, y0, y1 = roi

    # ROI corners in parent data coords (imshow pixel coords)
    roi_tl = (x0, y0)
    roi_br = (x1, y1)

    # Inset corners: use inset bbox in display coords, convert to parent data coords
    fig = ax.figure
    fig.canvas.draw()  # make sure bboxes are up to date

    bbox_disp = axins.get_window_extent()  # display coords
    # corners in display coords
    inset_tl_disp = (bbox_disp.x0, bbox_disp.y1)
    inset_br_disp = (bbox_disp.x1, bbox_disp.y0)

    # convert display -> parent data
    inset_tl = ax.transData.inverted().transform(inset_tl_disp)
    inset_br = ax.transData.inverted().transform(inset_br_disp)

    # draw lines in parent axes, in data coords
    ax.plot([inset_tl[0], roi_tl[0]], [inset_tl[1], roi_tl[1]], color=color, lw=lw)
    ax.plot([inset_br[0], roi_br[0]], [inset_br[1], roi_br[1]], color=color, lw=lw)


def add_zoom_inset(ax, img, roi, clim=None, cmap="Greys_r",
                   inset_loc="lower left", inset_size="40%",
                   borderpad=1, rect_color="red", rect_lw=2,
                   connector=True, conn_lw=2):

    x0, x1, y0, y1 = roi

    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                     linewidth=rect_lw, edgecolor=rect_color, facecolor="none")
    ax.add_patch(rect)

    axins = inset_axes(ax, width=inset_size, height=inset_size,
                       loc=inset_loc, borderpad=borderpad)

    axins.imshow(img, cmap=cmap, clim=clim)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y1, y0)
    axins.axis("off")

    if connector:
        connect_inset_to_roi(ax, axins, roi, color=rect_color, lw=conn_lw)

    return axins


Ny, Nx = (2500, 2500)
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

x0, x1 = 1200, 1350   # columns
y0, y1 = 1000, 1150    # rows
roi = (x0, x1, y0, y1)

rect = Rectangle(
    (x0, y0), x1 - x0, y1 - y0,
    linewidth=2, edgecolor='white', facecolor='none'
)

run_wfs = "561"
run_holo = "572"
type = "exp"

dir_main = "/Users/danielhodge/Desktop/"
dir_holo_preprocessed = "run" + run_holo + "_" + type + "_preprocessed/"
dir_wfs_preprocessed = "run" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo + "_exp_preprocessed/"

wfs_orig = np.array(imread(dir_main + dir_wfs_preprocessed + "run561_exp_evt_1_preprocessed.tiff"), dtype=np.float32)
wfs_no_Talbot = np.array(imread(dir_main + dir_wfs_preprocessed + "run561_exp_preprocessed_no_Talbot1.tiff"), dtype=np.float32)
holo_orig = np.array(imread(dir_main + dir_holo_preprocessed + "run572_exp_preprocessed.tiff"), dtype=np.float32)[0]
holos_no_Talbot = np.array(imread(dir_main + dir_holo_preprocessed + "run572_exp_preprocessed_no_Talbot.tiff"), dtype=np.float32)[0]

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Plot images with colorbars
im0 = axs[0, 0].imshow(wfs_orig, clim=(0, 1800), cmap="Greys_r")
axs[0, 0].set_title('White Field BG Corrected', size=16, loc='center', fontweight='bold')
add_zoom_inset(axs[0, 0], wfs_orig, roi, clim=(0, 1800), inset_loc="lower left")

im1 = axs[0, 1].imshow(holo_orig, clim=(0, 600), cmap="Greys_r")
axs[0, 1].set_title('Void Image BG Corrected', size=16, fontweight='bold')
add_zoom_inset(axs[0, 1], holo_orig, roi, clim=(0, 600), inset_loc="lower left")

im2 = axs[1, 0].imshow(wfs_no_Talbot, clim=(0, 1800), cmap="Greys_r")
axs[1, 0].set_title('White Field Talbot Removed', size=16, fontweight='bold')
add_zoom_inset(axs[1, 0], wfs_no_Talbot, roi, clim=(0, 1800), inset_loc="lower left")

im3 = axs[1, 1].imshow(holos_no_Talbot, clim=(0, 600), cmap="Greys_r")
axs[1, 1].set_title('Void Image Talbot Removed', size=16, fontweight='bold')
add_zoom_inset(axs[1, 1], holos_no_Talbot, roi, clim=(0, 600), inset_loc="lower left")

# Remove axes
for ax in axs.flatten():
    ax.axis("off")

# Define subplot labels
labels = ['(a)', '(b)', '(c)', '(d)']
x_pos, y_pos = 0.025, 0.95
for i, ax in enumerate(axs.flatten()):
    ax.text(x_pos, y_pos, labels[i], transform=ax.transAxes,
            fontsize=16, fontweight="bold", color='white', va="top", ha="left")

# ----------------
# SCALE BARS
# ----------------
for ax in axs.flatten():
    scalebar = ScaleBar(dx=dx_eff, units='m', location='upper right',
                        length_fraction=0.3, height_fraction=0.025,
                        box_alpha=0, color='white', font_properties={"size": 16})
    ax.add_artist(scalebar)

# ----------------
# COLORBARS
# ----------------
for im, ax in zip([im0, im1, im2, im3], axs.flatten()):
    cbar_ax = ax.inset_axes([0.80, 0.09, 0.15, 0.04])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=10, width=1, length=2, direction="out", pad=2, color='white', labelcolor='white')
    cbar.outline.set_edgecolor('white')



plt.tight_layout()
# Do not use bbox_inches_tight. It ruins the colorbar color
plt.savefig('/Users/danielhodge/Desktop/BG_Talbot_remove.pdf', dpi=300, transparent=False)
plt.show()


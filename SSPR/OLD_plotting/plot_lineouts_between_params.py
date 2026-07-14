import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from utilities import cropToCenter, create_circular_mask
from scipy import signal
from skimage.filters import gaussian


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

# # Load the data
# ph_gt = imread("/Users/danielhodge/Library/CloudStorage/Box-Box/meclx4819DATA/run306_sim/run306_sim_phase.tiff").astype(np.float32)
# ph_ctf_sim = imread("/Users/danielhodge/Desktop/run306_sim_new/run306_sim_phase_CTFfADMM_masked.tiff").astype(np.float32)
# ph_pgd_sim = imread("/Users/danielhodge/Desktop/run306_sim_new/run306_sim_phase_PGD_masked.tiff").astype(np.float32)
# ph_ctf_exp = imread("/Users/danielhodge/Desktop/run306_exp_new/run306_exp_phase_CTFfADMM_masked.tiff").astype(np.float32)
# ph_pgd_exp = imread("/Users/danielhodge/Desktop/run306_exp_new/run306_exp_phase_PGD_masked.tiff").astype(np.float32)

ph_gt = imread("/Users/danielhodge/Library/CloudStorage/Box-Box/meclx4819DATA/run307_sim/run307_sim_phis.tiff")[0].astype(np.float32)
ph_ctf_sim = imread("/Users/danielhodge/Desktop/run307_sim_new/run307_sim_dynamic_ph_fADMM_CTF.tiff").astype(np.float32)
ph_pgd_sim = imread("/Users/danielhodge/Desktop/run307_sim_new/run307_sim_dynamic_ph_PGD.tiff").astype(np.float32)
ph_ctf_exp = imread("/Users/danielhodge/Desktop/run307_exp_new/run307_exp_dynamic_ph_fADMM_CTF.tiff").astype(np.float32)
ph_pgd_exp = imread("/Users/danielhodge/Desktop/run307_exp_new/run307_exp_dynamic_ph_PGD.tiff").astype(np.float32)



# run302/303
mask1 = create_circular_mask(size=2496, percentage=0.56, smooth_pixels=1)
mask1[mask1 > 0] = 1
mask2 = create_circular_mask(size=2500, percentage=0.56, smooth_pixels=1)
mask2[mask2 > 0] = 1
# ph_gt = imread("/Users/danielhodge/Library/CloudStorage/Box-Box/meclx4819DATA/run302_sim/run302_sim_phase.tiff").astype(np.float32) * mask2
# ph_ctf_sim = imread("/Users/danielhodge/Desktop/run302_sim_new/run302_sim_static_ph_fADMM_CTF.tiff").astype(np.float32) * mask2
# ph_pgd_sim = imread("/Users/danielhodge/Desktop/run302_sim_new/run302_sim_static_ph_PGD.tiff").astype(np.float32) * mask2
# ph_ctf_exp = imread("/Users/danielhodge/Desktop/run302_exp_new/run302_exp_static_ph_fADMM_CTF.tiff").astype(np.float32) * mask1
# ph_pgd_exp = imread("/Users/danielhodge/Desktop/run302_exp_new/run302_exp_ph_PGD.tiff").astype(np.float32) * mask2

# ph_gt = imread("/Users/danielhodge/Library/CloudStorage/Box-Box/meclx4819DATA/run303_sim/run303_sim_phi.tiff").astype(np.float32) * mask2
# ph_ctf_sim = imread("/Users/danielhodge/Desktop/run303_sim_new/run303_sim_dynamic_ph_fADMM_CTF.tiff").astype(np.float32) * mask2
# ph_pgd_sim = imread("/Users/danielhodge/Desktop/run303_sim_new/run303simph_other.tiff").astype(np.float32) * mask2
# ph_ctf_exp = imread("/Users/danielhodge/Desktop/run303_exp_new/run303_exp_dynamic_ph_fADMM_CTF.tiff").astype(np.float32) * mask2
# ph_pgd_exp = imread("/Users/danielhodge/Desktop/run303_exp_new/ph_best_3.tiff").astype(np.float32) * mask2


# ph_gt = imread("/Users/danielhodge/Library/CloudStorage/Box-Box/meclx4819DATA/run307_sim/run307_sim_phis.tiff")[0].astype(np.float32) * mask2
# ph_ctf_sim = imread("/Users/danielhodge/Desktop/run307_sim_new/run307_sim_dynamic_ph_fADMM_CTF.tiff").astype(np.float32) * mask2
# ph_pgd_sim = imread("/Users/danielhodge/Desktop/run307_sim_new/run307_sim_dynamic_ph_PGD.tiff").astype(np.float32) * mask2
# ph_ctf_exp = imread("/Users/danielhodge/Desktop/run307_exp_new/run307_exp_dynamic_ph_fADMM_CTF.tiff").astype(np.float32) * mask2
# ph_pgd_exp = imread("/Users/danielhodge/Desktop/run307_exp_new/run307_exp_dynamic_ph_PGD.tiff").astype(np.float32) * mask2


ph_ctf_sim = cropToCenter(img=ph_ctf_sim, newSize=[1600, 1600])
ph_pgd_sim = cropToCenter(img=ph_pgd_sim, newSize=[1600, 1600])
ph_ctf_exp = cropToCenter(img=ph_ctf_exp, newSize=[1600, 1600])
ph_pgd_exp = cropToCenter(img=ph_pgd_exp, newSize=[1600, 1600])

ph_pgd_sim = gaussian(ph_pgd_sim, sigma=15 / 2.35, truncate=2)
ph_pgd_exp = gaussian(ph_pgd_exp, sigma=15 / 2.35, truncate=2)
ph_pgd_sim = signal.medfilt2d(ph_pgd_sim, kernel_size=15)
ph_pgd_exp = signal.medfilt2d(ph_pgd_exp, kernel_size=15)

ph_ctf_sim = gaussian(ph_ctf_sim, sigma=15 / 2.35, truncate=2)
ph_ctf_exp = gaussian(ph_ctf_exp, sigma=15 / 2.35, truncate=2)
ph_ctf_sim = signal.medfilt2d(ph_ctf_sim, kernel_size=15)
ph_ctf_exp = signal.medfilt2d(ph_ctf_exp, kernel_size=15)

# ph_ctf_exp = gaussian(ph_ctf_exp, sigma=15 / 2.35, truncate=2)
# ph_ctf_exp = signal.medfilt2d(ph_ctf_exp, kernel_size=15)

mask = create_circular_mask(size=2500, percentage=0.56, smooth_pixels=1)
mask[mask > 0] = 1
ph_gt = ph_gt * mask
ph_gt = cropToCenter(img=ph_gt, newSize=[1600, 1600])

# Get center indices
center_x, center_y = ph_gt.shape[1] // 2, ph_gt.shape[0] // 2

start = 200
end = 1400

midpoint = (start + end) // 2  # Compute the center of the extracted range
# Compute centered x and y positions in microns (um)
y_positions = (np.arange(start, end) - midpoint) * dy_eff * 1e6  # Center at 0
x_positions = (np.arange(start, end) - midpoint) * dx_eff * 1e6  # Center at 0

specific_row = 800  # was center_y
# Extract horizontal and vertical lineouts
horizontal_lineout_gt = ph_gt[specific_row, start:end]  # Middle row (horizontal slice)
vertical_lineout_gt = ph_gt[start:end, center_x]  # Middle column (vertical slice)

# # CTF
horizontal_lineout_sim_ctf = ph_ctf_sim[specific_row, start:end]  # Middle row (horizontal slice)
vertical_lineout_sim_ctf = ph_ctf_sim[start:end, center_x]  # Middle column (vertical slice)
horizontal_lineout_exp_ctf = ph_ctf_exp[specific_row, start:end]  # Middle row (horizontal slice)
vertical_lineout_exp_ctf = ph_ctf_exp[start:end, center_x]  # Middle column (vertical slice)

# PGD
horizontal_lineout_sim_pgd = ph_pgd_sim[specific_row, start:end]  # Middle row (horizontal slice)
vertical_lineout_sim_pgd = ph_pgd_sim[start:end, center_x]  # Middle column (vertical slice)
horizontal_lineout_exp_pgd = ph_pgd_exp[specific_row, start:end]  # Middle row (horizontal slice)
vertical_lineout_exp_pgd = ph_pgd_exp[start:end, center_x]  # Middle column (vertical slice)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # 1 row, 2 columns

# Plot Vertical Lineout (along y-axis)
axs[0].plot(y_positions, vertical_lineout_gt, color='black', label="Ground Truth")
axs[0].plot(y_positions, vertical_lineout_exp_pgd, color='blue', label="Experiment - PGD")
axs[0].plot(y_positions, vertical_lineout_exp_ctf, color='purple', linestyle='--', label="Simulation - CTF-fADMM")
axs[0].set_title("Vertical Lineout -- Experiment", size=14, fontweight='bold')
axs[0].set_ylabel("Phase (rad)", size=12, fontweight='bold')
axs[0].set_xlabel("y Position [um]", size=12, fontweight='bold')
axs[0].legend(loc="best")

# Plot Horizontal Lineout (along x-axis)
axs[1].plot(x_positions, horizontal_lineout_gt, color='black', label="Ground Truth")
axs[1].plot(x_positions, horizontal_lineout_exp_pgd, color='red', label="Experiment - PGD")
axs[1].plot(x_positions, horizontal_lineout_exp_ctf, color='orange', linestyle='--', label="Simulation - CTF-fADMM")
axs[1].set_title("Horizontal Lineout -- Experiment", size=14, fontweight='bold')
axs[1].set_ylabel("Phase (rad)", size=12, fontweight='bold')
axs[1].set_xlabel("x Position [um]", size=12, fontweight='bold')
axs[1].legend(loc="best")

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

# Improve layout
plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/tempPlot', bbox_inches='tight', dpi=300, transparent=False)
plt.show()

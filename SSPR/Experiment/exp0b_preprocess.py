"""

Code is used for finding peaks in Fourier space that correspond to periodic structures in real space (Talbot grating).
We masked out those peak regions and perform an inverse Fourier transform to recover the object without the grating.
No need to use this code if the Talbot grating or some other periodic pattern is not present.

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import peak_local_max
from tqdm import tqdm
import os
from scipy.ndimage import center_of_mass
import shutil
from tifffile import imwrite, imread

from SSPR.utilities import FFT, IFFT


def clear_directory(dir_main, dir_sub):
    """
    Clear all files in the specified directory.
    """
    dir_path = os.path.join(dir_main, dir_sub)

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        # Delete all files in the directory if it already exists
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


save = True
# Do not need to remove it multiple times so after running this once, set below to False
preprocess_wfs = True  # Preprocessing action needs to happen only once since 1 batch of wfs were for many runs

run_wfs = "561"
run_holo = "582"

# Peak finding parameters - this is data dependent so parameters have to be tuned
min_distance = 5  # The minimal allowed distance separating the peaks
threshold_abs = 30.3  # Minimum intensity of peaks for them to be considered peaks, 26.25 for static, 29.21 for dyn
num_peaks = 100  # Find the closest Fourier peaks with subpixel accuracy, 24
center_exclusion_radius = 330  # Exclude peaks very close to the center, consider only 8 exterior peaks
region_size = 3  # Define a small region for peak refinement
box_radius = 50  # Define the box radius for making 0 mask in Fourier domain
wf_num = 1  # Ranges from 0 to however many white fields you have

# Directory where files are saved
dir_main = "/Users/danielhodge/Desktop/"
# dir_wfs_preprocessed = "run" + run_wfs + "_exp_preprocessed/"
dir_wfs_preprocessed = "run" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo + "_exp_preprocessed/"
dir_holo_preprocessed = "run" + run_holo + "_exp_preprocessed/"

# Files to import
tiffs_wfs_preprocessed = "run" + run_wfs + "_exp_preprocessed.tiff"
tiffs_holo_preprocessed = "run" + run_holo + "_exp_preprocessed.tiff"

# Save files
tiffs_wfs_preprocessed_no_Talbot = "run" + run_wfs + "_exp_preprocessed_no_Talbot.tiff"
tiffs_holo_preprocessed_no_Talbot = "run" + run_holo + "_exp_preprocessed_no_Talbot.tiff"

wfs = np.array(imread(dir_main + dir_wfs_preprocessed + tiffs_wfs_preprocessed), dtype=np.float32)
holos = np.array(imread(dir_main + dir_holo_preprocessed + tiffs_holo_preprocessed), dtype=np.float32)

# Use the wf for peak finding
wf_event = wfs[wf_num, ...]

# Power spectrum
P = np.abs(FFT(wf_event)) ** 2

# Apply log to the power spectrum (P) to enhance visibility of peaks
log_P = np.log(P)

# Set up figure for all subplots
fig, axes = plt.subplots(1, 1, figsize=(14, 8))

# for i in range(len(P)):
# Find peaks in the Fourier space with integer pixel accuracy first
coordinates_all = peak_local_max(log_P, min_distance=min_distance, threshold_abs=threshold_abs)

# Calculate the initial integer center peak location
initial_center_y, initial_center_x = np.array(P.shape) // 2
# Refine the center peak position with subpixel accuracy
center_region = log_P[initial_center_y - region_size:initial_center_y + region_size + 1,
                initial_center_x - region_size:initial_center_x + region_size + 1]

# The center of mass function is able to find the peak to subpixel precision
center_y_subpixel, center_x_subpixel = center_of_mass(center_region)

# Add the region offset to get the subpixel-accurate center coordinates in the original array
center_y = initial_center_y - region_size + center_y_subpixel
center_x = initial_center_x - region_size + center_x_subpixel

# Calculate the distances from the center with subpixel accuracy
distances = np.sqrt((coordinates_all[:, 0] - center_y) ** 2 + (coordinates_all[:, 1] - center_x) ** 2)

# Filter out central peak and sort by distance
non_center_indices = np.where(distances > center_exclusion_radius)[0]
coordinates_filtered = coordinates_all[non_center_indices]
filtered_distances = distances[non_center_indices]
sorted_indices = np.argsort(filtered_distances)
closest_coordinates = coordinates_filtered[sorted_indices[:num_peaks]]

# Refine each peak position using subpixel accuracy via center of mass (interpolation)
def refine_peak(log_P, y, x, region_size=3):
    """Refine the peak position using center of mass."""
    region = log_P[y - region_size:y + region_size + 1, x - region_size:x + region_size + 1]
    offset_y, offset_x = center_of_mass(region)
    refined_y = y - region_size + offset_y
    refined_x = x - region_size + offset_x
    return refined_y, refined_x

# Apply subpixel refinement to the 8 closest Fourier peaks to the central Fourier peak
refined_coordinates = []
for y, x in closest_coordinates:
    refined_y, refined_x = refine_peak(log_P, y, x, region_size=region_size)
    refined_coordinates.append((refined_y, refined_x))
refined_coordinates = np.array(refined_coordinates, dtype=np.float32)

# Save refined coordinates in txt file
np.savetxt(
    dir_main + dir_holo_preprocessed + "refined_coordinates.txt",
    refined_coordinates,
    fmt="%.6f",
    header="y x"
)

# Plot the zoomed-in power spectrum with circles and annotations
ax = axes
ax.imshow(log_P, cmap='Greys_r')
ax.set_title("Log Power Spectrum")
ax.axis('off')  # Hide axes for cleaner visualization

# Draw circles around the peaks and annotate
for j, (y, x) in enumerate(refined_coordinates):
    circle = plt.Circle((x, y), radius=50, color='red', fill=False)
    ax.add_artist(circle)

plt.tight_layout()
if save:
    plt.savefig(dir_main + dir_holo_preprocessed + 'log_power_spectrum.pdf', bbox_inches='tight', dpi=300)
    imwrite(dir_main + dir_holo_preprocessed + 'log_power_spectrum.tiff', log_P)
plt.show()

mask = np.ones_like(log_P, dtype=np.uint8)
# Loop over each refined peak and zero out a 151x151 region
for y, x in refined_coordinates:
    y = int(round(y))
    x = int(round(x))
    y_start = max(y - box_radius, 0)
    y_end = min(y + box_radius + 1, P.shape[0])
    x_start = max(x - box_radius, 0)
    x_end = min(x + box_radius + 1, P.shape[1])
    mask[y_start:y_end, x_start:x_end] = 0
masked_peaks = log_P * mask

plt.figure()
plt.title("Masked Peaks")
plt.imshow(masked_peaks, cmap='Greys_r')
plt.tight_layout()
if save:
    plt.savefig(dir_main + dir_holo_preprocessed + 'masked_peaks.pdf', bbox_inches='tight', dpi=300)
    imwrite(dir_main + dir_holo_preprocessed + 'masked_peaks.tiff', masked_peaks)
plt.show()

if preprocess_wfs:
    print("Removing Talbot grating from the white fields")
    if preprocess_wfs:
        wfs_no_Talbot = []
        for i in tqdm(range(len(wfs))):
            wfs_Fourier_masked = FFT(wfs[i]) * mask
            I_removed_Talbot = np.abs(IFFT(wfs_Fourier_masked))
            wfs_no_Talbot.append(I_removed_Talbot)
        wfs_no_Talbot = np.array(wfs_no_Talbot, dtype=np.float32)

print("Removing Talbot grating from the void holograms")
holos_no_Talbot = []
for i in tqdm(range(len(holos))):
    wfs_Fourier_masked = FFT(holos[i]) * mask
    I_removed_Talbot = np.abs(IFFT(wfs_Fourier_masked))
    holos_no_Talbot.append(I_removed_Talbot)
holos_no_Talbot = np.array(holos_no_Talbot, dtype=np.float32)


if save:
    if preprocess_wfs:
        imwrite(dir_main + dir_wfs_preprocessed + tiffs_wfs_preprocessed_no_Talbot, wfs_no_Talbot,
                photometric='minisblack')
    imwrite(dir_main + dir_holo_preprocessed + tiffs_holo_preprocessed_no_Talbot, holos_no_Talbot,
            photometric='minisblack')

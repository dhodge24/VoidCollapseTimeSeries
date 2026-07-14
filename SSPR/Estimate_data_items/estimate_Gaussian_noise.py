"""

The purpose of this code is to estimate the standard deviation (sigma) of the gaussian noise that is present
in our experimental data.. Once the standard deviation (sigma) is obtained, we can use that value to apply Gaussian
noise to our simulated images so they become comparable to our experimental data (more realistic).

"""

import numpy as np
from tifffile import imread
from SSPR.utilities import estimate_noise


run_holo = "571"

# Main directories
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_holo = "run" + run_holo + "_exp_preprocessed/"  # Sub directory

tiffs_holo_preprocessed = "run" + run_holo + "_exp_preprocessed.tiff"  # Tiff image file

# Import tiff
img_noisy = np.array(imread(dir_main + dir_holo + tiffs_holo_preprocessed), dtype=np.float32)

# Compute per-image standard deviations
std_ests = [estimate_noise(img) for img in img_noisy]
std_ests = np.array(std_ests, dtype=np.float32)

# Compute per-image percentage noise (normalized by each image's max)
noise_percentages = np.array([s / np.max(img) * 100 for s, img in zip(std_ests, img_noisy)], dtype=np.float32)

# Compute average values
std_est = np.mean(std_ests, dtype=np.float32)
noise_percentage = np.mean(noise_percentages, dtype=np.float32)

print(f"Estimated Gaussian noise standard deviation: {std_est:.2f}")
print("Percent Gaussian Noise: ", noise_percentage)

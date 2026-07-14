import os
import numpy as np
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

# ----------------
# RUNS TO LOAD AND SAVE ORDER
# ----------------
# Requested output order:
# run 572 -> Time1
# run 582 -> Time2
# run 580 -> Time3
# run 590 -> Time4
run_time_map = [
    ("572", "Time1"),
    ("582", "Time2"),
    ("580", "Time3"),
    ("590", "Time4"),
]

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped"

save_dir = "/Users/danielhodge/Desktop/cropped_matched_intensity_inferno"
os.makedirs(save_dir, exist_ok=True)

# ----------------
# INPUT FILE SETTINGS
# ----------------
# input_suffix = "I_final.tiff"
input_suffix = "I_final_no_deconvolve.tiff"

# ----------------
# NORMALIZATION SETTINGS
# ----------------
# These percentiles are computed globally across all cropped images.
lower_percentile = 0.5
upper_percentile = 99.5

# ----------------
# BRIGHTNESS MATCHING SETTINGS
# ----------------
# This forces all images to have more similar average brightness and contrast.
# Good starting values:
target_mean = 0.45
target_std = 0.18

# ----------------
# DISPLAY BRIGHTNESS SETTINGS
# ----------------
# brightness_factor > 1 brightens the whole image.
# gamma < 1 brightens midtones.
# gamma > 1 darkens midtones.
brightness_factor = 1.0
gamma = 0.85

# ----------------
# COLORMAP
# ----------------
cmap_name = "Greys_r"
cmap = plt.get_cmap(cmap_name)

# ----------------
# LOAD IMAGES
# ----------------
imgs = {}
shapes = {}

for run, time_label in run_time_map:
    img_path = os.path.join(
        dir_main,
        f"run{run}_exp",
        f"run{run}_exp_{input_suffix}"
    )

    img = np.array(imread(img_path), dtype=np.float32)

    imgs[run] = img
    shapes[run] = img.shape

    print(f"Loaded run {run}: shape = {img.shape}")

# ----------------
# FIND SMALLEST COMMON SIZE
# ----------------
target_ny = min(shape[0] for shape in shapes.values())
target_nx = min(shape[1] for shape in shapes.values())

print(f"\nTarget cropped size: ny = {target_ny}, nx = {target_nx}")

# ----------------
# CROP FUNCTION
# ----------------
def crop_to_common_size(img, target_ny, target_nx):
    """
    Crop image to target_ny x target_nx.

    Y-direction:
        If cropping is needed, crop from the top only.
        This keeps the bottom of the image fixed.

    X-direction:
        Crop symmetrically from left/right.
    """

    ny, nx = img.shape

    # ----------------
    # Crop y from the top only
    # ----------------
    if ny > target_ny:
        y_start = ny - target_ny
        y_end = ny
    else:
        y_start = 0
        y_end = ny

    img_crop = img[y_start:y_end, :]

    # ----------------
    # Crop x symmetrically
    # ----------------
    ny_crop, nx_crop = img_crop.shape

    if nx_crop > target_nx:
        x_start = (nx_crop - target_nx) // 2
        x_end = x_start + target_nx
    else:
        x_start = 0
        x_end = nx_crop

    img_crop = img_crop[:, x_start:x_end]

    return img_crop

# ----------------
# GLOBAL NORMALIZATION FUNCTION
# ----------------
def normalize_with_global_range(img, global_low, global_high):
    """
    Normalize image using one global intensity range shared by all images.
    """

    if global_high <= global_low:
        raise ValueError(
            f"Invalid global range: global_low = {global_low}, global_high = {global_high}"
        )

    img_norm = (img - global_low) / (global_high - global_low)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm.astype(np.float32)

# ----------------
# BRIGHTNESS MATCHING FUNCTION
# ----------------
def match_mean_std(img_norm, target_mean=0.45, target_std=0.18):
    """
    Match each image to the same target mean and standard deviation.

    This is useful when the images still appear visually different in brightness
    after global normalization.

    The operation is:
        img_matched = (img - mean) / std
        img_matched = img_matched * target_std + target_mean
    """

    img_mean = np.mean(img_norm)
    img_std = np.std(img_norm)

    if img_std <= 1e-8:
        raise ValueError(
            f"Image standard deviation is too small for brightness matching: std = {img_std}"
        )

    img_matched = (img_norm - img_mean) / img_std
    img_matched = img_matched * target_std + target_mean
    img_matched = np.clip(img_matched, 0, 1)

    return img_matched.astype(np.float32), img_mean, img_std

# ----------------
# BRIGHTNESS / GAMMA FUNCTION
# ----------------
def apply_brightness_gamma(img_norm, brightness_factor=1.0, gamma=0.85):
    """
    Apply the same brightness and gamma correction to all images.

    brightness_factor:
        Multiplies the normalized image intensity.

    gamma:
        Applies nonlinear midtone correction.
        gamma < 1 brightens midtones.
        gamma > 1 darkens midtones.
    """

    img_display = img_norm.copy()

    # Linear brightness boost
    img_display = brightness_factor * img_display
    img_display = np.clip(img_display, 0, 1)

    # Gamma correction
    img_display = img_display ** gamma
    img_display = np.clip(img_display, 0, 1)

    return img_display.astype(np.float32)

# ----------------
# COLORMAP FUNCTION
# ----------------
def apply_colormap(img_display, cmap):
    """
    Apply a Matplotlib colormap to a normalized [0, 1] image.

    Returns an RGB uint8 image suitable for PNG or TIFF saving.
    """

    img_rgba = cmap(img_display)
    img_rgb = img_rgba[:, :, :3]
    img_rgb_uint8 = (255 * img_rgb).astype(np.uint8)

    return img_rgb_uint8

# ----------------
# CROP ALL IMAGES FIRST
# ----------------
cropped_imgs = {}

for run, time_label in run_time_map:
    img_crop = crop_to_common_size(
        imgs[run],
        target_ny=target_ny,
        target_nx=target_nx
    )

    cropped_imgs[run] = img_crop

    print(f"Cropped run {run}: shape = {img_crop.shape}")

# ----------------
# COMPUTE ONE GLOBAL INTENSITY RANGE
# ----------------
all_pixels = np.concatenate([
    cropped_imgs[run].ravel()
    for run, time_label in run_time_map
])

global_low, global_high = np.percentile(
    all_pixels,
    [lower_percentile, upper_percentile]
)

print("\nGlobal normalization range:")
print(f"  global_low  = {global_low:.6f}")
print(f"  global_high = {global_high:.6f}")

# ----------------
# NORMALIZE, MATCH BRIGHTNESS, COLORIZE, AND SAVE
# ----------------
for run, time_label in run_time_map:
    img_crop = cropped_imgs[run]

    # Normalize using the same global intensity range for all images
    img_norm = normalize_with_global_range(
        img_crop,
        global_low=global_low,
        global_high=global_high
    )

    # Match each image to the same visual brightness/contrast statistics
    img_matched, original_mean, original_std = match_mean_std(
        img_norm,
        target_mean=target_mean,
        target_std=target_std
    )

    # Apply same brightness/gamma correction after matching
    img_display = apply_brightness_gamma(
        img_matched,
        brightness_factor=brightness_factor,
        gamma=gamma
    )

    # Apply inferno colormap
    img_rgb = apply_colormap(img_display, cmap)

    # ----------------
    # SAVE PNG
    # ----------------
    png_save_path = os.path.join(
        save_dir,
        f"{time_label}_run{run}_intensity_cropped_matched_inferno.png"
    )

    imwrite(png_save_path, img_rgb)

    # ----------------
    # SAVE TIFF
    # ----------------
    tiff_save_path = os.path.join(
        save_dir,
        f"{time_label}_run{run}_intensity_cropped_matched_inferno.tiff"
    )

    imwrite(tiff_save_path, img_rgb)

    print(
        f"Saved {time_label}, run {run}: "
        f"shape = {img_rgb.shape}, "
        f"global range = ({global_low:.4f}, {global_high:.4f}), "
        f"original mean/std = ({original_mean:.4f}, {original_std:.4f}), "
        f"target mean/std = ({target_mean:.4f}, {target_std:.4f}), "
        f"brightness_factor = {brightness_factor}, "
        f"gamma = {gamma}, "
        f"colormap = {cmap_name}"
    )
    print(f"  PNG  -> {png_save_path}")
    print(f"  TIFF -> {tiff_save_path}")
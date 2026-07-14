"""

Code is used for:
1. dark field subtraction
2. shifting, reflecting, and cropping the images
3. Filtering outlier pixels

"""

import os
import numpy as np
from tifffile import imread, imwrite
from natsort import natsorted
import glob
import shutil
from tqdm import tqdm

from SSPR.utilities import shiftRotateMagnifyImage, padToSize, cropToCenter, remove_outliers_mult_imgs, reflect_image_2d, removeOutliers


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
reflect = False
preprocess_wfs = True  # Preprocessing action needs to happen only once since 1 batch of wfs were for many runs
N_pad = 6000
crop_initial_size = [2100, 2100]
crop_final_size = [2500, 2500]
# Shift image parameters to make it more centered
shift_y = 0
shift_x = 0

run_wfs = "561"
run_holo = "582"
run_darks = "381"

# Directory where files are saved/located
dir_main = "/Users/danielhodge/Desktop/"
dir_wfs = "run" + run_wfs + "_exp/"
dir_holo = "run" + run_holo + "_exp/"
dir_darks = "run" + run_darks + "_darks/"

# Files to import
all_tiffs = "*.tiff"

# Directories to create where we will save our files
dir_wfs_preprocessed = "run" + run_wfs + "_to_" + run_wfs + "_for_run" + run_holo + "_exp_preprocessed/"
# dir_wfs_preprocessed = "run" + run_wfs + "_exp_preprocessed/"
dir_holo_preprocessed = "run" + run_holo + "_exp_preprocessed/"

# Save Files
tiffs_wfs_preprocessed = "run" + run_wfs + "_exp_preprocessed.tiff"
tiffs_holo_preprocessed = "run" + run_holo + "_exp_preprocessed.tiff"

os.chdir(dir_main)  # Change directory to where you want to save the data
cwd = os.getcwd()

# Get the current working directory
print('Current working directory is: ', cwd)

if preprocess_wfs:
    print("Preprocessing white fields")
    run_loc_wfs = dir_main + dir_wfs_preprocessed
    if not os.path.exists(run_loc_wfs):
        os.makedirs(run_loc_wfs)
    if os.path.exists(dir_main + dir_wfs_preprocessed):
        clear_directory(dir_main=dir_main + dir_wfs_preprocessed, dir_sub="")

run_loc_static = dir_main + dir_holo_preprocessed
if not os.path.exists(run_loc_static):
    os.makedirs(run_loc_static)
if os.path.exists(dir_main + dir_holo_preprocessed):
    clear_directory(dir_main=dir_main + dir_holo_preprocessed, dir_sub="")


wfs = None  # Initialize
# Import data
if preprocess_wfs:
    wfs = glob.glob(dir_main + dir_wfs + all_tiffs)
    wfs = natsorted(wfs)
    wfs = np.array(imread(wfs), dtype=np.float32)

holos = glob.glob(dir_main + dir_holo + all_tiffs)
holos = natsorted(holos)
holos = np.array(imread(holos), dtype=np.float32)

if len(holos.shape) < 3:
    holos = np.expand_dims(holos, axis=0)  # For the single dynamic image case - to make it size (1, M, N)

darks = glob.glob(dir_main + dir_darks + all_tiffs)
darks = natsorted(darks)
darks = np.mean(np.array(imread(darks)), axis=0, dtype=np.float32)

# Subtract average dark field from all the images
holos -= darks
holos[holos < 0] = 0
if preprocess_wfs:
    wfs -= darks
    wfs[wfs < 0] = 0

# Remove outliers from holos
if len(holos.shape) < 3:
    holos = np.expand_dims(holos, axis=0)
    holos = remove_outliers_mult_imgs(holos, threshold=3, kernSize=3).squeeze(axis=0)
else:
    holos = remove_outliers_mult_imgs(holos, threshold=3, kernSize=3)

# Remove outliers from wfs
if preprocess_wfs:
    if len(wfs.shape) < 3:
        wfs = np.expand_dims(wfs, axis=0)
        # wfs = removeOutliers(wfs[0], threshold=3).squeeze(axis=0)
        wfs = remove_outliers_mult_imgs(wfs, threshold=3, kernSize=3).squeeze(axis=0)
    else:
        wfs = remove_outliers_mult_imgs(wfs, threshold=3, kernSize=3)
        if len(wfs.shape) < 3:
            moving = np.expand_dims(wfs, axis=0)

# imwrite("/Users/danielhodge/Desktop/holo_df_corr_remove_outliers.tiff", holos)
# imwrite("/Users/danielhodge/Desktop/wf_df_corr_remove_outliers.tiff", wfs[0])

holos_adjusted = []
save_path = dir_main + dir_holo_preprocessed + dir_holo_preprocessed
# Create or clear directory once before loop so it doesn't overwrite
if save:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        clear_directory(dir_main=dir_main + dir_holo_preprocessed, dir_sub=dir_holo_preprocessed)
for i in tqdm(range(len(holos))):
    if reflect:
        holo_cropped = cropToCenter(holos[i], crop_initial_size)
        holo_reflected = reflect_image_2d(holo_cropped)
        holo_shifted = shiftRotateMagnifyImage(img=holo_reflected,
                                               shifts=[shift_y, shift_x])
        holo_cropped = cropToCenter(img=holo_shifted,
                                    newSize=crop_final_size)
        if save:
            imwrite(dir_main + dir_holo_preprocessed + dir_holo_preprocessed + "run" + run_holo + f"_exp_evt_{i+1}_"
                                                                                                  f"preprocessed",
                    holo_cropped,
                    photometric='minisblack')
    else:
        holo_shifted = shiftRotateMagnifyImage(img=holos[i],
                                               shifts=[shift_y, shift_x])
        holo_padded = padToSize(img=holo_shifted,
                                outputSize=[N_pad, N_pad],
                                padMethod='constant',  # Was replicate
                                padType='both',
                                padValue=0)  # Was none
        holo_cropped = cropToCenter(img=holo_padded,
                                    newSize=crop_final_size)
        if save:
            imwrite(dir_main + dir_holo_preprocessed + dir_holo_preprocessed + "run" + run_holo + f"_exp_evt_{i+1}_"
                                                                                                  f"preprocessed",
                    holo_cropped,
                    photometric='minisblack')
    holos_adjusted.append(holo_cropped)
holos = np.array(holos_adjusted, dtype=np.float32)

if preprocess_wfs:
    wfs_adjusted = []

    # Clear or create directory before the loop so it does not overwrite
    save_path = dir_main + dir_wfs_preprocessed + dir_wfs_preprocessed
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            clear_directory(dir_main=dir_main + dir_wfs_preprocessed, dir_sub=dir_wfs_preprocessed)

    for i in tqdm(range(len(wfs))):
        if reflect:
            wf_cropped = cropToCenter(wfs[i], crop_initial_size)
            wf_reflected = reflect_image_2d(wf_cropped)
            wf_shifted = shiftRotateMagnifyImage(img=wf_reflected,
                                                   shifts=[shift_y, shift_x])
            wf_cropped = cropToCenter(img=wf_shifted,
                                        newSize=crop_final_size)
            if save:
                imwrite(dir_main + dir_wfs_preprocessed + dir_wfs_preprocessed + "run" + run_wfs + f"_exp_evt_{i+1}_"
                                                                                                   f"preprocessed",
                        wf_cropped,
                        photometric='minisblack')
        else:
            wf_shifted = shiftRotateMagnifyImage(img=wfs[i],
                                                 shifts=[shift_y, shift_x])
            wf_padded = padToSize(img=wf_shifted,
                                  outputSize=[N_pad, N_pad],
                                  padMethod='constant',  # Was replicate
                                  padType='both',
                                  padValue=0)  # Was none
            wf_cropped = cropToCenter(img=wf_padded,
                                      newSize=crop_final_size)
            if save:
                imwrite(dir_main + dir_wfs_preprocessed + dir_wfs_preprocessed + "run" + run_wfs + f"_exp_evt_{i+1}_"
                                                                                                   f"preprocessed",
                        wf_cropped,
                        photometric='minisblack')
        wfs_adjusted.append(wf_cropped)
    wfs = np.array(wfs_adjusted, dtype=np.float32)

if save:
    imwrite(dir_main + dir_holo_preprocessed + tiffs_holo_preprocessed, holos, photometric='minisblack')
    if preprocess_wfs:
        imwrite(dir_main + dir_wfs_preprocessed + tiffs_wfs_preprocessed, wfs, photometric='minisblack')

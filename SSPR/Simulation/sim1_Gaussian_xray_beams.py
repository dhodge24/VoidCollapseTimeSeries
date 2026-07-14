"""

First use code "estimate_beam_shifts_fwhms_amps_from_exp.py" to obtain illumination amplitude, beam shifts relative
to the center of the grid, and the width of the Gaussian (FWHM). Also, first determine point spread function (PSF)
parameters and Gaussian noise level

The purpose of this code is to simulate a Gaussian x-ray beam that will illuminate our object. The FWHM of this
beam is estimated based on a fitting a 2D Gaussian to our experimental white field images (not flat-field corrected).

"""


import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite
from SSPR.units import *
import os
from tqdm import tqdm
import shutil

from SSPR.utilities import create_circular_mask, cropToCenter


def make_Gaussian(size, fwhm=1200, center=None, exponent=1.0):
    """
    Make a square Gaussian kernel with adjustable sharpness.

    size: int
        The length of a side of the square.
    fwhm: float
        Full-width-half-maximum, which can be thought of as an effective radius.
    center: tuple of two ints
        The coordinates of the center of the Gaussian. If None, the center is the middle of the image.
    exponent: float
        A factor to control the sharpness of the falloff. Lower values result in faster falloff.
    """

    # Convert FWHM to standard deviation (sigma)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    x = np.arange(-size // 2, size // 2, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center

    return np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) ** exponent)


def make_top_hat(size, fwhm=1200, center=None):
    """
    Make a square top hat function with a specified width.

    size: int
        The length of a side of the square.
    fwhm: float
        Full-width-half-maximum, which can be thought of as an effective diameter.
    center: tuple of two ints
        The coordinates of the center of the top hat. If None, the center is the middle of the image.
    """

    radius = fwhm / 2  # Radius is half of the FWHM

    x = np.arange(-size // 2, size // 2, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center

    # Top hat function
    top_hat = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) <= radius

    return top_hat


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


# Have to run this code twice
save_plot = True
save_img = True
# True for wfs, False for holos. We do this since we have a different number of white fields and holos in our exp
make_Gaussian_beams_for_wfs = True
crop_size_y, crop_size_x = 2500, 2500

run_wfs = "561"
run_holo = "576"

# run_wfs = "562"
# run_holo = "589"

# Main directories
dir_main = "/Users/danielhodge/Desktop/"
dir_sim = dir_main + "run" + run_holo + "_sim/"

# Directories to create
if make_Gaussian_beams_for_wfs:
    dir_Gaussian_beams = "run" + run_wfs + "_Gaussian_beams/"
else:
    dir_Gaussian_beams = "run" + run_holo + "_Gaussian_beams/"

# Files to import
if make_Gaussian_beams_for_wfs:
    txt_beam_pos_filename = "beam_shifts_wfs.txt"
else:
    txt_beam_pos_filename = "beam_shifts_holos.txt"

# Files to save
if make_Gaussian_beams_for_wfs:
    tiff_gbeam_stack_filename = "run" + run_wfs + "_gbeam_stack.tiff"
else:
    tiff_gbeam_stack_filename = "run" + run_holo + "_gbeam_stack.tiff"


# Define experimental parameters
Ny, Nx = (2500, 2500)  # Max length of Zyla detector for x and y
E = np.array(18000.0, dtype=np.float32)  # Energy of the beam
z01 = np.array(120.41 * mm, dtype=np.float32)  # Distance from source to sample
z12 = np.array(4.668995, dtype=np.float32)  # Distance from sample to detector
scale_fac = np.array(4.0, dtype=np.float32)  # Scale factor we include to compensate for lens mag
det_pix_size = np.array(6.5 * um, dtype=np.float32)  # Detector pixel size
lam = (1240.0 / E) * nm  # Wavelength of the beam
z02 = z01 + z12  # Distance from source to detector
M = z02 / z01  # Magnification
dx_eff = det_pix_size / M / scale_fac  # Object effective pixel size in x, equals detector pixel size if no mag
dy_eff = det_pix_size / M / scale_fac  # Object effective pixel size in y, equals detector pixel size if no mag
z_eff = z12 / M  # Effective propagation distance for spherical illumination (geometry)
k0 = 2 * np.pi / lam
extent_x = Nx * dx_eff  # Physical grid length in x
extent_y = Ny * dx_eff  # Physical grid length in y
circle_grid_percentage = 1.2
smooth_outer_circle_edges = 20
Gaussian_exponential = 1.75
Gaussian_FWHM = 3600  # Pixels

# Load in the center Gaussian positions retrieved from the exp data. Add a center (0, 0) position.
beam_centers = np.loadtxt(dir_sim + txt_beam_pos_filename, delimiter='\t', skiprows=1)
center_beam_position = np.array([[0, 0]])
beam_centers = np.vstack([center_beam_position, beam_centers])

# Clear directory of images. Sometimes it is saved to the cloud and overwriting takes too much time
if os.path.exists(dir_main + dir_Gaussian_beams):
    clear_directory(dir_main=dir_main, dir_sub=dir_Gaussian_beams)

beams = []
top_hat = create_circular_mask(size=Ny,
                               percentage=circle_grid_percentage,
                               smooth_pixels=smooth_outer_circle_edges)
for i in tqdm(range(len(beam_centers))):
    # Calculate the 2D Gaussian distribution (x-ray beam that will illuminate the object)
    beam = make_Gaussian(size=Ny,
                         fwhm=Gaussian_FWHM,
                         center=tuple(beam_centers[i]),
                         exponent=Gaussian_exponential)
    beam = top_hat * beam
    beam = cropToCenter(img=beam, newSize=[crop_size_y, crop_size_x])
    beams.append(beam)

    if save_img:
        # Create the directory if it doesn't exist
        if not os.path.exists(dir_main + dir_Gaussian_beams):
            os.mkdir(dir_main + dir_Gaussian_beams)
        else:
            if make_Gaussian_beams_for_wfs:
                imwrite(dir_main + dir_Gaussian_beams + "run" + run_wfs + f"_sim_evt_{i}.tiff", beam)
            else:
                imwrite(dir_main + dir_Gaussian_beams + "run" + run_holo + f"_sim_evt_{i}.tiff", beam)

beams = np.stack(beams, axis=0, dtype=np.float32)

# Plot the centered Gaussian beam
plt.figure(figsize=(10, 8))
plt.imshow(beams[0],
           extent=[-extent_x / 2 / um, extent_y / 2 / um, -extent_x / 2 / um, extent_y / 2 / um],
           cmap='Greys_r')
plt.title('2D Gaussian X-ray Beam Profile')
plt.xlabel('x [um]')
plt.ylabel('y [um]')
plt.colorbar()
if save_plot:
    plt.savefig(dir_sim + '2D_Gaussian_xray_beam_profile',
                bbox_inches='tight',
                transparent=True)
plt.show()

if save_img:
    imwrite(dir_main + dir_Gaussian_beams + tiff_gbeam_stack_filename, beams, photometric='minisblack')

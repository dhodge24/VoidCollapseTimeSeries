"""

The purpose of this code is to use the recorded experimental white fields and fit a 2D Gaussian to each of the images.
We do this to obtain the relative peak shift from the center, find the amplitude of our beam, and determine the rough
FWHM of the beam (may need post adjustment) so we can construct our real space illumination function. The illumination
function is technically complex, but we assume it is real only as the beam phase is slowly varying and thus we assume
it to be a constant value.

"""

import numpy as np
from scipy.optimize import curve_fit
import glob
from tifffile import imread
from natsort import natsorted
from tqdm import tqdm
from SSPR.utilities import cropToCenter, padToSize, fadeoutImage, showImg
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter


def calculate_shifts(image, center):
    """
    :param image: X-ray beam
    :param center: Center of the beam. Default is (Ny//2, Nx//2)
    :return: Pixel shifts in x and y
    """
    com = center_of_mass(image)
    shift_x = com[1] - center[1]
    shift_y = com[0] - center[0]
    return shift_x, shift_y


def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, A):
    """2D Gaussian that is used for fitting the experimental data"""
    x, y = xy
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))


def fit_gaussian(image):
    """Fit the Gaussian to the experimental data"""
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((x.ravel(), y.ravel()))
    ydata = image.ravel()

    # Initial guess for the parameters: x0, y0, sigma_x, sigma_y, amplitude
    initial_guess = (center[1], center[0], 50, 50, image.max())

    popt, _ = curve_fit(gaussian_2d, xdata, ydata, p0=initial_guess)
    xcenter, ycenter, sigmaX, sigmaY, amp = popt[0], popt[1], popt[2], popt[3], popt[4]
    FWHM_x = np.abs(2 * sigmaX * np.sqrt(2 * np.log(2)))
    FWHM_y = np.abs(2 * sigmaY * np.sqrt(2 * np.log(2)))
    # Equivalent to the above FWHM definitions:
    # FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    # FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return xcenter, ycenter, FWHM_x, FWHM_y, amp


save_data = True  # True = save the data
sub_dfs = False  # True = subtract dark fields from the experimental images
apply_to_wfs = False  # True = apply the algorithm to both the void data AND the white fields

# dynamics
run_wfs = "561"
run_holo = "586"

# # statics
# run_wfs = "562"
# run_holo = "589"

# Main directories
dir_main = "/Users/danielhodge/Desktop/"
dir_sim = "run" + run_holo + "_sim/"

# For import
tiffs_wfs_run_name = "run" + run_wfs + "_exp_preprocessed/run" + run_wfs + "_exp_preprocessed.tiff"
tiffs_holos_run_name = "run" + run_holo + "_exp_preprocessed/run" + run_holo + "_exp_preprocessed.tiff"
tiffs_darks_run_name = "run381_darks/*.tiff"

# For export
txt_fwhm_xy_filename = "fwhm_xy.txt"
txt_beam_shift_pos_wfs_filename = "beam_shifts_wfs.txt"
txt_beam_shift_pos_holos_filename = "beam_shifts_holos.txt"
txt_amps_wfs_filename = "amps_wfs.txt"
txt_amps_holos_filename = "amps_holos.txt"

# Import white fields
if apply_to_wfs:
    num_images_wfs = 50  # Consider only a few white fields for computational speed
    file_path_wfs_exp = glob.glob(dir_main + tiffs_wfs_run_name)
    wf_images = natsorted(file_path_wfs_exp)
    wf_images = imread(wf_images)
    wf_images = wf_images[:num_images_wfs, :, :]

num_images_holos = 20  # Consider only a few white fields for computational speed
file_path_holos_exp = glob.glob(dir_main + tiffs_holos_run_name)
holos = natsorted(file_path_holos_exp)
holos = imread(holos)
holos = holos[:num_images_holos, :, :]

if sub_dfs:
    file_path_darks_exp = glob.glob(dir_main + tiffs_darks_run_name)
    # df_images = natsorted(file_path_wf_exp)
    df_images = imread(file_path_darks_exp)
    df_images = np.array(df_images, dtype=np.float32)
    df_image_mean = np.mean(df_images, axis=0)
    wf_images -= df_image_mean
    wf_images[wf_images < 0] = 0


######################################################################################################################
######################################################################################################################
################################################## HOLOS #############################################################
######################################################################################################################
######################################################################################################################

holos_preprocessed = []
i = 0
mask = None
for image in tqdm(holos):
    # img_shifted = shiftImage(img=image, shifts=[-75, 20])
    img_padded = padToSize(img=image, outputSize=[3000, 3000], padMethod='constant', padType='both', padValue=0)
    img_cropped = cropToCenter(img=img_padded, newSize=[2500, 2500])
    if i == 0:
        _, mask = fadeoutImage(img=img_cropped, fadeMethod='ellipse', transitionLength=[5, 5], ellipseSize=[1, 1])
        mask[mask > 0] = 1
    img_masked = img_cropped * mask
    img_masked = gaussian_filter(img_masked, sigma=2)
    # showImg(img_masked)
    holos_preprocessed.append(img_masked)
    i += 1

holos_preprocessed = np.array(holos_preprocessed, dtype=np.float32)
holos_shape = holos_preprocessed[-1].shape
center = (holos_shape[-2] / 2, holos_shape[-1] / 2)

amps_holos = []
shifts_holos_x = []
shifts_holos_y = []
for image in tqdm(holos_preprocessed):
    x0, y0, _, _, _ = fit_gaussian(image)
    amp_holo = np.median(image)
    shift_x = x0 - center[1]
    shift_y = y0 - center[0]
    shifts_holos_x.append(int(shift_x))
    shifts_holos_y.append(int(shift_y))
    amps_holos.append(amp_holo)

######################################################################################################################
######################################################################################################################
####################################################### WFS ##########################################################
######################################################################################################################
######################################################################################################################

if apply_to_wfs:
    wfs_preprocessed = []
    i = 0
    mask = None
    for image in tqdm(wf_images):
        # img_shifted = shiftImage(img=image, shifts=[-75, 20])
        img_padded = padToSize(img=image, outputSize=[3000, 3000], padMethod='constant', padType='both', padValue=0)
        img_cropped = cropToCenter(img=img_padded, newSize=[1800, 1800])
        if i == 0:
            _, mask = fadeoutImage(img=img_cropped, fadeMethod='ellipse', transitionLength=[5, 5], ellipseSize=[1, 1])
            mask[mask > 0] = 1
        img_masked = img_cropped * mask
        wfs_preprocessed.append(img_masked)
        i += 1

    wfs_preprocessed = np.array(wfs_preprocessed, dtype=np.float32)
    wfs_shape = wfs_preprocessed[-1].shape
    center = (wfs_shape[-2] / 2, wfs_shape[-1] / 2)

    shifts_wfs_x = []
    shifts_wfs_y = []
    fwhms_x = []
    fwhms_y = []
    amps_wfs = []
    for image in tqdm(wfs_preprocessed):
        x0, y0, fwhm_x, fwhm_y, amp_wf = fit_gaussian(image)
        shift_x = x0 - center[1]
        shift_y = y0 - center[0]
        shifts_wfs_x.append(int(shift_x))
        shifts_wfs_y.append(int(shift_y))
        fwhms_x.append(int(fwhm_x))
        fwhms_y.append(int(fwhm_y))
        amps_wfs.append(amp_wf)

if apply_to_wfs:
    average_fwhm_x = int(np.mean(fwhms_x))
    average_fwhm_y = int(np.mean(fwhms_y))
    print(f'Average fwhm in x: {average_fwhm_x:.2f} pixels')
    print(f'Average fwhm in y: {average_fwhm_y:.2f} pixels')

if apply_to_wfs:
    average_shift_wfs_x = int(np.mean(shifts_wfs_x))
    average_shift_wfs_y = int(np.mean(shifts_wfs_y))
    print(f'Average pointing shift in x direction from the center for wfs: {average_shift_wfs_x:.2f} pixels')
    print(f'Average pointing shift in y direction from the center for wfs: {average_shift_wfs_y:.2f} pixels')

average_shift_holos_x = int(np.mean(shifts_holos_x))
average_shift_holos_y = int(np.mean(shifts_holos_y))
print(f'Average pointing shift in x direction from the center for holos: {average_shift_holos_x:.2f} pixels')
print(f'Average pointing shift in y direction from the center for holos: {average_shift_holos_y:.2f} pixels')

average_amp_holos = np.mean(amps_holos)
print(f'Average amp holos: {average_amp_holos:.2f}')
if apply_to_wfs:
    average_amp_wfs = np.mean(amps_wfs)
    print(f'Average amp wfs: {average_amp_wfs:.2f}')

# Convert to columns
if apply_to_wfs:
    fwhms = np.column_stack((fwhms_x, fwhms_y))
    shifts_wfs = np.column_stack((shifts_wfs_x, shifts_wfs_y))
    amps_wfs = np.column_stack(amps_wfs).reshape(-1, 1)
shifts_holos = np.column_stack((shifts_holos_x, shifts_holos_y))
amps_holos = np.column_stack(amps_holos).reshape(-1, 1)

if save_data:
    if apply_to_wfs:
        np.savetxt(dir_main + dir_sim + txt_fwhm_xy_filename, fwhms,
                   fmt='%d',
                   delimiter='\t',
                   header='FWHM_x\tFWHM_y',
                   comments='')

    np.savetxt(dir_main + dir_sim + txt_beam_shift_pos_holos_filename, shifts_holos,
               fmt='%d',
               delimiter='\t',
               header='Shift_x\tShift_y',
               comments='')

    if apply_to_wfs:
        np.savetxt(dir_main + dir_sim + txt_beam_shift_pos_wfs_filename, shifts_wfs,
                   fmt='%d',
                   delimiter='\t',
                   header='Shift_x\tShift_y',
                   comments='')

    np.savetxt(dir_main + dir_sim + txt_amps_holos_filename, amps_holos,
               fmt='%d',
               delimiter='\t',
               header='Amps_holos',
               comments='')

    if apply_to_wfs:
        np.savetxt(dir_main + dir_sim + txt_amps_wfs_filename, amps_wfs,
                   fmt='%d',
                   delimiter='\t',
                   header='Amps_wfs',
                   comments='')
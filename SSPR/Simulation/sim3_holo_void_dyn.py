"""

The purpose of this code is to simulate an intensity pattern of the dynamic hollow SiO2 glass shell embedded within SU8.
This is done by using xRAGE simulations since the interactions and evolution of the shock compression is complex and
nonlinear. Then we can forward propagate this phase map with the addition of speckle to obtain a hologram that more
closely corresponds to experimental data. In addition to speckle being added, the point spread function (PSF),
Poisson noise, and Gaussian noise still need to be applied.

"""

import numpy as np
from scipy.signal import convolve2d
from skimage.transform import resize
import h5py
from tifffile import imwrite, imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from skimage.filters import gaussian

# Import custom modules
from SSPR.units import *
from SSPR.Propagation.SimulateHologram import simulate_hologram
from SSPR.utilities import (cropToCenter, voigt_2d, FFT, IFFT, shiftImage, rotateImage, shiftRotateMagnifyImage,
                            reflect_image_2d, padToSize)


def interpolate_phase_attenuation_maps(x, current_pixel_size=0, desired_pixel_size=np.inf, anti_aliasing_sigma=(3, 3)):
    """xRAGE generates phase and attenuation maps with 0.1um pixel size. We must rescale this to the experimental
    pixel size to have an accurate comparison. To do this we interpolate the images with some given resolution and
    scale it to a desired pixel size."""
    # scale_factor = current_pixel_size / desired_pixel_size  # This should work...but it is slightly off...
    scale_factor = 2.5  # This must be manually tuned
    new_shape = (int(x.shape[0] * scale_factor), int(x.shape[1] * scale_factor))
    print("The resized image is scaled up by this amount: ", scale_factor)
    print("The new image shape is size: ", new_shape)

    # Interpolate phase and attenuation maps to the new resolution
    interpolated_map = resize(x,
                              new_shape,
                              mode='constant',
                              order=3,
                              anti_aliasing=True,
                              anti_aliasing_sigma=anti_aliasing_sigma)

    # # temp, remove after
    # interpolated_map = gaussian(interpolated_map, sigma=50 / 2.35, truncate=2)

    return interpolated_map


def process_h5_file(h5_file, half_img=False):
    """Mirrors the phase and attenuation maps, rotates, and magnifies them to match experimental images"""
    mu = h5_file['/attenuation'][...]  # Attenuation
    phi = -h5_file['/phase'][...]  # Phase should be negative as phase advances in a material in the x-ray regime
    phi = phi[-2580:, :]
    mu = mu[-2580:, :]

    # MU = np.zeros((orig_shape_y, 2 * orig_shape_x))  # Shape is (2580, 2580)
    # PHI = np.zeros((orig_shape_y, 2 * orig_shape_x))  # Shape is (2580, 2580)
    #
    # # Mirror the arrays
    # MU[:, 0:orig_shape_x] = np.fliplr(mu)
    # MU[:, orig_shape_x:2*orig_shape_x] = mu  # Shape is (2580, 2580)
    # PHI[:, 0:orig_shape_x] = np.fliplr(phi)
    # PHI[:, orig_shape_x:2*orig_shape_x] = phi  # Shape is (2580, 2580)

    if half_img: # Assuming size (2580, 1290)
        orig_shape_y, orig_shape_x = phi.shape  # Shape is (2580, 1290)
        out_phi = np.zeros((orig_shape_y, 2 * orig_shape_x), dtype=phi.dtype)
        out_phi[:, :orig_shape_x] = np.fliplr(phi)
        out_phi[:, orig_shape_x:] = phi
        out_mu = np.zeros((orig_shape_y, 2 * orig_shape_x), dtype=mu.dtype)
        out_mu[:, :orig_shape_x] = np.fliplr(mu)
        out_mu[:, orig_shape_x:] = mu
        PHI = out_phi
        MU = out_mu
    else: # Assuming size (2580, 2580)
        PHI = phi[-2580:, :]
        MU = mu[-2580:, :]

    # Resize to prevent phase aliasing and to scale the image to experimental conditions
    phi = interpolate_phase_attenuation_maps(x=PHI,
                                             current_pixel_size=0.1*um,  # xRAGE resolution (pixel size)
                                             desired_pixel_size=dx_eff,  # Exp resolution (pixel size)
                                             anti_aliasing_sigma=(3, 3))
    mu = interpolate_phase_attenuation_maps(x=MU,
                                             current_pixel_size=0.1*um,  # xRAGE resolution (pixel size)
                                             desired_pixel_size=dx_eff,  # Exp resolution (pixel size)
                                             anti_aliasing_sigma=(3, 3))

    # Rotate image to match experimental shockwave direction
    phi = rotateImage(img=phi, rotAngleDegree=180)
    mu = rotateImage(img=mu, rotAngleDegree=180)
    phi = shiftImage(img=phi, shifts=[40, 0])  # Manually adjust to match experimental data
    mu = shiftImage(img=mu, shifts=[40, 0])  # Manually adjust to match experimental data
    phi = cropToCenter(img=phi, newSize=[Ny, Nx])
    mu = cropToCenter(img=mu, newSize=[Ny, Nx])

    return phi, mu

def check_aliasing(phi, mu, extent_x, extent_y, dx_eff, dy_eff):
    """Checks for aliasing in the phase maps and smooths those regions"""
    FX, FY = np.gradient(phi, dx_eff)
    FX *= dx_eff
    FY *= dy_eff
    max_FX = np.max(np.abs(FX))
    max_FY = np.max(np.abs(FY))

    mask_phase = np.zeros(np.shape(phi))  # For identifying aliased pixels
    flag_phase = 0  # Set to 1 if we exceed the phase aliasing limit
    filter_val = 5  # To smooth out regions where there is aliasing

    if max_FX > np.pi:
        flag_phase = 1
        print("WARNING: Phase aliasing in the x-direction")
        print("Aliased regions marked RED")
        bin_phase = np.abs(FX) > np.pi
        mask_phase[bin_phase] = 1.0
        h1 = np.ones((filter_val, filter_val), dtype=float) / filter_val ** 2
        new_phase = convolve2d(phi, h1, mode='same', boundary='symm')
        phi[bin_phase] = new_phase[bin_phase]

    if max_FY > np.pi:
        flag_phase = 1
        print("WARNING: Phase aliasing in the y-direction")
        print("Aliased regions marked RED")
        bin_phase = np.abs(FY) > np.pi
        mask_phase[bin_phase] = 1.0
        h1 = np.ones((filter_val, filter_val), dtype=float) / filter_val ** 2
        new_phase = convolve2d(phi, h1, mode='same', boundary='symm')
        phi[bin_phase] = new_phase[bin_phase]

    if flag_phase > 0:
        phase_temp = cropToCenter(phi, newSize=[Ny, Nx])
        mask_phase_temp = cropToCenter(mask_phase, newSize=[Ny, Nx])
        plt.figure()
        plt.title("Aliased Phase Regions")
        plt.imshow(phase_temp, cmap='Greys_r')
        cmap = LinearSegmentedColormap.from_list('bright_red', ['#FF0000', '#FF0000'])
        plt.imshow(mask_phase_temp, cmap=cmap, alpha=1.0 * (mask_phase_temp == 1))
        plt.show()

    # if save_img:
    #     imwrite("/Users/danielhodge/Desktop/xRAGE_SU8_shockwave_phase.tiff", phi)
    if plot_attenuation_map:
        plt.figure()
        plt.imshow(mu, extent=[-extent_x / 2 / um, extent_y / 2 / um, -extent_x / 2 / um, extent_y / 2 / um],
                   cmap='Greys_r')
        plt.show()

    if plot_phase_map:
        plt.figure()
        plt.imshow(phi, extent=[-extent_x / 2 / um, extent_y / 2 / um, -extent_x / 2 / um, extent_y / 2 / um],
                   cmap='Greys_r')
        plt.show()

    if plot_aliasing:
        loc = np.where(np.logical_or(np.abs(FY) > np.pi, np.abs(FX) > np.pi), 1, 0)
        plt.figure()
        plt.title('Pixels Greater Than $\Pi$')
        plt.imshow(loc, cmap='jet')
        plt.colorbar()
        plt.show()

    return phi, mu


def generate_hologram(phi, mu, beam, extent_x, extent_y, dx_eff, initial_lam, z_eff, psf, amp, add_speckle,
                      add_Gaussian_noise, add_Poisson_noise, phase_only_obj):
    if add_speckle:
        speckle = np.array(imread(dir_main + dir_sim + tiff_phase_speckle_filename))
    else:
        speckle = np.zeros_like(phi, dtype=np.float32)

    if phase_only_obj:
        holo, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * initial_lam),
                                    phase_image=(phi + speckle),
                                    absorption_image=None,
                                    beam=beam,
                                    size_pad=[Ny*2, Nx*2],
                                    size_out=[Ny, Nx])
    else:
        holo, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * initial_lam),
                                    phase_image=(phi + speckle),
                                    absorption_image=mu,
                                    beam=beam,
                                    size_pad=[Ny*2, Nx*2],
                                    size_out=[Ny, Nx])

    # Divided by 3 because the counts for the dynamic image were roughly 3-4x smaller than the white fields. This needs
    # to manually be changed but it is consistently ~3 times smaller across runs. A better way to do this is to take
    # the mean of the white field and the mean of the dynamic image and that is the number we divide by.
    holo *= amp

    if apply_psf:
        holo = np.real(IFFT(FFT(holo) * FFT(psf)))
    if add_Poisson_noise:
        holo = np.random.poisson(holo).astype(np.float64)
    if add_Gaussian_noise:
        # max_intensity = np.max(holo) - np.min(holo)
        # std_dev = percent_Gaussian_Noise * max_intensity
        # noise_Gaussian = rng.normal(0, std_dev, holo.shape)  # Mean=0, std_dev=0.033 (3.3% noise)
        # holo = np.abs(holo + noise_Gaussian)  # abs is what FIJI does
        mean_intensity = np.mean(holo)
        std_dev = percent_Gaussian_Noise * mean_intensity
        noise_Gaussian = rng.normal(0, std_dev, holo.shape)  # Mean=0, std_dev=0.033 (3.3% noise)
        holo += noise_Gaussian
        holo[holo < 0] = 0

    if plot_hologram:
        plt.figure()
        plt.imshow(holo,
                   extent=[-extent_x / 2 / um, extent_y / 2 / um, -extent_x / 2 / um, extent_y / 2 / um],
                   cmap='Greys_r')
        plt.show()

    return holo


# Save options
save_img = True

# Plotting options
plot_index_of_refrac = False
plot_phase_map = False
plot_attenuation_map = False
plot_aliasing = False
plot_hologram = False

# # Other options -- For replication of experiment
# apply_psf = True
# phase_only_obj = False
# add_speckle = True
# add_Poisson_noise = True
# add_Gaussian_noise = True
# use_cxro = True  # Use if you have internet access and if the website is up
# use_plane_wave = False
# reflect = True
# crop_initial_size = [2100, 2100]  # For cropping before reflecting
# crop_final_size = [2500, 2500]  # Final padding

# For perfect image
apply_psf = False
phase_only_obj = False
add_speckle = False
add_Poisson_noise = False
add_Gaussian_noise = False
use_cxro = True  # Use if you have internet access and if the website is up
use_plane_wave = True
reflect = True
crop_initial_size = [2100, 2100]  # For cropping before reflecting
crop_final_size = [2500, 2500]  # Final padding
half_img = False  # True if we use half sized xRAGE outputs instead of both left/right sides

# Shift image parameters
shift_y = 0
shift_x = 0

run_holo = "572"

# Main directory
dir_main = "/Users/danielhodge/Desktop/all_runs/"

# Directories to access
dir_sim = "run" + run_holo + "_sim/"
dir_holo_Gaussian_beams = "run" + run_holo + "_Gaussian_beams/"

# # Directories to create
# dir_sim_holos_with_speckle_orig = "run" + run_holo + "_sim_holos_with_speckle_orig/"
# dir_sim_holos_no_speckle_orig = "run" + run_holo + "_sim_holos_no_speckle_orig/"
# dir_sim_holos_with_speckle = "run" + run_holo + "_sim_holos_with_speckle/"
# dir_sim_holos_no_speckle = "run" + run_holo + "_sim_holos_no_speckle/"

# Files to import
tiff_phase_speckle_filename = "phase_speckle.tiff"
tiff_mu_speckle_filename = "attenuation_speckle.tiff"
txt_amps_holos_filename = "amps_holos.txt"
tiff_holos_gbeam_stack_filename = "run" + run_holo + "_gbeam_stack.tiff"

# h5_time1 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004300.h5"
# h5_time2 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004400.h5"
# h5_time3 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004500.h5"
# h5_time4 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004600.h5"

h5_time1 = "/Users/danielhodge/Desktop/XRAGEdata_60um_off/out/2/void-col-phase-attenuation-18.0-keV006500.h5"
# with h5py.File(h5_time1, "r") as f:
#     print("Top-level groups:")
#     print(list(f.keys()))
#     dset = f["/phase"]  # adjust key based on f.keys() output
#     print(dset.shape, dset.dtype)
#     data = dset[()]
#     print(data.shape)
# plt.figure()
# plt.imshow(data)
# plt.show()

# h5_time2 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004400.h5"
# h5_time3 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004500.h5"
# h5_time4 = "/Users/danielhodge/Library/CloudStorage/Box-Box/xpci_void_run307/out/2/" \
#           "void-col-phase-attenuation-18.0-keV004600.h5"

# Files to save. Orig = original before reflecting, shifting, and cropping
tiff_holos_no_speckle_orig_filename = "run" + run_holo + "_sim_holos_no_speckle_orig.tiff"
tiff_holos_with_speckle_orig_filename = "run" + run_holo + "_sim_holos_with_speckle_orig.tiff"
tiff_holos_no_speckle_filename = "run" + run_holo + "_sim_holos_no_speckle.tiff"
tiff_holos_with_speckle_filename = "run" + run_holo + "_sim_holos_with_speckle.tiff"
tiff_phase_orig_filename = "run" + run_holo + "_sim_phase_orig.tiff"
tiff_mu_orig_filename = "run" + run_holo + "_sim_mu_orig.tiff"
tiff_phase_filename = "run" + run_holo + "_sim_phase.tiff"
tiff_mu_filename = "run" + run_holo + "_sim_mu.tiff"

# Access the h5 files
f_h5_time1 = h5py.File(h5_time1)
# f_h5_time2 = h5py.File(h5_time2)
# f_h5_time3 = h5py.File(h5_time3)
# f_h5_time4 = h5py.File(h5_time4)

# Specify array size
Nx = 2500  # Pixels in x
Ny = 2500  # Pixels in y

# Array of beams with shifting positions and specified FWHM
if use_plane_wave:
    beam = np.ones((Ny, Nx))
    amp = 1.0
else:
    beams = np.array(imread(dir_main + dir_holo_Gaussian_beams + tiff_holos_gbeam_stack_filename), dtype=np.float32)
    beam = beams[0]  # Use centered beam for this case -- single shot
    # Various amplitudes from the experimental images
    amps = np.loadtxt(dir_main + dir_sim + txt_amps_holos_filename, delimiter='\t', skiprows=1)
    amp = np.mean(amps)  # Use single amplitude average for this case -- single shot

# Parameters
initial_E = 18000  # Initial energy of the beam in eV
initial_lam = 1240 / initial_E * nm
num_energies = 1
bandwidth = 18  # Corresponds to 0.2% natural bandwidth of an XFEL beam
beam_jitter = 0  # Currently a rough estimate of the energy that causes dilation of the image
energies = [initial_E] + list(np.random.uniform(initial_E - bandwidth - beam_jitter,
                                                initial_E + bandwidth + beam_jitter,
                                                num_energies - 1))
lams = [1240 / E * nm for E in energies]  # Wavelength
N_pad = 6000  # Pad size for propagation
z01 = 120.41 * mm  # Distance from source to sample
z12 = 4.668995 * m  # Distance from sample to detector
z02 = z01 + z12  # Distance from source to detector
M = z02 / z01  # Magnification
scale_fac = 4  # Scale factor we include to compensate for lens mag (optique peter)
detPixSize = 6.5 * um  # Detector pixel size
dx_eff = detPixSize / M / scale_fac  # Object pixel size in x, equals detector pixel size if no mag
dy_eff = detPixSize / M / scale_fac  # Object pixel size in y, equals detector pixel size if no mag
X = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=2 * np.pi / (Nx * dx_eff)))  # Spacing in real space
Y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=2 * np.pi / (Ny * dy_eff)))  # Spacing in real space
x, y = np.meshgrid(X, Y)
z_eff = z12 / M  # Effective propagation distance
extent_x = Nx * dx_eff
extent_y = Ny * dy_eff
k0 = 2 * np.pi / initial_lam
percent_Gaussian_Noise = 0.033  # Gaussian Noise to add to our images

# Parameters for the 2D Voigt profile -- Blur to incorporate effects of scintillator, finite source size, and partial
# degree of transverse coherence of the x-ray beam
sigma_g = 435.621e-9 / 2  # Gaussian sigma
gamma_l = 0e-9  # Lorentzian gamma
psf = voigt_2d(x, y, sigma_g, gamma_l)

# Access the h5 files corresponding to the xRAGE phase and attenuation maps
# h5_files = [f_h5_time1, f_h5_time2, f_h5_time3, f_h5_time4]
h5_files = [f_h5_time1]

# Extract phase and attenuation maps from the h5 files and adjust them to match experimental conditions
processed_data = [process_h5_file(f, half_img=half_img) for f in h5_files]

# Check if there is any aliasing and generate holograms for each of the dynamic images
holos_orig = []  # Original holos before shifting/reflecting/padding
holos = []  # Holos after shifting/reflecting/padding
phis = []
phis_orig = []
mus = []
mus_orig = []
rng = np.random.default_rng()
print("Generating dynamic void hologram with a varied x-ray beam energy")
for phi, mu in processed_data:
    phi, mu = check_aliasing(phi=phi,
                             mu=mu,
                             extent_x=extent_x,
                             extent_y=extent_y,
                             dx_eff=dx_eff,
                             dy_eff=dy_eff)

    holo = generate_hologram(phi=phi,
                             mu=mu,
                             beam=beam,
                             extent_x=extent_x,
                             extent_y=extent_y,
                             dx_eff=dx_eff,
                             initial_lam=initial_lam,
                             z_eff=z_eff,
                             psf=psf,
                             amp=amp,
                             add_speckle=add_speckle,
                             add_Gaussian_noise=add_Gaussian_noise,
                             add_Poisson_noise=add_Poisson_noise,
                             phase_only_obj=phase_only_obj)

    plt.figure()
    plt.imshow(holo)
    plt.title('orig')
    plt.show()

    if reflect:
        # Below is the original holo before any manipulation (cropping/reflecting/shifting - basically the raw image
        # if we are thinking about the experimental image) called "holo_orig". This should correspond to a
        # void image that is captured experimentally. Then we want to save these holos and then use these new
        # manipulated holos because that is what we did for the experimental images for better centered
        # phase reconstructions. We did this since the void was towards the edge of the computational domain.
        holo_orig = holo
        holo_cropped = cropToCenter(holo, crop_initial_size)
        holo_reflected = reflect_image_2d(holo_cropped)
        holo_shifted = shiftRotateMagnifyImage(img=holo_reflected,
                                               shifts=[shift_y, shift_x])
        holo_cropped = cropToCenter(img=holo_shifted,
                                    newSize=crop_final_size)
        holo = holo_cropped

        # Same applied to phi
        phi_orig = phi
        phi_cropped = cropToCenter(phi, crop_initial_size)
        phi_reflected = reflect_image_2d(phi_cropped)
        phi_shifted = shiftRotateMagnifyImage(img=phi_reflected,
                                              shifts=[shift_y, shift_x])
        phi_cropped = cropToCenter(img=phi_shifted,
                                   newSize=crop_final_size)
        phi = phi_cropped

        # Same applied to mu
        mu_orig = mu
        mu_cropped = cropToCenter(mu, crop_initial_size)
        mu_reflected = reflect_image_2d(mu_cropped)
        mu_shifted = shiftRotateMagnifyImage(img=mu_reflected,
                                             shifts=[shift_y, shift_x])
        mu_cropped = cropToCenter(img=mu_shifted,
                                  newSize=crop_final_size)
        mu = mu_cropped


    else:
        # Below is the original holo before any manipulation (cropping/reflecting/shifting - basically the raw image
        # if we are thinking about the experimental image) called "holo_orig". This should correspond to a
        # void image that is captured experimentally. Then we want to save these holos and then use these new
        # manipulated holos because that is what we did for the experimental images for better centered
        # phase reconstructions. We did this since the void was towards the edge of the computational domain.
        holo_orig = holo
        holo_shifted = shiftRotateMagnifyImage(img=holo,
                                               shifts=[shift_y, shift_x])
        holo_padded = padToSize(img=holo_shifted,
                                outputSize=[N_pad, N_pad],
                                padMethod='constant',  # Was 'replicate'
                                padType='both',
                                padValue=0)  # Was none
        holo_cropped = cropToCenter(img=holo_padded,
                                    newSize=crop_final_size)
        holo = holo_cropped

        # Same applied to phi
        phi_orig = phi
        phi_shifted = shiftRotateMagnifyImage(img=phi,
                                              shifts=[shift_y, shift_x])
        phi_padded = padToSize(img=phi_shifted,
                               outputSize=[N_pad, N_pad],
                               padMethod='constant',  # Was 'replicate'
                               padType='both',
                               padValue=0)  # Was none
        phi_cropped = cropToCenter(img=phi_padded,
                                   newSize=crop_final_size)
        phi = phi_cropped

        # Same applied to mu
        mu_orig = mu
        mu_shifted = shiftRotateMagnifyImage(img=mu,
                                             shifts=[shift_y, shift_x])
        mu_padded = padToSize(img=mu_shifted,
                              outputSize=[N_pad, N_pad],
                              padMethod='constant',  # Was 'replicate'
                              padType='both',
                              padValue=0)  # Was none
        mu_cropped = cropToCenter(img=mu_padded,
                                  newSize=crop_final_size)
        mu = mu_cropped

    # if save_img:
    #     if add_speckle:
    #         if not os.path.exists(dir_main + dir_sim + dir_sim_holos_with_speckle_orig):
    #             os.mkdir(dir_main + dir_sim + dir_sim_holos_with_speckle_orig)
    #         imwrite(dir_main + dir_sim + dir_sim_holos_with_speckle_orig + "run" + run_holo + f"_sim_evt_0.tiff",
    #                 holo_orig,
    #                 photometric='minisblack')
    #         if not os.path.exists(dir_main + dir_sim + dir_sim_holos_with_speckle):
    #             os.mkdir(dir_main + dir_sim + dir_sim_holos_with_speckle)
    #         imwrite(dir_main + dir_sim + dir_sim_holos_with_speckle + "run" + run_holo + f"_sim_evt_0.tiff", holo,
    #                 photometric='minisblack')
    #     else:
    #         if not os.path.exists(dir_main + dir_sim_holos_no_speckle_orig):
    #             os.mkdir(dir_main + dir_sim_holos_no_speckle_orig)
    #         imwrite(dir_main + dir_sim_holos_no_speckle_orig + "run" + run_holo + f"_sim_evt_0.tiff", holo_orig,
    #                 photometric='minisblack')
    #         if not os.path.exists(dir_main + dir_sim_holos_no_speckle):
    #             os.mkdir(dir_main + dir_sim_holos_no_speckle)
    #         imwrite(dir_main + dir_sim_holos_no_speckle + "run" + run_holo + f"_sim_evt_0.tiff", holo,
    #                 photometric='minisblack')

    phis.append(phi)
    phis_orig.append(phi_orig)
    mus.append(mu)
    mus_orig.append(mu_orig)
    holos_orig.append(holo_orig)
    holos.append(holo)

phis = np.stack(phis, axis=0).astype(np.float32)
phis_orig = np.stack(phis_orig, axis=0).astype(np.float32)
mus = np.stack(mus, axis=0).astype(np.float32)
mus_orig = np.stack(mus_orig, axis=0).astype(np.float32)
holos_orig = np.stack(holos_orig, axis=0).astype(np.float32)
holos = np.stack(holos, axis=0).astype(np.float32)

if save_img:
    # imwrite(dir_main + dir_sim + tiff_phase_filename, phis, photometric='minisblack')
    # imwrite(dir_main + dir_sim + tiff_phase_orig_filename, phis_orig, photometric='minisblack')
    # imwrite(dir_main + dir_sim + tiff_mu_filename, mus, photometric='minisblack')
    # imwrite(dir_main + dir_sim + tiff_mu_orig_filename, mus_orig, photometric='minisblack')
    if add_speckle:
        imwrite(dir_main + dir_sim + tiff_holos_with_speckle_filename, holos, photometric='minisblack')
        imwrite(dir_main + dir_sim + tiff_holos_with_speckle_orig_filename, holos_orig, photometric='minisblack')
    else:
        imwrite(dir_main + dir_sim + tiff_holos_no_speckle_filename, holos, photometric='minisblack')
        imwrite(dir_main + dir_sim + tiff_holos_no_speckle_orig_filename, holos_orig, photometric='minisblack')


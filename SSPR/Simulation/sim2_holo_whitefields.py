"""

The purpose of this code is to simulate white fields similar to those captured experimentally. This is done by
assuming a Gaussian-type beam that illuminates our object with a specified FWHM. Additionally, we consider an
aperture that limits the object field of view (FOV) since the Be CRLs in the experiment limits the object FOV on
the camera. We also consider the speckle on the Be CRLs and assume the projection approximation (meaning we can
assume a thin object such that there is no internal scattering and project the speckle to a 2D plane and obtain the
phase shift through the equation: phase = -k * delta * T. Similarly, we can obtain the attenuation through the equation:
mu = k * beta * T. Here, k is the wave number, delta is the real part of the refractive index, beta is the imaginary
part of the refractive index and T is the sample thickness. Delta and beta values are obtained from CXRO. Finally,
we include the point spread function (PSF) and Poisson noise (and perhaps Gaussian noise) to simulate blur and
noise which replicates experimental conditions.

"""

# Import libraries
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
import os
import shutil

# Import custom modules
from SSPR.Propagation.SimulateHologram import simulate_hologram
from SSPR.units import *
from SSPR.utilities import voigt_2d, FFT, IFFT, cropToCenter, reflect_image_2d, shiftRotateMagnifyImage, padToSize


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


save_img = True

apply_psf = True
phase_only = False
add_speckle = True
add_Poisson_noise = True
add_Gaussian_noise = True
reflect = True
crop_initial_size = [2100, 2100]  # For cropping before reflecting
crop_final_size = [2500, 2500]  # Final padding
# Shift image parameters to make it more centered
shift_y = 0
shift_x = 0

run_wfs = "561"
run_holo = "576"

# Directories
dir_main = "/Users/danielhodge/Desktop/"

# Directories to access
dir_wfs_Gaussian_beams = "run" + run_wfs + "_Gaussian_beams/"
# dir_sim = "run" + run_holo + "_sim/"
dir_sim = "run" + run_holo + "_sim_test/"

# Directories to create
dir_sim_wfs_with_speckle_orig = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_with_speckle_orig/"
dir_sim_wfs_no_speckle_orig = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_no_speckle_orig/"
dir_sim_wfs_with_speckle = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_with_speckle/"
dir_sim_wfs_no_speckle = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_no_speckle/"

# Files to import
tiff_phase_speckle_filename = "phase_speckle.tiff"
tiff_mu_speckle_filename = "attenuation_speckle.tiff"
txt_amps_wfs_filename = "amps_wfs.txt"
tiff_wfs_gbeam_stack_filename = "run" + run_wfs + "_gbeam_stack.tiff"

# Files to save
tiff_wfs_with_speckle_orig_filename = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_with_speckle_orig.tiff"
tiff_wfs_no_speckle_orig_filename = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_no_speckle_orig.tiff"
tiff_wfs_with_speckle_filename = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_with_speckle.tiff"
tiff_wfs_no_speckle_filename = "run" + run_wfs + "_for_run" + run_holo + "_sim_wfs_no_speckle.tiff"

# Array of beams with several shifting positions and specified FWHM
beams = np.array(imread(dir_main + dir_wfs_Gaussian_beams + tiff_wfs_gbeam_stack_filename))

# Various amplitudes from the experimental images
amps = np.loadtxt(dir_main + dir_sim + txt_amps_wfs_filename, delimiter='\t', skiprows=1)
amp_average = np.mean(amps)
amps = np.insert(amps, 0, amp_average)

initial_E = 18000  # Initial energy of the beam in eV
num_energies = len(beams[0])
bandwidth = 18  # Corresponds to 0.2% natural bandwidth of an XFEL beam (in eV)
beam_jitter = 0  # Rough estimate of the energy (in eV) that causes dilation/contraction of the image
energies = [initial_E] + list(np.random.uniform(initial_E - bandwidth - beam_jitter,
                                                initial_E + bandwidth + beam_jitter,
                                                num_energies - 1))
lams = [1240 / E * nm for E in energies]  # Wavelength
Nx = 2500  # Pixels in x
Ny = 2500  # Pixels in y
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
y, x = np.meshgrid(Y, X, indexing='ij')
z_eff = z12 / M  # Propagation distance
extent_x = Nx * dx_eff  # Field of view
extent_y = Ny * dy_eff  # Field of view

# Gaussian Noise to add to our images
percent_Gaussian_Noise = 0.033

# Parameters for the 2D Voigt profile -- Blur to incorporate effects of scintillator, finite source size, and partial
# degree of transverse coherence of the x-ray beam
sigma_g = 415.444e-9 / 2  # Gaussian sigma
gamma_l = 0e-9  # Lorentzian gamma
psf = voigt_2d(x, y, sigma_g, gamma_l)

# z_crit = [dx_eff ** 2/(z_eff * lam) for lam in lams]
# print("Fresnel Number or z_crit: ", z_crit)
# if z_eff > z_crit:
#     ValueError("WARNING: Propagation distance out of bounds for this method")

if add_speckle:
    print("Generating white field holograms with speckle using varying x-ray beam energies")
    phase_speckle = np.array(imread(dir_main + dir_sim + tiff_phase_speckle_filename), dtype=np.float32)
    mu_speckle = np.array(imread(dir_main + dir_sim + tiff_mu_speckle_filename), dtype=np.float32)
else:
    print("Generating white field holograms without speckle using varying x-ray beam energies")
    phase_speckle = np.zeros_like(beams[0], dtype=np.float32)
    mu_speckle = np.zeros_like(beams[0], dtype=np.float32)

# Clear directories of images
if add_speckle and os.path.exists(dir_main + dir_sim_wfs_with_speckle_orig):
    clear_directory(dir_main=dir_main, dir_sub=dir_sim_wfs_with_speckle_orig)

if not add_speckle and os.path.exists(dir_main + dir_sim_wfs_no_speckle_orig):
    clear_directory(dir_main=dir_main, dir_sub=dir_sim_wfs_no_speckle_orig)

if add_speckle and os.path.exists(dir_main + dir_sim_wfs_with_speckle):
    clear_directory(dir_main=dir_main, dir_sub=dir_sim_wfs_with_speckle)

if not add_speckle and os.path.exists(dir_main + dir_sim_wfs_no_speckle):
    clear_directory(dir_main=dir_main, dir_sub=dir_sim_wfs_no_speckle)


holo_wfs_orig = []  # Original wfs before shifting/reflecting/padding
holo_wfs = []  # Wfs before shifting/reflecting/padding
rng = np.random.default_rng()  # Random noise generator for consistency
for i in tqdm(range(len(beams))):
    # Generate hologram (intensity)
    if phase_only:
        holo_wf, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * lams[i]),
                                       phase_image=phase_speckle,
                                       absorption_image=None,
                                       beam=beams[i],
                                       size_pad=[N_pad, N_pad],
                                       size_out=[Ny, Nx])

    else:
        holo_wf, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * lams[i]),
                                       phase_image=phase_speckle,
                                       absorption_image=mu_speckle,
                                       beam=beams[i],
                                       size_pad=[N_pad, N_pad],
                                       size_out=[Ny, Nx])

    holo_wf *= amps[i]

    if apply_psf:
        holo_wf = np.real(IFFT(FFT(holo_wf) * FFT(psf)))
    if add_Poisson_noise:
        holo_wf = np.random.poisson(holo_wf).astype(np.float64)
    if add_Gaussian_noise:
        # max_intensity = np.max(holo_wf) - np.min(holo_wf)
        # std_dev = percent_Gaussian_Noise * max_intensity
        # noise_Gaussian = rng.normal(0, std_dev, holo_wf.shape)  # Mean=0, std_dev=0.033 (3.3% noise)
        # holo_wf = np.abs(holo_wf + noise_Gaussian)  # abs is what FIJI does
        mean_intensity = np.mean(holo_wf)
        std_dev = percent_Gaussian_Noise * mean_intensity
        noise_Gaussian = rng.normal(0, std_dev, holo_wf.shape)  # Mean=0, std_dev=0.033 (3.3% noise)
        holo_wf += noise_Gaussian
        holo_wf[holo_wf < 0] = 0

    if reflect:
        # Below is the original wf before any manipulation (cropping/reflecting/shifting - basically the raw image
        # if we are thinking about the experimental image) called "holo_wf_orig". This should correspond to a
        # white field that is captured experimentally. Then we want to save these wfs and then use these new
        # manipulated white fields because that is what we did for the experimental images for better centered
        # phase reconstructions. We did this since the void was towards the edge of the computational domain.
        holo_wf_orig = holo_wf

        holo_wf_cropped = cropToCenter(holo_wf, crop_initial_size)
        holo_wf_reflected = reflect_image_2d(holo_wf_cropped)
        holo_wf_shifted = shiftRotateMagnifyImage(img=holo_wf_reflected,
                                                  shifts=[shift_y, shift_x])
        holo_wf_cropped = cropToCenter(img=holo_wf_shifted,
                                       newSize=crop_final_size)
        holo_wf = holo_wf_cropped
    else:
        # Below is the original wf before any manipulation (cropping/reflecting/shifting - basically the raw image
        # if we are thinking about the experimental image) called "holo_wf_orig". This should correspond to a
        # white field that is captured experimentally. Then we want to save these wfs and then use these new
        # manipulated white fields because that is what we did for the experimental images for better centered
        # phase reconstructions. We did this since the void was towards the edge of the computational domain.
        holo_wf_orig = holo_wf

        holo_wf_shifted = shiftRotateMagnifyImage(img=holo_wf,
                                                  shifts=[shift_y, shift_x])
        holo_wf_padded = padToSize(img=holo_wf_shifted,
                                   outputSize=[N_pad, N_pad],
                                   padMethod='constant',  # Was replicate
                                   padType='both',
                                   padValue=0)  # Was none
        holo_wf_cropped = cropToCenter(img=holo_wf_padded,
                                       newSize=crop_final_size)
        holo_wf = holo_wf_cropped

    holo_wfs_orig.append(holo_wf_orig)
    holo_wfs.append(holo_wf)

    if save_img:
        if add_speckle:
            if not os.path.exists(dir_main + dir_sim_wfs_with_speckle_orig):
                os.mkdir(dir_main + dir_sim_wfs_with_speckle_orig)
            imwrite(dir_main + dir_sim_wfs_with_speckle_orig + "run" + run_wfs + f"_sim_evt_{i}.tiff", holo_wf_orig,
                    photometric='minisblack')
            if not os.path.exists(dir_main + dir_sim_wfs_with_speckle):
                os.mkdir(dir_main + dir_sim_wfs_with_speckle)
            imwrite(dir_main + dir_sim_wfs_with_speckle + "run" + run_wfs + f"_sim_evt_{i}.tiff", holo_wf,
                    photometric='minisblack')
        else:
            if not os.path.exists(dir_main + dir_sim_wfs_no_speckle_orig):
                os.mkdir(dir_main + dir_sim_wfs_no_speckle_orig)
            imwrite(dir_main + dir_sim_wfs_no_speckle_orig + "run" + run_wfs + f"_sim_evt_{i}.tiff", holo_wf_orig,
                    photometric='minisblack')
            if not os.path.exists(dir_main + dir_sim_wfs_no_speckle):
                os.mkdir(dir_main + dir_sim_wfs_no_speckle)
            imwrite(dir_main + dir_sim_wfs_no_speckle + "run" + run_wfs + f"_sim_evt_{i}.tiff", holo_wf,
                    photometric='minisblack')

holo_wfs = np.stack(holo_wfs, axis=0).astype(np.float32)
holos_wfs_orig = np.stack(holo_wfs_orig, axis=0).astype(np.float32)

if save_img:
    if add_speckle:
        imwrite(dir_main + dir_sim_wfs_with_speckle_orig + tiff_wfs_with_speckle_orig_filename, holo_wfs_orig,
                photometric='minisblack')
        imwrite(dir_main + dir_sim_wfs_with_speckle + tiff_wfs_with_speckle_filename, holo_wfs,
                photometric='minisblack')
    else:
        imwrite(dir_main + dir_sim_wfs_no_speckle_orig + tiff_wfs_no_speckle_orig_filename, holo_wfs_orig,
                photometric='minisblack')
        imwrite(dir_main + dir_sim_wfs_no_speckle + tiff_wfs_no_speckle_filename, holo_wfs,
                photometric='minisblack')

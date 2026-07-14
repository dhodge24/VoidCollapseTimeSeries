"""

The purpose of this code is to simulate an intensity pattern of the static hollow SiO2 glass shell embedded within SU8.
This is done by using the analytical equation of a sphere (assuming projection approximation, meaning we can assume
a thin object such that there is no internal scattering and project a 3D sphere to a 2D plane and obtain the phase
through the equation: phase = -k * delta * T. Here, k is the wave number, delta is the real part of the refractive
index, and T is the sample thickness) with delta values obtained from CXRO. Then we can forward propagate this phase
map with the addition of speckle to obtain a hologram that more closely corresponds to experimental data. In addition
to speckle being added, the point spread function (PSF), Poisson noise, and Gaussian noise still need to be applied.

"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from mechanize import Browser
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from skimage.transform import resize
import os

# Import custom modules
from SSPR.Propagation.SimulateHologram import simulate_hologram
from SSPR.units import *
from SSPR.utilities import cropToCenter, voigt_2d, FFT, IFFT, shiftRotateMagnifyImage, reflect_image_2d, padToSize

# Save options
save_img = True

# Plotting options
plot_index_of_refrac = False
plot_phase_map = False
plot_attenuation_map = False
plot_aliasing = False
plot_hologram = False

# Other options
apply_psf = True
phase_only_obj = False
add_speckle = True
add_Poisson_noise = True
add_Gaussian_noise = True
use_cxro = True  # Use if you have internet access and if the website is up
use_plane_wave = False
reflect = True
crop_initial_size = [2100, 2100]  # For cropping before reflecting
crop_final_size = [2500, 2500]  # Final padding
# Shift image parameters to make it more centered
shift_y = 0
shift_x = 0


run_wfs = "562"
run_holo = "586"

# Main directory
dir_main = "/Users/danielhodge/Desktop/"

# Directories to access
dir_sim = "run" + run_holo + "_sim/"
dir_holo_Gaussian_beams = "run" + run_holo + "_Gaussian_beams/"

# Directories to create
dir_sim_holos_with_speckle_orig = "run" + run_holo + "_sim_holos_with_speckle_orig/"
dir_sim_holos_no_speckle_orig = "run" + run_holo + "_sim_holos_no_speckle_orig/"
dir_sim_holos_with_speckle = "run" + run_holo + "_sim_holos_with_speckle/"
dir_sim_holos_no_speckle = "run" + run_holo + "_sim_holos_no_speckle/"

# Files to import
tiff_phase_speckle_filename = "phase_speckle.tiff"
tiff_mu_speckle_filename = "attenuation_speckle.tiff"
txt_amps_holos_filename = "amps_holos.txt"
tiff_holos_gbeam_stack_filename = "run" + run_holo + "_gbeam_stack.tiff"

# Files to save. Orig = original before reflecting, shifting, and cropping
tiff_holos_no_speckle_orig_filename = "run" + run_holo + "_sim_holos_no_speckle_orig.tiff"
tiff_holos_with_speckle_orig_filename = "run" + run_holo + "_sim_holos_with_speckle_orig.tiff"
tiff_holos_no_speckle_filename = "run" + run_holo + "_sim_holos_no_speckle.tiff"
tiff_holos_with_speckle_filename = "run" + run_holo + "_sim_holos_with_speckle.tiff"
tiff_phase_orig_filename = "run" + run_holo + "_sim_phase_orig.tiff"
tiff_mu_orig_filename = "run" + run_holo + "_sim_mu_orig.tiff"
tiff_phase_filename = "run" + run_holo + "_sim_phase.tiff"
tiff_mu_filename = "run" + run_holo + "_sim_mu.tiff"

# Specify array size
Nx = 2500  # Pixels in x
Ny = 2500  # Pixels in y

# Array of beams with shifting positions and specified FWHM
if use_plane_wave:
    beams = np.ones((Ny, Nx))
    beams = np.expand_dims(beams, axis=0)
    amps = np.array(1.0, dtype=np.float32)
    amps = np.expand_dims(amps, axis=0)
else:
    beams = np.array(imread(dir_main + dir_holo_Gaussian_beams + tiff_holos_gbeam_stack_filename), dtype=np.float32)
    # Various amplitudes from the experimental images
    amps = np.loadtxt(dir_main + dir_sim + txt_amps_holos_filename, delimiter='\t', skiprows=1)
    amp_average = np.mean(amps)
    amps = np.insert(amps, 0, amp_average)

initial_E = 18000  # Initial energy of the beam in eV
initial_lam = 1240 / initial_E * nm
num_energies = len(beams[0])
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
z_eff = z12 / M  # Propagation distance
extent_x = Nx * dx_eff
extent_y = Ny * dy_eff
k0 = 2 * np.pi / initial_lam

# Define SU8 and SiO2 parameters
d_void_out = np.array(40.707 * um, dtype=np.float32)  # Found using estimate_PSF_params_exp.py
t_shell = np.array(1.449 * um, dtype=np.float32)  # Found using estimate_PSF_params_exp.py
SU8_thickness = 400 * um

# Gaussian Noise to add to our images
percent_Gaussian_Noise = 0.033

y_c = -7.043e-06
x_c = -1.944e-06

# Parameters for the 2D Voigt profile -- Blur to incorporate effects of scintillator, finite source size, and partial
# degree of transverse coherence of the x-ray beam
sigma_g = 415.444e-9  # Gaussian sigma
gamma_l = 0e-9  # Lorentzian gamma
psf = voigt_2d(x, y, sigma_g, gamma_l)

# z_crit = dx_eff**2/(z_eff*lam)
# print("Fresnel Number or z_crit: ", z_crit)
# if z_eff > z_crit:
#     ValueError("WARNING: Propagation distance out of bounds for this method")

if use_cxro:
    # Find delta and beta parameters for SiO2 -- Form here: https://henke.lbl.gov/optical_constants/getdb2.html
    # SiO2 has density 2.2 g/cc
    br1 = Browser()  # Open browser
    br1.open("https://henke.lbl.gov/optical_constants/getdb2.html")  # Open to specified URL
    br1.form = list(br1.forms())[0]  # If the form has no name
    br1.form['Formula'] = 'SiO2'  # Choose element or combination of
    br1.form['Density'] = str(2.2)  # SiO2 density is 2.2 g/cc
    br1.form['Scan'] = ["Energy"]  # Specify energy
    br1.form['Min'] = str(30)  # Minimum x-ray energy to consider -- 30 eV is the min energy allowed
    br1.form['Max'] = str(30000)  # Max x-ray energy to consider -- 30000 eV is the max energy allowed
    br1.form['Npts'] = str(500)  # More points for more accuracy in interpolation -- 500 is maximum
    br1.form['Output'] = ["Text File"]  # Chooses text file so I can extract data
    req1 = br1.submit()  # Submit the form
    data_SiO2 = req1.get_data()  # Obtain the data
    data_SiO2 = data_SiO2.splitlines(True)  # Splits string into a list
    data_SiO2 = data_SiO2[2:]  # Remove title headers and extract only numerical values
    E_vals_SiO2, delta_vals_SiO2, beta_vals_SiO2 = np.loadtxt(data_SiO2, unpack=True)  # Obtain the delta + beta values

    # Find delta and beta parameters for SU8 -- Form here: https://henke.lbl.gov/optical_constants/getdb2.html
    # SU8 is C87 H118 O16 with density 1.185 g/cc or about ~1.2 g/cc
    br2 = Browser()  # Open browser
    br2.open("https://henke.lbl.gov/optical_constants/getdb2.html")  # Open to specified URL
    br2.form = list(br2.forms())[0]  # If the form has no name
    br2.form['Formula'] = 'C87H118O16'  # Choose element or combination of
    br2.form['Density'] = str(1.2)  # SU8 density is 1.2 g/cc
    br2.form['Scan'] = ["Energy"]  # Specify energy
    br2.form['Min'] = str(30)  # Minimum x-ray energy to consider -- 30 eV is the min energy allowed
    br2.form['Max'] = str(30000)  # Max x-ray energy to consider -- 30000 eV is the max energy allowed
    br2.form['Npts'] = str(500)  # More points for more accuracy in interpolation -- 500 is maximum
    br2.form['Output'] = ["Text File"]  # Chooses text file so I can extract data
    req2 = br2.submit()  # Submit the form
    data_SU8 = req2.get_data()  # Obtain the data
    data_SU8 = data_SU8.splitlines(True)  # Splits string into a list
    data_SU8 = data_SU8[2:]  # Remove title headers and extract only numerical values
    E_vals_SU8, delta_vals_SU8, beta_vals_SU8 = np.loadtxt(data_SU8, unpack=True)  # Obtain the delta and beta values

    if plot_index_of_refrac:
        # Plot the refractive index for SiO2
        plt.figure()
        plt.loglog(E_vals_SiO2, delta_vals_SiO2, color='blue', label='Real part (delta)')
        plt.loglog(E_vals_SiO2, beta_vals_SiO2, color='red', label='Imaginary part (beta)')
        plt.title('Energies vs delta/beta Values for SiO2', fontsize=18)
        plt.xlabel('Photon Energy [eV]', fontsize=14)
        plt.ylabel('beta/delta Values', fontsize=14)
        plt.legend()
        plt.show()

        # Plot the refractive index for SU8
        plt.figure()
        plt.loglog(E_vals_SU8, delta_vals_SU8, color='blue', label='Real part (delta)')
        plt.loglog(E_vals_SU8, beta_vals_SU8, color='red', label='Imaginary part (beta)')
        plt.title('Energies vs delta/beta Values for SU8', fontsize=18)
        plt.xlabel('Photon Energy [eV]', fontsize=14)
        plt.ylabel('beta/delta Values', fontsize=14)
        plt.legend()
        plt.show()

    # Interpolation to find the most accurate delta value for SiO2 corresponding to a specific beam energy
    interp_delta_SiO2 = interp1d(E_vals_SiO2, delta_vals_SiO2, kind='cubic')
    delta_SiO2 = interp_delta_SiO2(initial_E)
    interp_beta_SiO2 = interp1d(E_vals_SiO2, beta_vals_SiO2, kind='cubic')
    beta_SiO2 = interp_beta_SiO2(initial_E)

    # Interpolation to find the most accurate delta value for SU8 corresponding to a specific beam energy
    interp_delta_SU8 = interp1d(E_vals_SU8, delta_vals_SU8, kind='cubic')
    delta_SU8 = interp_delta_SU8(initial_E)
    interp_beta_SU8 = interp1d(E_vals_SU8, beta_vals_SU8, kind='cubic')
    beta_SU8 = interp_beta_SU8(initial_E)

# Manual values @ 18 keV in case CXRO is down
else:
    delta_SU8 = 8.32221474e-7
    beta_SU8 = 3.53186008e-10
    delta_SiO2 = 1.41223461e-6
    beta_SiO2 = 3.81504783e-9

# Create the void and embed it in the SU8 material
r_void_out = d_void_out / 2  # Radius of SiO2 void
SU8 = np.zeros((Ny, Nx), dtype=np.float32)  # Create 2D grid for SU8 material
SU8[:] = SU8_thickness  # Thickness of the SU8 along the x-ray propagation direction
arg_bead_out = (r_void_out ** 2 - (x - x_c) ** 2 - (y - y_c) ** 2)  # Argument for a spherical bead
arg_bead_out[arg_bead_out < 0] = 0
t_bead_out = 2.0 * np.real(np.sqrt(arg_bead_out))  # Projection of spherical bead thickness to 2D
assert t_shell >= 0
r_void_in = (d_void_out - t_shell) / 2  # Inner radius of SiO2 void
arg_bead_in = (r_void_in ** 2 - (x - x_c) ** 2 - (y - y_c) ** 2)  # Argument for spherical bead (smaller than tbead)
arg_bead_in[arg_bead_in < 0] = 0
t_bead_in = 2.0 * np.real(np.sqrt(arg_bead_in))  # Projection of spherical bead thickness to 2D
t_void = t_bead_out - t_bead_in
t_void[t_void < 0] = 0
SU8 = SU8 - t_bead_out  # Subtract the entire bead out from the SU8 material to get just SU8 thickness only
SU8[SU8 < 0] = 0

# Phase map of SU8 with embedded SiO2 void (Projection approximation) for a central energy
mu = k0 * (SU8 * beta_SU8 + t_void * beta_SiO2)  # Attenuation map
phase = -k0 * (SU8 * delta_SU8 + t_void * delta_SiO2)  # Phase map

if plot_phase_map:
    plt.figure()
    plt.imshow(phase, cmap="Greys_r")
    plt.show()

if plot_attenuation_map:
    plt.figure()
    plt.imshow(mu, cmap="Greys_r")
    plt.show()

# Used to prevent phase aliasing
mu = resize(image=mu,
            output_shape=(Ny, Nx),
            mode='constant',
            anti_aliasing=True)
phase = resize(image=phase,
               output_shape=(Ny, Nx),
               mode='constant',
               anti_aliasing=True)

# Check the gradients of the phase -- See "Digital simulation of scalar optical diffraction:
# revisiting chirp function sampling criteria and consequences" by David Voelz
FX, FY = np.gradient(phase, dx_eff)
FX *= dx_eff
FY *= dy_eff
max_FX = np.max(np.abs(FX))
max_FY = np.max(np.abs(FY))  # check x and y phase gradients, can't exceed pi according to manuscript

mask_phase = np.zeros(np.shape(phase))  # For identifying aliased pixels
flag_phase = 0  # Set to 1 if we exceed the phase aliasing limit
filter_val = 5  # To smooth out regions where there is aliasing

if max_FX > np.pi:
    flag_phase = 1
    print("WARNING: Phase aliasing in the x-direction")
    print("Aliased regions marked RED")
    bin_phase = np.abs(FX) > np.pi
    mask_phase[bin_phase] = 1.0
    h1 = np.ones((filter_val, filter_val), dtype=float) / filter_val ** 2
    new_phase = convolve2d(phase, h1, mode='same', boundary='symm')
    phase[bin_phase] = new_phase[bin_phase]

if max_FY > np.pi:
    flag_phase = 1
    print("WARNING: Phase aliasing in the y-direction")
    print("Aliased regions marked RED")
    bin_phase = np.abs(FY) > np.pi
    mask_phase[bin_phase] = 1.0
    h1 = np.ones((filter_val, filter_val), dtype=float) / filter_val ** 2
    new_phase = convolve2d(phase, h1, mode='same', boundary='symm')
    phase[bin_phase] = new_phase[bin_phase]

if flag_phase > 0:
    phase_temp = cropToCenter(phase, newSize=[Ny, Nx])
    mask_phase_temp = cropToCenter(mask_phase, newSize=[Ny, Nx])
    plt.figure()
    plt.title("Aliased Phase Regions")
    plt.imshow(phase_temp, cmap='Greys_r')
    # Define a custom colormap with a single color (bright red)
    cmap = LinearSegmentedColormap.from_list('bright_red', ['#FF0000', '#FF0000'])
    plt.imshow(mask_phase_temp, cmap=cmap, alpha=1.0 * (mask_phase_temp == 1))
    plt.show()

if plot_aliasing:
    # Find pixel values greater than pi and then plot it
    zeros = np.zeros_like(phase)
    loc = np.where(np.logical_or(np.abs(FY) > np.pi, np.abs(FX) > np.pi), 1, 0)
    plt.figure()
    plt.title("Pixels Greater Than $\pi$")
    plt.imshow(loc, cmap="jet")
    plt.colorbar()
    plt.show()

if add_speckle:
    phase_speckle = np.array(imread(dir_main + dir_sim + tiff_phase_speckle_filename), dtype=np.float32)
    mu_speckle = np.array(imread(dir_main + dir_sim + tiff_mu_speckle_filename), dtype=np.float32)
else:
    phase_speckle = np.zeros_like(phase, dtype=np.float32)
    mu_speckle = np.zeros_like(mu, dtype=np.float32)

holos_orig = []  # Original holos before shifting/reflecting/padding
holos = []  # Holos after shifting/reflecting/padding
rng = np.random.default_rng()
print("Generating void holograms with varying x-ray beam energies")
for i in tqdm(range(len(beams))):
    # Generate hologram (intensity)
    if phase_only_obj:
        holo, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * lams[i]),
                                    phase_image=(phase + phase_speckle),
                                    absorption_image=None,
                                    beam=beams[i],
                                    size_pad=[N_pad, N_pad],
                                    size_out=[Ny, Nx])
    else:
        holo, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * lams[i]),
                                    phase_image=(phase + phase_speckle),
                                    absorption_image=(mu + mu_speckle),
                                    beam=beams[i],
                                    size_pad=[N_pad, N_pad],
                                    size_out=[Ny, Nx])
    holo *= amps[i]

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

    # plt.figure()
    # plt.imshow(holo)
    # plt.show()

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

    holos_orig.append(holo_orig)
    holos.append(holo)

    if save_img:
        if add_speckle:
            if not os.path.exists(dir_main + dir_sim + dir_sim_holos_with_speckle_orig):
                os.mkdir(dir_main + dir_sim + dir_sim_holos_with_speckle_orig)
            imwrite(dir_main + dir_sim + dir_sim_holos_with_speckle_orig + "run" + run_holo + f"_sim_evt_{i}.tiff", holo_orig,
                    photometric='minisblack')
            if not os.path.exists(dir_main + dir_sim + dir_sim_holos_with_speckle):
                os.mkdir(dir_main + dir_sim + dir_sim_holos_with_speckle)
            imwrite(dir_main + dir_sim + dir_sim_holos_with_speckle + "run" + run_holo + f"_sim_evt_{i}.tiff", holo,
                    photometric='minisblack')
        else:
            if not os.path.exists(dir_main + dir_sim_holos_no_speckle_orig):
                os.mkdir(dir_main + dir_sim_holos_no_speckle_orig)
            imwrite(dir_main + dir_sim_holos_no_speckle_orig + "run" + run_holo + f"_sim_evt_{i}.tiff", holo_orig,
                    photometric='minisblack')
            if not os.path.exists(dir_main + dir_sim_holos_no_speckle):
                os.mkdir(dir_main + dir_sim_holos_no_speckle)
            imwrite(dir_main + dir_sim_holos_no_speckle + "run" + run_holo + f"_sim_evt_{i}.tiff", holo,
                    photometric='minisblack')

# Shift the phase and attenuation to match that of the intensity image so we can compare recons with ground truth
if reflect:
    phase_orig = phase
    phase_cropped = cropToCenter(phase, crop_initial_size)
    phase_reflected = reflect_image_2d(phase_cropped)
    phase_shifted = shiftRotateMagnifyImage(img=phase_reflected,
                                            shifts=[shift_y, shift_x])
    phase_cropped = cropToCenter(img=phase_shifted,
                                 newSize=crop_final_size)
    phase = phase_cropped

    mu_orig = mu
    mu_cropped = cropToCenter(mu, crop_initial_size)
    mu_reflected = reflect_image_2d(mu_cropped)
    mu_shifted = shiftRotateMagnifyImage(img=mu_reflected,
                                         shifts=[shift_y, shift_x])
    mu_cropped = cropToCenter(img=mu_shifted,
                              newSize=crop_final_size)
    mu = mu_cropped
else:
    phase_orig = phase
    phase_shifted = shiftRotateMagnifyImage(img=phase,
                                            shifts=[shift_y, shift_x])
    phase_padded = padToSize(img=phase_shifted,
                             outputSize=[N_pad, N_pad],
                             padMethod='constant',  # Was 'replicate'
                             padType='both',
                             padValue=0)  # Was none
    phase_cropped = cropToCenter(img=phase_padded,
                                 newSize=crop_final_size)
    phase = phase_cropped

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

# Save holos as stack
holos_orig = np.stack(holos_orig, axis=0, dtype=np.float32)
holos = np.stack(holos, axis=0, dtype=np.float32)

if save_img:
    imwrite(dir_main + dir_sim + tiff_phase_orig_filename, phase_orig, photometric='minisblack')
    imwrite(dir_main + dir_sim + tiff_mu_orig_filename, mu_orig, photometric='minisblack')
    imwrite(dir_main + dir_sim + tiff_phase_filename, phase, photometric='minisblack')
    imwrite(dir_main + dir_sim + tiff_mu_filename, mu, photometric='minisblack')
    if add_speckle:
        imwrite(dir_main + dir_sim + tiff_holos_with_speckle_filename, holos_orig, photometric='minisblack')
        imwrite(dir_main + dir_sim + tiff_holos_with_speckle_filename, holos, photometric='minisblack')
    else:
        imwrite(dir_main + dir_sim + tiff_holos_no_speckle_filename, holos_orig, photometric='minisblack')
        imwrite(dir_main + dir_sim + tiff_holos_no_speckle_filename, holos, photometric='minisblack')

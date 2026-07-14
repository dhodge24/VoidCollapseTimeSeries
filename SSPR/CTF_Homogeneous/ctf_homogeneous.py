"""

Code modified and translated from MATLAB to python from here:
https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/phaseRetrieval/holographic/phaserec_ctf.m?ref_type=heads


"""

import numpy as np
from tifffile import imread, imwrite
import scipy.special as sp
import time

from SSPR.Propagation.calc_Fresnel_geometry import calc_Fresnel_geometry
from SSPR.utilities import (showImg, FFT, IFFT, create_circular_mask, fadeoutImage, cropToCenter, padToSize, rotateImage,
                            shiftRotateMagnifyImage)
import matplotlib.pyplot as plt

start = time.process_time()


def smooth_from_row_to_constant_downward(image, start_row, constant_value=None, num_smooth_pixels=10):
    """
    Smoothly transitions from the values in a specified row to a constant value (e.g., 1)
    downwards, over the specified number of rows, replacing any existing values.

    Parameters:
        image (2D array): The input image.
        start_row (int): The row from which to start the smooth transition.
        constant_value (float): The value to which the transition should lead.
        num_smooth_pixels (int): The number of rows over which to smooth the transition.

    Returns:
        smoothed_image (2D array): The image with a smooth transition downward to the constant value.
    """
    # Copy the image to avoid modifying the original
    smoothed_image = np.copy(image)

    # Ensure the start_row is within the bounds of the image
    if start_row >= smoothed_image.shape[0] or start_row < 0:
        raise ValueError(f"start_row {start_row} is out of bounds for the image height.")

    # Get the values from the specified start row
    values_at_start_row = smoothed_image[start_row, :]

    # Calculate the range for smoothing (ensure it does not exceed bounds)
    end_smooth_row = min(smoothed_image.shape[0] - 1, start_row + num_smooth_pixels)

    # Smooth transition from the values in the start row to the constant value
    for i, row in enumerate(range(start_row, end_smooth_row + 1)):
        weight = i / num_smooth_pixels  # Linear interpolation weight
        smoothed_image[row, :] = (1 - weight) * values_at_start_row + weight * constant_value

    # Set all rows below the smoothing region to the constant value
    smoothed_image[end_smooth_row + 1:, :] = constant_value

    return smoothed_image


def get_topmost_pixel(img, mask):
    row_indices, col_indices = np.nonzero(mask)
    topmost_index = np.argmin(row_indices)
    topmost_row = row_indices[topmost_index]
    topmost_col = col_indices[topmost_index]
    topmost_pixel_value = img[topmost_row, topmost_col]
    return topmost_pixel_value


def smooth_from_row_to_constant_upward(image, start_row, mask, constant_value=None, num_smooth_pixels=10):
    """
    Smoothly transitions from the values in a specified row to a constant value (e.g., 1)
    upwards, over the specified number of rows, replacing any existing values.

    Parameters:
        image (2D array): The input image.
        start_row (int): The row from which to start the smooth transition.
        mask (2D array): The mask to apply to the image.
        constant_value (float): The value to which the transition should lead (e.g., 1).
        num_smooth_pixels (int): The number of rows over which to smooth the transition.

    Returns:
        smoothed_image (2D array): The image with a smooth transition upwards to the constant value,
                                   replacing existing values.
    """
    # Copy the image to avoid modifying the original
    smoothed_image = np.copy(image)

    if constant_value is None:
        constant_value = get_topmost_pixel(img=image, mask=mask)

    # Ensure the start_row is within the bounds of the image
    if start_row >= smoothed_image.shape[0] or start_row < 0:
        raise ValueError(f"start_row {start_row} is out of bounds for the image height.")

    # Get the values from the specified start row
    values_at_start_row = smoothed_image[start_row, :]

    # Calculate the range for smoothing (ensure it does not exceed bounds)
    end_smooth_row = max(0, start_row - num_smooth_pixels)

    # Smooth transition from the values in the start row to the constant value (e.g., 1)
    for i, row in enumerate(range(start_row, end_smooth_row - 1, -1)):
        weight = i / num_smooth_pixels  # Linear interpolation weight
        smoothed_image[row, :] = (1 - weight) * values_at_start_row + weight * constant_value

    # Set all rows above the smoothing region to the constant value
    smoothed_image[:end_smooth_row, :] = constant_value

    return smoothed_image


def extend_horizontally(image, mask, percentage_split=0.5):
    """
    The purpose of this function is to use a mask to find corresponding edges of a diffraction pattern
    confined to a circular aperture and extend the edges horizontally to fill the computational domain.
    The extension is applied only to the top percentage of the mask, based on the percentage_split parameter.

    :param image: Image to be extended horizontally such that it fills the computational domain
    :param mask: Circular mask used to find object edges confined to a circular aperture
    :param percentage_split: The percentage (0-1) of the mask height where the horizontal extension is applied
    (default 0.5)
    :return: Extended diffraction pattern
    """

    height, width = image.shape
    new_image = np.copy(image)

    # Find the last row where the mask has non-zero values
    last_mask_row = np.max(np.where(np.any(mask == 1, axis=1))[0])

    # Determine the row where the horizontal extension stops
    split_row = int(last_mask_row * percentage_split)

    # Find the leftmost non-zero pixel for each row in the top percentage of the mask
    left_indices = np.argmax(mask[:split_row, :], axis=1)
    left_indices = np.where(np.any(mask[:split_row, :], axis=1), left_indices, 0)

    # Find the rightmost non-zero pixel for each row in the top percentage of the mask
    right_indices = width - np.argmax(mask[:split_row, ::-1], axis=1) - 1
    right_indices = np.where(np.any(mask[:split_row, :], axis=1), right_indices, width - 1)

    # Calculate the average of the leftmost and rightmost values for each row in the top percentage
    left_values = new_image[np.arange(split_row), left_indices]
    right_values = new_image[np.arange(split_row), right_indices]
    avg_values = (left_values + right_values) / 2

    # Create a matrix with the averaged values propagated across the rows in the top percentage
    avg_propagated = np.tile(avg_values, (width, 1)).T

    # Create masks for the areas to the left of the leftmost and to the right of the rightmost non-zero pixels in the
    # top percentage
    left_mask = np.arange(width) < left_indices[:, None]
    right_mask = np.arange(width) > right_indices[:, None]

    # Update the top portion of the image symmetrically with the averaged values
    new_image[:split_row, :] = np.where(left_mask | right_mask, avg_propagated, new_image[:split_row, :])

    return new_image


class CtfHomogeneous:
    """
    Reconstructs the phase from one or several holograms for a pure phase- or homogeneous object based on the
    CTF-inversion approach by Peter Cloetens et al.

    lim1 : Default = 1e-3 -- Regularization parameter for low frequencies
    lim2 : Default = 1e-1 -- Regularization parameter for high frequencies
    beta_delta_ratio : Default = 0 -- Fixed ratio between absorption and phase shifts to be assumed in the
                                      reconstruction. The value 0 corresponds to assuming a pure phase object.

    """
    def __init__(self, measurements, fresnel_numbers, lim1=1e-3, lim2=1e-1, beta_delta_ratio=0,
                 initial_guess=None, max_iters=100, rho=1e-15, eta=0.999, min_phase=-np.inf, max_phase=np.inf,
                 support=None, tolerance=1e-3, verbose=False):

        self.initial_guess = initial_guess

        self.measurements = np.array(measurements, dtype=np.float32)
        self.fresnel_numbers = np.array(fresnel_numbers, dtype=np.float32)
        self.lim1 = lim1
        self.lim2 = lim2
        self.beta_delta_ratio = beta_delta_ratio
        self.num_dim = np.ndim(self.measurements)
        if self.num_dim == 2:
            self.measurements = np.expand_dims(self.measurements, axis=0)
        self.num_measurements = self.measurements.shape[0]

        self.max_iters = max_iters
        self.tolerance = tolerance
        self.rho = rho
        self.eta = eta
        self.min_phase = min_phase
        self.max_phase = max_phase
        if support is None:
            self.support = []
        else:
            self.support = support
        self.verbose = verbose
        self.do_iterative_optimization = (len(self.support) > 0 or self.min_phase > -np.inf or self.max_phase < np.inf)
        if self.verbose:
            print("Run iterative optimization: ", self.do_iterative_optimization)

        # SPECIAL CASE: if beta_delta_ratio is sufficiently high, regularization of low frequencies is omitted
        if self.lim1 < 2 * self.num_measurements * self.beta_delta_ratio ** 2:
            self.lim1 = 0

        # Check compatibility of Fresnel numbers
        if not isinstance(self.fresnel_numbers, (int, float, list, tuple, np.ndarray)):
            raise ValueError('fresnel_numbers must be a single number, list, tuple, or array.')
        if self.fresnel_numbers.shape[0] != self.num_measurements:
            raise ValueError('fresnel_numbers must be equal to the number of assigned holograms measurements).')

        self.Ny = self.measurements.shape[-2]
        self.Nx = self.measurements.shape[-1]

        self.it_num = 0

    @staticmethod
    def FFT(x):
        """
        2D Fourier transform
        see: https://github.com/numpy/numpy/issues/13442
        see: https://stackoverflow.com/questions/33846123/when-should-i-use-fftshiftfftfftshiftx-and-when-fftx
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))  # Provides correct magnitude AND phase output

    @staticmethod
    def IFFT(x):
        """
        2D inverse Fourier transform
        see: https://github.com/numpy/numpy/issues/13442
        see: https://stackoverflow.com/questions/33846123/when-should-i-use-fftshiftfftfftshiftx-and-when-fftx
        """
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))  # Provides correct magnitude AND phase output

    def ctf_Reg_Weights(self, fresnel_numbers, lim1, lim2):
        """
        Computes regularization weights in Fourier space, suitable for regularized inversion of the CTF and
        similar reconstruction methods. The main idea, originally proposed by Peter Cloetens, is to create a mask
        that smoothly transitions from one (typically low) regularization parameter lim1 at low Fourier frequencies
        around the central CTF-mimimum to a second value lim2 that determines the regularization in
        the higher Fourier frequencies beyond the first CTF-maximum
        """

        FX = np.fft.fftshift(np.fft.fftfreq(self.Nx, d=1.0))  # Pixel size included in Fresnel number, so d=1.0
        FY = np.fft.fftshift(np.fft.fftfreq(self.Ny, d=1.0))  # Pixel size included in Fresnel number, so d=1.0
        fx, fy = np.meshgrid(FX, FY)  # Spatial frequency map

        arg = np.pi * (fx ** 2 + fy ** 2) / fresnel_numbers  # Argument of sin and cos for CTF

        sigma = (np.sqrt(2) - 1) / 8  # was 2 on denominator, 8 works better for some cases
        w = 1 / 2 * sp.erfc((np.sqrt(arg) - 1) / (np.sqrt(2) * sigma))
        reg_weights = lim1 * w + lim2 * (1 - w)

        # For deeply holographic data
        # fresnel_numbers_FOV = fresnel_numbers * self.Nx * self.Ny
        # freq_cutoff_FOV = np.pi * (fresnel_numbers_FOV / self.Nx)
        # w_cutoff_FOV = 1 / 2 * sp.erfc((np.sqrt(fx**2 + fy**2) / freq_cutoff_FOV - 1) / 0.1)
        # reg_weights = w_cutoff_FOV * reg_weights + (1 - w_cutoff_FOV) * 10

        return reg_weights

    def ctf_Calc(self):
        """
        This function computes the phase using the Contrast Transfer Function (CTF) approach
        :return: Phase
        """

        FX = np.fft.fftshift(np.fft.fftfreq(self.Nx, d=1.0))  # Pixel size included in Fresnel number, so d=1.0
        FY = np.fft.fftshift(np.fft.fftfreq(self.Ny, d=1.0))  # Pixel size included in Fresnel number, so d=1.0
        fx, fy = np.meshgrid(FX, FY)  # Spatial frequency map

        sum_CTF_holograms = 0
        sum_CTF_sq = 0
        for i in range(self.num_measurements):
            arg = np.pi * (fx ** 2 + fy ** 2) / self.fresnel_numbers[i]  # Argument of sin and cos for CTF
            # The sign in front of cos is positive because mu = -beta/delta * phi
            # See Eq. 7 in: "Evaluation of phase retrieval approaches in magnified X-ray phase nano computerized
            # tomography applied to bone tissue" by B. Yu
            ctf = 2 * np.sin(arg) + 2 * self.beta_delta_ratio * np.cos(arg)
            sum_CTF_holograms = sum_CTF_holograms + ctf * CtfHomogeneous.FFT(self.measurements[i] - 1.0)  # Numerator
            sum_CTF_sq = sum_CTF_sq + ctf ** 2  # Denominator

        # Add regularization weights in Fourier-space that smoothly transition from the value lim1 in the
        # low-frequency regime (around the central CTF-minimum) to lim2 at larger spatial frequencies beyond
        # the first CTF-maximum
        sum_CTF_sq_reg = sum_CTF_sq + self.ctf_Reg_Weights(fresnel_numbers=np.mean(a=self.fresnel_numbers,
                                                                                   axis=0),
                                                           lim1=self.lim1,
                                                           lim2=self.lim2)
        if self.do_iterative_optimization:

            def prox_CTF(f, sigma):

                x = np.real(IFFT((sum_CTF_holograms + FFT((1.0 / sigma) * f)) / (sum_CTF_sq_reg + 1.0 / sigma)))
                return x

            # Assemble proximal map that implements the optional support- and min/max-constraints
            def Id(f):
                return f

            if self.min_phase > -np.inf:
                def prox_Min(f):
                    return np.maximum(f, self.min_phase)
            else:
                prox_Min = Id

            if self.max_phase < np.inf:
                def prox_Max(f):
                    return np.minimum(f, self.max_phase)
            else:
                prox_Max = Id

            if len(self.support):
                # If padding is applied to the data, the support has to be padded as well
                # support = pad(self.support, ((self.pady, self.pady), (self.padx, self.padx)))
                pass

                def prox_Support(f):
                    # original code
                    # f = self.support * f

                    # # For static image
                    # f, _ = fadeoutImage(img=f,
                    #                     fadeMethod='ellipse',
                    #                     ellipseSize=[0.65, 0.65],  #[0.84, 0.84] 0.6
                    #                     transitionLength=[40, 40],
                    #                     fadeToVal=-30,
                    #                     numSegments=None,
                    #                     bottomApply=False)
                    # showImg(f)

                    # # For dynamic image
                    # mask = create_circular_mask(size=f.shape[0], percentage=0.8, smooth_pixels=50)
                    # f = extend_horizontally(image=f, mask=mask, percentage_split=1)  # Needed for circular FOV
                    # f[2790:, ...] = -19.55  # 1945

                    f = smooth_from_row_to_constant_downward(f,
                                                             start_row=2800,
                                                             constant_value=-19.55,
                                                             num_smooth_pixels=100)

                    # f = smooth_from_row_to_constant_downward(f,
                    #                                          start_row=2500,
                    #                                          constant_value=-19.55,
                    #                                          num_smooth_pixels=50)

                    # f = smooth_from_row_to_constant_downward(f,
                    #                                          start_row=2075,
                    #                                          constant_value=-19.55,
                    #                                          num_smooth_pixels=100)

                    # showImg(f)


                    # f = smooth_from_row_to_constant_downward(f,
                    #                                          start_row=2250,
                    #                                          constant_value=-19.55,
                    #                                          num_smooth_pixels=100)


                    # f = smooth_from_row_to_constant_upward(f,
                    #                                        start_row=300,
                    #                                        mask=None,
                    #                                        constant_value=-56.0,
                    #                                        num_smooth_pixels=50)
                    return f
            else:
                prox_Support = Id

            # Proximal operator of the total constraint functional G
            def prox_Constraints(f):
                return prox_Min(prox_Max(prox_Support(f)))

            def fADMM(prox_H, prox_G, v_hat_start):
                v_hat = v_hat_start
                v_old = 0
                lamm_hat = 0
                lamm_old = 0
                c_old = np.inf
                t = 1

                x = None
                it = 0
                for it in range(self.max_iters):

                    if self.verbose:
                        print('[FADMM_Restart] Iteration: ', it)

                    # Perform ADMM - step
                    x = prox_H(v_hat - lamm_hat)
                    v = prox_G(x + lamm_hat, 1.0 / self.rho)
                    lamm = lamm_hat + x - v
                    resi_primal = np.linalg.norm(x - v)
                    resi_dual = np.linalg.norm(v - v_hat)
                    c = self.rho * (resi_primal ** 2 + resi_dual ** 2)

                    # Compute relative residuals
                    norm_x = np.linalg.norm(x)
                    norm_v = np.linalg.norm(v)
                    norm_v_hat = np.linalg.norm(v_hat)
                    resi_primal = resi_primal / np.maximum(norm_x, norm_v)
                    resi_dual = resi_dual / np.maximum(norm_v, norm_v_hat)

                    if self.verbose:
                        print(f" Relative residuals: primal = {resi_primal:.2e}, dual = {resi_dual:.2e}")

                    # Check stopping conditions
                    if np.maximum(resi_primal, resi_dual) < self.tolerance:
                        print("Condition met...exiting algorithm")
                        break

                    # Check convergence behavior
                    if c < self.eta * c_old:
                        # Case "converging": CONTINUE with extrapolation step
                        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
                        v_hat = v + ((t - 1) / t_new) * (v - v_old)
                        lamm_hat = lamm + ((t - 1) / t_new) * (lamm - lamm_old)
                        lamm_old = lamm
                        c_old = c
                        v_old = v
                        t = t_new
                    else:
                        # Case "not converging": RESTART algorithm with initial step length t = 1
                        t = 1
                        v_hat = v_old
                        lamm_hat = lamm_old
                        c = c_old / self.eta

                result = x
                num_it = it

                return result, num_it

            # Compute minimizer using a fast ADMM algorithm
            result, _ = fADMM(prox_Constraints, prox_CTF, np.zeros(sum_CTF_holograms.shape, dtype=np.float32))
            return result

        else:
            # End result will be IFFT(CTF * [FFT(I) - 1.0]/[CTF^2 + Reg_Weights])
            result = np.real(CtfHomogeneous.IFFT(sum_CTF_holograms / sum_CTF_sq_reg))
            return result


run_holo = "572"
sim = True
save = False
plot_results = False
pure_phase = True

if sim:
    # sim
    dir_main = "/Users/danielhodge/Desktop/all_runs/"
    dir_sim = "run" + run_holo + "_sim/"
    # dir_sim = "run" + run_holo + "_sim_test/"
    # tiff_I = "run" + run_holo + "_sim_holos_with_speckle_FFC_extended_decon.tiff"
    tiff_I = "run" + run_holo + "_sim_holos_with_speckle_FFC_extended.tiff"
    tiff_ph = "run" + run_holo + "_sim_phase_CTF2.tiff"
    if not pure_phase:
        tiff_atten = "run" + run_holo + "_sim_atten_CTF.tiff"
else:
    # exp
    dir_main = "/Users/danielhodge/Desktop/"
    dir_exp = "run" + run_holo + "_exp_preprocessed/"
    tiff_I = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended_decon.tiff"
    tiff_ph = "run" + run_holo + "_exp_phase_CTF3.tiff"
    if not pure_phase:
        tiff_atten = "run" + run_holo + "_exp_atten_CTF.tiff"


# Import hologram intensity
if sim:
    I = np.array(imread(dir_main + dir_sim + tiff_I), dtype=np.float32)
    # I = np.array(imread('/Users/danielhodge/Desktop/all_runs/run572_sim/run572_sim_holos_no_speckle.tiff'), dtype=np.float32)[0]
else:
    I = np.array(imread(dir_main + dir_exp + tiff_I), dtype=np.float32)

# I = cropToCenter(img=I, newSize=[2000, 2000])
I = padToSize(img=I, outputSize=[3500, 3500], padMethod='replicate', padType='both', padValue=None)
# showImg(I, clim=(0, 2))

# I = np.array(imread("/Users/danielhodge/Desktop/run576_sim/run576_sim_holos_no_speckle.tiff"))[0]

# Experimental parameters
Ny, Nx = I.shape
E = 18000  # Initial energy of the beam in eV
lam = (1240 / E) * 1e-9  # Wavelength
# z01 = 63.58816e-3  # Distance from source to sample
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


# CTF parameters
lim1 = 2e-5  # Regularization for low spatial freq. Going up seems to make it more flat -- 8e-6 best perfect
lim2 = 0.5  # Regularization for high spatial freq.
beta_delta_ratio = 0.0  # (0.000424389 for SU8 and 0.00270143 for SiO2)
max_iters = 100
min_phase = -65
max_phase = -15
rho = 1e-3  # 2e-3 best perfect
eta = 0.999
verbose = True

# support = create_circular_mask(size=I.shape[0], percentage=0.855, smooth_pixels=1)
support = np.ones_like(I)

fresnel_numbers = calc_Fresnel_geometry(lam=lam,
                                       det_pixel_size=det_pixel_size,
                                       z01=z01,
                                       z02=z02,
                                       z12=None,
                                       scale_fac=scale_fac)

ctf_alg = CtfHomogeneous(measurements=I,
                          fresnel_numbers=fresnel_numbers,
                          support=support,
                          lim1=lim1,  # Regularization for low spatial freq.
                          lim2=lim2,  # Regularization for high spatial freq.
                          beta_delta_ratio=beta_delta_ratio,
                          max_iters=max_iters,
                          min_phase=min_phase,
                          max_phase=max_phase,
                          rho=rho,
                          eta=eta,
                          verbose=verbose)

ph = ctf_alg.ctf_Calc()
ph = cropToCenter(img=ph, newSize=[2500, 2500])

# ph = rotateImage(img=ph, rotAngleDegree=-4.25)
# ph = shiftRotateMagnifyImage(img=ph, shifts=[-171, 0], padMethod='replicate')

if plot_results:
    plt.figure()
    plt.imshow(ph, cmap='Greys_r', extent=(-extent_x / 2e-6, extent_x / 2e-6, -extent_y / 2e-6 , extent_y / 2e-6))
    plt.title("CTF-fADMM Phase Reconstruction")
    plt.xlabel("Microns [um]")
    plt.ylabel("Microns [um]")
    plt.colorbar()
    plt.show()

if not pure_phase:
    atten = -ph / (1 / beta_delta_ratio)  # Assuming a single material

showImg(ph)
if not pure_phase:
    showImg(atten)

imwrite("/Users/danielhodge/Desktop/run572_sim_test_no_deconvolve.tiff", ph)

# Save result
if save:
    if sim:
        imwrite(dir_main + dir_sim + tiff_ph, ph)
        if not pure_phase:
            imwrite(dir_main + dir_sim + tiff_ph, atten)
    else:
        imwrite(dir_main + dir_exp + tiff_ph, ph)
        if not pure_phase:
            imwrite(dir_main + dir_exp + tiff_ph, atten)
print(time.process_time() - start)
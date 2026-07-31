"""

This code was developed and constructed with significant help from Taylor Buckway


"""


import numpy as np
from scipy.ndimage import shift, center_of_mass
from tifffile import imread, imwrite
from utilities import cropToCenter
import matplotlib.pyplot as plt
from loess import loess_1d


def FFT(img):
    """2D Fourier transform"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))  # Provides correct magnitude and phase output


def IFFT(img):
    """2D inverse Fourier transform"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))  # Provides correct magnitude and phase output


def cross_correlation_fft(img1, img2):
    """Performs cross-correlation between 2 images in Fourier space. Aligns img2 to img1."""
    F = FFT(img1)
    G = FFT(img2)
    cross_correlation = np.real(IFFT(F * np.conj(G)))
    peak_pos = center_of_mass(cross_correlation)  # Enforce sub-pixel precision for peak location
    shift_values = peak_pos - np.array(img1.shape) / 2  # Shift only 1 of the images to align to the other
    print("Relative shift values between img1 and img2 (y, x): ", shift_values)
    img2_aligned = shift(img2, shift_values, mode='constant')  # Shifts img2 to align to img1
    return img2_aligned, cross_correlation


def threshold_criteria(N, criteria='one-bit'):
    if criteria == 'one-bit':
        return (0.5 + 2.4142 / np.sqrt(N)) / (1.5 + 1.4142 / np.sqrt(N))
    elif criteria == 'half-bit':
        return (0.2071 + 1.9102 / np.sqrt(N)) / (1.2071 + 0.9102 / np.sqrt(N))
    elif criteria == '1/7':
        return 1 / 7
    else:
        raise ValueError("Invalid criteria. Use 'one-bit', 'half-bit', or '1/7'.")


def Fourier_ring_correlation(img1, img2, criteria='one-bit'):
    F1 = FFT(img1)
    F2 = FFT(img2)
    shape = img1.shape
    center = np.array([shape[0] // 2, shape[1] // 2])
    y, x = np.indices(shape)
    # Creates a grid where the center is value 0 - values increase radially and are the same value at each radius
    r_map = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    index_map = np.round(r_map).astype(int)
    # max_ind ensures that the outmost ring used to evaluate the FRC will be contained within the square image domain
    max_index = shape[0] // 2
    S12 = np.conjugate(F1) * F2  # Cross power spectrum for F1 and F2
    S1 = np.real(np.conjugate(F1) * F1)  # Power spectrum for F1
    S2 = np.real(np.conjugate(F2) * F2)  # Power spectrum for F2

    FRC = np.zeros(max_index, dtype=np.float32)
    T = np.zeros(max_index, dtype=np.float32)
    for i in range(max_index):
        ring_mask = index_map == i  # Grabs all values i in index_map (same value at each radius, so we obtain rings)
        N = np.count_nonzero(ring_mask)  # Counts the number of non-zero values in ring_mask, corresponding to each ring
        T[i] = threshold_criteria(N, criteria=criteria)
        num = np.real(np.mean(S12[ring_mask]))  # Numerator for FRC
        denom = np.sqrt(np.mean(S1[ring_mask]) * np.mean(S2[ring_mask]))  # Denominator for FRC
        FRC[i] = num / denom if denom != 0 else 0
    FRC /= np.max(FRC)  # Normalize such that the max value is 1

    return FRC, T


def plot_results(FRC, T, pixel_size, pixel_number, criteria='one-bit'):
    xf = np.fft.fftshift(np.fft.fftfreq(pixel_number, d=pixel_size))[pixel_number // 2:]
    smooth_xf, smooth_FRC, _ = loess_1d.loess_1d(xf, FRC, degree=2, frac=0.5)
    smooth_FRC = smooth_FRC / np.max(smooth_FRC)

    # Difference between the FRC and threshold curve - Find where this is minimized (value 0 or close to 0)
    diff = np.abs(smooth_FRC - T)
    valid_range = slice(1, -1)  # Excludes the first and last points as intersection options
    intersect = xf[valid_range][diff[valid_range] == np.min(diff[valid_range])]

    plt.figure(figsize=(12, 8))
    plt.title('Fourier Ring Correlation (FRC) Results')
    plt.plot(xf, FRC, 'b', label='FRC')
    plt.plot(smooth_xf, smooth_FRC, 'r', label='Smoothed FRC')
    plt.plot(smooth_xf, T, 'g--', label=f'Threshold criteria: {criteria}')
    plt.axvline(intersect, color='k', linestyle='--', label=f'{intersect[0]:.3f} nm⁻¹')
    plt.text(
        0.006, 0.6, f'Resolution: {int(round(1 / intersect[0]))} nm',  #FIRE: Fourier Image REsolution in microns.
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.7),
        fontsize=12
    )
    plt.xlabel(r'Spatial frequency $(nm^{-1})$', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(1, 2))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plt.show()


img1 = np.array(imread("/Users/danielhodge/Desktop/WillowStuff/Run_408_evt_1_Zyla_xray_east.tiff"), dtype=np.float32)
img2 = np.array(imread("/Users/danielhodge/Desktop/WillowStuff/Run_408_evt_2_Zyla_xray_east.tiff"), dtype=np.float32)
# img1 = np.array(imread("/Users/danielhodge/Desktop/WillowStuff/Run_847_evt_1_Zyla_xray_east.tiff"), dtype=np.float32)
# img2 = np.array(imread("/Users/danielhodge/Desktop/WillowStuff/Run_847_evt_98_Zyla_xray_east.tiff"), dtype=np.float32)
img1 = cropToCenter(img1, newSize=[2000, 2000])
img2 = cropToCenter(img2, newSize=[2000, 2000])

dx = 77.3  # In nanometers
criteria = 'half-bit'

img2, cc = cross_correlation_fft(img1, img2)
FRC, T = Fourier_ring_correlation(img1, img2, criteria)

plot_results(FRC, T, pixel_number=img1.shape[0], pixel_size=dx, criteria=criteria)

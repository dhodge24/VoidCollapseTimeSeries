"""

References:
1) "Spatial resolution in Bragg-magnified X-ray images as determined by Fourier analysis" by P. Modregger
2) "Single-distance phase retrieval algorithm for Bragg magnifier microscope" by S. Hrivnak

The purpose of this code is to determine the spatial resolution of the reconstructed object using the Fourier
power spectral method from the manuscript "Spatial resolution in bragg-magnified x-ray images as determined by fourier
analysis" by P. Modregger (Ref. 1). Ref. 2 is another example that applies their paper.

How they define their resolution criterion:
"The effective resolution limit is given by the maximum spatial frequency at which the spectral power of the measured
signal equals the spectral power of the underlying noise". Essentially, this mean that as you move to higher spatial
frequencies, you are looking for smaller and finer details in the image. Eventually, the measured variations become
no stronger than the random noise. The effective resolution limit is the highest spatial frequency where the real image
signal is still equal to, or distinguishable from, the noise. In essence:

1 - Below this frequency, the image contains meaningful structural information.
2 - At this frequency, signal and noise are equally strong.
3 - Above this frequency, the apparent fine detail is dominated by noise and cannot be trusted.

Note:
The choice on where you choose the noise level to be is somewhat arbitrary. So if you state a resolution, be sure
to include the plot and supporting evidence to back up your resolution claim. Additionally, it would be beneficial
to use other methods for calculating resolution to confirm that the resolution is roughly the same.

"""

import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from skimage.filters import gaussian


plot_image = False
plot_profile = True
plot_power_spectrum = True
apply_Gaussian_filter = False
# Choose either "horizontal" or "vertical"
profile_direction = "vertical"

img = np.array(imread("/Users/danielhodge/Desktop/time_series_recons_cropped/run576_exp/run576_exp_ph_final.tiff"),
               dtype=np.float32)

if apply_Gaussian_filter:
    img = gaussian(img, sigma=15 / 2.35, truncate=2)

# Experimental parameters
Ny, Nx = img.shape
E = 18000  # Initial energy of the beam in eV
lam = (1240 / E) * 1e-9  # Wavelength
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

if plot_image:
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='Greys_r', extent=(-Nx * dx_eff / 2e-6,
                                            Nx * dx_eff / 2e-6,
                                            -Ny * dy_eff / 2e-6,
                                            Ny * dy_eff / 2e-6))
    plt.xlabel("X [um])")
    plt.ylabel("Y [um]")
    plt.show()

Ny, Nx = img.shape

if profile_direction == "horizontal":
    # Fixed row, varying columns
    line_row = Ny // 2
    start_col, end_col = 0, Nx

    line_profile = img[line_row, start_col:end_col]

elif profile_direction == "vertical":
    # Fixed column, varying rows
    line_col = Nx // 2
    start_row, end_row = 0, Ny

    line_profile = img[start_row:end_row, line_col]

else:
    raise ValueError(
        "profile_direction must be either 'horizontal' or 'vertical'"
    )


if plot_profile:
    plt.figure(figsize=(10, 8))

    plt.imshow(
        img,
        cmap="Greys_r",
        extent=(
            -Nx * dx_eff / 2e-6,
             Nx * dx_eff / 2e-6,
            -Ny * dy_eff / 2e-6,
             Ny * dy_eff / 2e-6
        )
    )

    plt.colorbar(label="Phase")

    if profile_direction == "horizontal":
        # Convert pixel coordinates to micrometers
        x_start = (start_col - Nx / 2) * dx_eff / 1e-6
        x_end = (end_col - Nx / 2) * dx_eff / 1e-6
        y_pos = (line_row - Ny / 2) * dy_eff / 1e-6

        plt.plot(
            [x_start, x_end],
            [y_pos, y_pos],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Horizontal Line Profile"
        )

    elif profile_direction == "vertical":
        # Convert pixel coordinates to micrometers
        x_pos = (line_col - Nx / 2) * dx_eff / 1e-6
        y_start = (start_row - Ny / 2) * dy_eff / 1e-6
        y_end = (end_row - Ny / 2) * dy_eff / 1e-6

        plt.plot(
            [x_pos, x_pos],
            [y_start, y_end],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Vertical Line Profile"
        )

    plt.legend()
    plt.title(f"Image with {profile_direction.capitalize()} Line Profile")
    plt.xlabel("X [µm]")
    plt.ylabel("Y [µm]")
    plt.tight_layout()
    plt.show()

ft_line_profile = np.fft.fft(line_profile)
power_spectrum = np.abs(ft_line_profile)**2

# Create a moving average kernel with a width of 15 pixels
kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size
power_spectrum = np.convolve(power_spectrum, kernel, mode='same')

frequencies = np.fft.fftfreq(len(line_profile), d=dx_eff)
frequencies /= 1e6  # Convert to um^-1
# The Fourier transform returns both positive and negative frequency components, creating a symmetric spectrum about 0
# frequency. In a real-valued input signal, this symmetry means that both positive and negative frequencies contain
# the same information (redundant information), thus we consider only the positive frequencies
positive_freq_indices = frequencies > 0
frequencies = frequencies[positive_freq_indices]
power_spectrum = power_spectrum[positive_freq_indices]


if plot_power_spectrum:
    plt.figure(figsize=(10, 8))
    plt.plot(
        power_spectrum,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="Power Spectrum"
    )

    plt.yscale("log")
    plt.xlabel("Spatial Frequency Index")
    plt.ylabel("Spectral Power")
    plt.title("Power Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Calculate the noise level as the mean of spectral power for frequencies > noise_threshold_freq
noise_threshold_freq = 4  # Threshold frequency in um^-1. This is manually chosen based off the power spectrum plot.
high_freq_indices = frequencies > noise_threshold_freq
# mu_N is determined as the mean spectral power of higher frequencies considered as noise (in our case we chose this
# to be more than 'noise_threshold_freq' based on the power spectrum plot where the signal ends and the noise begins)
mu_N = np.mean(power_spectrum[high_freq_indices])  # Noise level
threshold_level = 2 * mu_N  # Threshold spectral power


# # The Rose criterion is a rule of thumb that describes the minimum signal-to-noise ratio (SNR) required to detect
# # image features with certainty. A signal must be five standard deviations above background to be detectable to a human
# # observer
# Rose_criterion = 5 * np.sqrt(mu_N)


def find_crossing_frequency(frequencies, power_spectrum, threshold_level, start_index, end_index):
    """Use linear interpolation to find the frequency where power_spectrum crosses the threshold level."""
    # Select the two points around the threshold crossing
    f_subset = frequencies[start_index:end_index + 1]
    p_subset = power_spectrum[start_index:end_index + 1]
    interp_func = interp1d(p_subset, f_subset, kind='linear', fill_value="interpolate")
    return float(interp_func(threshold_level))


# ### For the Rose Criterion ###
# first_rose_crossing_frequency = None
# last_rose_crossing_frequency = None
# # Find the highest frequency fulfilling the Rose criterion (crossing downwards)
# above_rose_threshold = np.where(power_spectrum >= Rose_criterion)[0]
# if above_rose_threshold.size > 0 and above_rose_threshold[-1] + 1 < len(frequencies):
#     last_rose_crossing_frequency = find_crossing_frequency(
#         frequencies, power_spectrum, Rose_criterion, above_rose_threshold[-1], above_rose_threshold[-1] + 1
#     )
#     print(f"The highest frequency fulfilling the Rose criterion (approx): {last_rose_crossing_frequency:.4f} um^-1")
#
# # Find the smallest frequency fulfilling the Rose criterion (crossing upwards)
# below_rose_threshold = np.where(power_spectrum <= Rose_criterion)[0]
# if below_rose_threshold.size > 0 and below_rose_threshold[0] > 0:
#     first_rose_crossing_frequency = find_crossing_frequency(
#         frequencies, power_spectrum, Rose_criterion, below_rose_threshold[0] - 1, below_rose_threshold[0]
#     )
#     print(f"The smallest frequency fulfilling the Rose criterion (approx): {first_rose_crossing_frequency:.4f} um^-1")

### For the Resolution threshold ###
first_crossing_frequency = None
last_crossing_frequency = None
# Find the highest frequency fulfilling the resolution criterion
above_threshold = np.where(power_spectrum >= threshold_level)[0]
if above_threshold.size > 0 and above_threshold[-1] + 1 < len(frequencies):
    last_crossing_frequency = find_crossing_frequency(
        frequencies, power_spectrum, threshold_level, above_threshold[-1], above_threshold[-1] + 1
    )
    print(f"The highest frequency fulfilling the resolution criterion (approx): {last_crossing_frequency:.4f} um^-1")

# Find the smallest frequency fulfilling the resolution criterion
below_threshold = np.where(power_spectrum <= threshold_level)[0]
if below_threshold.size > 0 and below_threshold[0] > 0:
    first_crossing_frequency = find_crossing_frequency(
        frequencies, power_spectrum, threshold_level, below_threshold[0] - 1, below_threshold[0]
    )
    print(f"The smallest frequency fulfilling the resolution criterion (approx): {first_crossing_frequency:.4f} um^-1")

# Calculate the average frequency and uncertainty in Fourier space. The uncertainty is given by the highest and
# lowest frequencies satisfying the resolution criterion.
average_frequency = (first_crossing_frequency + last_crossing_frequency) / 2
uncertainty = np.abs(last_crossing_frequency - first_crossing_frequency) / 2

print(f"Fourier space resolution: {average_frequency:.4f} um^-1 ± {uncertainty:.4f} um^-1")

# Calculate the real-space resolution and the associated uncertainty -- Given in Equation 6 in the manuscript:
# "Spatial resolution in Bragg-magnified X-ray images as determined by Fourier analysis" by P. Modregger
real_space_resolution = 1 / average_frequency
real_space_uncertainty = 1 / average_frequency**2 * uncertainty
print(f"Real-space resolution: {real_space_resolution:.4f} um ± {real_space_uncertainty:.4f} um")

plt.figure(figsize=(10, 6))
plt.plot(frequencies, power_spectrum, label='Power Spectrum', color='black')
plt.vlines(x=average_frequency, ymin=0, ymax=threshold_level, colors='purple', linestyles='-', label=f'{average_frequency} um^-1')
plt.vlines(x=first_crossing_frequency, ymin=0, ymax=threshold_level, colors='green', linestyles='--', label=f'{first_crossing_frequency} um^-1')
plt.vlines(x=last_crossing_frequency, ymin=0, ymax=threshold_level, colors='orange', linestyles='--', label=f'{last_crossing_frequency} um^-1')
plt.vlines(x=noise_threshold_freq, ymin=0, ymax=np.max(power_spectrum), colors='cyan', linestyles='-', label='Separation between signal and noise')
plt.axhline(y=mu_N, color='red', linestyle='-', label=r'$\mu_N$')
plt.axhline(y=threshold_level, color='blue', linestyle='-', label=r'$2\mu_N$')
# plt.axhline(y=Rose_criterion, color='magenta', linestyle='-.', label=r'Rose Criterion')
plt.yscale("log")
plt.xlabel("Spatial Frequency [um$^{-1}$]")
plt.ylabel("Log Power Spectrum [dB]")
plt.title("Power Spectrum with Noise Threshold")
plt.legend()
plt.xlim(0, 11)
plt.grid(True)
plt.show()

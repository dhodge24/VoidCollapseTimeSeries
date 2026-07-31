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
from scipy.signal import detrend
from scipy.signal.windows import tukey

# ============================================================
# User settings
# ============================================================

plot_image = False
plot_profile = True
plot_power_spectrum = True

# Threshold frequency in um^-1.
# This is manually selected from the power-spectrum plot (where it falls flat and is comparable to noise)
# val_freq_threshold_min = 5.5
# val_freq_threshold_max = 7.2
val_freq_threshold_min = 7.0
val_freq_threshold_max = 10

# Kernel value to smooth the power spectrum line profile
kern_val = 5

# Choose either "horizontal" or "vertical"
profile_direction = "vertical"


# ============================================================
# Load image and define experimental parameters
# ============================================================

img = np.array(imread("/Users/danielhodge/Desktop/time_series_recons_cropped/run572_exp/run572_exp_ph_final.tiff"),
    dtype=np.float32)

# img = np.array(imread("/Users/danielhodge/Desktop/run306_exp_ph_PGD.tiff"), dtype=np.float32)

Ny, Nx = img.shape

E = 18000                    # X-ray energy [eV]
lam = (1240 / E) * 1e-9     # Wavelength [m]

# z01 = 63.58816e-3
z01 = 120.41e-3             # Source-to-sample distance [m]
z12 = 4.668995              # Sample-to-detector distance [m]
z02 = z01 + z12             # Source-to-detector distance [m]

M = z02 / z01               # Geometric magnification
z_eff = z12 / M             # Effective propagation distance [m]

# scale_fac = 2
scale_fac = 4               # Lens magnification at scintillator
det_pixel_size = 6.5e-6     # Detector pixel size [m]

dx_eff = det_pixel_size / M / scale_fac
dy_eff = det_pixel_size / M / scale_fac
print("Pixel size: ", dx_eff)

extent_x = Nx * dx_eff
extent_y = Ny * dy_eff


# img = np.array(
#     imread("/Users/danielhodge/Desktop/obj_ramp_removed.tiff"),
#     dtype=np.float32
# )
#
# Ny, Nx = img.shape
# # Effective object-plane pixel sizes [m/pixel]
# dx_eff = 30e-9
# dy_eff = 30e-9

# ============================================================
# Plot the full image
# ============================================================

if plot_image:
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
    plt.xlabel(r"X [$\mu$m]")
    plt.ylabel(r"Y [$\mu$m]")
    plt.title("Reconstructed Image")
    plt.tight_layout()
    plt.show()


# ============================================================
# Extract horizontal or vertical line profile
# ============================================================

if profile_direction == "horizontal":

    # Fixed row, varying columns
    line_row = Ny // 2
    start_col = 0
    end_col = Nx

    line_profile = img[
        line_row,
        start_col:end_col
    ]

    # Physical sampling interval for the horizontal profile
    pixel_size_m = dx_eff

elif profile_direction == "vertical":

    # Fixed column, varying rows
    line_col = Nx // 2
    start_row = 0
    end_row = Ny

    line_profile = img[
        start_row:end_row,
        line_col
    ]

    # Physical sampling interval for the vertical profile
    pixel_size_m = dy_eff

else:
    raise ValueError(
        "profile_direction must be either 'horizontal' or 'vertical'"
    )


# Pixel size in micrometers per pixel
pixel_size_um = pixel_size_m / 1e-6

print(
    f"Profile direction: {profile_direction}"
)

print(
    f"Profile length: {len(line_profile)} pixels"
)

print(
    f"Effective pixel size: {pixel_size_um:.6f} um/pixel"
)


# ============================================================
# Plot image and selected line profile
# ============================================================

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
    plt.title(
        f"Image with {profile_direction.capitalize()} Line Profile"
    )
    plt.xlabel(r"X [$\mu$m]")
    plt.ylabel(r"Y [$\mu$m]")
    plt.tight_layout()
    plt.show()


# ============================================================
# Calculate the one-sided Fourier power spectrum
# ============================================================

N = len(line_profile)

# rfft is appropriate because line_profile is real-valued.
# It returns only the non-negative frequency components.
ft_line_profile = np.fft.rfft(line_profile)
power_spectrum_full = np.abs(ft_line_profile) ** 2



# N = len(line_profile)
#
# # Remove the constant and linear phase background
# profile_processed = detrend(
#     line_profile.astype(np.float64),
#     type="linear"
# )
#
# # Reduce FFT endpoint discontinuities
# window = tukey(N, alpha=0.1)
# profile_processed *= window
#
# # One-sided Fourier transform
# ft_line_profile = np.fft.rfft(profile_processed)
# power_spectrum_full = np.abs(ft_line_profile) ** 2

# ============================================================
# Smooth the one-sided power spectrum
# ============================================================

kernel_size = kern_val
kernel = np.ones(kernel_size) / kernel_size

power_spectrum_full = np.convolve(power_spectrum_full, kernel, mode="same")

# ============================================================
# Create both spatial-frequency axes
# ============================================================

# Spatial frequency in cycles per meter, converted to cycles/um
frequencies_um_full = (np.fft.rfftfreq(N, d=pixel_size_m) / 1e6)

# Spatial frequency in cycles per pixel
frequencies_pixel_full = np.fft.rfftfreq(N, d=1.0)

# Remove the zero-frequency/DC component.
#
# This mask is applied exactly once to all full-length arrays.
positive_freq_indices = frequencies_pixel_full > 0
frequencies_um = frequencies_um_full[positive_freq_indices]
frequencies_pixel = frequencies_pixel_full[positive_freq_indices]

power_spectrum = power_spectrum_full[positive_freq_indices]

# Preserve your original variable name for the resolution calculations below. Therefore, all thresholds continue to
# be interpreted in inverse micrometers.
frequencies = frequencies_um


# Confirm that all plotted arrays have the same length
print(f"frequencies_um length: {len(frequencies_um)}")
print(f"frequencies_pixel length: {len(frequencies_pixel)}")
print(f"power_spectrum length: {len(power_spectrum)}")


# ============================================================
# Determine the noise level
# ============================================================

# Threshold frequency in um^-1.
# This is manually selected from the power-spectrum plot.
noise_threshold_freq = val_freq_threshold_min
high_freq_indices = frequencies_um > noise_threshold_freq

if not np.any(high_freq_indices):
    raise ValueError(
        "No frequencies are greater than noise_threshold_freq. "
        "Choose a smaller noise_threshold_freq."
    )

# Mean spectral power in the high-frequency noise region
mu_N = np.mean(power_spectrum[high_freq_indices])

# Resolution threshold
threshold_level = 2 * mu_N


# # Threshold frequency in um^-1.
# # This is manually selected from the power-spectrum plot.
# noise_threshold_freq = val_freq_threshold_min
#
# noise_min_freq = val_freq_threshold_min  # um^-1
# noise_max_freq = val_freq_threshold_max  # um^-1
#
# noise_indices = (
#     (frequencies_um >= noise_min_freq)
#     & (frequencies_um <= noise_max_freq)
# )
#
# mu_N = np.mean(power_spectrum[noise_indices])
# threshold_level = 2 * mu_N


# ============================================================
# Function for interpolating a threshold crossing
# ============================================================

def find_crossing_frequency(
        frequency_axis,
        spectrum,
        threshold,
        start_index,
        end_index
):
    """
    Use linear interpolation to estimate the frequency at which
    the power spectrum crosses the specified threshold.
    """

    # Select the two points surrounding the threshold crossing
    f_subset = frequency_axis[start_index:end_index + 1]

    p_subset = spectrum[start_index:end_index + 1]

    # If the two spectral-power values are identical, interpolation
    # is undefined. Return the midpoint of the two frequencies.
    if np.isclose(p_subset[0], p_subset[-1]):
        return float(np.mean(f_subset))

    interp_func = interp1d(
        p_subset,
        f_subset,
        kind="linear",
        fill_value="extrapolate"
    )

    return float(
        interp_func(threshold)
    )


# ============================================================
# Optional Rose criterion
# ============================================================

# The Rose criterion is a rule of thumb that describes the
# minimum signal-to-noise ratio required to detect a feature.

# Rose_criterion = 5 * mu_N
#
# first_rose_crossing_frequency = None
# last_rose_crossing_frequency = None
#
# # Find the highest frequency fulfilling the Rose criterion
# above_rose_threshold = np.where(
#     power_spectrum >= Rose_criterion
# )[0]
#
# if (
#     above_rose_threshold.size > 0
#     and above_rose_threshold[-1] + 1 < len(frequencies_um)
# ):
#     last_rose_crossing_frequency = find_crossing_frequency(
#         frequencies_um,
#         power_spectrum,
#         Rose_criterion,
#         above_rose_threshold[-1],
#         above_rose_threshold[-1] + 1
#     )
#
#     print(
#         "Highest frequency fulfilling the Rose criterion: "
#         f"{last_rose_crossing_frequency:.4f} um^-1"
#     )
#
# # Find the smallest frequency fulfilling the Rose criterion
# below_rose_threshold = np.where(
#     power_spectrum <= Rose_criterion
# )[0]
#
# if (
#     below_rose_threshold.size > 0
#     and below_rose_threshold[0] > 0
# ):
#     first_rose_crossing_frequency = find_crossing_frequency(
#         frequencies_um,
#         power_spectrum,
#         Rose_criterion,
#         below_rose_threshold[0] - 1,
#         below_rose_threshold[0]
#     )
#
#     print(
#         "Smallest frequency fulfilling the Rose criterion: "
#         f"{first_rose_crossing_frequency:.4f} um^-1"
#     )


# ============================================================
# Find resolution-threshold crossings
# ============================================================

first_crossing_frequency = None
last_crossing_frequency = None


# Find the highest frequency fulfilling the resolution criterion
above_threshold = np.where(power_spectrum >= threshold_level)[0]

if above_threshold.size > 0 and above_threshold[-1] + 1 < len(frequencies_um):
    last_crossing_frequency = find_crossing_frequency(
        frequencies_um,
        power_spectrum,
        threshold_level,
        above_threshold[-1],
        above_threshold[-1] + 1
    )

    print(
        "Highest frequency fulfilling the resolution criterion: "
        f"{last_crossing_frequency:.4f} um^-1"
    )


# Find the smallest frequency fulfilling the resolution criterion
below_threshold = np.where(power_spectrum <= threshold_level)[0]

if below_threshold.size > 0 and below_threshold[0] > 0:
    first_crossing_frequency = find_crossing_frequency(
        frequencies_um,
        power_spectrum,
        threshold_level,
        below_threshold[0] - 1,
        below_threshold[0]
    )

    print(
        "Smallest frequency fulfilling the resolution criterion: "
        f"{first_crossing_frequency:.4f} um^-1"
    )


# Make sure both crossings were found
if first_crossing_frequency is None or last_crossing_frequency is None:
    raise RuntimeError(
        "Both resolution-threshold crossings could not be found. "
        "Inspect the power spectrum and threshold level."
    )


# ============================================================
# Calculate frequency-space resolution
# ============================================================

average_frequency = (first_crossing_frequency + last_crossing_frequency) / 2
frequency_uncertainty = np.abs(last_crossing_frequency - first_crossing_frequency) / 2

print(
    "Fourier-space resolution (inverse microns): "
    f"{average_frequency:.4f} um^-1 "
    f"+/- {frequency_uncertainty:.4f} um^-1"
)


# ============================================================
# Calculate real-space resolution
# ============================================================

real_space_resolution = 1 / average_frequency
real_space_uncertainty = (frequency_uncertainty / average_frequency ** 2)

print(
    "Real-space resolution (microns): "
    f"{real_space_resolution:.4f} um "
    f"+/- {real_space_uncertainty:.4f} um"
)


# ============================================================
# Convert important frequencies to inverse pixels
# ============================================================

# Because:
#
# frequency [pixel^-1]
#     = frequency [um^-1] * pixel size [um/pixel]

noise_threshold_freq_pixel = noise_threshold_freq * pixel_size_um
first_crossing_frequency_pixel = first_crossing_frequency * pixel_size_um
last_crossing_frequency_pixel = last_crossing_frequency * pixel_size_um
average_frequency_pixel = average_frequency * pixel_size_um
frequency_uncertainty_pixel = frequency_uncertainty * pixel_size_um

print(
    "Fourier-space resolution (inverse pixels): "
    f"{average_frequency_pixel:.6f} pixel^-1 "
    f"+/- {frequency_uncertainty_pixel:.6f} pixel^-1"
)


# Real-space resolution expressed in pixels
real_space_resolution_pixels = (1 / average_frequency_pixel)
real_space_uncertainty_pixels = (frequency_uncertainty_pixel / average_frequency_pixel ** 2)

print(
    "Real-space resolution (pixels): "
    f"{real_space_resolution_pixels:.4f} pixels "
    f"+/- {real_space_uncertainty_pixels:.4f} pixels"
)


# ============================================================
# Prepare safe values for logarithmic plots
# ============================================================

# Do not modify the power spectrum used in calculations.
# This clipped copy is only used for plotting on a log scale.
smallest_positive_float = np.finfo(float).tiny
power_spectrum_plot = np.clip(power_spectrum, smallest_positive_float,None)
# minimum_plot_power = np.min(power_spectrum_plot)


# ============================================================
# Plot power spectrum in inverse micrometers
# ============================================================

if plot_power_spectrum:
    plt.figure(figsize=(10, 6))

    plt.plot(
        frequencies_um,
        power_spectrum_plot,
        color="black",
        linewidth=1.5,
        label="Power Spectrum"
    )

    plt.yscale("log")
    # Explicit logarithmic y-axis limits
    y_bottom = np.min(power_spectrum_plot) * 0.5
    y_top = np.max(power_spectrum_plot) * 2
    plt.ylim(y_bottom, y_top)

    plt.vlines(
        x=average_frequency,
        ymin=y_bottom,
        ymax=threshold_level,
        colors="purple",
        linestyles="-",
        label=(
            f"{average_frequency:.4f} "
            r"$\mu$m$^{-1}$"
        )
    )

    # Spectral power where the black curve intersects the purple line
    intersection_power = np.interp(
        average_frequency,
        frequencies_um,
        power_spectrum_plot
    )

    plt.annotate(
        f"Resolution = {real_space_resolution:.4f} "
        r"$\pm$ "
        f"{real_space_uncertainty:.4f} "
        r"$\mu$m",
        xy=(average_frequency, intersection_power),
        xytext=(0.2, 0.4),
        textcoords="axes fraction",
        color="purple",
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color="purple"
        ),
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="purple"
        )
    )

    plt.vlines(
        x=first_crossing_frequency,
        ymin=y_bottom,
        ymax=threshold_level,
        colors="green",
        linestyles="--",
        label=(
            f"{first_crossing_frequency:.4f} "
            r"$\mu$m$^{-1}$"
        )
    )


    plt.vlines(
        x=last_crossing_frequency,
        ymin=y_bottom,
        ymax=threshold_level,
        colors="orange",
        linestyles="--",
        label=(
            f"{last_crossing_frequency:.4f} "
            r"$\mu$m$^{-1}$"
        )
    )

    plt.vlines(
        x=noise_threshold_freq,
        ymin=y_bottom,
        ymax=y_top,
        colors="cyan",
        linestyles="-",
        label="Separation Between Signal and Noise"
    )

    plt.axhline(
        y=mu_N,
        color="red",
        linestyle="-",
        label=r"$\mu_N$"
    )

    plt.axhline(
        y=threshold_level,
        color="blue",
        linestyle="-",
        label=r"$2\mu_N$"
    )

    # plt.axhline(
    #     y=Rose_criterion,
    #     color="magenta",
    #     linestyle="-.",
    #     label="Rose Criterion"
    # )

    plt.yscale("log")
    plt.xlabel(r"Spatial Frequency [$\mu$m$^{-1}$]")
    plt.ylabel("Spectral Power")
    plt.title("Power Spectrum with Noise Threshold " r"[$\mu$m$^{-1}$]"
    )

    plt.xlim(
        0,
        frequencies_um[-1] + 1
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot the same power spectrum in inverse pixels
# ============================================================

if plot_power_spectrum:
    plt.figure(figsize=(10, 6))

    plt.plot(
        frequencies_pixel,
        power_spectrum_plot,
        color="black",
        linewidth=1.5,
        label="Power Spectrum"
    )

    plt.yscale("log")
    # Use the same explicit logarithmic limits
    y_bottom = np.min(power_spectrum_plot) * 0.5
    y_top = np.max(power_spectrum_plot) * 2
    plt.ylim(y_bottom, y_top)

    plt.vlines(
        x=average_frequency_pixel,
        ymin=y_bottom,
        ymax=threshold_level,
        colors="purple",
        linestyles="-",
        label=(
            f"{average_frequency_pixel:.6f} "
            r"pixel$^{-1}$"
        )
    )

    intersection_power = np.interp(
        average_frequency_pixel,
        frequencies_pixel,
        power_spectrum_plot
    )

    plt.annotate(
        f"Resolution = {real_space_resolution:.4f} "
        r"$\pm$ "
        f"{real_space_uncertainty:.4f} "
        r"$\mu$m",
        xy=(average_frequency_pixel, intersection_power),
        xytext=(0.2, 0.4),
        textcoords="axes fraction",
        color="purple",
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color="purple"
        ),
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="purple"
        )
    )

    plt.vlines(
        x=first_crossing_frequency_pixel,
        ymin=y_bottom,
        ymax=threshold_level,
        colors="green",
        linestyles="--",
        label=(
            f"{first_crossing_frequency_pixel:.6f} "
            r"pixel$^{-1}$"
        )
    )

    plt.vlines(
        x=last_crossing_frequency_pixel,
        ymin=y_bottom,
        ymax=threshold_level,
        colors="orange",
        linestyles="--",
        label=(
            f"{last_crossing_frequency_pixel:.6f} "
            r"pixel$^{-1}$"
        )
    )

    plt.vlines(
        x=noise_threshold_freq_pixel,
        ymin=y_bottom,
        ymax=y_top,
        colors="cyan",
        linestyles="-",
        label="Separation Between Signal and Noise"
    )

    plt.axhline(
        y=mu_N,
        color="red",
        linestyle="-",
        label=r"$\mu_N$"
    )

    plt.axhline(
        y=threshold_level,
        color="blue",
        linestyle="-",
        label=r"$2\mu_N$"
    )

    # plt.axhline(
    #     y=Rose_criterion,
    #     color="magenta",
    #     linestyle="-.",
    #     label="Rose Criterion"
    # )

    plt.yscale("log")
    plt.xlabel(r"Spatial Frequency [pixel$^{-1}$]")
    plt.ylabel("Spectral Power")
    plt.title("Power Spectrum with Noise Threshold " r"[pixel$^{-1}$]")

    # Nyquist frequency is 0.5 cycles/pixel
    plt.xlim(0, 0.55)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""

This code was constructed using the manuscript:
1) "Determining the nanostructure of polymer foams using 3D ptycho-tomography for inertial fusion energy applications"
by Hancock et al.


"""



import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from loess import loess_1d


# ============================================================
# User settings
# ============================================================

image_path = ("/Users/danielhodge/Desktop/time_series_recons_cropped/run572_exp/run572_exp_ph_final.tiff")

criteria = "half-bit"

# Show the individual pairwise FRC curves.
plot_pairwise_curves = False

# LOESS smoothing fraction.
loess_frac = 0.03

# False:
#     Use the raw averaged FRC to determine the threshold crossing.
#
# True:
#     Use the smoothed averaged FRC to determine the crossing.
use_smoothed_curve_for_crossing = True

# Exclude the first frequency ring from crossing detection.
start_crossing_index = 1

# Remove the constant image offset before calculating the FRC.
subtract_image_mean = False


# ============================================================
# Centered Fourier transform
# ============================================================

def FFT(img):
    """
    Calculate a centered two-dimensional Fourier transform.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

# ============================================================
# FRC threshold criteria
# ============================================================

def threshold_criteria(N, criteria="half-bit"):
    """
    Calculate the FRC threshold for a ring containing N samples.

    Parameters
    ----------
    N : int
        Number of Fourier coefficients in the ring.

    criteria : str
        Available options:
            "one-bit"
            "half-bit"
            "1/7"

    Returns
    -------
    threshold : float
        Threshold value.
    """
    if N <= 0:
        return np.nan

    if criteria == "one-bit":
        return (0.5 + 2.4142 / np.sqrt(N)) / (1.5 + 1.4142 / np.sqrt(N))

    elif criteria == "half-bit":
        return (0.2071 + 1.9102 / np.sqrt(N)) / (1.2071 + 0.9102 / np.sqrt(N))

    elif criteria == "1/7":
        return 1.0 / 7.0

    else:
        raise ValueError("Invalid criteria. Use 'one-bit', 'half-bit', or '1/7'.")


# ============================================================
# Split image into four full-size alternating-pixel images
# ============================================================

def split_single_image(img):
    """
    Split one image into four full-size alternating-pixel subimages.

    Every subimage retains the dimensions and pixel spacing of the
    original image. Pixels that do not belong to the selected parity
    group are set to zero.

    Pixel assignments
    -----------------
    S1 = even rows, even columns
    S2 = odd rows,  odd columns
    S3 = odd rows,  even columns
    S4 = even rows, odd columns

    The single-image FRC is

        FRC_single =
            [FRC(S1, S2) + FRC(S3, S4)] / 2.

    Parameters
    ----------
    img : ndarray
        Original two-dimensional image.

    Returns
    -------
    S1, S2, S3, S4 : ndarray
        Four full-size masked subimages.
    """
    img = np.asarray(img, dtype=np.float64)

    if img.ndim != 2:
        raise ValueError(f"The input image must be two-dimensional. "f"Received shape {img.shape}.")

    if not np.all(np.isfinite(img)):
        raise ValueError("The input image contains NaN or infinite values.")

    ny, nx = img.shape

    # Make both image dimensions even.
    ny_even = ny - (ny % 2)
    nx_even = nx - (nx % 2)

    if ny_even != ny or nx_even != nx:
        print(f"Cropping image from {img.shape} to "f"{(ny_even, nx_even)} so both dimensions are even.")

        # Center crop by at most one pixel in either dimension.
        y_start = (ny - ny_even) // 2
        x_start = (nx - nx_even) // 2

        img = img[y_start:y_start + ny_even, x_start:x_start + nx_even]

    S1 = np.zeros_like(
        img,
        dtype=np.float64
    )

    S2 = np.zeros_like(
        img,
        dtype=np.float64
    )

    S3 = np.zeros_like(
        img,
        dtype=np.float64
    )

    S4 = np.zeros_like(
        img,
        dtype=np.float64
    )

    # Preserve selected pixels at their original locations.
    S1[0::2, 0::2] = img[0::2, 0::2]
    S2[1::2, 1::2] = img[1::2, 1::2]
    S3[1::2, 0::2] = img[1::2, 0::2]
    S4[0::2, 1::2] = img[0::2, 1::2]

    print("Image shape used:", img.shape)
    print("Masked subimage shape:", S1.shape)

    return S1, S2, S3, S4


# ============================================================
# Two-image Fourier ring correlation
# ============================================================

def Fourier_ring_correlation(
        img1,
        img2,
        pixel_size_y_um,
        pixel_size_x_um,
        criteria="half-bit"
):
    """
    Calculate the FRC between two images using physical-frequency
    rings.

    This implementation supports rectangular images.

    Parameters
    ----------
    img1, img2 : ndarray
        Two images with identical dimensions.

    pixel_size_y_um : float
        Vertical pixel spacing in micrometers per pixel.

    pixel_size_x_um : float
        Horizontal pixel spacing in micrometers per pixel.

    criteria : str
        FRC threshold criterion.

    Returns
    -------
    FRC : ndarray
        Fourier ring correlation curve.

    threshold : ndarray
        Threshold value for each frequency ring.

    ring_counts : ndarray
        Number of Fourier coefficients in each ring.

    spatial_frequency_um : ndarray
        Mean radial spatial frequency of each ring in um^-1.
    """
    img1 = np.asarray(
        img1,
        dtype=np.float64
    )

    img2 = np.asarray(
        img2,
        dtype=np.float64
    )

    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError(
            "Both input images must be two-dimensional."
        )

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have identical shapes. "
            f"Received {img1.shape} and {img2.shape}."
        )

    if pixel_size_x_um <= 0 or pixel_size_y_um <= 0:
        raise ValueError(
            "Pixel sizes must be greater than zero."
        )

    if not np.all(np.isfinite(img1)):
        raise ValueError(
            "img1 contains NaN or infinite values."
        )

    if not np.all(np.isfinite(img2)):
        raise ValueError(
            "img2 contains NaN or infinite values."
        )

    ny, nx = img1.shape

    if ny < 2 or nx < 2:
        raise ValueError(
            "Input images must contain at least two rows "
            "and two columns."
        )

    F1 = FFT(img1)
    F2 = FFT(img2)

    # Physical spatial-frequency axes in inverse micrometers.
    fy_um = np.fft.fftshift(
        np.fft.fftfreq(
            ny,
            d=pixel_size_y_um
        )
    )

    fx_um = np.fft.fftshift(
        np.fft.fftfreq(
            nx,
            d=pixel_size_x_um
        )
    )

    FX_um, FY_um = np.meshgrid(
        fx_um,
        fy_um
    )

    radial_frequency_um = np.sqrt(
        FX_um ** 2 + FY_um ** 2
    )

    df_x_um = np.abs(
        fx_um[1] - fx_um[0]
    )

    df_y_um = np.abs(
        fy_um[1] - fy_um[0]
    )

    # Use the coarser Fourier increment as the ring width.
    ring_width_um = max(
        df_x_um,
        df_y_um
    )

    # Restrict calculation to complete circular rings.
    maximum_complete_frequency_um = min(
        np.max(np.abs(fx_um)),
        np.max(np.abs(fy_um))
    )

    # Keep floor-based ring assignment.
    ring_index_map = np.floor(
        radial_frequency_um / ring_width_um
    ).astype(np.int32)

    number_of_rings = (
        int(
            np.floor(
                maximum_complete_frequency_um / ring_width_um
            )
        )
        + 1
    )

    cross_spectrum = (
        np.conj(F1) * F2
    )

    power_spectrum1 = (
        np.abs(F1) ** 2
    )

    power_spectrum2 = (
        np.abs(F2) ** 2
    )

    FRC = np.full(
        number_of_rings,
        np.nan,
        dtype=np.float64
    )

    threshold = np.full(
        number_of_rings,
        np.nan,
        dtype=np.float64
    )

    ring_counts = np.zeros(
        number_of_rings,
        dtype=np.int64
    )

    spatial_frequency_um = np.full(
        number_of_rings,
        np.nan,
        dtype=np.float64
    )

    for i in range(number_of_rings):

        ring_mask = (
            (ring_index_map == i)
            & (
                radial_frequency_um
                <= maximum_complete_frequency_um
            )
        )

        N = np.count_nonzero(
            ring_mask
        )

        ring_counts[i] = N

        if N == 0:
            continue

        spatial_frequency_um[i] = np.mean(
            radial_frequency_um[ring_mask]
        )

        threshold[i] = threshold_criteria(
            N,
            criteria=criteria
        )

        numerator = np.real(
            np.sum(
                cross_spectrum[ring_mask]
            )
        )

        denominator = np.sqrt(
            np.sum(
                power_spectrum1[ring_mask]
            )
            * np.sum(
                power_spectrum2[ring_mask]
            )
        )

        if denominator > 0:
            FRC[i] = (
                numerator / denominator
            )

    # FRC is already normalized by its definition.
    FRC = np.clip(
        FRC,
        -1.0,
        1.0
    )

    return (
        FRC,
        threshold,
        ring_counts,
        spatial_frequency_um
    )


# ============================================================
# Single-image FRC
# ============================================================

def single_image_FRC(
        img,
        original_pixel_size_y_um,
        original_pixel_size_x_um=None,
        criteria="half-bit"
):
    """
    Calculate single-image FRC from four full-size parity subimages.

    No registration, shifting, compression, or interpolation is applied.

    Parameters
    ----------
    img : ndarray
        Original image.

    original_pixel_size_y_um : float
        Vertical pixel spacing in micrometers per pixel.

    original_pixel_size_x_um : float or None
        Horizontal pixel spacing in micrometers per pixel.

    criteria : str
        FRC threshold criterion.

    Returns
    -------
    results : dict
        FRC curves, threshold, frequency axis, and metadata.
    """
    if original_pixel_size_x_um is None:
        original_pixel_size_x_um = (
            original_pixel_size_y_um
        )

    S1, S2, S3, S4 = split_single_image(
        img
    )

    (
        FRC12,
        threshold12,
        ring_counts12,
        spatial_frequency_um12
    ) = Fourier_ring_correlation(
        S1,
        S2,
        pixel_size_y_um=original_pixel_size_y_um,
        pixel_size_x_um=original_pixel_size_x_um,
        criteria=criteria
    )

    (
        FRC34,
        threshold34,
        ring_counts34,
        spatial_frequency_um34
    ) = Fourier_ring_correlation(
        S3,
        S4,
        pixel_size_y_um=original_pixel_size_y_um,
        pixel_size_x_um=original_pixel_size_x_um,
        criteria=criteria
    )

    if not np.array_equal(
            ring_counts12,
            ring_counts34
    ):
        raise RuntimeError(
            "The two FRC calculations produced different "
            "ring populations."
        )

    if not np.allclose(
            spatial_frequency_um12,
            spatial_frequency_um34,
            equal_nan=True
    ):
        raise RuntimeError(
            "The two FRC calculations produced different "
            "spatial-frequency axes."
        )

    FRC_average = 0.5 * (
        FRC12 + FRC34
    )

    threshold_average = 0.5 * (
        threshold12 + threshold34
    )

    return {
        "FRC": FRC_average,
        "threshold": threshold_average,
        "FRC12": FRC12,
        "FRC34": FRC34,
        "ring_counts": ring_counts12,
        "spatial_frequency_um": spatial_frequency_um12,
        "subimages": (
            S1,
            S2,
            S3,
            S4
        ),
        "original_pixel_size_y_um": original_pixel_size_y_um,
        "original_pixel_size_x_um": original_pixel_size_x_um
    }


# ============================================================
# Threshold-crossing calculation
# ============================================================

def find_first_downward_crossing(
        x,
        curve,
        threshold,
        start_index=1,
        print_diagnostics=True
):
    """
    Find the first downward FRC-threshold crossing.

    A downward crossing satisfies

        FRC[i] >= threshold[i]

    followed by

        FRC[i + 1] < threshold[i + 1].

    Linear interpolation is used between the surrounding samples.

    Returns
    -------
    crossing_frequency : float or None
        Interpolated crossing frequency.

    crossing_status : str
        "crossing"
        "above_through_limit"
        "below_everywhere"
        "no_downward_crossing"
    """
    x = np.asarray(
        x,
        dtype=np.float64
    )

    curve = np.asarray(
        curve,
        dtype=np.float64
    )

    threshold = np.asarray(
        threshold,
        dtype=np.float64
    )

    if not (
        len(x)
        == len(curve)
        == len(threshold)
    ):
        raise ValueError(
            "x, curve, and threshold must have equal lengths."
        )

    finite = (
        np.isfinite(x)
        & np.isfinite(curve)
        & np.isfinite(threshold)
    )

    x = x[finite]
    curve = curve[finite]
    threshold = threshold[finite]

    if len(x) < 2:
        raise RuntimeError(
            "Not enough valid FRC points to evaluate a crossing."
        )

    start_index = max(
        int(start_index),
        0
    )

    if start_index >= len(x) - 1:
        raise ValueError(
            "start_index is too large for the available FRC data."
        )

    difference = (
        curve - threshold
    )

    tested_difference = difference[
        start_index:
    ]

    if print_diagnostics:
        print()
        print("Threshold-crossing diagnostics")
        print("--------------------------------")

        print(
            f"Minimum FRC - threshold: "
            f"{np.min(tested_difference):.6f}"
        )

        print(
            f"Maximum FRC - threshold: "
            f"{np.max(tested_difference):.6f}"
        )

        print(
            f"First tested FRC - threshold: "
            f"{tested_difference[0]:.6f}"
        )

        print(
            f"Final FRC - threshold: "
            f"{tested_difference[-1]:.6f}"
        )

    crossing_indices = np.where(
        (difference[:-1] >= 0)
        & (difference[1:] < 0)
    )[0]

    crossing_indices = crossing_indices[
        crossing_indices >= start_index
    ]

    if crossing_indices.size > 0:

        i = crossing_indices[0]

        x0 = x[i]
        x1 = x[i + 1]

        d0 = difference[i]
        d1 = difference[i + 1]

        if np.isclose(
                d1,
                d0
        ):
            crossing_frequency = x1

        else:
            crossing_frequency = (
                x0
                - d0
                * (x1 - x0)
                / (d1 - d0)
            )

        return (
            crossing_frequency,
            "crossing"
        )

    if np.all(
            tested_difference >= 0
    ):
        return (
            None,
            "above_through_limit"
        )

    if np.all(
            tested_difference < 0
    ):
        return (
            None,
            "below_everywhere"
        )

    return (
        None,
        "no_downward_crossing"
    )


# ============================================================
# Prepare plotting and resolution data
# ============================================================

def prepare_plot_data(
        results,
        loess_frac=0.03,
        use_smoothed_curve=False,
        start_crossing_index=1
):
    """
    Prepare FRC curves for plotting and calculate resolution.
    """
    FRC = np.asarray(
        results["FRC"],
        dtype=np.float64
    )

    threshold = np.asarray(
        results["threshold"],
        dtype=np.float64
    )

    FRC12 = np.asarray(
        results["FRC12"],
        dtype=np.float64
    )

    FRC34 = np.asarray(
        results["FRC34"],
        dtype=np.float64
    )

    spatial_frequency_um = np.asarray(
        results["spatial_frequency_um"],
        dtype=np.float64
    )

    valid = (
        np.isfinite(spatial_frequency_um)
        & np.isfinite(FRC)
        & np.isfinite(threshold)
        & np.isfinite(FRC12)
        & np.isfinite(FRC34)
    )

    spatial_frequency_um = spatial_frequency_um[
        valid
    ]

    FRC = FRC[
        valid
    ]

    threshold = threshold[
        valid
    ]

    FRC12 = FRC12[
        valid
    ]

    FRC34 = FRC34[
        valid
    ]

    if len(spatial_frequency_um) < 3:
        raise RuntimeError(
            "Not enough valid FRC rings were generated."
        )

    order = np.argsort(
        spatial_frequency_um
    )

    spatial_frequency_um = spatial_frequency_um[
        order
    ]

    FRC = FRC[
        order
    ]

    threshold = threshold[
        order
    ]

    FRC12 = FRC12[
        order
    ]

    FRC34 = FRC34[
        order
    ]

    pixel_size_x_um = results[
        "original_pixel_size_x_um"
    ]

    pixel_size_y_um = results[
        "original_pixel_size_y_um"
    ]

    if not np.isclose(
            pixel_size_x_um,
            pixel_size_y_um,
            rtol=1e-6,
            atol=0.0
    ):
        raise ValueError(
            "The inverse-pixel radial axis requires equal x and y "
            "pixel sizes. Current values are "
            f"{pixel_size_x_um:.8f} and "
            f"{pixel_size_y_um:.8f} um/pixel."
        )

    pixel_size_um = 0.5 * (
        pixel_size_x_um + pixel_size_y_um
    )

    # Convert inverse micrometers to inverse pixels:
    #
    # f[pixel^-1] =
    #     f[um^-1] * pixel size[um/pixel]
    spatial_frequency_pixel = (
        spatial_frequency_um
        * pixel_size_um
    )

    # ========================================================
    # Smooth averaged FRC
    # ========================================================

    try:
        (
            smooth_frequency_um,
            smooth_FRC,
            _
        ) = loess_1d.loess_1d(
            spatial_frequency_um,
            FRC,
            degree=2,
            frac=loess_frac
        )

        smooth_frequency_um = np.asarray(
            smooth_frequency_um,
            dtype=np.float64
        )

        smooth_FRC = np.asarray(
            smooth_FRC,
            dtype=np.float64
        )

        smooth_valid = (
            np.isfinite(smooth_frequency_um)
            & np.isfinite(smooth_FRC)
        )

        smooth_frequency_um = smooth_frequency_um[
            smooth_valid
        ]

        smooth_FRC = smooth_FRC[
            smooth_valid
        ]

        smooth_order = np.argsort(
            smooth_frequency_um
        )

        smooth_frequency_um = smooth_frequency_um[
            smooth_order
        ]

        smooth_FRC = smooth_FRC[
            smooth_order
        ]

        smooth_threshold = np.interp(
            smooth_frequency_um,
            spatial_frequency_um,
            threshold
        )

    except Exception as error:

        print(
            "LOESS smoothing failed. The raw averaged FRC "
            "will be used instead."
        )

        print(
            "LOESS error:",
            error
        )

        smooth_frequency_um = (
            spatial_frequency_um.copy()
        )

        smooth_FRC = (
            FRC.copy()
        )

        smooth_threshold = (
            threshold.copy()
        )

    smooth_frequency_pixel = (
        smooth_frequency_um
        * pixel_size_um
    )

    # ========================================================
    # Select curve for crossing detection
    # ========================================================

    if use_smoothed_curve:

        crossing_x_um = smooth_frequency_um
        crossing_curve = smooth_FRC
        crossing_threshold = smooth_threshold
        crossing_curve_name = "smoothed averaged FRC"

    else:

        crossing_x_um = spatial_frequency_um
        crossing_curve = FRC
        crossing_threshold = threshold
        crossing_curve_name = "raw averaged FRC"

    (
        crossing_frequency_um,
        crossing_status
    ) = find_first_downward_crossing(
        crossing_x_um,
        crossing_curve,
        crossing_threshold,
        start_index=start_crossing_index,
        print_diagnostics=True
    )

    crossing_x_pixel = (
        crossing_x_um
        * pixel_size_um
    )

    if crossing_status == "crossing":

        resolution_um = (
            1.0 / crossing_frequency_um
        )

        crossing_frequency_pixel = (
            crossing_frequency_um
            * pixel_size_um
        )

        resolution_pixels = (
            1.0 / crossing_frequency_pixel
        )

    else:

        crossing_frequency_um = None
        crossing_frequency_pixel = None

        resolution_um = np.nan
        resolution_pixels = np.nan

    return {
        "spatial_frequency_um": spatial_frequency_um,
        "spatial_frequency_pixel": spatial_frequency_pixel,
        "FRC": FRC,
        "FRC12": FRC12,
        "FRC34": FRC34,
        "threshold": threshold,
        "smooth_frequency_um": smooth_frequency_um,
        "smooth_frequency_pixel": smooth_frequency_pixel,
        "smooth_FRC": smooth_FRC,
        "smooth_threshold": smooth_threshold,
        "crossing_x_um": crossing_x_um,
        "crossing_x_pixel": crossing_x_pixel,
        "crossing_curve": crossing_curve,
        "crossing_curve_name": crossing_curve_name,
        "crossing_frequency_um": crossing_frequency_um,
        "crossing_frequency_pixel": crossing_frequency_pixel,
        "resolution_um": resolution_um,
        "resolution_pixels": resolution_pixels,
        "crossing_status": crossing_status,
        "pixel_size_um": pixel_size_um
    }


# ============================================================
# Plot one FRC figure
# ============================================================

def plot_frc(
        frequency_axis,
        smooth_frequency_axis,
        crossing_x_axis,
        plot_data,
        x_label,
        title,
        crossing_frequency,
        frequency_unit,
        x_limit=None
):
    """
    Plot one FRC figure using the supplied frequency axis.
    """
    FRC = plot_data[
        "FRC"
    ]

    FRC12 = plot_data[
        "FRC12"
    ]

    FRC34 = plot_data[
        "FRC34"
    ]

    threshold = plot_data[
        "threshold"
    ]

    smooth_FRC = plot_data[
        "smooth_FRC"
    ]

    crossing_status = plot_data[
        "crossing_status"
    ]

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    # ========================================================
    # Plot pairwise and averaged FRC curves
    # ========================================================

    if plot_pairwise_curves:

        ax.plot(
            frequency_axis,
            FRC12,
            linestyle=":",
            alpha=0.65,
            label=r"FRC$(S_1,S_2)$"
        )

        ax.plot(
            frequency_axis,
            FRC34,
            linestyle=":",
            alpha=0.65,
            label=r"FRC$(S_3,S_4)$"
        )

    ax.plot(
        frequency_axis,
        FRC,
        linewidth=1.25,
        label="Average FRC"
    )

    ax.plot(
        smooth_frequency_axis,
        smooth_FRC,
        linewidth=2,
        label="Smoothed average FRC"
    )

    # ========================================================
    # Display-only threshold correction
    # ========================================================
    #
    # Force only the first displayed half-bit threshold point to:
    #
    #     frequency = 0
    #     threshold = 1
    #
    # This does not alter the threshold used for calculating
    # the crossing or resolution.

    threshold_plot_frequency = np.array(
        frequency_axis,
        dtype=np.float64,
        copy=True
    )

    threshold_plot_values = np.array(
        threshold,
        dtype=np.float64,
        copy=True
    )

    if (
        criteria == "half-bit"
        and len(threshold_plot_values) > 0
    ):
        threshold_plot_frequency[0] = 0.0
        threshold_plot_values[0] = 1.0

    ax.plot(
        threshold_plot_frequency,
        threshold_plot_values,
        linestyle="--",
        linewidth=1.5,
        label=f"{criteria} threshold"
    )

    # ========================================================
    # Plot crossing marker, vertical line, arrow, and resolution
    # ========================================================

    if crossing_status == "crossing":

        crossing_correlation = np.interp(
            crossing_frequency,
            crossing_x_axis,
            plot_data["crossing_curve"]
        )

        if frequency_unit == "um":

            annotation_text = (
                f"Resolution = "
                f"{plot_data['resolution_um']:.4f} "
                r"$\mu$m"
            )

            crossing_label = (
                f"Crossing = "
                f"{crossing_frequency:.4f} "
                r"$\mu$m$^{-1}$"
            )

            # Position of the annotation text in axes coordinates.
            annotation_position = (
                0.5,
                0.4
            )

        elif frequency_unit == "pixel":

            annotation_text = (
                f"Resolution = "
                f"{plot_data['resolution_pixels']:.3f} pixels\n"
                f"({plot_data['resolution_um']:.4f} "
                r"$\mu$m)"
            )

            crossing_label = (
                f"Crossing = "
                f"{crossing_frequency:.6f} "
                r"pixel$^{-1}$"
            )

            # Position of the annotation text in axes coordinates.
            annotation_position = (
                0.66,
                0.58
            )

        else:
            raise ValueError(
                "frequency_unit must be 'um' or 'pixel'."
            )

        # Vertical line through the threshold crossing.
        ax.axvline(
            crossing_frequency,
            linestyle="--",
            linewidth=1.5,
            label=crossing_label
        )

        # Marker at the FRC-threshold intersection.
        ax.scatter(
            crossing_frequency,
            crossing_correlation,
            s=45,
            zorder=6
        )

        # Arrow pointing from the resolution annotation to the
        # calculated threshold intersection.
        ax.annotate(
            annotation_text,

            # Arrow tip location.
            xy=(
                crossing_frequency,
                crossing_correlation
            ),

            # Text position in axes-fraction coordinates.
            xytext=annotation_position,
            textcoords="axes fraction",

            horizontalalignment="left",
            verticalalignment="center",
            fontsize=12,

            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.5,
                connectionstyle="arc3,rad=0.05"
            ),

            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.35",
                alpha=0.8
            )
        )

    elif crossing_status == "above_through_limit":

        ax.text(
            0.97,
            0.60,
            (
                "No half-bit crossing\n"
                "Resolution not determined"
            ),
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.3",
                alpha=0.7
            ),
            fontsize=12
        )

    elif crossing_status == "below_everywhere":

        ax.text(
            0.97,
            0.60,
            (
                "FRC below threshold\n"
                "Resolution not determined"
            ),
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.3",
                alpha=0.7
            ),
            fontsize=12
        )

    else:

        ax.text(
            0.97,
            0.60,
            (
                "No downward crossing\n"
                "Resolution not determined"
            ),
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.3",
                alpha=0.7
            ),
            fontsize=12
        )

    # ========================================================
    # Plot formatting
    # ========================================================

    ax.set_title(
        title
    )

    ax.set_xlabel(
        x_label,
        fontsize=12
    )

    ax.set_ylabel(
        "Correlation",
        fontsize=12
    )

    if x_limit is None:
        ax.set_xlim(
            left=0
        )

    else:
        ax.set_xlim(
            x_limit
        )

    ax.set_ylim(
        -0.1,
        1.05
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=11,
        loc="best"
    )

    fig.tight_layout()
    plt.show()


# ============================================================
# Print numerical results
# ============================================================

def print_results(plot_data):
    """
    Print FRC resolution in physical and pixel units.
    """
    maximum_frequency_um = np.max(
        plot_data["spatial_frequency_um"]
    )

    maximum_frequency_pixel = np.max(
        plot_data["spatial_frequency_pixel"]
    )

    print()
    print("Single-image FRC results")
    print("------------------------")

    print(
        f"Crossing calculated from: "
        f"{plot_data['crossing_curve_name']}"
    )

    print(
        f"Pixel size: "
        f"{plot_data['pixel_size_um']:.8f} um/pixel"
    )

    print(
        f"Highest analyzed frequency: "
        f"{maximum_frequency_um:.6f} um^-1"
    )

    print(
        f"Highest analyzed normalized frequency: "
        f"{maximum_frequency_pixel:.6f} pixel^-1"
    )

    if plot_data["crossing_status"] == "crossing":

        print(
            f"Threshold crossing frequency: "
            f"{plot_data['crossing_frequency_um']:.6f} um^-1"
        )

        print(
            f"Threshold crossing frequency: "
            f"{plot_data['crossing_frequency_pixel']:.6f} pixel^-1"
        )

        print(
            f"Real-space resolution: "
            f"{plot_data['resolution_um']:.6f} um"
        )

        print(
            f"Real-space resolution: "
            f"{plot_data['resolution_pixels']:.6f} pixels"
        )

    elif plot_data["crossing_status"] == "above_through_limit":

        print(
            "The FRC remained above the threshold throughout "
            "the analyzed spatial-frequency range."
        )

        print(
            "No numerical resolution was assigned because no "
            "half-bit threshold crossing was observed."
        )

    elif plot_data["crossing_status"] == "below_everywhere":

        print(
            "The FRC remained below the threshold throughout "
            "the analyzed spatial-frequency range."
        )

        print(
            "No valid FRC resolution could be assigned."
        )

    else:

        print(
            "The FRC and threshold did not exhibit a standard "
            "downward crossing."
        )

        print(
            "No numerical resolution was assigned."
        )

    print(
        f"Crossing status: "
        f"{plot_data['crossing_status']}"
    )

    print()


# ============================================================
# Load reconstructed image
# ============================================================

img = np.asarray(
    imread(
        image_path
    ),
    dtype=np.float64
)

if img.ndim != 2:
    raise ValueError(
        f"Expected a two-dimensional image, but received "
        f"shape {img.shape}."
    )

if not np.all(np.isfinite(img)):
    raise ValueError(
        "The loaded image contains NaN or infinite values."
    )

Ny, Nx = img.shape

print(
    "Loaded image shape:",
    img.shape
)


# ============================================================
# Remove constant phase/background offset
# ============================================================

if subtract_image_mean:

    original_mean = np.mean(
        img
    )

    img = (
        img - original_mean
    )

    print(
        f"Removed image mean: "
        f"{original_mean:.6e}"
    )

    print(
        f"Mean after subtraction: "
        f"{np.mean(img):.6e}"
    )


# ============================================================
# Experimental geometry and effective pixel size
# ============================================================

E = 18000                    # X-ray energy [eV]
lam = (1240.0 / E) * 1e-9   # X-ray wavelength [m]

z01 = 120.41e-3             # Source-to-sample distance [m]
z12 = 4.668995              # Sample-to-detector distance [m]
z02 = z01 + z12             # Source-to-detector distance [m]

M = z02 / z01               # Geometric magnification
z_eff = z12 / M             # Effective propagation distance [m]

scale_fac = 4               # Lens magnification at scintillator
det_pixel_size = 6.5e-6     # Detector pixel size [m]

dx_eff = (
    det_pixel_size
    / M
    / scale_fac
)

dy_eff = (
    det_pixel_size
    / M
    / scale_fac
)

extent_x = (
    Nx * dx_eff
)

extent_y = (
    Ny * dy_eff
)

print(
    f"Magnification: "
    f"{M:.6f}"
)

print(
    f"Effective propagation distance: "
    f"{z_eff:.6e} m"
)

print(
    f"Effective x pixel size: "
    f"{dx_eff:.6e} m"
)

print(
    f"Effective y pixel size: "
    f"{dy_eff:.6e} m"
)

print(
    f"Image x extent: "
    f"{extent_x:.6e} m"
)

print(
    f"Image y extent: "
    f"{extent_y:.6e} m"
)


# Convert meters per pixel to micrometers per pixel.
dx_um = (
    dx_eff / 1e-6
)

dy_um = (
    dy_eff / 1e-6
)

print(
    f"Effective x pixel size: "
    f"{dx_um:.8f} um/pixel"
)

print(
    f"Effective y pixel size: "
    f"{dy_um:.8f} um/pixel"
)


# ============================================================
# Calculate single-image FRC
# ============================================================

results = single_image_FRC(
    img,
    original_pixel_size_y_um=dy_um,
    original_pixel_size_x_um=dx_um,
    criteria=criteria
)


# ============================================================
# Prepare plotting and resolution results
# ============================================================

plot_data = prepare_plot_data(
    results,
    loess_frac=loess_frac,
    use_smoothed_curve=use_smoothed_curve_for_crossing,
    start_crossing_index=start_crossing_index
)


# ============================================================
# Print numerical results
# ============================================================

print_results(
    plot_data
)


# ============================================================
# Plot 1: spatial frequency in inverse micrometers
# ============================================================

plot_frc(
    frequency_axis=plot_data["spatial_frequency_um"],
    smooth_frequency_axis=plot_data["smooth_frequency_um"],
    crossing_x_axis=plot_data["crossing_x_um"],
    plot_data=plot_data,
    x_label=r"Spatial frequency [$\mu$m$^{-1}$]",
    title=(
        "Single-Image Fourier Ring Correlation "
        r"[$\mu$m$^{-1}$]"
    ),
    crossing_frequency=plot_data["crossing_frequency_um"],
    frequency_unit="um",
    x_limit=None
)


# ============================================================
# Plot 2: spatial frequency in inverse pixels
# ============================================================

plot_frc(
    frequency_axis=plot_data["spatial_frequency_pixel"],
    smooth_frequency_axis=plot_data["smooth_frequency_pixel"],
    crossing_x_axis=plot_data["crossing_x_pixel"],
    plot_data=plot_data,
    x_label=r"Spatial frequency [pixel$^{-1}$]",
    title=(
        "Single-Image Fourier Ring Correlation "
        r"[pixel$^{-1}$]"
    ),
    crossing_frequency=plot_data["crossing_frequency_pixel"],
    frequency_unit="pixel",

    # The one-dimensional Nyquist frequency is
    # 0.5 cycles per pixel.
    x_limit=(0.0, 0.55)
)


# ============================================================
# Final result
# ============================================================

print(
    "Final crossing status:",
    plot_data["crossing_status"]
)

if plot_data["crossing_status"] == "crossing":

    print(
        f"Final crossing frequency: "
        f"{plot_data['crossing_frequency_um']:.6f} um^-1"
    )

    print(
        f"Final crossing frequency: "
        f"{plot_data['crossing_frequency_pixel']:.6f} pixel^-1"
    )

    print(
        f"Final real-space resolution: "
        f"{plot_data['resolution_um']:.6f} um"
    )

    print(
        f"Final real-space resolution: "
        f"{plot_data['resolution_pixels']:.6f} pixels"
    )

else:

    print(
        "No numerical resolution was assigned because a valid "
        "downward half-bit threshold crossing was not observed."
    )
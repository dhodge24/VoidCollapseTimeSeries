import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Marker pixel coordinates from the image
# ------------------------------------------------------------

black_pts_px = np.array([
    [221.08150470219437, 588.9278996865205],
    [396.08150470219425, 466.8667711598747],
    [556.269592476489,   352.9373040752353],
    [731.8181818181818,  230.23040752351088],
    [892.0062695924764,  111.8777429467085],
])

gray_pts_px = np.array([
    [377.9780564263322, 156.60501567398117],
    [432.8369905956113, 196.9263322884012],
    [500.0391849529781, 261.1285266457682],
    [568.3385579937304, 320.7648902821317],
    [638.5579937304076, 379.6583072100312],
    [712.3432601880878, 435.77429467084653],
    [794.0830721003135, 502.0736677115989],
    [861.2852664576802, 560.241379310345],
    [900.7836990595611, 591.8652037617557],
])

# ------------------------------------------------------------
# Original traced line ENDPOINTS from the attached figure
# These are the anchor points that stay fixed
# ------------------------------------------------------------

black_line_endpoints_px = np.array([
    [189.5, 610.1818181818181],   # left end
    [917.0, 98.5],                # right end
])

gray_line_endpoints_px = np.array([
    [284.59717868338566, 85.0],   # left end
    [916.25, 601.5],              # right end
])

# ------------------------------------------------------------
# Optional marker-position tuning
# Leave as zeros unless you want to move the plotted dots slightly
# ------------------------------------------------------------

black_marker_y_offsets_px = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
])

gray_marker_y_offsets_px = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
])

black_pts_px_plot = black_pts_px.copy()
gray_pts_px_plot = gray_pts_px.copy()

black_pts_px_plot[:, 1] += black_marker_y_offsets_px
gray_pts_px_plot[:, 1] += gray_marker_y_offsets_px

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def make_log_axis_converter(pixel_1, value_1, pixel_2, value_2):
    """
    Convert pixel coordinate to data coordinate for a logarithmic axis.
    """
    a = (np.log10(value_2) - np.log10(value_1)) / (pixel_2 - pixel_1)
    b = np.log10(value_1) - a * pixel_1

    def converter(pixel):
        return 10 ** (a * pixel + b)

    return converter


def make_inverse_log_axis_converter(pixel_1, value_1, pixel_2, value_2):
    """
    Convert data coordinate to pixel coordinate for a logarithmic axis.
    """
    a = (np.log10(value_2) - np.log10(value_1)) / (pixel_2 - pixel_1)
    b = np.log10(value_1) - a * pixel_1

    def inverse_converter(value):
        return (np.log10(value) - b) / a

    return inverse_converter


def straight_line_from_pixel_endpoints(
    endpoints_px,
    px_to_y,
    x_min,
    x_max,
    npts=500
):
    """
    Construct a perfectly straight line in pixel space using two endpoint anchors,
    then extend that straight line to the full x-range of the plot.

    Because the plotted axes are logarithmic, this becomes a straight line
    on the log-log plot as well.
    """
    x1_px, y1_px = endpoints_px[0]
    x2_px, y2_px = endpoints_px[1]

    m_px = (y2_px - y1_px) / (x2_px - x1_px)
    b_px = y1_px - m_px * x1_px

    x_data = np.logspace(np.log10(x_min), np.log10(x_max), npts)
    x_line_px = data_to_px_x(x_data)
    y_line_px = m_px * x_line_px + b_px
    y_data = px_to_y(y_line_px)

    return x_data, y_data, m_px, b_px


# ------------------------------------------------------------
# Axis calibration
# ------------------------------------------------------------

# Bottom x-axis calibration:
# x_pixel = 396.0815 corresponds to v/c_s = 0.1
# x_pixel = 731.8182 corresponds to v/c_s = 1.0

px_to_x = make_log_axis_converter(
    pixel_1=396.08150470219425,
    value_1=0.1,
    pixel_2=731.8181818181818,
    value_2=1.0
)

data_to_px_x = make_inverse_log_axis_converter(
    pixel_1=396.08150470219425,
    value_1=0.1,
    pixel_2=731.8181818181818,
    value_2=1.0
)

# Left y-axis calibration for black data
px_to_y_left = make_log_axis_converter(
    pixel_1=469.8667711598747,
    value_1=1e-2,
    pixel_2=231.23040752351088,
    value_2=1.0
)

# Right y-axis calibration for gray data
px_to_y_right = make_log_axis_converter(
    pixel_1=95.0,
    value_1=1.0,
    pixel_2=552.0,
    value_2=1e-5
)

# ------------------------------------------------------------
# Convert marker coordinates
# ------------------------------------------------------------

x_black_pts = px_to_x(black_pts_px_plot[:, 0])
y_black_pts = px_to_y_left(black_pts_px_plot[:, 1])

x_gray_pts = px_to_x(gray_pts_px_plot[:, 0])
y_gray_pts = px_to_y_right(gray_pts_px_plot[:, 1])

# ------------------------------------------------------------
# Plot limits
# ------------------------------------------------------------

x_min = 0.02
x_max = 3.6

y1_min = 5e-4
y1_max = 15

y2_min = 2e-6
y2_max = 1.2

# ------------------------------------------------------------
# Straight anchored lines extended to the plot edges
# ------------------------------------------------------------

x_black_fit, y_black_fit, black_m_px, black_b_px = straight_line_from_pixel_endpoints(
    black_line_endpoints_px,
    px_to_y_left,
    x_min,
    x_max,
    npts=500
)

x_gray_fit, y_gray_fit, gray_m_px, gray_b_px = straight_line_from_pixel_endpoints(
    gray_line_endpoints_px,
    px_to_y_right,
    x_min,
    x_max,
    npts=500
)

# ------------------------------------------------------------
# Print extracted values
# ------------------------------------------------------------

print("Black marker data:")
for x, y in zip(x_black_pts, y_black_pts):
    print(f"{x:.6g}, {y:.6g}")

print("\nGray marker data:")
for x, y in zip(x_gray_pts, y_gray_pts):
    print(f"{x:.6g}, {y:.6g}")

print("\nBlack line in pixel space:")
print(f"y_px = {black_m_px:.6f} x_px + {black_b_px:.6f}")

print("\nGray line in pixel space:")
print(f"y_px = {gray_m_px:.6f} x_px + {gray_b_px:.6f}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(12, 8))

# Left axis
ax1.set_xscale("log")
ax1.set_yscale("log")

# Black straight anchored line
ax1.plot(
    x_black_fit,
    y_black_fit,
    color="black",
    linewidth=0.9,
    zorder=2
)

# Black markers
ax1.plot(
    x_black_pts,
    y_black_pts,
    linestyle="None",
    marker="o",
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
    markersize=6,
    zorder=3
)

ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y1_min, y1_max)

ax1.set_xlabel(r"$v/c_s$", fontsize=16)
ax1.set_ylabel(
    r"$E_{\rm rot}/(\rho_{\rm ICM}V_{\rm bubble}c_s^2/2)$",
    fontsize=16
)

ax1.tick_params(
    axis="both",
    which="major",
    labelsize=14,
    direction="in",
    length=7
)

ax1.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=4
)

# Right axis
ax2 = ax1.twinx()
ax2.set_yscale("log")

# Gray straight anchored line
ax2.plot(
    x_gray_fit,
    y_gray_fit,
    color="0.6",
    linewidth=0.5,
    zorder=1
)

# Gray markers
ax2.plot(
    x_gray_pts,
    y_gray_pts,
    linestyle="None",
    marker="s",
    color="0.6",
    markerfacecolor="0.6",
    markeredgecolor="0.6",
    markersize=5,
    zorder=2
)

ax2.set_ylim(y2_min, y2_max)
ax2.set_ylabel("efficiency, g", fontsize=16, color="0.55")

ax2.tick_params(
    axis="y",
    which="major",
    labelsize=14,
    colors="0.55",
    direction="in",
    length=7
)

ax2.tick_params(
    axis="y",
    which="minor",
    colors="0.55",
    direction="in",
    length=4
)

# Top axis for visual recreation
ax_top = ax1.twiny()
ax_top.set_xscale("log")
ax_top.set_xlim(0.007, 20)

ax_top.set_xlabel(r"$R/\lambda$", fontsize=16, color="0.55")
ax_top.set_xticks([0.01, 0.1, 1.0, 10.0])
ax_top.set_xticklabels(["0.01", "0.10", "1.00", "10.00"])

ax_top.tick_params(
    axis="x",
    which="major",
    labelsize=14,
    colors="0.55",
    direction="in",
    length=7
)

ax_top.tick_params(
    axis="x",
    which="minor",
    colors="0.55",
    direction="in",
    length=4
)

# Spine styling
for spine in ax1.spines.values():
    spine.set_linewidth(0.8)

for spine in ax2.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("0.55")

for spine in ax_top.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("0.55")

plt.tight_layout()
plt.show()
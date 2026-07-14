import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1D Contrast Transfer Function Plot
#
# CTF equation:
# I_hat_z(u) ≈ δ(u)
#              - 2 cos(pi lambda z |u|^2) mu_hat(u)
#              + 2 sin(pi lambda z |u|^2) phi_hat(u)
#
# This plot uses the scaled coordinate:
# x = sqrt(pi lambda z u^2)
#
# Therefore:
# pi lambda z u^2 = x^2
#
# So the plotted transfer functions are:
# phase transfer      = sin(x^2)
# absorption transfer = cos(x^2)
#
# The TIE low-frequency approximation is:
# sin(pi lambda z u^2) ≈ pi lambda z u^2
#
# Since x = sqrt(pi lambda z u^2), the TIE line is:
# TIE line = x^2
# ============================================================

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

save_figure = True
output_basename = "/Users/danielhodge/Desktop/CTF_1DPlot"

# Set this to True if you want the full CTF coefficients:
# phase term:       2 sin(pi lambda z u^2)
# absorption term: -2 cos(pi lambda z u^2)
use_full_ctf_coefficients = False

# Add TIE approximation line
show_tie_line = True

# If True, y-axis expands to show the full TIE line.
# If False, y-axis stays like your original plot and the TIE line is clipped.
expand_yaxis_for_tie = False

# Optional line showing y = x = sqrt(pi lambda z u^2)
# This is not a CTF transfer function. It only shows the
# scaled coordinate on the same plot.
show_sqrt_coordinate_line = False

# Plot range for x = sqrt(pi lambda z u^2)
x_min = 0.0
x_max = 3.0
num_points = 1000

# ------------------------------------------------------------
# Generate scaled coordinate
# ------------------------------------------------------------

x = np.linspace(x_min, x_max, num_points)

# Since x = sqrt(pi lambda z u^2),
# the CTF argument is pi lambda z u^2 = x^2
ctf_argument = x**2

# TIE low-frequency approximation term
tie_line = ctf_argument

# ------------------------------------------------------------
# Compute CTF transfer functions
# ------------------------------------------------------------

if use_full_ctf_coefficients:
    phase_tf = 2.0 * np.sin(ctf_argument)
    absorption_tf = -2.0 * np.cos(ctf_argument)

    # Full TIE phase coefficient corresponding to 2 sin(q) ≈ 2q
    tie_tf = 2.0 * tie_line

    phase_label = r'Phase -- $2\sin(\pi \lambda z u^2) = \sin(x^2)$'
    absorption_label = r'Absorption -- $-2\cos(\pi \lambda z u^2) = \cos(x^2)$'
    tie_label = r'TIE approx. -- $2\pi \lambda z u^2 = x^2$'

    y_min = -2.2
    y_max = 2.2

else:
    phase_tf = np.sin(ctf_argument)
    absorption_tf = np.cos(ctf_argument)

    # Normalized TIE approximation corresponding to sin(q) ≈ q
    tie_tf = tie_line

    phase_label = r'Phase -- $\sin(\pi \lambda z u^2) = \sin(x^2)$'
    absorption_label = r'Amplitude -- $\cos(\pi \lambda z u^2) = \cos(x^2)$'
    tie_label = r'TIE approx. -- $\pi\lambda z u^2 = x^2$'

    y_min = -2.0
    y_max = 2.0

# Expand y-axis if desired
if show_tie_line and expand_yaxis_for_tie:
    y_max = 1.05 * np.max(tie_tf)
    y_min = min(y_min, -0.5)

# ------------------------------------------------------------
# Make plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    x,
    phase_tf,
    color='red',
    linewidth=2.5,
    label=phase_label
)

ax.plot(
    x,
    absorption_tf,
    color='blue',
    linewidth=2.5,
    label=absorption_label
)

# Purple TIE approximation line
if show_tie_line:
    ax.plot(
        x,
        tie_tf,
        color='purple',
        linewidth=2.5,
        linestyle='-',
        label=tie_label
    )

# Optional coordinate line
if show_sqrt_coordinate_line:
    ax.plot(
        x,
        x,
        color='black',
        linestyle=':',
        linewidth=2.0,
        label=r'$x=\sqrt{\pi \lambda z u^2}$'
    )

# Horizontal zero line
ax.axhline(
    0,
    color='black',
    linewidth=1.0,
    linestyle='--',
    alpha=0.6
)

# ------------------------------------------------------------
# Labels, title, legend, and formatting
# ------------------------------------------------------------

ax.set_title('1D Contrast Transfer Function', fontsize=18)

ax.set_xlabel(
    r'$x = \sqrt{\pi \lambda z u^2}$',
    fontsize=16
)

ax.set_ylabel(
    'Contrast Transfer Function',
    fontsize=16
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.grid(
    True,
    which='major',
    linewidth=0.8,
    alpha=0.7
)

ax.legend(
    fontsize=11,
    loc='upper right',
    frameon=True
)

ax.tick_params(
    axis='both',
    labelsize=13
)

plt.tight_layout()

# ------------------------------------------------------------
# Save and show
# ------------------------------------------------------------

if save_figure:
    plt.savefig(f'{output_basename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_basename}.png', dpi=300, bbox_inches='tight')

plt.show()
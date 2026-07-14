import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the data
# pgd_sim_txt = np.loadtxt("/Users/danielhodge/Desktop/run306_sim_new/loss.txt", skiprows=1)
pgd_exp_txt = np.loadtxt("/Users/danielhodge/Desktop/loss_run576sim2.txt", skiprows=1)

# Create a larger figure for better spacing
fig, ax = plt.subplots(figsize=(8, 6))

# Define custom log-spaced tick positions
custom_ticks = np.logspace(0, -10, num=10)  # 10 evenly spaced points from 10^0 to 10^-10

# Plot both datasets on the same axis
# ax.plot(np.arange(len(pgd_sim_txt)), pgd_sim_txt, color='blue', label="Simulation (Static)")
ax.plot(np.arange(len(pgd_exp_txt)), pgd_exp_txt, color='red', label="Experiment (Static)")

# Formatting
ax.set_title("PGD Loss: Simulation vs. Experiment", size=16, fontweight='bold')
ax.set_ylabel("Mean Squared Error (MSE)", size=16, fontweight='bold')
ax.set_xlabel("Iteration", size=16, fontweight='bold')
ax.set_yscale("log")  # Use log scale
ax.set_ylim(10**-10, 10**-2)  # Set limits to 10^-2 to 10^-13
ax.set_yticks(custom_ticks)  # Apply custom tick positions

# Format y-axis labels to scientific notation (10^-3, 10^-5, etc.)
formatter = ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$')
ax.yaxis.set_major_formatter(formatter)

# Add legend
ax.legend(fontsize=14, loc='upper right')

# Improve layout
plt.tight_layout()
# plt.savefig('/Users/danielhodge/Desktop/tempPlot_combined', bbox_inches='tight', dpi=300, transparent=False)
plt.show()

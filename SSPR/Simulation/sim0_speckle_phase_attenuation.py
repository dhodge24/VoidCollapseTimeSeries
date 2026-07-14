"""
Author: Daniel S. Hodge

The purpose of this code is to create speckle (small particles like dust) that is seen in the experiment (on the
lenses, for example). To accurately represent this speckle in simulation, we simulate a 3D box and randomly place
ellipsoids within it. The ellipsoids represent our dust particles. Then, by the projection approximation (that our
sample is sufficiently thin enough to avoid scattering within the material), we can sum along the x-ray propagation
direction (z-axis) and obtain the total phase shift and attenuation induced by these ellipsoids (particles). The
phase shift and attenuation will be added to our object phase and attenuation maps.

Note: The larger the particles, the longer the code will take to run because a local 3D meshgrid is created for each
particle in the 3D box. If much larger particles are required, see "simulate_speckle_large_particles.py". However,
these particles have to be much, much bigger so it will probably not be needed.

"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from mechanize import Browser
from scipy.interpolate import interp1d
from tqdm import tqdm
from tifffile import imwrite

# Import custom modules
from SSPR.units import *


# Save and plot options
save = True
plot_index_of_refrac = True

run_holo = "582"

# Saving directories
dir_main = "/Users/danielhodge/Desktop/"
dir_sim = "run" + run_holo + "_sim/"

# Save file name
tiff_phase_speckle = "phase_speckle.tiff"
tiff_attenuation_speckle = "attenuation_speckle.tiff"

# Define experimental parameters
Ny, Nx = (2500, 2500)  # Grid size
E = np.array(18000.0, dtype=np.float32)  # Energy of the beam
z01 = np.array(120.41 * mm, dtype=np.float32)  # Distance from source to sample
z12 = np.array(4.668995, dtype=np.float32)  # Distance from sample to detector
scale_fac = np.array(4.0, dtype=np.float32)  # Scale factor we include to compensate for 2x lens magnification
det_pix_size = np.array(6.5 * um, dtype=np.float32)  # Detector pixel size
lam = (1240.0 / E) * nm  # Beam wavelength
z02 = z01 + z12  # Distance from the source to detector
M = z02 / z01  # Geometric magnification
dx_eff = det_pix_size / M / scale_fac  # Object effective pixel size in x, equals detector pixel size if no mag
dy_eff = det_pix_size / M / scale_fac  # Object effective pixel size in y, equals detector pixel size if no mag
dz_eff = dx_eff
z_eff = z12 / M  # Effective propagation distance for spherical illumination (geometry)
k0 = 2 * np.pi / lam  # Wave number
extent_x = Nx * dx_eff  # Physical grid length in x
extent_y = Ny * dx_eff  # Physical grid length in y

# Define box physical and pixel length to contain ellipsoids
Lz = 84.5 * um  # Length of the box in z-direction (propagation direction)
Nz = int(Lz / dx_eff)  # Define the number of grid points (pixels) in the z-direction

# Define SiO2 density
SiO2_density = 2.2  # g/cc

# Find delta and beta parameters from CXRO
# Form here: https://henke.lbl.gov/optical_constants/getdb2.html
br = Browser()  # Open browser
br.open("https://henke.lbl.gov/optical_constants/getdb2.html")  # Open to specified URL
br.form = list(br.forms())[0]  # If the form has no name
br.form['Formula'] = 'SiO2'  # Choose element or combination of
br.form['Density'] = str(SiO2_density)  # SiO2 density is 2.2 g/cc
br.form['Scan'] = ["Energy"]  # Specify energy
br.form['Min'] = str(30)  # Minimum x-ray energy to consider -- 30 eV is the min energy allowed
br.form['Max'] = str(30000)  # Max x-ray energy to consider -- 30000 eV is the max energy allowed
br.form['Npts'] = str(500)  # More points for more accuracy in interpolation -- 500 is maximum
br.form['Output'] = ["Text File"]  # Chooses text file so I can extract data
req = br.submit()  # Submit the form
data = req.get_data()  # Obtain the data
data = data.splitlines(True)  # Splits string into a list
data = data[2:]  # Remove title headers and extract only numerical values
E_vals, delta_vals, beta_vals = np.loadtxt(data, unpack=True)  # Obtain the delta and beta values

if plot_index_of_refrac:
    # Plot the refractive index
    plt.figure()
    plt.loglog(E_vals, delta_vals, color='blue', label='Real part (delta)')
    plt.loglog(E_vals, beta_vals, color='red', label='Imaginary part (beta)')
    plt.title('Energies vs delta/beta Values for SiO2', fontsize=18)
    plt.xlabel('Photon Energy [eV]', fontsize=14)
    plt.ylabel('beta/delta Values', fontsize=14)
    plt.legend()
    plt.show()

# Interpolation to find the most accurate delta/beta values corresponding to a specific beam energy
func_delta = interp1d(E_vals, delta_vals, kind='cubic')
delta = func_delta(E)  # Phase -- refractive portion in the refractive index n=1-delta+i*beta
func_beta = interp1d(E_vals, beta_vals, kind='cubic')
beta = func_beta(E)  # Attenuation -- absorption in the refractive index n=1-delta+i*beta

# Define the semi-axes lengths of the ellipsoids in meters
a = (1.4 / 2) * um  # Semi-major axis in x direction
b = (0.5 / 2) * um  # Semi-major axis in y direction
c = (2.0 / 2) * um  # Semi-major axis in z direction

# Convert semi-axes lengths from meters to pixels
a_pixels = a / dx_eff
b_pixels = b / dy_eff
c_pixels = c / dz_eff

# Calculate the volume of the ellipsoid
volume_ellipsoid_pixels = (4/3) * np.pi * a * b * c / (dx_eff * dy_eff * dz_eff)

# Define the packing density and calculate the number of ellipsoids
packing_density = 0.03  # Multiply by 100 to get the percentage
volume_box_pixels = Nx * Ny * Nz
num_ellipsoids = int(packing_density * (volume_box_pixels / volume_ellipsoid_pixels))
print("Number of ellipsoids (dust particles) that will be placed in the box: ", num_ellipsoids)

# Generate unique random x, y, z positions for the ellipsoids within the box
x_positions = np.random.choice(np.linspace(0, Nx, num_ellipsoids), num_ellipsoids, replace=False)
y_positions = np.random.choice(np.linspace(0, Ny, num_ellipsoids), num_ellipsoids, replace=False)
z_positions = np.random.choice(np.linspace(0, Nz, num_ellipsoids), num_ellipsoids, replace=False)
positions = np.vstack((y_positions, x_positions, z_positions)).T

# Preallocate memory for a 3D boolean box that will hold 3D ellipsoids. Any other method requires insane memory and/or
# takes forever to run.
box = np.zeros((Ny, Nx, Nz), dtype=bool)  # Ny = vertical, Nx = horizontal, Nz = Depth

def make_ellipsoid(x_coords, y_coords, z_coords, a_length, b_length, c_length):
    """
    :param x_coords: Contains the x-coordinates for each point in the bounding box
    :param y_coords: Contains the y-coordinates for each point in the bounding box
    :param z_coords: Contains the z-coordinates for each point in the bounding box
    :param a_length: Length of the semi-major axis in x in pixels
    :param b_length: Length of the semi-major axis in y in pixels
    :param c_length: Length of the semi-major axis in z in pixels

    :return: An ellipsoid shape (the voxels for the ellipsoid) within a bounding box are set to true (1's)
    """
    return (x_coords / a_length) ** 2 + (y_coords / b_length) ** 2 + (z_coords / c_length) ** 2 <= 1


# Insert ellipsoids into the box
for pos in tqdm(positions):

    # Random x, y, z center position in the 3D box where we will construct an ellipsoid
    y0, x0, z0 = pos

    # Define the bounding box for the ellipsoid that we will construct. We use max/min to keep the ellipsoid within
    # the box
    x_min = np.maximum(0, int(x0 - a_pixels))
    x_max = np.minimum(Nx, int(x0 + a_pixels))
    y_min = np.maximum(0, int(y0 - b_pixels))
    y_max = np.minimum(Ny, int(y0 + b_pixels))
    z_min = np.maximum(0, int(z0 - c_pixels))
    z_max = np.minimum(Nz, int(z0 + c_pixels))

    # Create a local grid for the ellipsoid -- By using the maximum and minimum pixel lengths in each dimension we can
    # construct the correct sized box for the ellipsoid within this 3D box. The ellipsoid's center is at the origin of
    # of this local grid, which is defined to be (y0, x0, z0). A local grid will be made for each new ellipsoid
    # inserted into the box. Code later in the script enforces the constraint that the ellipsoids cannot overlap. We
    # want this constraint because speckles/dust do/does not overlap on the CRLs.
    X = np.linspace(x_min - x0, x_max - x0, x_max - x_min)
    Y = np.linspace(y_min - y0, y_max - y0, y_max - y_min)
    Z = np.linspace(z_min - z0, z_max - z0, z_max - z_min)
    y, x, z = np.meshgrid(Y, X, Z, indexing='ij')

    # Use the ellipsoid volume equation to turn certain voxels (the ellipsoid volume within the local grid) to true
    # within the local bounding box
    mask = make_ellipsoid(x, y, z, a_pixels, b_pixels, c_pixels)

    # Update the 3D box locally and prevent ellipsoid overlap and/or rewriting. The 'logical_or' operation ensures that
    # if any voxel in the box is already true, it remains true, and if a voxel in the mask is true, it sets the
    # corresponding voxel in the box to true.
    box[y_min:y_max, x_min:x_max, z_min:z_max] = np.logical_or(box[y_min:y_max, x_min:x_max, z_min:z_max], mask)

# Projection Approximation. We project the 3D volume containing all of our ellipsoids to 2D to obtain a 2D phase
# and attenuation maps representing our CRL speckle at the object plane (right after the CRLs).
box_2d_proj = np.sum(box, axis=-1)  # Sum along the z direction (last axis) of the box containing the ellipsoids
phase = -k0 * delta * box_2d_proj * dx_eff  # thickness = box_2d_proj * dx_eff
attenuation = k0 * beta * box_2d_proj * dx_eff

plt.figure()
plt.imshow(phase,
           extent=[-extent_x / 2 / um, extent_y / 2 / um, -extent_x / 2 / um, extent_y / 2 / um],
           cmap='Greys_r')
plt.colorbar()
plt.xlabel('x [um]')
plt.ylabel('y [um]')
plt.title('2D Phase Shift')
plt.show()

plt.figure()
plt.imshow(attenuation,
           extent=[-extent_x / 2 / um, extent_y / 2 / um, -extent_x / 2 / um, extent_y / 2 / um],
           cmap='Greys_r')
plt.colorbar()
plt.xlabel('x [um]')
plt.ylabel('y [um]')
plt.title('2D Attenuation')
plt.show()

if save:
    imwrite(dir_main + dir_sim + tiff_phase_speckle, phase, photometric='minisblack')
    imwrite(dir_main + dir_sim + tiff_attenuation_speckle, attenuation, photometric='minisblack')

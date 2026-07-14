import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread
from utilities import binArray, shiftRotateMagnifyImage, cropToCenter, create_circular_mask, make_Gaussian
from Propagation.SimulateHologram import simulate_hologram


# def reflect_tile(image):
#     flip_ud = np.flipud(image)  # up/down flip
#     flip_lr = np.fliplr(image)  # left/right flip
#     flip_udlr = np.flipud(flip_lr)  # up/down + left/right flip
#
#     # Create the 3 rows
#     top_row = np.hstack([flip_udlr, flip_ud, flip_udlr])
#     middle_row = np.hstack([flip_lr, image, flip_lr])
#     bottom_row = np.hstack([flip_udlr, flip_ud, flip_udlr])
#
#     # Stack rows vertically
#     full_image = np.vstack([top_row, middle_row, bottom_row])
#
#     return full_image

save = True

# data_file = "/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Lab_Research_Team/ProjectFolders/SingleShotImaging/nanowire_data/nanowire_series/nanowire_-61.7fs/NanoDataSLAC_-61.7fs.dat"
# data_file = "/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Lab_Research_Team/ProjectFolders/SingleShotImaging/nanowire_data/nanowire_series/nanowire_-8.3fs/NanoDataSLAC_-8.3fs.dat"
data_file = "/Users/danielhodge/Desktop/NanoDataSLAC_98.3fs.dat"
# data_file = "/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Lab_Research_Team/ProjectFolders/SingleShotImaging/nanowire_data/nanowire_series/nanowire_325.0fs/NanoDataSLAC_325.0fs.dat"
# data_file = "/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Lab_Research_Team/ProjectFolders/SingleShotImaging/nanowire_data/nanowire_series/nanowire_591.7fs/NanoDataSLAC_591.7fs.dat"

# Obtain time step value/number
folder_name = os.path.basename(os.path.dirname(data_file))  # 'nanowire_-61.7fs'
time_step = folder_name.replace('nanowire_', '')  # '-61.7fs'

with open(data_file, 'r') as f:
    lines = f.readlines()

header_lines = lines[:12]
print("HEADER INFO:")
for line in header_lines:
    print(line.strip())

print("---------------------------------------------------------------------------------------------------------")

with open(data_file, 'r') as f:
    for line in f:
        if "Dimensions" in line:
            parts = line.strip().split(":")[-1].split()
            Ny, Nz, Nx = map(int, parts)

        if "Sizes (cm)" in line:
            parts = line.strip().split(":")[-1].split()
            fov_y, fov_z, fov_x = map(float, parts)

        if "Max electron density" in line:
            parts = line.strip().split(":")[-1].split()
            max_electron_density = np.array(parts, np.float64)[0]

print(f"iy_len = {Ny}, iz_len = {Nz}, ix_len = {Nx}")
print(f"Y = {fov_y}, Z = {fov_z}, X = {fov_x}")
print(f"max_electron_density = {max_electron_density}")

print("---------------------------------------------------------------------------------------------------------")

data = np.loadtxt(data_file, skiprows=13)
iy, iz, ix = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2].astype(int)
electron_densities, ion1_densities, charge1, ion2_densities, charge2 = data[:, 3:].T  # Electron density in units atoms/cc

dx = fov_x / Nx  # cm -- 8 nm
dy = fov_y / Ny  # cm -- 8 nm
dz = fov_z / Nz  # cm
extent_x = Nx * dx * 1e7  # nm
extent_y = Ny * dy * 1e7  # nm
extent_z = Nz * dz * 1e7  # nm

E = 9000  # Beam energy - eV
r_e = 2.82e-13  # Classical electron radius - cm
lam_um = 1240 / E * 1e-3  # Wavelength - um
lam_cm = 1240 / E * 1e-7  # Wavelength - cm
k0 = 2 * np.pi / lam_cm

Nc = 1.11e21 / lam_um**2  # Critical electron density -- Equation 8.112b on page 349 - e/cc

delta = electron_densities / (2 * Nc)  # Page 350/368 Atwood
delta = delta.reshape(Nx, Ny, Nz)
delta_2d = np.sum(delta, axis=0)
phase = -k0 * delta_2d * dx

ds_size = 6  # Downsample/Bin size
binStep = ds_size
binSize = ds_size
phase = binArray(phase, axis=0, binStep=binStep, binSize=binSize, func=np.mean)
phase = binArray(phase, axis=1, binStep=binStep, binSize=binSize, func=np.mean)
Ny_binned = int(Ny / ds_size)
Nz_binned = int(Nz / ds_size)
dy_binned = dy * ds_size * 1e7  # nm -- 48 nm with 6x downsample
dz_binned = dz * ds_size * 1e7  # nm
extent_y_binned = Ny_binned * dy_binned  # nm
extent_z_binned = Nz_binned * dz_binned  # nm


plt.figure()
plt.imshow(phase,
           cmap='Greys_r',
           extent=(-extent_y_binned / 2, extent_y_binned / 2, -extent_z_binned / 2, extent_z_binned / 2))
plt.title(f"Phase (Time Step: {time_step})")
plt.xlabel("y [nm]")
plt.ylabel("z [nm]")
plt.colorbar()
if save:
    plt.savefig('/Users/danielhodge/Desktop/' + str(time_step) + ".png",
                bbox_inches='tight',
                dpi=300,
                transparent=False)
plt.show()


print("Pixel sizes for y and z in nm: ", (dy_binned, dz_binned))


grid_size_nm = 90000  # 90x90um grid in nm - Holds all the wires
# After binning the natural phase map extent is 624 nm, but we crop the phase map to have our desired center-to-center spacing
desired_extent = 400  # nm - Desired phase map extent in nm. Also the center-to-center spacing between wires.
desired_Ny = int(np.ceil(desired_extent / dy_binned))  # Number of pixels to achieve desired extent or center-to-center spacing
desired_Nz = int(np.ceil(desired_extent / dz_binned))  # Number of pixels to achieve desired extent or center-to-center spacing

# Calculate start and end indices for cropping
start_y = (Ny_binned - desired_Ny) // 2
start_z = (Nz_binned - desired_Nz) // 2

# Crop to 400nm extent
phase_cropped = phase[start_z:start_z + desired_Nz, start_y:start_y + desired_Ny]

tile_size_y_nm = desired_extent
tile_size_z_nm = desired_extent

n_tiles_y = int(grid_size_nm // tile_size_y_nm)
n_tiles_z = int(grid_size_nm // tile_size_z_nm)

phase_nanowire_array = np.tile(phase_cropped, (n_tiles_z, n_tiles_y))
# phase_nanowire_array = reflect_tile(phase_nanowire_array)
# phase_nanowire_array = cropToCenter(img=phase_nanowire_array, newSize=[int(extent_y // dy_binned), int(extent_z // dz_binned)])
print(phase_nanowire_array.shape)

extent_y = tile_size_y_nm * n_tiles_y * 1e-3  # um
extent_z = tile_size_z_nm * n_tiles_z * 1e-3  # um

# phase_nanowire_array = shiftRotateMagnifyImage(img=phase_nanowire_array, rotAngleDegree=5)

plt.figure(figsize=(10, 8))
plt.imshow(phase_nanowire_array,
           cmap='Greys_r',
           extent=(-extent_y/2, extent_y/2, -extent_z/2, extent_z/2))
plt.title(f"Nanowire Array Phase Map ({desired_extent} nm spacing)")
plt.xlabel("y [um]")
plt.ylabel("z [um]")
plt.colorbar()
if save:
    plt.savefig(f'/Users/danielhodge/Desktop/Nanowire_array_{time_step}.png',
                bbox_inches='tight',
                dpi=300,
                transparent=False)
plt.show()

# phase_speckle = np.array(imread("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Lab_Research_Team/ProjectFolders/SingleShotImaging/nanowire_data/run_sim/phase_speckle.tiff"))


z_eff = 15e-3  # Adjust
circle_grid_percentage = 1.5
smooth_outer_circle_edges = 20
Gaussian_exponential = 0.65
Gaussian_FWHM = 4000  # Pixels

beam_center = np.array([0, 0])
top_hat = create_circular_mask(size=2025,
                               percentage=circle_grid_percentage,
                               smooth_pixels=smooth_outer_circle_edges)
beam = make_Gaussian(size=2025,
                     fwhm=Gaussian_FWHM,
                     center=beam_center,
                     exponent=Gaussian_exponential)
beam = top_hat * beam

lam = 1240 / E * 1e-9
holo, _ = simulate_hologram(fresnel_number=(dz_binned / 1e9) ** 2 / (z_eff * lam),
                               phase_image=(phase_nanowire_array),
                               absorption_image=None,
                               beam=beam,
                               size_pad=[3000, 3000],
                               size_out=[2025, 2025])

holo = holo * 2000
holo = np.random.poisson(holo).astype(np.float32)

plt.figure(figsize=(10, 8))
plt.imshow(holo,
           cmap='Greys_r',
           clim=(0, 3500),
           extent=(-extent_y/2, extent_y/2, -extent_z/2, extent_z/2))
plt.title("Intensity")
plt.xlabel("y [um]")
plt.ylabel("z [um]")
plt.colorbar()
if save:
    plt.savefig(f'/Users/danielhodge/Desktop/Nanowire_intensity.png',
                bbox_inches='tight',
                dpi=300,
                transparent=False)
plt.show()

# plt.figure(figsize=(10, 8))
# plt.imshow(holo, clim=(0, 3500), cmap="Greys_r")
# plt.show()

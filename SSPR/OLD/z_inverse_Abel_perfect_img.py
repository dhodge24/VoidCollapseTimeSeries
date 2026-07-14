"""

References:
    1) "Abel inversion of a holographic interferogram for determination of the density profile of a sheared-flow pinch"
     by S. L. Jackson et al. (See Eq. 1)
    2) "X-Ray Phase-Contrast Imaging" by M. Endrizzi (See Eqs. 5 and 7)
    3) "Quantitative biological imaging by ptychographic x-ray diffraction microscopy" by K. Giewekemeyer et al.
    4) "Single-Pulse Phase-Contrast Imaging at Free-Electron Lasers in the Hard X-Ray Regime" by J. Hagemann et al.
        (See Figure 9, third row)
    5) "Quantitative X-Ray Phase Nanotomography" by A. Diaz et al. (see Eqs. 1-3)
    6) "Radiation and heat transport in divergent shock–bubble interactions" by K. Kurzer-Ogul (see Table 1)

The purpose of this code is to calculate the projected electron density (n_e, 1/m^2) and areal density (ρ_areal, g/cm^2)
of our samples given a single phase map, φ. To calculate the projected electron density you would use the equation:
∫n_e dz = -φ / (r_e * λ), using Eqs. 5 and 7 in Reference 2 or Eqs. 1 and 2 in Reference 5. Here, r_e and λ are the
classical electron radius and laser wavelength, respectively. Alternatively, you can use Eq. 1 in Reference 1, which
gives the same result. This equation is defined as ∫n_e dz = λ * n_c * -φ / π, where n_c is the plasma cutoff density
above which the laser light will not propagate. To obtain the areal density, you need to use Eq. 3 in Reference 5,
which is: ρ_areal = n_e * A / (N_A * Z) --> This assumes a single material, no combination or mixing. Here, A is the
molar mass in units g/mol, N_A is Avogadro's number in units of mol^-1, and Z is the total number of electrons in a
molecule.

So we have 2 options we can do:
1) Compute the projected electron density map from the experimental and simulated phase maps and compare these
to the projected electron density map from the xRAGE hydrodynamic code
2) Assume all the material is a single material (SU-8) and compute the areal density map from the experimental and
simulated phase maps and compare it to the xRAGE hydrodynamic code. This requires 1) as we need the total projected
electron density map to compute the areal density map. This assumption is valid if the SiO2 mass is significantly
smaller than the SU8 total mass.

"""

# Import python modules
import numpy as np
import abel
from matplotlib import pyplot as plt
from tifffile import imread, imwrite
from skimage.transform import resize

from SSPR.utilities import shiftRotateMagnifyImage, cropToCenter, padToSize

def interpolate_maps(x, scale_factor=2.5, anti_aliasing_sigma=(1, 1)):
    """xRAGE generates phase and attenuation maps with 0.1um pixel size. We must rescale this to the experimental
    pixel size to have an accurate comparison. To do this we interpolate the images with some given resolution and
    scale it to a desired pixel size."""
    new_shape = (int(x.shape[0] * scale_factor), int(x.shape[1] * scale_factor))
    print("The resized image is scaled up by this amount: ", scale_factor)
    print("The new image shape is size: ", new_shape)

    # Interpolate phase and attenuation maps to the new resolution
    interpolated_map = resize(x,
                              new_shape,
                              mode='constant',
                              order=3,
                              anti_aliasing=True,
                              anti_aliasing_sigma=anti_aliasing_sigma)

    return interpolated_map

run_holo = "572"
dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped/"
sim = True
type = "sim"
save = False


Ny, Nx = (2580, 2580)
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

# Experimental parameters
E = 18000  # Initial energy of the beam in eV
lam = (1240 / E) * 1e-9  # Wavelength
c = 2.9979e8  # Speed of light in m/s
m_e = 9.1094e-31  # Electron mass in kg
eps0 = 8.852e-12  # Permittivity of free space in units C^2 / (N * m^2)
e = 1.6022e-19  # Charge of an electron in C
r_e = 2.82e-15  # Classical electron radius in meters
N_A = 6.022e23  # Avogadro's number in mol^-1

m_to_nm = 1e-9  # To put the scaling in # of electrons per nm^2
num_elec = 10e6  # Scaling the electrons for a more reasonable looking plot
cmPerPx = 9.998679161071777e-06

# Seems that the best is when the padding size is twice the cropped size
end_size = [2100, 2100]
out_size = [2 * end_size[0], 2 * end_size[1]]
scale_factor = 2.5  # We must scale the pixel size in the transform for the values to be accurate

if sim:
    dir_sim = "run" + run_holo + "_sim/"
    tiff_ph = "run" + run_holo + "_" + type + "_ph_gt_final.tiff"
    ph = np.array(imread(dir_main + dir_sim + tiff_ph), dtype=np.float32)
else:
    dir_exp = "run" + run_holo + "_exp/"
    tiff_ph = "run" + run_holo + "_" + type + "_ph_final.tiff"
    ph = np.array(imread(dir_main + dir_exp + tiff_ph), dtype=np.float32)


# Electron density GT
electron_density_total = (
    np.array(imread("/Users/danielhodge/Desktop/electron_density_total_GT.tiff"), dtype=np.float32))
electron_density_total = electron_density_total[-2580:, :]
electron_density_total = shiftRotateMagnifyImage(img=electron_density_total, rotAngleDegree=180)
electron_density_total = interpolate_maps(x=electron_density_total, scale_factor=scale_factor)
electron_density_total = cropToCenter(img=electron_density_total, newSize=end_size)
plt.figure()
plt.imshow(electron_density_total, clim=(0, 1000))
plt.show()
if save:
    imwrite("/Users/danielhodge/Desktop/electron_density_total_cropped.tiff", electron_density_total)

# Perfect data
projected_electron_density_total = (
    np.array(imread("/Users/danielhodge/Desktop/projected_electron_density_total.tiff"), dtype=np.float32))

projected_electron_density_total = projected_electron_density_total[-2580:, :]

projected_electron_density_total = shiftRotateMagnifyImage(img=projected_electron_density_total,
                                                           rotAngleDegree=180)

projected_electron_density_total = interpolate_maps(x=projected_electron_density_total,
                                                    scale_factor=scale_factor)

projected_electron_density_total = cropToCenter(img=projected_electron_density_total,
                                                newSize=end_size)

if save:
    imwrite("/Users/danielhodge/Desktop/ed_perf.tiff", projected_electron_density_total)

projected_electron_density_total = padToSize(img=projected_electron_density_total,
                                             outputSize=out_size,
                                             padMethod='replicate',
                                             padType='both',
                                             padValue=None)
electron_density_total_recon = abel.Transform(projected_electron_density_total,
                                              direction='inverse',
                                              method='daun',
                                              origin='slice',
                                              symmetry_axis=0,
                                              symmetrize_method='fourier',
                                              transform_options=dict(reg=100, dr=cmPerPx / scale_factor),
                                              recast_as_float64=True).transform
electron_density_total_recon = electron_density_total_recon #* num_elec * (1e7 ** 2) / cmPerPx / 10e20 * scale_factor
# This must be manually tuned after output. The Abel transform does not know the absolute 0 so we adjust the
# image globally based off a known value that is unperturbed by the shockwave. Additionally, we know from the
# ground truth that inside the void should be ~0 e-/cm^3 and/or the SU8 should be ~390 x 10^20 e-/cm^3.
electron_density_total_recon -= 60
electron_density_total_recon[electron_density_total_recon < 0] = 0
electron_density_total_recon = cropToCenter(img=electron_density_total_recon, newSize=end_size)
plt.figure()
plt.imshow(electron_density_total_recon, clim=(0, 1000))
plt.show()
if save:
    imwrite("/Users/danielhodge/Desktop/electron_density_total_recon_cropped.tiff",
            electron_density_total_recon)




# time = 6.5
# plt.figure()
# plt.imshow(electron_density_total_recon)
# cbar = plt.colorbar()
# cbar.set_label(r'Electron Density ($\rm 10^{20}$ e$^-$/cm$^3$)')
# plt.title(f'Electron Density Recon @ t = {time:.1f} ns')
# # plt.xlim((-60, 60))
# # plt.ylim((20, 140))
# plt.xlabel('Pixels')
# plt.ylabel('Pixels')
# timeStr = int(time * 10)
# savePng = "/Users/danielhodge/Desktop/" + f'electron_density_recon_{timeStr}.png'
# plt.savefig(savePng, dpi=300, bbox_inches='tight', transparent=True)
# plt.show()
#
# gt = np.array(imread("/Users/danielhodge/Desktop/electron_density_total_GT.tiff"), dtype=np.float32)
# gt = gt[-2580:, :]
# gt = shiftRotateMagnifyImage(img=gt,
#                              rotAngleDegree=180)
#
# time = 6.5
# plt.figure()
# plt.imshow(gt)
# cbar = plt.colorbar()
# cbar.set_label(r'Electron Density ($\rm 10^{20}$ e$^-$/cm$^3$)')
# plt.title(f'Electron Density GT @ t = {time:.1f} ns')
# # plt.xlim((-60, 60))
# # plt.ylim((20, 140))
# plt.xlabel('Pixels')
# plt.ylabel('Pixels')
# timeStr = int(time * 10)
# savePng = "/Users/danielhodge/Desktop/" + f'electron_density_gt_{timeStr}.png'
# plt.savefig(savePng, dpi=300, bbox_inches='tight', transparent=True)
# plt.show()




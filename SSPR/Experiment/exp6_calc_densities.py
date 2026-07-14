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
from tifffile import imread, imwrite
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

# Import custom modules
from SSPR.utilities import create_circular_mask, cropToCenter


run_holo = "590"
type = "sim"
sim = True
save = False
use_mask = False
plot = False

if sim:
    # sim
    dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped/"
    dir_sim = "run" + run_holo + "_sim/"
    tiff_ph = "run" + run_holo + "_" + type + "_ph_final.tiff"
    ph = np.array(imread(dir_main + dir_sim + tiff_ph), dtype=np.float32)
    tiff_ph_gt = "run" + run_holo + "_" + type + "_ph_gt_final.tiff"
    # tiff_mu_gt = "run" + run_holo + "_" + type + "_mu_gt_final.tiff"
    ph_gt = np.array(imread(dir_main + dir_sim + tiff_ph_gt), dtype=np.float32)
    # mu_gt = np.array(imread(dir_main + dir_sim + tiff_mu_gt), dtype=np.float32)

    # Sampling test items, uncomment if that is the case
    # dir_sim = "run" + run_holo + "_sim_sampling_test/"
    # tiff_ph = "ph_final_run576sim_test.tiff"
    # ph = np.array(imread("/Users/danielhodge/Desktop/time_series_new_recons/run576_sim_sampling_test/ph_final_run576sim_test.tiff"), dtype=np.float32)
else:
    # exp
    dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped/"
    dir_exp = "run" + run_holo + "_exp/"
    tiff_ph = "run" + run_holo + "_" + type + "_ph_final.tiff"
    ph = np.array(imread(dir_main + dir_exp + tiff_ph), dtype=np.float32)


Ny, Nx = ph.shape
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


mask_percentage = 0.555
smooth_pixels = 1  # Number of pixels to smooth by for the mask edges

# ph = np.array(imread("/Users/danielhodge/Library/CloudStorage/Box-Box/meclx4819DATA/run307_exp_preprocessed/run307_exp_new/run307_exp_dynamic_ph_PGD.tiff"), dtype=np.float32)
# ph = np.array(imread(dir_main + "run" + run_holo + "_sim/run" + run_holo + "_sim_phase.tiff"), dtype=np.float32)

E = 18000  # Energy of the x-ray beam in eV
c = 2.9979e8  # Speed of light in m/s
m_e = 9.1094e-31  # Electron mass in kg
eps0 = 8.852e-12  # Permittivity of free space in units C^2 / (N * m^2)
e = 1.6022e-19  # Charge of an electron in C
r_e = 2.82e-15  # Classical electron radius in meters
N_A = 6.022e23  # Avogadro's number in mol^-1
if use_mask:
    mask = create_circular_mask(size=ph.shape[0], percentage=mask_percentage, smooth_pixels=smooth_pixels)
else:
    mask = np.ones_like(ph)

m_to_nm = 1e-9
num_elec = 10e6
print("Calculating and plotting the projected electron density map...")
n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e**2  # Using Eq. 2 in Reference 1 in units of m^-3
if sim:
    n_e_gt = -ph_gt * lam * n_c / np.pi * m_to_nm**2 / num_elec #* mask
n_e = -ph * lam * n_c / np.pi * m_to_nm**2 / num_elec #* mask  # Using Eq. 1 in Reference 1
# n_e = cropToCenter(img=n_e, newSize=[1600, 1600])
# n_e = -ph / (lam * r_e) * m_to_nm**2 / num_elec  # in 10e6e-/nm^2 -- Alternative def using References 2 and 5


m_to_um = 1e6
x_um = Nx * dx_eff * m_to_um
y_um = Ny * dy_eff * m_to_um
extent = (-x_um / 2,  x_um / 2, -y_um / 2,  y_um / 2)
if plot:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(n_e, cmap="plasma", extent=extent)  # Log the projected electron density?
    ax.set_ylabel("Y Position [$\mathbf{\mu}$m]", size=12, fontweight='bold')
    ax.set_xlabel("X Position [$\mathbf{\mu}$m]", size=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.042, pad=0.04)
    cbar.set_label(r"Projected Electron Density ($10^6$ e$^-$/nm$^2$)")
    plt.title("Projected Electron Density Map")
    # Use below if extent is not used for imshow
    # scalebar = ScaleBar(dx=dx_eff, units='m', location='upper right',
    #                     length_fraction=0.3, height_fraction=0.025,
    #                     box_alpha=0, color='black', font_properties={"size": 16})
    scalebar = ScaleBar(
        dx=1,
        units='m',
        dimension='si-length',
        location='upper right',
        length_fraction=0.3,
        height_fraction=0.025,
        box_alpha=0,
        color='black',
        font_properties={"size": 16}
    )
    ax.add_artist(scalebar)
    plt.tight_layout()
    # if save:
        # imwrite("/Users/danielhodge/Desktop/proj_elec_density_run" + run_holo + type + ".tiff", n_e)
        # plt.savefig("/Users/danielhodge/Desktop/Projected_Electron_Density.pdf", transparent=False)
    plt.show()

# imwrite("/Users/danielhodge/Desktop/proj_elec_density_run" + run_holo + type + ".tiff", n_e)
# imwrite("/Users/danielhodge/Desktop/proj_elec_density_run" + run_holo + type + "_gt.tiff", n_e_gt)


m_to_cm = 1e-2
print("Calculating and plotting the areal mass density map assuming purely SU-8 material...")
# We may need to assume SU8 only, which is a good approximation for the areal mass density (g/cm^2)
n_e = -ph * lam * n_c / np.pi * m_to_cm**2  # in e-/cm^2
# n_e = -ph / (lam * r_e) * 1 / 1e2**2  # Alternative def using References 2 and 5 in units e-/cm^2
# A_SiO2 = 1 * 28.09 + 2 * 16  # in g/mol
# We assume only SU8 with chemical composition: C87 H118 O16, the same used for XPCI forward modeling
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16  # in g/mol
# Z_SiO2 = 1 * 14 + 2 * 8
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8

# Defined in Kelin's paper in Table 1:
# A_SU8 = 22 * 12.011 + 14 * 1.0079 + 2 * 14.007 + 3 * 16  # in g/mol
# Z_SU8 = 22 * 6 + 14 * 1 + 2 * 7 + 3 * 8

A_total = A_SU8  # + A_SiO2  # in g/mol
Z_total = Z_SU8  # + Z_SiO2
rho_areal = n_e * A_total / (N_A * Z_total) #* mask

# imwrite("/Users/danielhodge/Desktop/test.tiff", rho_areal)

# rho_areal = cropToCenter(img=rho_areal, newSize=[1600, 1600])
plt.imshow(rho_areal, cmap="viridis")
cbar = plt.colorbar()
cbar.set_label(r"Areal Mass Density (g/cm$^2$)")
plt.title("Areal Mass Density Map")
# if save:
#     plt.savefig("/Users/danielhodge/Desktop/Areal_Mass_Density.png", bbox_inches='tight', transparent=True)
plt.show()

imwrite("/Users/danielhodge/Desktop/time_series_recon_cropped_areal_dens/"
        "areal_density_run" + run_holo + type + ".tiff", rho_areal)





# print("Checking the validity of assuming only SU-8 material...")
# z01 = 120.41 * mm  # Distance from source to sample
# z12 = 4.668995 * m  # Distance from sample to detector
# z02 = z01 + z12  # Distance from source to detector
# M = z02 / z01  # Magnification
# scale_fac = 4  # Scale factor we include to compensate for lens mag (optique peter)
# detPixSize = 6.5 * um  # Detector pixel size
# dx_eff = detPixSize / M / scale_fac  # Object pixel size in x, equals detector pixel size if no mag
# dy_eff = detPixSize / M / scale_fac  # Object pixel size in y, equals detector pixel size if no mag
# Nx, Ny = 1150, 1150  # Region we choose containing the shell and only the required SU-8
# r_SiO2 = 6.03 * um  # Thickness of the thickest SiO2 shell found using the minimization procedure
# t_SU8 = 400 * um  # Thickness of the SU-8 along the x-ray beam propagation direction
#
# V_SiO2_max = 4/3 * np.pi * (r_SiO2 / m_to_cm)**3  # 4/3 * π * r^3 volume
# V_SU8 = (Nx * dx_eff) * (Ny * dy_eff) * t_SU8 / m_to_cm**3  # l x w x h volume
# rho_SiO2 = 2.2  # Density of the SiO2 material - g/cm^3
# rho_SU8 = 1.185  # Density of the SU8 material - g/cm^3
# m_SiO2 = rho_SiO2 * V_SiO2_max  # Mass of the SiO2 material in grams
# m_SU8 = rho_SU8 * V_SU8  # Mass of the SU8 material in grams
# m_frac = m_SiO2 / (m_SiO2 + m_SU8) * 100
# print(f"The fraction of SiO2 within the {Ny}x{Nx} region is {m_frac:.4f}%")
#
# # Recalculate n_e and rho_areal without the mask to consider the entire Ny x Nx region where the void is
# n_e = -ph * lam * n_c / np.pi * m_to_cm**2
# rho_areal = n_e * A_total / (N_A * Z_total)
# rho_areal = cropToCenter(img=rho_areal, newSize=[Ny, Nx])
# plt.imshow(rho_areal, cmap="viridis")
# cbar = plt.colorbar()
# cbar.set_label(r"Areal Mass Density (g/cm$^2$)")
# plt.title(f"Areal Mass Density Map for {Ny}x{Nx} ROI")
# if save:
#     plt.savefig("/Users/danielhodge/Desktop/Areal_Mass_Density_ROI.png", bbox_inches='tight', transparent=True)
# plt.show()

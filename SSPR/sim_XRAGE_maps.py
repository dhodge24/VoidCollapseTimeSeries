"""

The author of this code is Pawel M. Kozlowski (@ LANL).

References:
    1) "X-Rays and Extreme Ultraviolet Radiation" by D. Atwood and A. Sakdinawat
    2) "X-Ray Phase-Contrast Imaging" by M. Endrizzi (see Eqs. 5, 7, and 9)
    3) "Quantitative X-Ray Phase Nanotomography" by A. Diaz et al. (see Eqs. 1-3)
    4) CXRO website - https://henke.lbl.gov/optical_constants/getdb2.html
    5) Abel Transform - https://en.wikipedia.org/wiki/Abel_transform and
    https://pyabel.readthedocs.io/en/latest/transform_methods/comparison.html
    6) "Abel inversion of a holographic interferogram for determination of the density profile of a sheared-flow pinch"
     by S. L. Jackson et al. (See Eq. 1)

Generate forward Abel projections of the density maps from simulated LCLS void collapse experiments. Rad-hydro density
maps are used as inputs. Note: xRAGE only produces density maps.

This script treats the plasma as multiple materials.

Essentially the Henke tables (CXRO) are calculating delta and beta from the atomic scattering factors f1 and f2 using:
delta = f1 * rho_lookup * N_A * r_e * lambda^2 / (2 * pi * m_a)
beta = f2 * rho_lookup * N_A * r_e * lambda^2 / (2 * pi * m_a)
These equations are constructed using Eqs. 3.13a/3.13b and 3.25 in Reference 1. Here, f1 and f2 are the real and
imaginary portion of the complex atomic scattering factor for forward scattering (θ<<1 or θ=0, see pgs 54-57 in
Reference 1) that describe the phase velocity variation and wave amplitude decay due to absorption, respectively.
rho_lookup is the constant density (g/cm^3) of a specific material that is found online, N_A is Avogadro's
number (mol^-1), r_e is the classical electron radius (cm), lambda is the x-ray wavelength (cm), and m_a is the molar
mass (g/mol).

So the equation becomes (see Reference 2)
phi = (2 * pi / lambda) * delta * T(x,z) * rho(x,z) / rho_lookup
mu = (2 * pi / lambda) * beta * T(x,z) * rho(x,z) / rho_lookup
Since delta and beta are reported on CXRO using the reference density (rho_lookup), we scale delta and beta by
rho(x,z) / rho_lookup, using the defined density given by xRAGE [rho(x,y)]. Here, delta and beta describe the phase
velocity variation and wave amplitude decay due to absorption, respectively. lambda is the x-ray wavelength (m), T is
the projected thickness of the material (m), rho is the mass density (g/cm^3) of material/materials given by xRAGE at
each position, and rho_lookup is the constant density (g/cm^3) of a specific material that is found online.

Since T(x,z) = rho_areal(x,z) / rho(x,z) this reduces to:
phi = (2 * pi / lambda) * delta * rho_areal(x,z) / rho_lookup
mu = (2 * pi / lambda) * beta * rho_areal(x,z) / rho_lookup
Here, rho_areal(x,z) is the areal mass density (g/cm^2), rho_lookup is the constant density (g/cm^3) of a specific
material that is found online.

We calculate these phase and attenuation maps for each material (after it has been Abel forward projected to produce
an areal density map) and then the phase and attenuation maps simply sum together to form the total phase and
attenuation maps. This is the same as treating the input electric field into one material as the attenuated/phase
shifted output from the other material.

SU-8 (epoxy) composition is: C87 H118 O16 (For XPCI) / C22 H14 N2 O3 (For xRAGE maps) --> density 1.2 g/cc
SiO2 (silica) composition is: Si O2 --> density 2.2 g/cc

From Kelin Kurzer-Ogul about the different SU-8 compositions:
C87 H118 O16 is not what is used in the xRAGE simulations but it is closer to the actual SU8 composition for calculating
the phase. Since xRAGE just produces density maps and doesn't know anything about XPCI, it doesn't really matter if the
EOS used in xRAGE matches the material composition in the forward modeling (obviously assuming the materials are
similar enough to evolve the same way). I talked to David and Pawel about this in the past and they said it was
appropriate to use that formula in the forward modeling script. I'm not sure if it's the actual SU8 formula. Pawel
might know, I inherited that part from him.

Sometimes you will see the word "opacity", which refers to how the light absorption and scattering  properties
(the “opacity”) of a material are modeled using its molecular composition. So if a code, model, or paper says it
“uses C22 H10 N2 O5 opacity,” it means they are treating this molecule as the absorber/scatterer and are using its
tabulated or calculated cross-sections (absorption coefficients, extinction coefficients, etc.) to determine how
radiation propagates through it.

"""

# %% Import python modules
import abel
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import h5py as h5
import astropy.units as u
from tifffile import imwrite
from skimage.transform import resize
from SSPR.utilities import shiftRotateMagnifyImage


def loadSim(filePath):
    """
    Loads rad-hydro simulation data from xRAGE simulations.

    filePath : str
        Full path to HDF5 file containing density data.
    """
    with h5.File(filePath, 'r') as f:
        keys = list(f.keys())
        densityMap = f['data'][...]
        xSimData = f['x'][...]
        ySimData = f['y'][...]
    return densityMap, xSimData, ySimData, keys


def mirrorImg(img, x):
    """
    Given a right-hand side half of an image (positive radial values), mirrors the image to create a radially symmetric
    image. This is useful for preparing an Abel transformation

    img : numpy.ndarray
        2D image array.

    x : numpy.ndarray
        1D radial coordinates. Positive valued, and excluding zero.
    """
    # Flip image to negative radial values
    flipOrig = np.fliplr(img)
    # Combine the original and flipped images
    mirrored = np.hstack((flipOrig, img))
    # Generate negative radial coordinates and combine this with positive radial coordinates to form new radial axis
    xMirrd = np.hstack((-1 * x[::-1], x))
    return mirrored, xMirrd


# def interpolate_maps(x, scale_factor):
#     """xRAGE generates phase and attenuation maps with 0.1um pixel size. We must rescale this to the experimental
#     pixel size to have an accurate comparison. To do this we interpolate the images with some given resolution and
#     scale it to a desired pixel size."""
#     new_shape = (int(x.shape[0] * scale_factor), int(x.shape[1] * scale_factor))
#     print("The resized image is scaled up by this amount: ", scale_factor)
#     print("The new image shape is size: ", new_shape)
#
#     interpolated_map = resize(x,
#                               new_shape,
#                               mode='constant',
#                               order=3,
#                               anti_aliasing=True,
#                               anti_aliasing_sigma=None)
#     return interpolated_map


# %% Simulation files from xRAGE
offset = '60'
off_dir = f'XRAGEdata_{offset}um_off'
simsDir = "/Users/danielhodge/Desktop/" + off_dir + "/sims/1/"
saveDir = "/Users/danielhodge/Desktop/" + off_dir + "/out/2/"
saveFlagPlots = False
saveFlagData = True
cut_img = False
plot = False

# # Temporary
# zoom_factor = 2.3
# deg_rotate = 180

# Select time steps needed (in ns)
ns_time1 = 9.8
ns_time2 = 6.6
ns_time3 = 6.7
ns_time4 = 6.8
# time1 = '12802'
# time2 = '12901'
# time3 = '13001'
# time4 = '13101'
time1 = '09800'
time2 = '06600'
time3 = '06700'
time4 = '06800'

# Files describing fraction of materials. This is used to recover the mass density of the specified material.
densityFilesDict = {ns_time1: f"data/vc{offset}-h1002rho0{time1}",
                    ns_time2: f"data/vc{offset}-h1002rho0{time2}",
                    ns_time3: f"data/vc{offset}-h1002rho0{time3}",
                    ns_time4: f"data/vc{offset}-h1002rho0{time4}"
                    }

# Ablator -- Kapton ablator with composition C22 H10 N2 O5
fraction2FilesDict = {ns_time1: f'data/vc{offset}-h1012c020{time1}',
                      ns_time2: f'data/vc{offset}-h1012c020{time2}',
                      ns_time3: f'data/vc{offset}-h1012c020{time3}',
                      ns_time4: f'data/vc{offset}-h1012c020{time4}'
                      }

# fraction2FilesDict = {ns_time1: f'data/vc{offset}-h1012c020{time1}',
#                       ns_time2: f'data/vc{offset}-h1012c020{time2}',
#                       ns_time3: f'data/vc{offset}-h1012c020{time3}',
#                       ns_time4: f'data/vc{offset}-h1012c020{time4}'
#                       }

# SU-8 epoxy with composition C22 H14 N2 O3
fraction4FilesDict = {ns_time1: f'data/vc{offset}-h1014c040{time1}',
                      ns_time2: f'data/vc{offset}-h1014c040{time2}',
                      ns_time3: f'data/vc{offset}-h1014c040{time3}',
                      ns_time4: f'data/vc{offset}-h1014c040{time4}'
                      }

# SiO2 (silica) shell with composition Si O2
fraction5FilesDict = {ns_time1: f'data/vc{offset}-h1015c050{time1}',
                      ns_time2: f'data/vc{offset}-h1015c050{time2}',
                      ns_time3: f'data/vc{offset}-h1015c050{time3}',
                      ns_time4: f'data/vc{offset}-h1015c050{time4}'
                      }

# %% Plot raw density

for i in range(1):
    # Selecting a particular time step to load density and material fractions
    timeIdx = i
    times = list(densityFilesDict.keys())
    time = times[timeIdx]

    fileDensityImg = densityFilesDict[time]
    fileDensityPath = simsDir + fileDensityImg

    fileFrac2Img = fraction2FilesDict[time]
    fileFrac2Path = simsDir + fileFrac2Img

    fileFrac4Img = fraction4FilesDict[time]
    fileFrac4Path = simsDir + fileFrac4Img

    fileFrac5Img = fraction5FilesDict[time]
    fileFrac5Path = simsDir + fileFrac5Img

    # Loading simulation data
    densityMap, ySimData, xSimData, keys = loadSim(filePath=fileDensityPath)
    frac2Map, xSimData2, ySimData2, keys2 = loadSim(filePath=fileFrac2Path)
    frac4Map, xSimData4, ySimData4, keys4 = loadSim(filePath=fileFrac4Path)
    frac5Map, xSimData5, ySimData5, keys5 = loadSim(filePath=fileFrac5Path)

    # print(densityMap.shape)
    # print(xSimData.shape)
    # print(ySimData.shape)

    # Extracting density maps specific to each material by using the overall density and the material fraction maps.
    densityMapEpoxy = densityMap * (frac2Map + frac4Map)
    densityMapSilica = densityMap * frac5Map


    # Opacity values for SU-8 (epoxy) and SiO2 (silica) at different photon energies

    # XFEL probe photon energies in keV
    photonEnergies = np.array([18]) * u.keV
    # Converting photon energies in keV to wavelengths in cm
    wavelengths = photonEnergies.to(u.cm, equivalencies=u.spectral()).value

    # Lookup refractive indices delta and beta for epoxy and silica in Henke
    # CXRO tables https://henke.lbl.gov/optical_constants/getdb2.html

    # Epoxy at 1.2 g/cc density. Refractive index saved as (delta, beta) tuple.
    # Composition C87 H118 O16
    epoxyDensityLookup = 1.2  # g/cc
    epoxyRefrIndices = {8: (4.22454605E-06, 7.84673038E-09),
                        8.2: (4.0204086E-06, 7.09273218E-09),
                        9: (6.10971392E-06, 8.87505891E-09),
                        16: (1.05349852E-06, 5.23986354E-10),
                        18: (8.32221474E-07, 3.53186008E-10),
                        24: (4.67954038E-07, 1.56729255E-10)}

    # Silica at 2.2 g/cc density. Refractive index saved as (delta, beta) tuple.
    # Composition Si O2
    silicaDensityLookup = 2.2  # g/cc
    silicaRefrIndices = {8: (7.21225524E-06, 9.45756256E-08),
                         8.2: (6.86200519E-06, 8.57948237E-08),
                         9: (5.68284486E-06, 5.92769709E-08),
                         16: (1.78878975E-06, 6.06715389E-09),
                         18: (1.41223461E-06, 3.81504783E-09),
                         24: (7.93199433E-07, 1.25793087E-09)}

    if plot:
        # Plot raw simulation density map (before interpolation, cropping, and mirroring the image)
        plt.imshow(densityMap)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Density Map Raw @ t = {time:.1f} ns')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()

        plt.imshow(densityMapEpoxy)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Density Map SU-8 Epoxy Raw @ t = {time:.1f} ns')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()

        plt.imshow(densityMapSilica)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Density Map Silica Shell Raw @ t = {time:.1f} ns')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()

    # %% Dealing with unevenly spaced data

    # Extracting spacing along each axis.
    xDiff = np.diff(xSimData)
    yDiff = np.diff(ySimData)

    # Getting the smallest amount of spacing in case we need to re-interpolate data onto linear axes.
    xDiffMin = np.min(xDiff)
    yDiffMin = np.min(yDiff)

    # Setting step sizes for new interpolated axes such that we don't lose resolution. These are the smallest step
    # sizes found across all simulation files.
    # interpXStep = 9.998679e-06  # Before
    # interpYStep = 9.9996105e-06  # Before
    interpXStep = xDiffMin
    interpYStep = yDiffMin

    # Spatial extents in nm
    extent = np.array([np.min(xSimData), np.max(xSimData), np.min(ySimData), np.max(ySimData)])
    # Convert spatial extent into um for plotting
    extentum = extent * 1e4

    xDataInterp = np.arange(start=extent[0],
                            stop=extent[1],
                            step=interpXStep)
    yDataInterp = np.arange(start=extent[2],
                            stop=extent[3],
                            step=interpYStep)

    # 2D interpolation of epoxy density map onto new axes
    interpDensityEpoxyFunc = RectBivariateSpline(x=xSimData,
                                                 y=ySimData,
                                                 z=densityMapEpoxy)
    densityMapEpoxyInterp = interpDensityEpoxyFunc(xDataInterp, yDataInterp)

    # 2D interpolation of silica density map onto new axes
    interpDensitySilicaFunc = RectBivariateSpline(x=xSimData,
                                                  y=ySimData,
                                                  z=densityMapSilica)
    densityMapSilicaInterp = interpDensitySilicaFunc(xDataInterp, yDataInterp)

    # Negative densities are not physical -- set negative values to 0
    densityMap[densityMap < 0] = 0
    densityMapEpoxyInterp[densityMapEpoxyInterp < 0] = 0
    densityMapSilicaInterp[densityMapSilicaInterp < 0] = 0

    # %% Plot interpolated and cropped density maps. We crop radially to omit sharp artifacts in the simulations

    # Select which pixels to crop at radially. These values correspond to the pixels values within the
    # interpolated image.
    radialMinPx = 0
    radialMaxPx = 1290

    # Crop the density maps
    densityMapCrop = densityMap[:, radialMinPx:radialMaxPx]  # For mirror
    densityMapEpoxyInterpCrop = densityMapEpoxyInterp[:, radialMinPx:radialMaxPx]
    densityMapSilicaInterpCrop = densityMapSilicaInterp[:, radialMinPx:radialMaxPx]

    # Adjust the extent and xData ranges
    xDataInterpCrop = xDataInterp[radialMinPx:radialMaxPx]

    extentumCrop = extentum.copy()
    extentumCrop[0] = radialMinPx * (extentum[1] - extentum[0]) / np.shape(densityMapEpoxyInterp)[1]
    extentumCrop[1] = radialMaxPx * (extentum[1] - extentum[0]) / np.shape(densityMapEpoxyInterp)[1]

    if plot:
        plt.imshow(densityMapEpoxyInterpCrop, extent=extentumCrop)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Epoxy Interpolated Cropped @ t = {time:.1f} ns')
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        plt.tight_layout()
        plt.show()

        plt.imshow(densityMapSilicaInterpCrop, extent=extentumCrop)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Silica Interpolated Cropped @ t = {time:.1f} ns')
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        plt.tight_layout()
        plt.show()

    # %% Mirroring the density maps to negative radial values

    # Mirror interpolated density map epoxy
    densityEpoxyInterpCropMirrd, _ = mirrorImg(img=densityMapEpoxyInterpCrop,
                                               x=xDataInterpCrop)
    # Include negative radial values in extent
    extentumMirrd = extentumCrop.copy()
    extentumMirrd[0] = -1 * extentumCrop[1]

    if plot:
        plt.imshow(densityEpoxyInterpCropMirrd, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Epoxy Interpolated Cropped Mirrored @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'mass_density_epoxy_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # Mirror interpolated density map Silica
    densitySilicaInterpCropMirrd, _ = mirrorImg(img=densityMapSilicaInterpCrop,
                                                x=xDataInterpCrop)
    # Include negative radial values in extent
    extentumMirrd = extentumCrop.copy()
    extentumMirrd[0] = -1 * extentumCrop[1]

    if plot:
        plt.imshow(densitySilicaInterpCropMirrd, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Silica Interpolated Cropped Mirrored @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'mass_density_silica_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # Original 3D density map
    if cut_img:
        density_total_GT = densityEpoxyInterpCropMirrd + densitySilicaInterpCropMirrd
        cutIdx = int(np.shape(densityEpoxyInterpCropMirrd)[1] / 2)
        density_total_GT = density_total_GT[:, cutIdx:]
    else:
        density_total_GT = densityEpoxyInterpCropMirrd + densitySilicaInterpCropMirrd
    imwrite("/Users/danielhodge/Desktop/3D_density_GT.tiff", density_total_GT)

    if plot:
        plt.imshow(density_total_GT, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Epoxy+Silica Interpolated Cropped Mirrored @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'mass_density_total_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()


    # %% Calculate the electron density maps (1/cm^3) for the epoxy, silica, and silica + epoxy

    N_A = 6.022e23  # Avogadro's number in 1/mol
    A_silica = 1 * 28.09 + 2 * 16  # in g/mol
    Z_silica = 1 * 14 + 2 * 8  # Number of electrons (unitless)
    A_epoxy = 87 * 12.011 + 118 * 1.0079 + 16 * 16  # in g/mol
    Z_epoxy = 87 * 6 + 118 * 1 + 16 * 8  # Number of electrons (unitless)
    electron_density_epoxy = densityEpoxyInterpCropMirrd * N_A * Z_epoxy / A_epoxy / 10e20  # Units of 10^20 e-/cm^3
    electron_density_silica = densitySilicaInterpCropMirrd * N_A * Z_silica / A_silica / 10e20  # Units of 10^20 e-/cm^3

    # Original 3D electron density map using the individual densities of the materials
    if cut_img:
        electron_density_total_GT = electron_density_epoxy + electron_density_silica
        cutIdx = int(np.shape(electron_density_epoxy)[1] / 2)
        electron_density_total_GT = electron_density_total_GT[:, cutIdx:]
    else:
        electron_density_total_GT = electron_density_epoxy + electron_density_silica
    imwrite("/Users/danielhodge/Desktop/electron_density_total_GT.tiff", electron_density_total_GT)

    if plot:
        plt.imshow(electron_density_epoxy, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Electron Density ($\rm 10^{20}$ e$^-$/cm$^3$)')
        plt.title(f'Epoxy Electron Density @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'electron_density_epoxy_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

        plt.imshow(electron_density_silica, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Electron Density ($\rm 10^{20}$ e$^-$/cm$^3$)')
        plt.title(f'Silica Electron Density @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'electron_density_silica_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

        plt.imshow(electron_density_total_GT, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Electron Density ($\rm 10^{20}$ e$^-$/cm$^3$)')
        plt.title(f'Epoxy+Silica Electron Density @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'electron_density_total_GT_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # %% Forward Abel transform --> 3D to 2D assuming cylindrical symmetry

    # In units of px * g / cm^3
    forward_abel_epoxy = abel.Transform(densityEpoxyInterpCropMirrd,
                                        direction='forward',
                                        method='hansenlaw').transform
    forward_abel_silica = abel.Transform(densitySilicaInterpCropMirrd,
                                         direction='forward',
                                         method='hansenlaw').transform

    # Conversion factor for getting the correct units after the Abel transform
    umPerPx = interpXStep * 1e4
    cmPerPx = umPerPx * 1e2 / 1e6

    # Converting units to get from Abel transform in units of px*g/cm^3 to areal density in units of g/cm^2
    areal_density_epoxy = forward_abel_epoxy * cmPerPx
    areal_density_silica = forward_abel_silica * cmPerPx

    ########################
    # Mirrored raw density #
    ########################
    # Mirror interpolated density map epoxy
    densityMirrd, _ = mirrorImg(img=densityMapCrop,
                                x=xDataInterpCrop)
    # include negative radial values in extent
    extentumMirrd = extentumCrop.copy()
    extentumMirrd[0] = -1 * extentumCrop[1]

    if plot:
        plt.imshow(densityMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Density Map Raw Crop Mirrored @ t = {time:.1f} ns')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'Raw_density_total_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # %% Plotting forward abel transformed simulation data

    # Plot epoxy
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        im1 = axs[0].imshow(densityEpoxyInterpCropMirrd,
                            clim=(0, np.max(densityEpoxyInterpCropMirrd) * 1.0),
                            origin='upper',
                            extent=extentumMirrd)
        im2 = axs[1].imshow(areal_density_epoxy,
                            clim=(0, np.max(areal_density_epoxy) * 0.8),
                            origin='upper',
                            extent=extentumMirrd)
        # Color bars
        cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.044, pad=0.04)
        cbar1.set_label(r'Mass Density ($\rm g / cm^3$)')
        cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.044, pad=0.04)
        cbar2.set_label(r'Areal Mass Density ($\rm g / cm^2$)')
        # Labels
        axs[1].set_title(f'Epoxy Forward Abel @ t = {time:.1f} ns')
        # Line below is the interpolated/cropped/mirrored Epoxy that is forward Abel transformed
        axs[0].set_title(f'Epoxy Processed Before Abel Transform @ t = {time:.1f} ns')
        axs[1].set_xlabel(r'Radial ($\rm \mu m$)')
        axs[1].set_ylabel(r'Axial ($\rm \mu m$)')
        axs[0].set_xlabel(r'Radial ($\rm \mu m$)')
        axs[0].set_ylabel(r'Axial ($\rm \mu m$)')
        # # Cropping
        # axs[0].set_xlim((-60, 60))
        # axs[0].set_ylim((20, 140))
        # axs[1].set_xlim((-60, 60))
        # axs[1].set_ylim((20, 140))
        plt.tight_layout()
        plt.show()

        # Plot silica
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        im1 = axs[0].imshow(densitySilicaInterpCropMirrd,
                            clim=(0, np.max(densitySilicaInterpCropMirrd) * 1.0),
                            origin='upper',
                            extent=extentumMirrd)
        im2 = axs[1].imshow(areal_density_silica,
                            clim=(0, np.max(areal_density_silica) * 0.8),
                            origin='upper',
                            extent=extentumMirrd)
        # Color bars
        cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.044, pad=0.04)
        cbar1.set_label(r'Mass Density ($\rm g / cm^3$)')
        cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.044, pad=0.04)
        cbar2.set_label(r'Areal Mass Density ($\rm g / cm^2$)')
        # Labels
        axs[1].set_title(f'Silica Forward Abel @ t = {time:.1f} ns')
        # Line below is the interpolated/cropped/mirrored Silica that is forward Abel transformed
        axs[0].set_title(f'Silica Processed Before Abel Transform @ t = {time:.1f} ns')
        axs[1].set_xlabel(r'Radial ($\rm \mu m$)')
        axs[1].set_ylabel(r'Axial ($\rm \mu m$)')
        axs[0].set_xlabel(r'Radial ($\rm \mu m$)')
        axs[0].set_ylabel(r'Axial ($\rm \mu m$)')
        # # Cropping
        # axs[0].set_xlim((-60, 60))
        # axs[0].set_ylim((20, 140))
        # axs[1].set_xlim((-60, 60))
        # axs[1].set_ylim((20, 140))
        plt.tight_layout()
        plt.show()

    # %% Plot areal density with just positive radial values and save as .png
    # Same plots as previous 2 plots, but we can cut the image to how we want it and save it

    if cut_img:
        cutIdx = int(np.shape(areal_density_epoxy)[1] / 2)
        areal_density_epoxy = areal_density_epoxy[:, cutIdx:]
        areal_density_silica = areal_density_silica[:, cutIdx:]
    else:
        areal_density_epoxy = areal_density_epoxy
        areal_density_silica = areal_density_silica

    # Epoxy
    if plot:
        if cut_img:
            plt.imshow(areal_density_epoxy, extent=extentumCrop, vmin=0, vmax=0.08)
        else:
            plt.imshow(areal_density_epoxy, extent=extentumMirrd, vmin=0, vmax=0.08)
        cbar = plt.colorbar()
        cbar.set_label(r'Areal Mass Density ($\rm g / cm^2$)')
        plt.title(f'Epoxy Forward Abel @ t = {time:.1f} ns')
        # plt.xlim((0, 120))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'rho_areal_epoxy_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

        # Silica
        if cut_img:
            plt.imshow(areal_density_silica, extent=extentumCrop, vmin=0, vmax=0.08)
        else:
            plt.imshow(areal_density_silica, extent=extentumMirrd, vmin=0, vmax=0.08)
        cbar = plt.colorbar()
        cbar.set_label(r'Areal Mass Density ($\rm g / cm^2$)')
        plt.title(f'Silica Forward Abel @ t = {time:.1f} ns')
        # plt.xlim((0, 120))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'rho_areal_silica_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # Total areal density obtained using the forward Abel transform. We used the original, but separate
    # material densities and then added up their contributions
    areal_density_total = areal_density_silica + areal_density_epoxy

    # Areal density
    if plot:
        if cut_img:
            plt.imshow(areal_density_total, extent=extentumCrop)
        else:
            plt.imshow(areal_density_total, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Areal Density ($\rm g / cm^2$)')
        plt.title(f'Epoxy + Silica Forward Abel @ t = {time:.1f} ns')
        # plt.xlim((0, 120))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'rho_areal_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # %% Inverse Abel transform -- A check to see if we can recover our initial density in g/cm^3

    # RECONSTRUCTION TO DETERMINE IF WE CAN GET 3D DENSITY FROM THE 2D AREAL DENSITY USING
    # THE INVERSE ABEL TRANSFORM. REQUIRES ENTIRE IMAGE, NOT HALF OF THE IMAGE
    if cut_img:
        areal_density_total, _ = mirrorImg(img=areal_density_total,
                                           x=xDataInterpCrop)
        density_total_recon = abel.Transform(areal_density_total,
                                             direction='inverse',
                                             method='hansenlaw').transform
    else:
        density_total_recon = abel.Transform(areal_density_total,
                                             direction='inverse',
                                             method='hansenlaw').transform
    if cut_img:
        cutIdx = int(np.shape(density_total_recon)[1] / 2)
        # Converting units to get from the inverse Abel transform in units of g/cm^2 to density units of px*g/cm^3
        density_total_recon = np.abs(density_total_recon) / cmPerPx
        density_total_recon = density_total_recon[:, cutIdx:]
        areal_density_total = areal_density_total[:, cutIdx:]
    else:
        areal_density_total = areal_density_total
        density_total_recon = np.abs(density_total_recon) / cmPerPx
    imwrite("/Users/danielhodge/Desktop/areal_density_total.tiff", areal_density_total)
    imwrite("/Users/danielhodge/Desktop/density_total_recon.tiff", density_total_recon)

    if plot:
        plt.imshow(density_total_recon, extent=extentumMirrd)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass Density ($\rm g / cm^3$)')
        plt.title(f'Inverse Abel Transform - Mass Density @ t = {time:.1f} ns')
        # plt.xlim((-60, 60))
        # plt.ylim((20, 140))
        plt.xlabel(r'Radial ($\rm \mu m$)')
        plt.ylabel(r'Axial ($\rm \mu m$)')
        if saveFlagPlots:
            timeStr = int(time * 10)
            savePng = saveDir + f'density_total_recon_{timeStr}.png'
            plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    # %% Combine density maps of Epoxy and Silica with corresponding real and imaginary components of refractive
    # indices for the materials and produce maps of phase (phi) and attenuation (mu)

    totalPhaseAttenMaps = {}
    for idx, wavelength in enumerate(wavelengths):
        photonEnergy = photonEnergies[idx].value
        # Lookup refractive indices
        deltaEpoxy, betaEpoxy = epoxyRefrIndices[photonEnergy]
        deltaSilica, betaSilica = silicaRefrIndices[photonEnergy]
        # Produce phase and attenuation maps for each material. We need to scale divide out the lookup density used
        # for getting the refractive indices in the Henke tables and replace it with the actual density map.
        waveVec = (2 * np.pi / wavelength)
        phaseEpoxy = waveVec * deltaEpoxy * areal_density_epoxy / epoxyDensityLookup
        phaseSilica = waveVec * deltaSilica * areal_density_silica / silicaDensityLookup
        attenuationEpoxy = waveVec * betaEpoxy * areal_density_epoxy / epoxyDensityLookup
        attenuationSilica = waveVec * betaSilica * areal_density_silica / silicaDensityLookup
        # Adding maps together to form total phase and total attenuation maps.
        phaseTotal = phaseEpoxy + phaseSilica
        attenuationTotal = attenuationEpoxy + attenuationSilica
        # Saving maps to dict
        totalPhaseAttenMaps[photonEnergy] = (phaseTotal, attenuationTotal)

    # %% saving plots to png files and phase/attenuation maps to hdf5 files.

    for photonEnergy in photonEnergies.value:
        phase, attenuation = totalPhaseAttenMaps[photonEnergy]

        # Plot phase map -- Phase is negative for x-rays (phase advance in materials relative to vacuum)
        if plot:
            if cut_img:
                plt.imshow(-phase, extent=extentumCrop)
            else:
                plt.imshow(-phase, extent=extentumMirrd)
            cbar = plt.colorbar()
            cbar.set_label(r'Phase (dimensionless)')
            plt.title(f'Total Phase @ {photonEnergy} keV @ t = {time:.1f} ns')
            # plt.xlim((0, 120))
            # plt.ylim((20, 140))
            plt.xlabel(r'Radial ($\rm \mu m$)')
            plt.ylabel(r'Axial ($\rm \mu m$)')
            if saveFlagPlots:
                timeStr = int(time * 10)
                savePng = saveDir + f'phase_phi_{photonEnergy}_keV_{timeStr}.png'
                plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()

            # Plot attenuation map
            if cut_img:
                plt.imshow(attenuation, extent=extentumCrop)
            else:
                plt.imshow(attenuation, extent=extentumMirrd)
            cbar = plt.colorbar()
            cbar.set_label(r'Attenuation (dimensionless)')
            plt.title(f'Total Attenuation @ {photonEnergy} keV @ t = {time:.1f} ns')
            # plt.xlim((0, 120))
            # plt.ylim((20, 140))
            plt.xlabel(r'Radial ($\rm \mu m$)')
            plt.ylabel(r'Axial ($\rm \mu m$)')
            if saveFlagPlots:
                timeStr = int(time * 10)
                savePng = saveDir + f'attenuation_mu_{photonEnergy}_keV_{timeStr}.png'
                plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()

        # Calculate the projected electron density given the phase
        E = photonEnergy * 1e3  # Energy of the x-ray beam in eV
        c = 2.9979e8  # Speed of light in m/s
        m_e = 9.1094e-31  # Electron mass in kg
        eps0 = 8.852e-12  # Permittivity of free space in units C^2 / (N * m^2)
        e = 1.6022e-19  # Charge of an electron in C
        lam = (1239.84 / E) * 1e-9  # Wavelength of the x-ray beam in meters
        r_e = 2.82e-15  # Classical electron radius in meters
        N_A = 6.022e23  # Avogadro's number in mol^-1
        m_to_nm = 1e-9  # To put the scaling in # of electrons per nm^2
        num_elec = 10e6  # Scaling the electrons for a more reasonable looking plot
        n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e ** 2  # Using Eq. 2 in Reference 6 in units of m^-3
        # Projected electron density in e-/nm^2 - Eq. 1 in Reference 6
        projected_electron_density_total = phase * lam * n_c / np.pi * m_to_nm ** 2 / num_elec

        # # e-/cm^2
        # m_to_cm = 1e-2
        # projected_electron_density_total = phase * lam * n_c / np.pi * m_to_cm ** 2 / num_elec

        # Plot projected electron density
        if plot:
            if cut_img:
                plt.imshow(projected_electron_density_total, extent=extentumCrop)
            else:
                plt.imshow(projected_electron_density_total, extent=extentumMirrd)
            cbar = plt.colorbar()
            cbar.set_label(r'Projected electron density ($10^6$ e$^-$/nm$^2$)')
            plt.title(f'Projected Electron Density @ {photonEnergy} keV @ t = {time:.1f} ns')
            # plt.xlim((0, 120))
            # plt.ylim((20, 140))
            plt.xlabel(r'Radial ($\rm \mu m$)')
            plt.ylabel(r'Axial ($\rm \mu m$)')
            if saveFlagPlots:
                timeStr = int(time * 10)
                savePng = saveDir + f'projected_electron_density_total_{photonEnergy}_keV_{timeStr}.png'
                plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()

        # RECONSTRUCTION TO DETERMINE IF WE CAN GET 3D ELECTRON DENSITY FROM THE 2D ELECTRON DENSITY USING
        # AN INVERSE ABEL TRANSFORM. REQUIRES ENTIRE IMAGE, NOT HALF OF THE IMAGE
        if cut_img:
            projected_electron_density_total, _ = mirrorImg(img=projected_electron_density_total,
                                                            x=xDataInterpCrop)
            electron_density_total_recon = abel.Transform(projected_electron_density_total,
                                                          direction='inverse',
                                                          method='hansenlaw').transform
        else:
            electron_density_total_recon = abel.Transform(projected_electron_density_total,
                                                          direction='inverse',
                                                          method='hansenlaw').transform

        if cut_img:
            cutIdx = int(np.shape(electron_density_total_recon)[1] / 2)
            # Convert the projected electron density to units 10^20 e-/cm^3
            electron_density_total_recon = np.abs(electron_density_total_recon) * num_elec * (
                        1e7 ** 2) / cmPerPx / 10e20
            electron_density_total_recon = electron_density_total_recon[:, cutIdx:]
            projected_electron_density_total = projected_electron_density_total[:, cutIdx:]
        else:
            electron_density_total_recon = np.abs(electron_density_total_recon) * num_elec * (
                        1e7 ** 2) / cmPerPx / 10e20
            projected_electron_density_total = projected_electron_density_total
        imwrite("/Users/danielhodge/Desktop/projected_electron_density_total.tiff",
                projected_electron_density_total)
        imwrite("/Users/danielhodge/Desktop/electron_density_total_recon.tiff",
                electron_density_total_recon)

        if plot:
            plt.imshow(electron_density_total_recon, extent=extentumMirrd)
            cbar = plt.colorbar()
            cbar.set_label(r'Electron Density ($10^{20}$ e$^-$/cm$^3$)')
            # cbar.set_label(r'Projected electron density (1/cm$^3$)')
            plt.title(f'Inverse Abel Transform - Electron Density @ t = {time:.1f} ns')
            # plt.xlim((-60, 60))
            # plt.ylim((20, 140))
            plt.xlabel(r'Radial ($\rm \mu m$)')
            plt.ylabel(r'Axial ($\rm \mu m$)')
            if saveFlagPlots:
                timeStr = int(time * 10)
                savePng = saveDir + f'inverse_abel_electron_density_{timeStr}.png'
                plt.savefig(savePng, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()


        # # save phase
        # if saveFlag:
        #     saveHdf = saveDir + f'void-col-phase-attenuation-{photonEnergy}-keV' + fileDensityImg[-6:] + '.h5'
        #     with h5.File(saveHdf, "w") as f:
        #         # save phase
        #         dset_phase = f.create_dataset("phase", data=phase, dtype='<f4')
        #         dset_phase.attrs['units'] = 'phi (dimensionless)'
        #         # save attenuation
        #         dset_atten = f.create_dataset("attenuation", data=attenuation, dtype='<f4')
        #         dset_atten.attrs['units'] = 'mu (dimensionless)'
        #         # save coordinate system
        #         dset_x = f.create_dataset("x", data=xDataInterpCrop, dtype='<f4')
        #         dset_x.attrs['units'] = 'cm'
        #         dset_y = f.create_dataset("y", data=yDataInterp, dtype='<f4')
        #         dset_y.attrs['units'] = 'cm'
        #         # save photon energy
        #         dset_phot = f.create_dataset("photon energy", data=photonEnergy, dtype='<f4')
        #         dset_phot.attrs['units'] = 'keV'

        # Save phase, attenuation, and the projected electron density
        if saveFlagData:
            saveHdf = saveDir + f'void-col-phase-attenuation-{photonEnergy}-keV' + fileDensityImg[-6:] + '.h5'
            with h5.File(saveHdf, "w") as f:

                # Save phase
                dset_phase = f.create_dataset("phase",
                                              data=phase,
                                              dtype='<f4')
                dset_phase.attrs['units'] = 'phi (dimensionless)'

                # Save attenuation
                dset_attenuation = f.create_dataset("attenuation",
                                                    data=attenuation,
                                                    dtype='<f4')
                dset_attenuation.attrs['units'] = 'mu (dimensionless)'

                # ----------------------------------------------------------------------------------------- #

                # Save epoxy+silica (total) density
                dset_density_GT = f.create_dataset("density_total_GT",
                                                   data=density_total_GT,
                                                   dtype='<f4')
                dset_density_GT.attrs['units'] = 'g/cm^3'

                # Save epoxy+silica (total) areal density which was created using forward Abel transform
                dset_areal_density = f.create_dataset("areal_density_total",
                                                      data=areal_density_total,
                                                      dtype='<f4')
                dset_areal_density.attrs['units'] = 'g/cm^2'

                # Save inverse Abel transform of the epoxy+silica (total) areal density (g/cm^2) --> to get
                # epoxy+silica (total) density (px*g/cm^3)
                dset_density_recon = f.create_dataset("density_total_recon",
                                                      data=density_total_recon,
                                                      dtype='<f4')
                dset_density_recon.attrs['units'] = 'g/cm^3'

                # ----------------------------------------------------------------------------------------- #

                # Save epoxy+silica (total) electron density
                dset_electron_density_GT = f.create_dataset("electron_density_total_GT",
                                                            data=electron_density_total_GT,
                                                            dtype='<f4')
                dset_electron_density_GT.attrs['units'] = '10^{20} e^-/cm^3'

                # Save epoxy+silica (total) projected electron density --> obtained directly from the phase
                # which was made from the forward Abel transform
                dset_projected_electron_density = f.create_dataset("projected_electron_density_total",
                                                                   data=projected_electron_density_total,
                                                                   dtype='<f4')
                dset_projected_electron_density.attrs['units'] = '10^6 e^-/nm^2'

                # Save inverse Abel transform of the epoxy+silica (total) projected electron density --> to get
                # epoxy+silica (total) electron density
                dset_electron_density_recon = f.create_dataset("electron_density_total_recon",
                                                                      data=electron_density_total_recon,
                                                                      dtype='<f4')
                dset_electron_density_recon.attrs['units'] = '10^{20} e-/cm^3'

                # ----------------------------------------------------------------------------------------- #

                # Save coordinate system
                dset_x = f.create_dataset("x", data=xDataInterpCrop, dtype='<f4')
                dset_x.attrs['units'] = 'cm'
                dset_y = f.create_dataset("y", data=yDataInterp, dtype='<f4')
                dset_y.attrs['units'] = 'cm'

                # Save photon energy
                dset_phot = f.create_dataset("photon energy", data=photonEnergy, dtype='<f4')
                dset_phot.attrs['units'] = 'keV'


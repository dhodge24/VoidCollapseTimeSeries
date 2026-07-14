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
from skimage.filters import gaussian
from scipy.ndimage import shift


from SSPR.utilities import shiftRotateMagnifyImage, cropToCenter, padToSize, padFadeOut, reflect_image_2d, fadeoutImage

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info


from scipy.ndimage import fourier_shift
from numpy.fft import fftn, ifftn

class PyAbelForward(linop.LinearOperator):
    def __init__(self, input_shape, method, symmetry_axis, reg, dr, recast_as_float64):
        self.method = method
        self.symmetry_axis = symmetry_axis
        self.reg = reg
        self.dr = dr
        self.recast_as_float64 = recast_as_float64
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, x):
        x = np.asarray(x, dtype=np.float32)
        proj = abel.Transform(
            x,
            direction="forward",
            method=self.method,
            symmetry_axis=self.symmetry_axis,
            transform_options=dict(reg=self.reg, dr=self.dr),
            recast_as_float64=self.recast_as_float64
        ).transform
        return proj


class PyAbelInverse(linop.LinearOperator):
    def __init__(self, input_shape, method, symmetry_axis, reg, dr, recast_as_float64):
        self.method = method
        self.symmetry_axis = symmetry_axis
        self.reg = reg
        self.dr = dr
        self.recast_as_float64 = recast_as_float64
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, y):
        y = np.asarray(y, dtype=np.float32)
        recon = abel.Transform(
            y,
            direction="inverse",
            method=self.method,
            symmetry_axis=self.symmetry_axis,
            transform_options=dict(reg=self.reg, dr=self.dr),
            recast_as_float64=self.recast_as_float64
        ).transform
        return recon


class PyAbelOperator(linop.LinearOperator):
    def __init__(self, input_shape, method, symmetry_axis, reg, dr, recast_as_float64):
        self.forward = PyAbelForward(input_shape, method, symmetry_axis, reg, dr, recast_as_float64)
        self.inverse = PyAbelInverse(input_shape, method, symmetry_axis, reg, dr, recast_as_float64)
        super().__init__(input_shape=input_shape, output_shape=input_shape)

    def _eval(self, x):
        return self.forward(x)

    def adj(self, y):
        return self.inverse(y)


def best_symmetry_axis(img, search_halfwidth=None, mask=None, ds=2, subpixel=True):
    """
    Fast left-right symmetry axis finder.
    Returns axis (float, in original pixel coords) and score (lower is better).
    """
    I = np.asarray(img, dtype=np.float32)

    # downsample for speed
    if ds > 1:
        I = I[::ds, ::ds]
        if mask is not None:
            mask = mask[::ds, ::ds].astype(bool)

    ny, nx = I.shape
    if mask is None:
        mask = np.ones_like(I, dtype=bool)

    c0 = nx // 2
    if search_halfwidth is None:
        search_halfwidth = nx // 4
    cmin = max(1, c0 - search_halfwidth)
    cmax = min(nx - 2, c0 + search_halfwidth)

    best_c, best_s = c0, np.inf
    scores = {}

    for c in range(cmin, cmax + 1):
        L = I[:, :c]
        R = I[:, c+1:]
        m = min(L.shape[1], R.shape[1])
        if m < 8:
            continue

        # take last m cols of L and first m cols of R, mirror R
        A = L[:, -m:]
        B = R[:, :m][:, ::-1]

        M = mask[:, :c][:, -m:] & mask[:, c+1:][:, :m][:, ::-1]
        if not M.any():
            continue

        d = (A - B)[M]
        s = float(np.mean(d * d))
        scores[c] = s

        if s < best_s:
            best_s, best_c = s, c

    axis = float(best_c)

    # subpixel refinement with a parabola through (c-1,c,c+1)
    if subpixel and (best_c - 1 in scores) and (best_c + 1 in scores):
        s1, s2, s3 = scores[best_c - 1], scores[best_c], scores[best_c + 1]
        denom = (s1 - 2*s2 + s3)
        if abs(denom) > 1e-12:
            delta = 0.5 * (s1 - s3) / denom   # in units of (downsampled) pixels
            axis = best_c + np.clip(delta, -1.0, 1.0)

    # convert back to original pixel coords
    axis *= ds
    return axis, best_s





# def best_symmetry_axis(img, search_halfwidth=100, mask=None):
#     """
#     Find the integer column c that best serves as the left-right symmetry axis
#     by minimizing mean squared error between left half and mirrored right half.
#     """
#
#     I = np.asarray(img, dtype=np.float64)  # Convert input image to float64 NumPy array
#     ny, nx = I.shape                       # Get image height (ny) and width (nx)
#
#     if mask is None:                       # If no mask is provided
#         mask = np.ones_like(I, dtype=bool) # Create a mask that includes all pixels
#
#     c0 = nx // 2                           # Initial guess for symmetry axis (image center column)
#
#     # Define search bounds around center while avoiding edges
#     cmin = max(1, c0 - search_halfwidth)   # Minimum candidate column index
#     cmax = min(nx - 1, c0 + search_halfwidth)  # Maximum candidate column index
#
#     best_c, best_score = c0, np.inf        # Initialize best axis and best score (start with infinite error)
#
#     # Loop over all candidate symmetry axis columns
#     for c in range(cmin, cmax + 1):
#
#         L = I[:, :c]                       # Left half of image (columns before c)
#         R = I[:, c+1:]                     # Right half of image (columns after c)
#
#         m = min(L.shape[1], R.shape[1])    # Number of overlapping columns available
#
#         Lm = L[:, -m:]                     # Take rightmost m columns of left half
#         Rm = R[:, :m][:, ::-1]             # Take leftmost m columns of right half and mirror horizontally
#         print(Rm)
#
#         # Build mask for overlapping region and mirror it on the right side
#         Mm = mask[:, :c][:, -m:] & mask[:, c+1:][:, :m][:, ::-1]
#
#         d = (Lm - Rm)[Mm]                  # Compute pixel-wise difference only in masked region
#
#         score = np.mean(d * d)             # Compute mean squared error (MSE)
#
#         if score < best_score:             # If this axis gives lower error
#             best_score = score             # Update best score
#             best_c = c                     # Update best axis
#
#     return best_c, best_score             # Return best axis column and corresponding MSE
#
# def shift_x_subpixel(img, dx):
#     """
#     Subpixel horizontal shift using Fourier shift (no interpolation blur).
#     Positive dx shifts image to the right.
#     """
#     F = fftn(img)
#     shifted = np.real(ifftn(fourier_shift(F, shift=(0.0, dx))))
#     return shifted
#
# def center_by_symmetry(img, search_halfwidth=300, mask=None, subpixel=True):
#     """
#     Align image so its best left-right symmetry axis is centered.
#     Returns aligned_img, dx, axis_col
#     """
#     I = np.asarray(img, dtype=np.float64)
#     ny, nx = I.shape
#     target = nx // 2
#
#     axis_col, score = best_symmetry_axis(I, search_halfwidth=search_halfwidth, mask=mask)
#     dx = target - axis_col  # shift to put axis at center
#
#     if subpixel:
#         I2 = shift_x_subpixel(I, dx)
#     else:
#         I2 = np.roll(I, int(round(dx)), axis=1)
#
#     return I2, dx, axis_col, score

run_holo = "572"
dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped/"
sim = False
type = "exp"


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
end_size = [1875, 2050]
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

n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e ** 2
projected_electron_density_total = -ph * lam * n_c / np.pi * m_to_nm ** 2 / num_elec

# ny, nx = projected_electron_density_total.shape
# mask = np.zeros_like(projected_electron_density_total, dtype=bool)
# mask[:, nx//2-500:nx//2+500] = True   # focus scoring on central band
# projected_electron_density_total, dx, axis_col, score = center_by_symmetry(projected_electron_density_total, search_halfwidth=400, mask=mask, subpixel=True)
#print("Axis column was:", axis_col, "Shift applied dx:", dx, "Score:", score)

# axis, score = best_symmetry_axis(projected_electron_density_total, ds=4, search_halfwidth=400)
# ny, nx = projected_electron_density_total.shape
# dx = (nx - 1)/2 - axis
# print(dx)

method = "daun"
symmetry_axis = 0
dr = cmPerPx / scale_factor  # Pixel size for the transform to use -- default is 1
reg = 20  # Smoothing regularization parameter in PyAbel -- best is 20
recast_as_float64 = False


dx = 24.9727
projected_electron_density_total = shift(projected_electron_density_total, shift=(0, dx), order=1, mode='nearest')

projected_electron_density_total = cropToCenter(img=projected_electron_density_total,
                                                newSize=end_size)

projected_electron_density_total = padToSize(img=projected_electron_density_total,
                                             outputSize=out_size,
                                             padMethod='replicate',
                                             padType='both',
                                             padValue=None)


A = PyAbelOperator(input_shape=projected_electron_density_total.shape,
                   method=method,
                   symmetry_axis=symmetry_axis,
                   reg=reg,
                   dr=dr,
                   recast_as_float64=recast_as_float64)

y = projected_electron_density_total
x_inv = A.inverse(y)
f = loss.SquaredL2Loss(y=y, A=A)
lamm = 0.2  # L1 norm regularization parameter -- best is 0.2
g = lamm * functional.L1Norm()
C = linop.FiniteDifference(input_shape=y.shape)

rho = 20  # ADMM penalty parameter -- best is 20
maxiter = 10  # number of ADMM iterations
cg_tol = 1e-6  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration -- best is 200


solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[rho],
    x0=x_inv,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 10},
)

print(f"Solving on {device_info()}\n")
solver.solve()
x_tv = solver.x

x_tv = cropToCenter(img=x_tv,
                    newSize=end_size)

plt.figure()
plt.imshow(x_tv, clim=(0, 1000))
plt.show()

# dx = 24.9727
# projected_electron_density_total = shift(projected_electron_density_total, shift=(0, dx), order=1, mode='nearest')
#
#
# projected_electron_density_total = cropToCenter(img=projected_electron_density_total,
#                                                 newSize=end_size)
#
# projected_electron_density_total = padToSize(img=projected_electron_density_total,
#                                              outputSize=out_size,
#                                              padMethod='replicate',
#                                              padType='both',
#                                              padValue=None)





# electron_density_total_recon = abel.Transform(projected_electron_density_total,
#                                               direction='inverse',
#                                               method='daun',
#                                               symmetry_axis=0,
#                                               symmetrize_method='fourier',
#                                               transform_options=dict(reg=100, dr=cmPerPx / scale_factor),
#                                               recast_as_float64=True).transform
# electron_density_total_recon = electron_density_total_recon #* num_elec * (1e7 ** 2) / cmPerPx / scale_factor / 10e20
# # This must be manually tuned after output. The Abel transform does not know the absolute 0 so we adjust the
# # image globally based off a known value that is unperturbed by the shockwave. Additionally, we know from the
# # ground truth that inside the void should be ~0 e-/cm^3 and/or the SU8 should be ~390 x 10^20 e-/cm^3.
# # electron_density_total_recon -= 67
# # electron_density_total_recon[electron_density_total_recon < 0] = 0
# electron_density_total_recon = cropToCenter(img=electron_density_total_recon, newSize=end_size)
# plt.figure()
# plt.imshow(electron_density_total_recon, clim=(0, 1000))
# plt.show()





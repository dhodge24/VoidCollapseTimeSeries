import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import h5py
from tifffile import imread, imwrite
from scipy.interpolate import griddata
from scipy.special import lpmv

# Import custom modules
from SSPR.utilities import rotateImage


# =========================================================
# PREPROCESSING / XRAGE MAP HANDLING
# =========================================================

def interpolate_maps(x, scale_factor):
    new_shape = (int(x.shape[0] * scale_factor), int(x.shape[1] * scale_factor))
    print("The resized image is scaled up by this amount:", scale_factor)
    print("The new image shape is:", new_shape)

    interpolated_map = resize(
        x,
        new_shape,
        mode='constant',
        order=3,
        anti_aliasing=True,
        anti_aliasing_sigma=(1, 1)
    )
    return interpolated_map


def geom_transform(arr, scale_fac, crop_rows=2580, half_img=False):
    if half_img:
        # Use only half image and build full image by reflection
        arr = arr[-crop_rows:, :]

        ny, nx = arr.shape
        out = np.zeros((ny, 2 * nx), dtype=arr.dtype)
        out[:, :nx] = np.fliplr(arr)
        out[:, nx:] = arr
    else:
        out = arr[-crop_rows:, :]

    out = interpolate_maps(x=out, scale_factor=scale_fac)
    out = rotateImage(img=out, rotAngleDegree=180)

    return out


def process_h5_file(h5_file, scale_fac, half_img=False):
    data = {
        "density_total_GT": h5_file["/density_total_GT"][...],
        "areal_density_total": h5_file["/areal_density_total"][...],
        "electron_density_total_GT": h5_file["/electron_density_total_GT"][...],
        "projected_electron_density_total": h5_file["/projected_electron_density_total"][...],
    }

    for k in data:
        print(f"{k}: {data[k].shape}")

    for k in data:
        data[k] = geom_transform(data[k], scale_fac=scale_fac, half_img=half_img)

    return (
        data["density_total_GT"],
        data["areal_density_total"],
        data["electron_density_total_GT"],
        data["projected_electron_density_total"],
    )


# =========================================================
# ABEL / POLAR / LEGENDRE HELPERS
# =========================================================

def abel_transform(image):
    """
    Forward Abel transform using the matrix Abel approach.
    Input:
        image : (ny, nx) or (ny, nx, n_images)
    Returns:
        image_abel, A
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        squeeze_output = True
    elif image.ndim == 3:
        squeeze_output = False
    else:
        raise ValueError("Input image must be 2D or 3D.")

    ny, nx, n_images = image.shape

    if nx % 2 != 0:
        raise ValueError("Image width must be even for this Abel implementation.")

    w = nx // 2
    image_abel = np.zeros_like(image, dtype=np.float64)
    A = np.zeros((w, w), dtype=np.float64)

    for r in range(1, w + 1):
        for y in range(1, w + 1):
            if y == r:
                A[r - 1, y - 1] = (
                    -0.5 * np.sqrt(-1 + 2 * r) * r
                    + 0.5 * np.sqrt(-1 + 2 * r)
                    - 0.5 * r**2 * np.arcsin((r - 1) / r)
                    + 0.25 * r**2 * np.pi
                )
            elif y < r:
                pass
            else:
                A[r - 1, y - 1] = (
                    (-0.5 * y**2 + y - 0.5) * np.arcsin(r / (y - 1))
                    + (0.5 * y**2 - y + 0.5) * np.arcsin((r - 1) / (y - 1))
                    + 0.5 * np.sqrt(y**2 - 2 * y + 2 * r - r**2) * r
                    - 0.5 * np.sqrt(y**2 + 2 * r - 1 - r**2) * r
                    - 0.5 * np.sqrt(y**2 - 2 * y + 2 * r - r**2)
                    + 0.5 * np.sqrt(y**2 + 2 * r - 1 - r**2)
                    + 0.5 * np.sqrt(y**2 - r**2) * r
                    - 0.5 * y**2 * np.arcsin((r - 1) / y)
                    + 0.5 * y**2 * np.arcsin(r / y)
                    - 0.5 * np.sqrt(y**2 - 2 * y + 1 - r**2) * r
                )

    for n in range(n_images):
        image_working = image[:, :, n].T

        image_abel[w - 1::-1, :, n] = 2 * A @ image_working[w - 1::-1, :]
        image_abel[w:2 * w, :, n] = 2 * A @ image_working[w:2 * w, :]
        image_abel[:, :, n] = image_abel[:, :, n].T

    if squeeze_output:
        image_abel = image_abel[:, :, 0]

    return image_abel, A


def cart_2_polar(I_cart, res):
    """
    Convert Cartesian image to polar coordinates.

    Parameters
    ----------
    I_cart : ndarray (ny, nx)
    res : (nr, ntheta)

    Returns
    -------
    I_polar : ndarray (ntheta, nr)
    """
    I_cart = np.asarray(I_cart, dtype=np.float64)

    ny, nx = I_cart.shape
    x = np.linspace(-0.5 * nx, 0.5 * nx, nx)
    y = np.linspace(-0.5 * ny, 0.5 * ny, ny)
    X_cart, Y_cart = np.meshgrid(x, y)

    nr, ntheta = res
    r = np.linspace(0, 0.5 * nx, nr)
    theta = np.linspace(0, 2 * np.pi, ntheta)
    R, T = np.meshgrid(r, theta)

    X_pol = R * np.cos(T)
    Y_pol = R * np.sin(T)

    points = np.column_stack((X_cart.ravel(), Y_cart.ravel()))
    values = I_cart.ravel()

    I_polar = griddata(
        points,
        values,
        (X_pol, Y_pol),
        method="cubic",
        fill_value=0
    )

    return I_polar


def polar_2_cart(I_polar, res):
    """
    Convert polar image to Cartesian coordinates.

    Parameters
    ----------
    I_polar : ndarray (ntheta, nr)
    res : int

    Returns
    -------
    I_cart : ndarray (res, res)
    """
    I_polar = np.asarray(I_polar, dtype=np.float64)

    t_res, r_res = I_polar.shape

    r = np.linspace(0, r_res, r_res)
    theta = np.linspace(-np.pi, np.pi, t_res)
    R, T = np.meshgrid(r, theta)

    x = np.linspace(-r_res, r_res, res)
    X_cart, Y_cart = np.meshgrid(x, x)

    R_cart = np.sqrt(X_cart**2 + Y_cart**2)
    T_cart = np.arctan2(Y_cart, X_cart)

    points = np.column_stack((R.ravel(), T.ravel()))
    values = I_polar.ravel()

    I_cart = griddata(
        points,
        values,
        (R_cart, T_cart),
        method="cubic",
        fill_value=0
    )

    I_cart = np.nan_to_num(I_cart)
    return I_cart


def daLt(direction, data, l_max, crop_ang=0):
    """
    Discrete associated Legendre transform.
    """
    data = np.asarray(data, dtype=np.float64)

    if direction == "inverse":
        crop_ang = 0

    theta_full = np.linspace(0, np.pi, data.shape[1])

    crop_ang_rad = np.deg2rad(crop_ang)
    mask = (theta_full >= crop_ang_rad) & (theta_full <= (np.pi - crop_ang_rad))

    theta = theta_full[mask]
    data = data[:, mask]

    legendre_set = np.zeros((len(theta), l_max + 1), dtype=np.float64)

    # MATLAB used legendre(m, cos(theta), 'unnorm') and then row 2
    # That corresponds to associated Legendre order 1, degree m
    for m in range(1, l_max + 1):
        legendre_set[:, m] = lpmv(1, m, np.cos(theta))

    legendre_set = legendre_set.T

    if direction == "forward":
        legendre_transform = np.linalg.pinv(legendre_set)
        L = data @ legendre_transform
    elif direction == "inverse":
        L = legendre_set.T @ data
    else:
        raise ValueError("Direction input not supported - forward/inverse only")

    return L, legendre_set


def dLt_filter_odd(image, l_max, crop_ang):
    """
    Odd Legendre filtering.
    """
    image = np.asarray(image, dtype=np.float64)

    res = image.shape[0] // 2

    pol_raw = cart_2_polar(image, (res, 2 * res))
    pol_raw = pol_raw[:res, :]

    L, _ = daLt("forward", pol_raw.T, l_max, crop_ang)
    pol_filtered, _ = daLt("inverse", L.T, l_max, 0)

    pol_full = np.vstack([
        -pol_filtered,
        np.flipud(pol_filtered)
    ])

    image_filtered = polar_2_cart(pol_full, 2 * res)
    image_filtered = np.flipud(image_filtered).T

    return image_filtered, L


# =========================================================
# MAIT
# =========================================================

def MAIT(data, l=None, center_eps=1e-6):
    """
    Modified Abel Inversion Transform.

    Parameters
    ----------
    data : ndarray
        Input 2D projected data
    l : int or None
        Optional Legendre filtering order
    center_eps : float
        Small value to avoid singular divide at x=0

    Returns
    -------
    data_inv : ndarray
    """
    data = np.asarray(data, dtype=np.float64)

    # MATLAB gradient(Data) with one output corresponds to derivative along x/columns
    deriv = np.gradient(data, axis=1)

    if l is not None:
        deriv = dLt_filter_odd(deriv.T, l + 1, 0)[0].T

    nx = deriv.shape[1]
    x = np.linspace(-1, 1, nx)
    x_safe = np.where(np.abs(x) < center_eps, np.sign(x) * center_eps, x)
    x_safe[x == 0] = center_eps

    integrand = deriv / x_safe[None, :]

    data_abel, _ = abel_transform(integrand)
    data_inv = -(1 / (2 * np.pi)) * data_abel

    return data_inv


# =========================================================
# MAIN
# =========================================================

offset = '60'
time = '06500'
dir_xrage_data = f"XRAGEdata_{offset}um_off"

mask_percentage = 0.555
smooth_pixels = 1
deg_rotate = 180
scale_fac = 1.0
crop_initial_size = [2100, 2100]
crop_final_size = [2500, 2500]
half_img = False

h5_time1 = f"/Users/danielhodge/Desktop/{dir_xrage_data}/out/2/void-col-phase-attenuation-18.0-keV0{time}.h5"

E = 18000
c = 2.9979e8
m_e = 9.1094e-31
eps0 = 8.852e-12
e = 1.6022e-19
lam = (1239.84 / E) * 1e-9
r_e = 2.82e-15
N_A = 6.022e23

with h5py.File(h5_time1, "r") as f_h5_time1:
    processed_data = [process_h5_file(f_h5_time1, scale_fac=scale_fac, half_img=half_img)]

# unpack first file
density_total_GT, areal_density_total, electron_density_total_GT, projected_electron_density_total = processed_data[0]

# ---------------------------------------------------------
# Choose the map to invert with MAIT
# Most likely this is the projected electron density map
# ---------------------------------------------------------
input_map = projected_electron_density_total

# Optional crop to square/even size if needed
ny, nx = input_map.shape
nmin = min(ny, nx)
if nmin % 2 != 0:
    nmin -= 1

y0 = (ny - nmin) // 2
x0 = (nx - nmin) // 2
input_map = input_map[y0:y0+nmin, x0:x0+nmin]

# Run MAIT
recon_mait = MAIT(input_map, l=None, center_eps=1e-6)
# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_map, cmap="viridis")
plt.title("Input Projected Electron Density")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(recon_mait, cmap="viridis")
plt.title("MAIT Reconstruction")
plt.colorbar()

plt.tight_layout()
plt.show()
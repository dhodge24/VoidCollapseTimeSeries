import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import h5py
from tifffile import imwrite

# Import custom modules
from SSPR.utilities import rotateImage, cropToCenter, padToSize, padFadeOut


# =========================================================
# PREPROCESSING / XRAGE MAP HANDLING
# =========================================================

def interpolate_maps(x, zoom_factor):
    new_shape = (int(x.shape[0] * zoom_factor), int(x.shape[1] * zoom_factor))
    print("The resized image is scaled up by this amount:", zoom_factor)
    print("The new image shape is:", new_shape)

    interpolated_map = resize(
        x,
        new_shape,
        mode='constant',
        order=3,
        anti_aliasing=True,
        anti_aliasing_sigma=(1, 1),
        preserve_range=True,
    )
    return interpolated_map


def geom_transform(arr, zoom_factor, crop_rows=2580, half_img=False):
    if half_img:
        # Use only half image and build full image by reflection
        arr = arr[-crop_rows:, :]

        ny, nx = arr.shape
        out = np.zeros((ny, 2 * nx), dtype=arr.dtype)
        out[:, :nx] = np.fliplr(arr)
        out[:, nx:] = arr
    else:
        out = arr[-crop_rows:, :]

    out = interpolate_maps(x=out, zoom_factor=zoom_factor)
    out = rotateImage(img=out, rotAngleDegree=180)
    out = cropToCenter(img=out, newSize=[2500, 2500])

    return out


def process_h5_file(h5_file, zoom_factor, half_img=False):
    data = {
        "density_total_GT": h5_file["/density_total_GT"][...],
        "areal_density_total": h5_file["/areal_density_total"][...],
        "electron_density_total_GT": h5_file["/electron_density_total_GT"][...],
        "projected_electron_density_total": h5_file["/projected_electron_density_total"][...],
    }

    for k in data:
        print(f"{k}: {data[k].shape}")

    for k in data:
        data[k] = geom_transform(data[k], zoom_factor=zoom_factor, half_img=half_img)

    return (
        data["density_total_GT"],
        data["areal_density_total"],
        data["electron_density_total_GT"],
        data["projected_electron_density_total"],
    )


# =========================================================
# ABEL TRANSFORM
# =========================================================

def abel_transform(image):
    """
    Forward Abel transform using the matrix Abel approach.

    Parameters
    ----------
    image : ndarray
        Shape (ny, nx) or (ny, nx, n_images)

    Returns
    -------
    image_abel : ndarray
    A : ndarray
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


# =========================================================
# MAIT
# =========================================================

def MAIT(data, unitsPerPx, center_eps=1e-6):
    data = np.asarray(data, dtype=np.float64)

    # derivative with respect to units
    deriv = np.gradient(data, unitsPerPx, axis=1)  # e-/units^3 or g/units^3 -- units here could be nm, cm, etc.

    nx = deriv.shape[1]
    x = (np.arange(nx) - (nx - 1) / 2) * unitsPerPx  # Scale is in units -- units here could be nm, cm, etc.

    x_safe = x.copy().astype(np.float64)
    x_safe[np.abs(x_safe) < center_eps] = center_eps

    integrand = deriv / x_safe[None, :]  # e-/units^4 or g/units^4

    # Abel matrix still integrates in PIXEL units, so multiply by unitsPerPix -- units here could be nm, cm, etc.
    data_abel, _ = abel_transform(integrand)

    data_inv = -(unitsPerPx / (2 * np.pi)) * data_abel  # e-/units^3 or g/units^3

    return data_inv


# =========================================================
# MAIN
# =========================================================

offset = '60'
time = '06500'
dir_xrage_data = f"XRAGEdata_{offset}um_off"

deg_rotate = 180
zoom_fac = 2.0
half_img = False
save_results = False

h5_time1 = f"/Users/danielhodge/Desktop/{dir_xrage_data}/out/2/void-col-phase-attenuation-18.0-keV0{time}.h5"

# --------------------------------------------------
# Parameters
# --------------------------------------------------

# E = 18000
# c = 2.9979e8
# m_e = 9.1094e-31
# eps0 = 8.852e-12
# e = 1.6022e-19
# lam = (1239.84 / E) * 1e-9
# r_e = 2.82e-15
# N_A = 6.022e23

num_elec = 10e6
cmPerPx = 9.998679161071777e-06 / zoom_fac
nmPerPx = cmPerPx * 1e7

# --------------------------------------------------
# MAIT
# --------------------------------------------------

with h5py.File(h5_time1, "r") as f_h5_time1:
    processed_data = [process_h5_file(f_h5_time1, zoom_factor=zoom_fac, half_img=half_img)]

# Unpack file
density_total_GT, areal_density_total, electron_density_total_GT, projected_electron_density_total = processed_data[0]

# Optional crop to square/even size if needed
ny, nx = projected_electron_density_total.shape
nmin = min(ny, nx)
if nmin % 2 != 0:
    nmin -= 1

y0 = (ny - nmin) // 2
x0 = (nx - nmin) // 2
projected_electron_density_total = projected_electron_density_total[y0:y0+nmin, x0:x0+nmin]
areal_density_total = areal_density_total[y0:y0+nmin, x0:x0+nmin]


# end_size = [projected_electron_density_total.shape[0], projected_electron_density_total.shape[1]]
# out_size = [2 * end_size[0], 2 * end_size[1]]

# Run MAIT on the projected electron density
recon_mait_electron_density = MAIT(projected_electron_density_total, unitsPerPx=nmPerPx, center_eps=1e-6)
recon_mait_electron_density[recon_mait_electron_density < 0] = 0
# Multiply by num_elec since I scaled (divided) the projected electron density by this value earlier
recon_mait_electron_density = recon_mait_electron_density * num_elec
# recon_mait_electron_density = cropToCenter(img=recon_mait_electron_density, newSize=end_size)

# Run MAIT on the areal density
recon_mait_density = MAIT(areal_density_total, unitsPerPx=cmPerPx, center_eps=1e-6)
recon_mait_density[recon_mait_density < 0] = 0
recon_mait_density = recon_mait_density
# recon_mait_density = cropToCenter(img=recon_mait_density, newSize=end_size)

# Plot
plt.figure(figsize=(16, 8))

# Electron Density
plt.subplot(2, 3, 1)
plt.imshow(projected_electron_density_total,
           cmap="viridis")
plt.title("Projected Electron Density ($10^6$ e$^-$/nm$^2$)")
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(recon_mait_electron_density,
           cmap="viridis",
           clim=(np.min(electron_density_total_GT), np.max(electron_density_total_GT)))
plt.title("MAIT Reconstruction Electron Density (e$^-$/cm$^3$)")
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(electron_density_total_GT,
           cmap="viridis")
plt.title("Ground Truth Electron Density (e$^-$/cm$^3$)")
plt.colorbar()

# Density
plt.subplot(2, 3, 4)
plt.imshow(areal_density_total,
           cmap="viridis")
plt.title("Areal Density (g/cm$^2$)")
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(recon_mait_density,
           cmap="viridis",
           clim=(np.min(density_total_GT), np.max(density_total_GT)))
plt.title("MAIT Reconstruction Mass Density (g/cm$^3$)")
plt.colorbar()

plt.subplot(2, 3, 6)
plt.imshow(density_total_GT,
           cmap="viridis")
plt.title("Ground Truth Mass Density (g/cm$^3$)")
plt.colorbar()

plt.tight_layout()
plt.savefig(f"/Users/danielhodge/Desktop/MAIT_3D_density_plots.pdf",
            dpi=300,
            transparent=False)
plt.show()

if save_results:
    imwrite("/Users/danielhodge/Desktop/MAIT_electron_density_recon.tiff", recon_mait_electron_density)
    imwrite("/Users/danielhodge/Desktop/MAIT_density_recon.tiff", recon_mait_density)
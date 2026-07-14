"""

Get images from xRAGE first. Use "sim_XRAGE_maps.py"

"""


# Import python modules
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import h5py
from tifffile import imread, imwrite

# Import custom modules
from SSPR.utilities import (create_circular_mask, cropToCenter, rotateImage, shiftImage, reflect_image_2d,
                            shiftRotateMagnifyImage)


def interpolate_maps(x, scale_factor):
    new_shape = (int(x.shape[0] * scale_factor), int(x.shape[1] * scale_factor))
    print("The resized image is scaled up by this amount: ", scale_factor)
    print("The new image shape is size: ", new_shape)

    # Interpolate maps to the new resolution
    interpolated_map = resize(x,
                              new_shape,
                              mode='constant',
                              order=3,
                              anti_aliasing=True,
                              anti_aliasing_sigma=(1, 1))

    return interpolated_map


def geom_transform(arr, scale_fac, shifts=(0, 0), crop_rows=2580, crop_cols=1290, half_img=False):
    if half_img:
        # Use only half image and build full image by reflection
        arr = arr[-crop_rows:, :]

        ny, nx = arr.shape
        out = np.zeros((ny, 2 * nx), dtype=arr.dtype)
        out[:, :nx] = np.fliplr(arr)
        out[:, nx:] = arr
    else:
        # Use full image directly
        out = arr[-crop_rows:, :]

    # Resize / rotate / shift
    out = interpolate_maps(x=out, scale_factor=scale_fac)
    out = rotateImage(img=out, rotAngleDegree=180)
    out = shiftImage(img=out, shifts=shifts)

    return out


def process_h5_file(h5_file, scale_fac, shifts=(0, 0), half_img=False):
    # Load everything
    data = {
        "phi": -h5_file["/phase"][...],
        "mu": h5_file["/attenuation"][...],
        "density_total_GT": h5_file["/density_total_GT"][...],
        "areal_density_total": h5_file["/areal_density_total"][...],
        "density_total_recon": h5_file["/density_total_recon"][...],
        "electron_density_total_GT": h5_file["/electron_density_total_GT"][...],
        "projected_electron_density_total": h5_file["/projected_electron_density_total"][...],
        "electron_density_total_recon": h5_file["/electron_density_total_recon"][...],
    }

    for k in data:
        print(data[k].shape)

    # Apply the SAME transform to all
    for k in data:
        data[k] = geom_transform(data[k], scale_fac=scale_fac, shifts=shifts, half_img=half_img)

    # Return in your original order
    return (
        data["phi"],
        data["mu"],
        data["density_total_GT"],
        data["areal_density_total"],
        data["density_total_recon"],
        data["electron_density_total_GT"],
        data["projected_electron_density_total"],
        data["electron_density_total_recon"],
    )


def postprocess_map(img, crop_initial_size, crop_final_size, shifts2):
    out = img  # don't mutate input reference
    out = cropToCenter(out, crop_initial_size)
    out = reflect_image_2d(out)
    out = cropToCenter(img=out, newSize=[2100, 2100])

    # # For run 590, uncomment the below
    # out = shiftRotateMagnifyImage(img=out, shifts=[100, -150], padMethod='constant')  # For run 590, uncomment
    # out = cropToCenter(img=out, newSize=[2100, 2100])  # For run 590, uncomment
    # out = shiftRotateMagnifyImage(img=out, shifts=[110, -5], padMethod='constant')  # For run 590, uncomment
    # out = out[abs(200):2100 - 100, abs(200):]  # For run 590, uncomment

    # For run 590, comment all this out below
    out = shiftRotateMagnifyImage(img=out, shifts=shifts2, padMethod='constant')

    # out = out[shifts2[0]:, :-shifts2[1]]  # If both shifts are positive
    # out = out[shifts2[0]:, -shifts2[1]:]  # If 1st shift is positive and the 2nd shift is negative
    # # out = out[:1900, :] # ONLY FOR RUN 584
    out = out[:shifts2[0], :-shifts2[1]]  # If 1st shift is negative and the 2nd shift is positive

    return out


run_holo = "582"
offset = '60'
time = '09800'
dir_xrage_data = f"XRAGEdata_{offset}um_off"
# save = False
# use_mask = True

mask_percentage = 0.555
smooth_pixels = 1  # Number of pixels to smooth by for the mask edges
deg_rotate = 180
shifts1 = [-550, 0]
scale_fac = 2.9
shifts2 = [-350, 50]
crop_initial_size = [2100, 2100]
crop_final_size = [2500, 2500]
half_img = False  # Set to true if we are using xRAGE output half images instead of the full reflected image

# # for run 590
# shift_y_590 = 200
# shift_x_590 = -200
# len_x_590 = 2100
# len_y_590 = 2100

# Main directories
# dir_main = "/Users/danielhodge/Desktop/"
h5_time1 = f"/Users/danielhodge/Desktop/{dir_xrage_data}/out/2/void-col-phase-attenuation-18.0-keV0{time}.h5"

E = 18000  # Energy of the x-ray beam in eV
c = 2.9979e8  # Speed of light in m/s
m_e = 9.1094e-31  # Electron mass in kg
eps0 = 8.852e-12  # Permittivity of free space in units C^2 / (N * m^2)
e = 1.6022e-19  # Charge of an electron in C
lam = (1239.84 / E) * 1e-9  # Wavelength of the x-ray beam in meters
r_e = 2.82e-15  # Classical electron radius in meters
N_A = 6.022e23  # Avogadro's number in mol^-1

f_h5_time1 = h5py.File(h5_time1)
h5_files = [f_h5_time1]
processed_data = [process_h5_file(f, scale_fac=scale_fac, shifts=shifts1, half_img=half_img) for f in h5_files]

processed_data = [
    tuple(postprocess_map(a, crop_initial_size, crop_final_size, shifts2) for a in tup)
    for tup in processed_data
]


names = [
    "phi",
    "mu",
    "density_total_GT",
    "areal_density_total",
    "density_total_recon",
    "electron_density_total_GT",
    "projected_electron_density_total",
    "electron_density_total_recon",
]

out_dir = "/Users/danielhodge/Desktop/"

for file_idx, tup in enumerate(processed_data):
    # tup is the 7-tuple of processed arrays for this file
    for name, img in zip(names, tup):
        suffix = f"_file{file_idx:02d}" if len(processed_data) > 1 else ""
        out_path = f"{out_dir}{name}_adjusted{suffix}.tiff"
        imwrite(out_path, img.astype(np.float32))
        print("Saved:", out_path)

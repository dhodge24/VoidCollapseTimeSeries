
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

def ssim_map(img1, img2, *, win_size=11, gaussian_weights=True, sigma=1.5, data_range=None):
    """
    Compute a 2D SSIM similarity map between two 2D images.

    Returns
    -------
    ssim_mean : float
        Global mean SSIM.
    ssim_img : (H, W) float ndarray
        Per-pixel SSIM map (computed over local windows).
    """
    a = np.asarray(img1, dtype=np.float32)
    b = np.asarray(img2, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # If you don't specify data_range and your data are floats, SSIM can be wrong.
    if data_range is None:
        # robust-ish default: use combined min/max
        data_range = float(np.max([a.max(), b.max()]) - np.min([a.min(), b.min()]))

        if data_range == 0:
            # images are constant and identical -> SSIM is 1 everywhere
            return 1.0, np.ones_like(a, dtype=np.float32)

    ssim_mean, ssim_img = ssim(
        a, b,
        data_range=data_range,
        win_size=win_size,
        gaussian_weights=gaussian_weights,
        sigma=sigma,
        full=True
    )
    return float(ssim_mean), ssim_img.astype(np.float32)


img1 = np.array(imread("/Users/danielhodge/Desktop/electron_density_total_cropped.tiff"))
img2 = np.array(imread("/Users/danielhodge/Desktop/electron_density_total_recon_cropped.tiff"))

_, ssim_map = ssim_map(img1, img2)

plt.figure()
plt.imshow(ssim_map, cmap="Greys_r")
plt.colorbar()
plt.show()
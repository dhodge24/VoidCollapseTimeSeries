import torch
import numpy as np
from tifffile import imread, imwrite
from torchvision import transforms
import matplotlib.pyplot as plt


from scipy.ndimage import gaussian_filter

def soften_repaired_band(arr, c0, c1, pad=6, sigma=4.0):
    out = arr.copy()
    blurred = gaussian_filter(out, sigma=sigma / 2.35, truncate=2)

    L = max(c0 - pad, 0)
    R = min(c1 + pad, arr.shape[1])

    # smooth taper mask
    w = R - L
    t = np.linspace(0, 1, w)
    mask = np.sin(np.pi * t)**2
    mask[(c0 - L):(c1 - L)] = 1.0

    out[:, L:R] = (1 - mask[None, :]) * out[:, L:R] + mask[None, :] * blurred[:, L:R]
    return out

def smooth_vertical_inpaint(arr, c0, c1, blend=6):
    """
    Smoothly replace vertical band [c0:c1) by interpolation plus tapered blending.

    c0, c1 : bad band
    blend  : number of columns on each side used for smooth transition
    """
    out = arr.copy()
    ny, nx = arr.shape

    L = max(c0 - blend, 0)
    R = min(c1 + blend, nx)

    if L == 0 or R == nx:
        return out

    left_vals = arr[:, L]
    right_vals = arr[:, R - 1]

    width = R - L
    interp = np.zeros((ny, width), dtype=arr.dtype)

    # build smooth interpolated patch across the whole blended region
    for i in range(width):
        t = i / (width - 1)
        # smoothstep for smoother slope transition
        s = 3 * t ** 2 - 2 * t ** 3
        interp[:, i] = (1 - s) * left_vals + s * right_vals

    # cosine blend between original and interpolated patch
    alpha = np.zeros(width)
    for i in range(width):
        x = i / (width - 1)
        if x < 0.5:
            alpha[i] = 0.5 * (1 - np.cos(np.pi * min(x / 0.5, 1)))
        else:
            alpha[i] = 0.5 * (1 - np.cos(np.pi * min((1 - x) / 0.5, 1)))

    alpha = np.clip(alpha, 0, 1)

    original_patch = arr[:, L:R]
    out[:, L:R] = (1 - alpha[None, :]) * original_patch + alpha[None, :] * interp

    # force full replacement in the actual bad band
    out[:, c0:c1] = interp[:, (c0 - L):(c1 - L)]

    return out

def DWT(x):
    """Discrete wavelet transform. Used for yielding a more precise reconstruction with lower distortion. This type of
    transform can extract localized information better than a Fourier transform, where a Fourier transform gives the
    global average over the entire signal. This function is used to down sample the given image in x and y.

    Input shape: (1, C, H, W)
    Output shape: (1, 4*C, H/2, W/2)

    """

    # Just like the Fourier transform constant out front -- the constant can be split between the forward and inverse
    # operations or have the whole constant on either the forward or inverse transform (value 1/4). Here we split the
    # value (1/4) so the constant is on both the forward and inverse operations (value 1/2). The 1/4 on one or 1/2 on
    # both is to ensure symmetry between operations.
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    # Size in PyTorch is (N, C, H, W). N is batch, C is number of channels, H is rows, and W is columns
    xLL = x1 + x2 + x3 + x4  # Size (N, C, H/2, W/2) -- Lower resolution image LL
    xLH = x1 - x2 + x3 - x4  # Size (N, C, H/2, W/2) -- Horizontal features LH
    xHL = x1 + x2 - x3 - x4  # Size (N, C, H/2, W/2) -- Vertical features HL
    xHH = x1 - x2 - x3 + x4  # Size (N, C, H/2, W/2) -- Diagonal features HH

    return torch.cat((xLL, xLH, xHL, xHH), 1)


def IDWT(x):
    """Inverse discrete wavelet transform. Used for yielding a more precise reconstruction with lower distortion."""

    inBatch, inChannel, inH, inW = x.size()  # Size in PyTorch is (N, C, H, W)
    r = 2  # Factor for upsampling x and y dimensions by 2 and for downsampling channel dimensions by this value squared
    outBatch, outChannel, outH, outW = inBatch, int(inChannel / r**2), r * inH, r * inW
    x1 = x[:, 0:outChannel, :, :] / 2  # Lower resolution image LL
    x2 = x[:, outChannel:outChannel*2, :, :] / 2  # Horizontal features LH
    x3 = x[:, outChannel*2:outChannel*3, :, :] / 2  # Vertical features HL
    x4 = x[:, outChannel*3:outChannel*4, :, :] / 2  # Diagonal features HH

    h = torch.zeros([outBatch, outChannel, outH, outW]).float()

    h[:, :, 0::2, 0::2] = x1 + x2 + x3 + x4
    h[:, :, 1::2, 1::2] = x1 - x2 - x3 + x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4

    return h


convert_to_tensor = transforms.ToTensor()
img = np.array(imread("/Users/danielhodge/Desktop/I_mait.tiff"), dtype=np.float32)
# Convert to torch tensor
img = transforms.ToTensor()(img).float()   # shape: (1, H, W) for grayscale
img = img.unsqueeze(0)                     # shape: (1, 1, H, W)

# -------------------------
# FORWARD WAVELET TRANSFORM
# -------------------------
x = DWT(img)   # expected shape: (1, 4, H/2, W/2)

# Zero out bad vertical line in selected subbands
x[0, 0, :, 98:102] = 0
x[0, 2, :, 98:102] = 0

# -------------------------
# DENOISE SELECTED SUBBAND
# -------------------------
# Convert to NumPy for SCICO processing
x_np = x.detach().cpu().numpy()

print("Subband shape:", x_np[0, 0].shape)

x_np[0, 0] = smooth_vertical_inpaint(x_np[0, 0], 98, 102, blend=8)
# x_np[0, 0] = soften_repaired_band(x_np[0, 0], 98, 102, pad=8, sigma=4.0)
x_np[0, 2] = smooth_vertical_inpaint(x_np[0, 2], 98, 102, blend=8)
# x_np[0, 2] = soften_repaired_band(x_np[0, 2], 98, 102, pad=8, sigma=4.0)

# Convert back to torch tensor
x = torch.from_numpy(x_np).float()

print("Wavelet coeff shape:", x.shape)

# -------------------------
# INVERSE WAVELET TRANSFORM
# -------------------------
out = IDWT(x)                  # shape likely (1, 1, H, W)
out = torch.squeeze(out)       # shape (H, W)


out = out.detach().cpu().numpy()
# out = gaussian_filter(out, sigma=4 / 2.35, truncate=2)

# -------------------------
# DISPLAY
# -------------------------
plt.figure()
plt.imshow(out, cmap="viridis")
plt.axis("off")
plt.show()


"""
This file contains functions that are frequently used

Many functions here were converted to python from MATLAB and/or were modified from existing code from:
1) https://gitlab.gwdg.de/irp/holotomotoolbox
2) https://github.com/rafael-fuente/diffractsim

If a citation is missing, let me know

"""

import numpy as np
import cv2 as cv2
from scipy import ndimage
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy.special import voigt_profile
import threading
import progressbar
import time
from matplotlib.widgets import Slider, Button


def FFT(img):
    """2D Fourier transform"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))  # Provides correct magnitude and phase output


def IFFT(img):
    """2D inverse Fourier transform"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))  # Provides correct magnitude and phase output


def propagator(wavefrontIn, lam, dx, dy, dist):
    """Propagation function -- see J. Goodman "Fourier Optics"

    Parameters
    -----------
    :param wavefrontIn : Original complex wavefront [N, M]
    :param lam : Wavelength
    :param dx : Pixel size in x dimension
    :param dy : Pixel size in y dimension
    :param dist : Propagation distance

    Returns
    -----------
    :returns wavefrontOut : Complex wavefront after propagation [N, M]

    """

    Ny, Nx = np.shape(wavefrontIn)
    k0 = 2 * np.pi / lam  # Wave number
    # Recall: k = 2*pi*f where f is a SPATIAL frequency
    dkx = 2 * np.pi / (Nx * dx)  # Pixel size (x-dimension) in k-space (spatial frequency)
    dky = 2 * np.pi / (Ny * dy)  # Pixel size (y-dimension) in k-space (spatial frequency)
    KX = np.arange(-Nx / 2, Nx / 2, 1) * dkx
    KY = np.arange(-Ny / 2, Ny / 2, 1) * dky
    kx, ky = np.meshgrid(KX, KY)
    arg = k0 ** 2 - kx ** 2 - ky ** 2
    arg[arg < 0] = 0
    kz = np.sqrt(np.real(arg))
    H = np.exp(1j * kz * dist)  # forward propagation kernel -- take conjugate for backward propagation
    wavefrontOut = IFFT(FFT(wavefrontIn) * H)
    return wavefrontOut


def removeOutliers(originalImage, threshold=2):
    """

    Parameters
    -----------

    :param originalImage: [N, M] image for outlier removal
    :param threshold: How many standard deviations away to consider when fixing pixels

    Returns
    -----------
    :returns Fixed image [N, M]

    """

    # Handles NaN
    originalImage[np.isnan(originalImage)] = np.inf

    # Remove hot and cold pixels
    filteredImage = median_filter(originalImage, mode='nearest', size=(3, 3))
    differenceImg = (originalImage - filteredImage)  # difference between original and filtered image

    # standard deviation of gray values in difference-image, excluding infinite values
    differenceImageFinite = differenceImg[np.logical_not(np.isinf(differenceImg))]
    stdGrayVal = differenceImageFinite.std(axis=0)  # Same as std() in Matlab
    # stdGrayVal = np.std(differenceImageFinite[:], ddof=1)  # Same as std() in Matlab

    # Find pixels to correct
    pixelsToCorrect = np.abs(differenceImg) > (threshold * stdGrayVal)

    # Replace faulty pixels
    correctedImage = originalImage
    correctedImage[pixelsToCorrect] = filteredImage[pixelsToCorrect]
    return correctedImage


def remove_outliers_mult_imgs(original_images, threshold, kernSize):
    """Remove outliers from a batch of images"""

    num_images, height, width = original_images.shape
    corrected_images = np.empty_like(original_images, dtype=np.float32)

    for i in range(num_images):
        original_image = original_images[i, :, :].astype(np.float32)

        # Handles NaN
        original_image[np.isnan(original_image)] = np.inf

        # Remove hot and cold pixels
        filtered_image = median_filter(original_image, mode='nearest', size=(kernSize, kernSize))
        difference_img = original_image - filtered_image  # difference between original and filtered image

        # Standard deviation of gray values in difference-image, excluding infinite values
        difference_image_finite = difference_img[np.logical_not(np.isinf(difference_img))]  # Consider non-inf vals
        std_gray_val = difference_image_finite.std()  # Same as std() in Matlab

        # Find pixels to correct
        pixels_to_correct = abs(difference_img) > (threshold * std_gray_val)

        # Replace faulty pixels
        corrected_image = original_image
        corrected_image[pixels_to_correct] = filtered_image[pixels_to_correct]
        corrected_images[i] = corrected_image

    return corrected_images


def cropToCenter(img, newSize):
    """Code converted from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/imageProcessing/cropPadWindow/cropToCenter.m

    cropToCenter returns the central part of a 2d or 3d input array. It trims downs the input image and leaves
    the  coordinate grid untouched. If a resampling is required, use cropToCenterSubpixel, to interpolate the output
    (over 1 pixel difference)

        Parameters
        -----------
        :param img : Numerical array to crop of size [N, M]
        :param newSize : Size in which to crop the numerical array [N, M]

        Returns
        -----------
        :returns croppedImg : Numerical array that is cropped
    """
    img = np.array(img)
    imageSize = np.array(np.shape(img), dtype=np.float32)  # Image shape
    #[y, x] = imageSize[0], imageSize[1]
    y, x = imageSize[0], imageSize[1]

    assert newSize[0] <= y, 'Output height is larger than the input.'
    assert newSize[1] <= x, 'Output width is larger than the input.'

    # Note: if N or M is even and N or M odd, then the (M/2)+1 element is the center, the same holds for N. This is the
    # location of the 0th order in a FFT of an image with even number of pixels.

    newSizey = newSize[0]
    newSizex = newSize[1]
    rawOffset = (np.asarray([y, x]) - np.asarray(newSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(np.ceil(rawOffsety)) - 1
    offsetx = int(np.ceil(rawOffsetx)) - 1
    offsetEndy = offsety + newSizey
    offsetEndx = offsetx + newSizex
    croppedImg = img[offsety:offsetEndy, offsetx: offsetEndx]

    return croppedImg


def cropToCenterSubPixel(img, newSize):
    """ This function returns the central part of a 2d input array with sub-pixel precision.

        img : Numerical array to crop of size [N, M]
        newSize : Size in which to crop the numerical array [N, M]

        returns
        --------
        croppedImg : Numerical array that is cropped
    """

    imageSize = np.array(np.shape(img))  # Image shape
    y, x = imageSize[0], imageSize[1]

    assert newSize[0] <= y, 'Output height is larger than the input.'
    assert newSize[1] <= x, 'Output width is larger than the input.'

    newSizey = newSize[0]
    newSizex = newSize[1]
    rawOffset = (np.array([y, x]) - np.array(newSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(np.ceil(rawOffsety)) - 1
    offsetx = int(np.ceil(rawOffsetx)) - 1

    shiftDisty = offsety - rawOffsety
    shiftDistx = offsetx - rawOffsetx

    if (np.array([shiftDisty, shiftDistx])).any():
        img = shiftImage(img, -np.array([shiftDisty, shiftDistx]))

    offsetEndy = offsety + newSizey
    offsetEndx = offsetx + newSizex

    croppedImg = img[offsety:offsetEndy, offsetx: offsetEndx]

    return croppedImg


def croppedArray(array, crop_pre, crop_post=None):
    """Code converted from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/auxiliary/croparray.m

    This function crops a given numerical array by a specified amount of rows/columns at the beginning and/or
    end of the array

    :param array: Input image to be cropped [N, M] or [N, N]
    :param crop_pre: Tuple of non-negative integers. Amount of rows/columns/etc to crop at the beginning of the array
    along the different dimensions
    :param crop_post: Tuple of non-negative integers, optional Amount rows/columns/etc to crop at the end of the array
    along the different dimensions. If not assigned, cropPost = cropPre (--> symmetric cropping) is assumed.

    :return: Cropped image
    """

    N = array.shape
    num_dimensions = len(N)

    if crop_post is None:
        crop_post = crop_pre

    # If crop_pre and/or crop_post have fewer entries than the number of dimensions of the array, fill with zeros
    crop_pre = np.concatenate((crop_pre, np.zeros(num_dimensions - len(crop_pre)))).astype(int)
    crop_post = np.concatenate((crop_post, np.zeros(num_dimensions - len(crop_post)))).astype(int)

    cropped_array = array.copy()
    for dim in range(num_dimensions):
        if crop_pre[dim] > 0 or crop_post[dim] > 0:
            idx_start = crop_pre[dim]
            idx_end = N[dim] - crop_post[dim]
            idx = tuple(slice(None) if i != dim else slice(idx_start, idx_end) for i in range(num_dimensions))
            cropped_array = cropped_array[idx]
    return cropped_array


def replaceCenterSubPixel(largeArray, smallArray):
    """Replace larger array with a portion of a smaller array"""

    largeSize = np.shape(largeArray)
    smallSize = np.shape(smallArray)

    assert largeSize[0] >= smallSize[0], 'Height is larger than the total.'
    assert largeSize[1] >= smallSize[1], 'Width is larger than the total.'

    rawOffset = (np.array(largeSize) - np.array(smallSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(np.floor(rawOffset[0]))
    offsetx = int(np.floor(rawOffset[1]))

    shiftDisty = int(rawOffsety - offsety)
    shiftDistx = int(rawOffsetx - offsetx)
    shiftDist = np.array([shiftDisty, shiftDistx])
    innerSize = (smallSize + np.ceil(shiftDist)).astype(int)

    innerSizey = innerSize[0]
    innerSizex = innerSize[1]
    offsetEndy = int(offsety + innerSizey)
    offsetEndx = int(offsetx + innerSizex)

    if shiftDistx and shiftDisty is None:
        innerArray = smallArray
    else:
        innerArray = np.zeros(innerSize, dtype=np.float32)
        innerMask = np.zeros(innerSize, dtype=np.float32)
        innerArray[0:smallSize[0], 0:smallSize[1]] = smallArray
        innerMask[0:smallSize[0], 0:smallSize[1]] = 1
        innerArray = shiftImage(innerArray, shifts=shiftDist)
        innerMask = shiftImage(innerMask, shifts=shiftDist)
        innerArray = innerArray + (1 - innerMask) * largeArray[offsety:offsetEndy, offsetx:offsetEndx]

    result = largeArray
    result[offsety:offsetEndy, offsetx:offsetEndx] = innerArray

    return result


def padToSize(img, outputSize, padMethod, padType, padValue):
    """
    padToSize pads or crops an image (either real or complex) to a given size.
    This exactly replicates the MATLAB padarray function -- See: https://www.mathworks.com/help/images/ref/padarray.html


    Parameters
    -----------
    :param img : Numerical real or complex array to be padded -- [N, M]
    :param outputSize : Padded image size -- [P, Q]
    :param padMethod : string, Options are: 'replicate', 'constant'
    :param padType : string, Options are: 'pre', 'post', 'both', 'preandpost', 'postandpre'
              pre:  Pad before the first array element along each dimension.
              post: Pad after the last array element along each dimension.
              both: Pads before the first array element and after the last array element along each dimension.
    :param padValue : constant number/value: e.g, mean value of the image, 0, 1, etc...

    Returns
    -----------
    :returns imgPadded : Numerical array that is padded or cropped
    """

    img = np.array(img)
    imageSize = np.shape(img)
    y, x = imageSize[0], imageSize[1]
    ynew, xnew = outputSize  # Desired output size
    ypad = ynew - y  # padding to add in y
    xpad = xnew - x  # padding to add in x

    if padValue is None:
        padValue = 0.0

    # Crop the image if padding becomes negative
    if (np.array([ypad, xpad]) < 0).any():  # any is a logical OR operation
        imageSize = np.minimum(imageSize, outputSize)
        imgCropped = cropToCenter(img, imageSize)
        return imgCropped

    # Pad the image if padding is positive
    else:
        if padMethod == 'replicate' and padType == 'pre':
            paddedImg = np.pad(img,
                               pad_width=[(int(ypad), 0), (int(xpad), 0)],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'post':
            paddedImg = np.pad(img,
                               pad_width=[(0, int(ypad)), (0, int(xpad))],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'both':
            pad_top = int(ypad // 2)
            pad_bottom = int(ypad // 2 + ypad % 2)
            pad_left = int(xpad // 2)
            pad_right = int(xpad // 2 + xpad % 2)
            paddedImg = np.pad(img,
                               pad_width=[(pad_top, pad_bottom), (pad_left, pad_right)],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'preandpost':
            pad_top = int(np.ceil(ypad / 2))
            pad_bottom = int(np.floor(ypad / 2))
            pad_left = int(np.ceil(xpad / 2))
            pad_right = int(np.floor(xpad / 2))
            paddedImg = np.pad(img,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='edge')
            paddedImg = np.pad(paddedImg,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'postandpre':
            pad_top = int(np.floor(ypad / 2))
            pad_bottom = int(np.ceil(ypad / 2))
            pad_left = int(np.floor(xpad / 2))
            pad_right = int(np.ceil(xpad / 2))
            paddedImg = np.pad(img,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='edge')
            paddedImg = np.pad(paddedImg,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='edge')
            return paddedImg

        if padMethod == 'constant' and padType == 'pre':
            paddedImg = np.pad(img,
                               pad_width=[(int(ypad), 0), (int(xpad), 0)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg
        if padMethod == 'constant' and padType == 'post':
            paddedImg = np.pad(img,
                               pad_width=[(0, int(ypad)), (0, int(xpad))],
                               mode='constant',
                               constant_values=padValue)

            return paddedImg
        if padMethod == 'constant' and padType == 'both':
            pad_top = int(ypad // 2)
            pad_bottom = int(ypad // 2 + ypad % 2)
            pad_left = int(xpad // 2)
            pad_right = int(xpad // 2 + xpad % 2)
            paddedImg = np.pad(img,
                               pad_width=[(pad_top, pad_bottom), (pad_left, pad_right)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg
        if padMethod == 'constant' and padType == 'preandpost':
            pad_top = int(np.ceil(ypad / 2))
            pad_bottom = int(np.floor(ypad / 2))
            pad_left = int(np.ceil(xpad / 2))
            pad_right = int(np.floor(xpad / 2))
            paddedImg = np.pad(img,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='constant',
                               constant_values=padValue)
            paddedImg = np.pad(paddedImg,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg
        if padMethod == 'constant' and padType == 'postandpre':
            pad_top = int(np.floor(ypad / 2))
            pad_bottom = int(np.ceil(ypad / 2))
            pad_left = int(np.floor(xpad / 2))
            pad_right = int(np.ceil(xpad / 2))
            paddedImg = np.pad(img,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='constant',
                               constant_values=padValue)
            paddedImg = np.pad(paddedImg,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg


# Old, does not work for complex data
# def padToSize(img, outputSize, padMethod, padType, padValue):
#     """
#     padToSize pads or crops an image to a given size
#
#     Parameters
#     -----------
#     :param img : Numerical array image which is padded -- [P, Q]
#     :param outputSize : Padded image size -- [N, M]
#     :param padMethod : string, Options are: 'replicate', 'constant'
#     :param padType : string, Options are: 'pre', 'post', 'both', 'preandpost', 'postandpre'
#               pre:  Pad before the first array element along each dimension.
#               post: Pad after the last array element along each dimension.
#               both: Pads before the first array element and after the last array element along each dimension.
#     :param padValue : constant number/value: e.g, mean value of the image, 0, 1, etc...
#
#     Returns
#     -----------
#     :returns imgPadded : Numerical array that is padded or cropped -- [N, M]
#     """
#
#     # These outputs below is a copy of the MATLAB padarray function.
#     # top, bottom, left, right  --  Filling in values for all slots is 'both'.
#     # Filling in values for 1st and 3rd slot is 'pre'. Filling in values for the 2nd and 4th slot is 'post'
#
#     img = np.array(img)
#     imageSize = np.shape(img)  # Image shape
#     y, x = imageSize[0], imageSize[1]
#     ynew, xnew = outputSize  # Desired output size
#     ypad = ynew - y  # padding to add in y
#     xpad = xnew - x  # padding to add in x
#
#     # Crop the image if padding becomes negative
#     if (np.array([ypad, xpad]) < 0).any():  # any is a logical OR operation
#         imageSize = np.minimum(imageSize, outputSize)
#         imgCropped = cropToCenter(img, imageSize)
#         return imgCropped
#
#     # Pad the image if padding is positive
#     else:
#         if padMethod == 'replicate' and padType == 'pre':
#             paddedImg = cv2.copyMakeBorder(img, int(ypad), 0, int(xpad), 0, cv2.BORDER_REPLICATE, value=None)
#             return paddedImg
#         if padMethod == 'replicate' and padType == 'post':
#             paddedImg = cv2.copyMakeBorder(img, 0, int(ypad), 0, int(xpad), cv2.BORDER_REPLICATE, value=None)
#             return paddedImg
#         if padMethod == 'replicate' and padType == 'both':
#             paddedImg = cv2.copyMakeBorder(img, int(ypad // 2), int(ypad // 2 + ypad % 2), int(xpad // 2),
#                                            int(xpad // 2 + xpad % 2), cv2.BORDER_REPLICATE, value=None)
#             return paddedImg
#         if padMethod == 'replicate' and padType == 'preandpost':
#             paddedImg = cv2.copyMakeBorder(img, int(np.ceil(ypad / 2)), 0, int(np.ceil(xpad / 2)), 0,
#                                            cv2.BORDER_REPLICATE, value=None)
#             paddedImg = cv2.copyMakeBorder(paddedImg, 0, int(np.floor(ypad / 2)), 0, int(np.floor(xpad / 2)),
#                                            cv2.BORDER_REPLICATE, value=None)
#             return paddedImg
#         if padMethod == 'replicate' and padType == 'postandpre':
#             paddedImg = cv2.copyMakeBorder(img, 0, int(np.ceil(ypad / 2)), 0, int(np.ceil(xpad / 2)),
#                                            cv2.BORDER_REPLICATE, value=None)
#             paddedImg = cv2.copyMakeBorder(paddedImg, int(np.floor(ypad / 2)), 0, int(np.floor(xpad / 2)), 0,
#                                            cv2.BORDER_REPLICATE, value=None)
#             return paddedImg
#
#         if padMethod == 'constant' and padType == 'pre':
#             if padValue is None:
#                 padValue = 0.0
#                 paddedImg = cv2.copyMakeBorder(img, int(ypad), 0, int(xpad), 0, cv2.BORDER_CONSTANT,
#                                                value=padValue + 0.0)
#             else:
#                 paddedImg = cv2.copyMakeBorder(img, int(ypad), 0, int(xpad), 0, cv2.BORDER_CONSTANT,
#                                                value=padValue + 0.0)
#             return paddedImg
#         if padMethod == 'constant' and padType == 'post':
#             if padValue is None:
#                 padValue = 0.0
#                 paddedImg = cv2.copyMakeBorder(img, 0, int(ypad), 0, int(xpad), cv2.BORDER_CONSTANT,
#                                                value=padValue + 0.0)
#             else:
#                 paddedImg = cv2.copyMakeBorder(img, 0, int(ypad), 0, int(xpad), cv2.BORDER_CONSTANT,
#                                                value=padValue + 0.0)
#             return paddedImg
#         if padMethod == 'constant' and padType == 'both':
#             if padValue is None:
#                 padValue = 0.0
#                 paddedImg = cv2.copyMakeBorder(img, int(ypad // 2), int(ypad // 2 + ypad % 2), int(xpad // 2),
#                                                int(xpad // 2 + xpad % 2), cv2.BORDER_CONSTANT, value=padValue + 0.0)
#             else:
#                 paddedImg = cv2.copyMakeBorder(img, int(ypad // 2), int(ypad // 2 + ypad % 2), int(xpad // 2),
#                                                int(xpad // 2 + xpad % 2), cv2.BORDER_CONSTANT, value=padValue + 0.0)
#             return paddedImg
#         if padMethod == 'constant' and padType == 'preandpost':
#             if padValue is None:
#                 padValue = 0.0
#                 paddedImg = cv2.copyMakeBorder(img, int(np.ceil(ypad / 2)), 0, int(np.ceil(xpad / 2)), 0,
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#                 paddedImg = cv2.copyMakeBorder(paddedImg, 0, int(np.floor(ypad / 2)), 0, int(np.floor(xpad / 2)),
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#             else:
#                 paddedImg = cv2.copyMakeBorder(img, int(np.ceil(ypad / 2)), 0, int(np.ceil(xpad / 2)), 0,
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#                 paddedImg = cv2.copyMakeBorder(paddedImg, 0, int(np.floor(ypad / 2)), 0, int(np.floor(xpad / 2)),
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#             return paddedImg
#         if padMethod == 'constant' and padType == 'postandpre':
#             if padValue is None:
#                 padValue = 0.0
#                 paddedImg = cv2.copyMakeBorder(img, 0, int(np.ceil(ypad / 2)), 0, int(np.ceil(xpad / 2)),
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#                 paddedImg = cv2.copyMakeBorder(paddedImg, int(np.floor(ypad / 2)), 0, int(np.floor(xpad / 2)), 0,
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#             else:
#                 paddedImg = cv2.copyMakeBorder(img, 0, int(np.ceil(ypad / 2)), 0, int(np.ceil(xpad / 2)),
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#                 paddedImg = cv2.copyMakeBorder(paddedImg, int(np.floor(ypad / 2)), 0, int(np.floor(xpad / 2)), 0,
#                                                cv2.BORDER_CONSTANT, value=padValue + 0.0)
#             return paddedImg


def centeredGrid(N, dx=None, computeMeshgrid=False):
    """
    centeredGrid creates a centered coordinate-grid of given size.

    Parameters
    -----------
    :param N : Size of grid -- [N, M]
    :param dx : Grid spacing in x and y -- [n, m]
    :param computeMeshgrid : Boolean for computing meshgrid

    Returns
    -----------
    :returns Centered grid
    """
    if dx is None:
        dx = [1]  # Step size of 1

    dx = np.transpose(dx[:]) * np.ones([1, len(N)]).flatten()
    ndim = len(N)

    # Optionally assemble meshgrid (same as MATLAB ndgrid if index is ij) (occupies more memory!)
    if computeMeshgrid:
        Y = np.arange(-N[0] / 2 + 0.5, N[0] / 2, 1) * dx[0]
        X = np.arange(-N[1] / 2 + 0.5, N[1] / 2, 1) * dx[1]
        xnew, ynew = np.meshgrid(X, Y)
        xnew = xnew.astype(np.float32)
        ynew = ynew.astype(np.float32)
        return xnew, ynew

    elif ndim > 1:
        Y = np.arange(-N[0] / 2 + 0.5, N[0] / 2, 1).flatten().reshape(1, N[0]) * dx[0]
        X = np.arange(-N[1] / 2 + 0.5, N[1] / 2, 1).flatten().reshape(N[1], 1) * dx[1]
        xnew = X.astype(np.float32)
        ynew = Y.astype(np.float32)
        return xnew, ynew


def padFadeOut(img, outputSize, transitionLength=None, padValue=None, fadeoutParallel=False, computeMeshgrid=False):
    """ Code converted/modified from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/imageProcessing/cropPadWindow/padFadeout.m

    padFadeOut smoothly pads an input image to a given output size. The length of this smooth transition is determined
    by transitionLength.

    Parameters
    -----------
    :param img : Numerical array image to be padded

    :param outputSize : Tuple of the size [N, M] or [N, N] -- [height, width]

    :param transitionLength : Length of the transition to the constant padding-value in pixels.
                       Default = [], --> transitionLength = [padPre, PadPre]

    :param padValue : Image fades to this constant selected pad value
             Default = [], --> padVal = np.mean(im[:]])
             Or you can input a number, e.g padVal = 1 -- Fades out to this selected pad value

    :param fadeoutParallel : Fades out perpendicular to image

    :param computeMeshgrid : Computes meshgrid


    If a 2-tuple([p, q]) is assigned, the two values are interpreted as (possibly different) lengths along along the
    different image-dimensions.
    If empty ([]) , transitionLength = np.ceil((sizePadded-size[img])/2) is used, so that the transition-region fills
    the whole padding area between the original boundaries of the original image and the padded one.

    fadeoutParallel : Default = false, Determines whether, the replicated boundary-values of the image are also faded
    out parallel to the image edge, by increasingly smoothing the replicated rows and columns the further away the
    padding proceeds from the edges of the original image. This option results in a visually 'cleaner' fade-out, but
    is also computationally more costly.

    Returns
    -----------
    :returns Padded image that is faded

    """

    img = np.array(img, dtype=np.float32)
    imageSize = np.asarray(np.shape(img))
    outputSize = np.asarray(outputSize)
    padPre = np.ceil((np.asarray(outputSize) - imageSize[0:2]) / 2)
    padPost = np.floor((np.asarray(outputSize) - imageSize[0:2]) / 2)

    if padValue is None:
        # Default: Padding-value is taken as mean of the image if not assigned
        padValue = img.mean() + 0.0
    if transitionLength is None:
        # Default: Transition spreads over total pad-region
        transitionLengthy = int(padPre[0])
        transitionLengthx = int(padPre[1])
        transitionLength = np.asarray([transitionLengthy, transitionLengthx])
    else:
        transitionLength = np.asarray([transitionLength[0], transitionLength[1]])

    # Step 1: Construct raw padded image

    # Variant 1.1: replicate padding with additional fade-out parallel to the image-edges by repetitively
    # smoothing the replicated boundary-values of the image

    if fadeoutParallel:
        imgPadded = padToSize(img, outputSize=outputSize, padMethod='constant', padType='preandpost', padValue=padValue)
        imgPadRegion1 = imgPadded
        imgPadRegion2 = np.transpose(imgPadded)
        imgPadRegion = np.asarray([imgPadRegion1, imgPadRegion2])

        for dim in [0, 1]:
            otherDim = 2 - dim
            imgTemp = np.fft.fft(imgPadRegion[dim], axis=0)
            if dim == 0:
                kernel = np.real(np.fft.fft(np.hstack([1 / 3, 1 / 3, np.zeros([outputSize[0] - 3]), 1 / 3]), axis=0))
            else:
                kernel = np.real(np.fft.fft(np.hstack([1 / 3, 1 / 3, np.zeros([outputSize[1] - 3]), 1 / 3]), axis=0))
            list1 = np.flip(np.arange(padPre[otherDim - 1]))
            list1 = [int(x) for x in list1]
            list2 = np.flip(np.arange(padPost[otherDim - 1]))
            list2 = [int(x) for x in list2]
            for jj in np.flip(range(len(list1))):
                imgTemp[:, jj] = kernel * imgTemp[:, jj + 1]
            for jj in np.flip(range(len(list2))):
                imgTemp[:, -1 - jj] = kernel * imgTemp[:, -1 - jj - 1]
            imgTemp = np.fft.ifft(imgTemp, axis=0)
            if np.isreal(imgTemp.any()):
                imgTemp = np.real(imgTemp)
            imgPadRegion[dim] = imgTemp

        imgPadded = imgPadRegion[0] + np.transpose(imgPadRegion[1]) - imgPadded

    # Variant 1.2 (default case): no fade-out parallel to the edges. Raw padding is simple replicate-operation

    else:
        imgPadded = padToSize(img, outputSize=outputSize, padMethod='replicate', padType='preandpost', padValue=None)

    #  Step 2: Fadeout perpendicular to the image boundaries

    #  Construct mask that is 1 within the region covered by the original image and decays to zero as the distance
    #  to the image-boundaries increases. The length of the transition from 1 to 0 is defined by transitionLength
    Y, X = centeredGrid(outputSize, dx=[1, 1], computeMeshgrid=computeMeshgrid)
    eps = 0.000000000000001  # Some small number to prevent division by 0 if input image == output size
    X = np.minimum((np.pi / (transitionLength[1] + eps)) * np.maximum(np.abs(X) - imageSize[1] / 2, 0), np.pi)
    Y = np.minimum((np.pi / (transitionLength[0] + eps)) * np.maximum(np.abs(Y) - imageSize[0] / 2, 0), np.pi)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    transitionMask = (0.25 * (1 + np.cos(X))) * (1 + np.cos(Y))

    # Superimpose constant pad-value and replicate-padded image with the transition mask
    imgPadded = transitionMask * imgPadded + (1 - transitionMask) * padValue
    return imgPadded


def shiftGrid(X, Y, shifts):
    """Shifting function for shiftRotateMagnifyImage function"""
    shifts = np.array(shifts, dtype=np.float32)
    Y = Y + shifts[0]
    X = X - shifts[1]
    return Y, X


def rotateGrid(X, Y, rotAngleDegree):
    """Rotating function for shiftRotateMagnifyImage function"""
    rotAngleDegree = np.array(rotAngleDegree, dtype=np.float32)
    YTemp = np.cos(np.radians(rotAngleDegree)) * Y - np.sin(np.radians(rotAngleDegree)) * X
    XTemp = np.sin(np.radians(rotAngleDegree)) * Y + np.cos(np.radians(rotAngleDegree)) * X
    Y = YTemp
    X = XTemp
    return Y, X


def magnifyGrid(X, Y, magnify):
    """Magnifying function for shiftRotateMagnifyImage function"""
    magnify = np.array([magnify], dtype=np.float32)
    magnify = (magnify[:] * np.ones((1, 2))).flatten()
    Y = magnify[0] * Y
    X = magnify[1] * X
    return Y, X


def shiftImage(img, shifts):
    """

    This is part of the main function shiftRotateMagnifyImage

    :param img: Input image array to shift vertically/horizontally
    :param shifts: Integer number to shift image. Input as [p, q]
    :returns Shifted image [N, M]
    """
    img = np.array(img, dtype=np.float32)
    imgTransformed = shiftRotateMagnifyImage(img, shifts=shifts)
    return imgTransformed


def magnifyImage(img, magnify):
    """

    This is part of the main function shiftRotateMagnifyImage

    :param img: Input image array to magnify
    :param magnify: Integer number to magnify image. Input as [p, q]
    :returns Magnified image [N, M]
    """
    img = np.array(img, dtype=np.float32)
    imgTransformed = shiftRotateMagnifyImage(img, magnify=magnify)
    return imgTransformed


def rotateImage(img, rotAngleDegree):
    """

    This is part of the main function shiftRotateMagnifyImage

    :param img: Input image array to rotate
    :param rotAngleDegree: Single integer number to specify degree of rotation for image
    :returns Rotated image [N, M]
    """
    img = np.array(img, dtype=np.float32)
    imgTransformed = shiftRotateMagnifyImage(img, rotAngleDegree=rotAngleDegree)
    return imgTransformed


def shiftRotateImage(img, shifts, rotAngleDegree):
    """

    This uses the main function shiftRotateMagnifyImage

    :param img: Input image array to rotate
    :param shifts: Integer number to shift image. Input as [p, q]
    :param rotAngleDegree: Single integer number to specify degree of rotation for image
    :returns Shifted and rotated image [N, M]
    """
    img = np.array(img, dtype=np.float32)
    imgTransformed = shiftRotateMagnifyImage(img, shifts=shifts, rotAngleDegree=rotAngleDegree)
    return imgTransformed


def shiftRotateMagnifyImage(img, magnify=None, rotAngleDegree=None, shifts=None, padMethod='replicate', order=5,
                            invertTransform=False):
    """Code converted/modified from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/imageProcessing/alignment/shiftRotateMagnifyImage

    Parameters
    ----------
    :param img : Numerical 2d-array image to be shifted and/or rotated
    :param magnify : float or 2-tuple of floats geometrical magnification of the image. If a tuple is assigned,
                     the values are interpreted as possibly different magnification applied along the different
                     dimensions.
    :param rotAngleDegree : float angle of the rotation to be applied to the image (in degrees)
    :param shifts : Tuple of floats shifts to be applied to the image (in units of pixel-lengths)
    :param padMethod : Default = 'replicate' -- Determines the type of the padding that is applied to the input image
                       prior to shifting and rotating.
    :param order : Default = 3 -- Interpolation-method used in the image transformation.
    :param invertTransform : Default = false -- Determines whether to invert the assigned geometrical transform before
                      application. This allows to magnify-rotate-shift back and forth without manually computing the
                      inverse transformation

    Returns
    -------
    :returns imgTransformed : numerical 2d-array the scaled and/or shifted and/or rotated image


    """

    if magnify is None:
        magnify = [1, 1]
    if rotAngleDegree is None:
        rotAngleDegree = 0
    if shifts is None:
        shifts = [0, 0]

    N = np.shape(img)
    X, Y = centeredGrid(N, dx=[1, 1], computeMeshgrid=True)
    if invertTransform:
        Y, X = magnifyGrid(X, Y, np.array(magnify, dtype=np.float32))
        Y, X = rotateGrid(X, Y, np.array(rotAngleDegree, dtype=np.float32))
        Y, X = shiftGrid(X, Y, np.array(shifts, dtype=np.float32))
    else:
        Y, X = shiftGrid(X, Y, np.array([-1 * s for s in shifts], dtype=np.float32))
        Y, X = rotateGrid(X, Y, -np.array(rotAngleDegree, dtype=np.float32))
        Y, X = magnifyGrid(X, Y, np.array([1 / mag for mag in magnify], dtype=np.float32))

    # Pad image according to the minimum/maximum values of the transformed coordinates
    # This is showing how to pad given different sizes for pre and post
    padPre = np.maximum(0, [np.ceil(-(N[0] - 1) / 2 - np.min(Y[:])), np.ceil(-(N[1] - 1) / 2 - np.min(X[:]))])
    padPost = np.maximum(0, [np.ceil(np.max(Y[:]) - (N[0] - 1) / 2), np.ceil(np.max(X[:]) - (N[1] - 1) / 2)])
    img = padToSize(img,
                    outputSize=[int(padPre[0] + N[0]), int(padPre[1] + N[1])],
                    padMethod=padMethod,
                    padType='pre',
                    padValue=None)
    img = padToSize(img,
                    outputSize=[int(padPre[0] + padPost[0] + N[0]), int(padPre[1] + padPost[1] + N[1])],
                    padMethod=padMethod,
                    padType='post',
                    padValue=None)

    # Perform image-transformation by interpolating the padded image on the transformed grid

    # flatten or ravel -- ravel doesn't occupy memory, but you have to be careful with manipulating array
    # map_coordinates -- use even array only for img
    imgTransformed = ndimage.map_coordinates(img,
                                             coordinates=[(Y + (0.5 * (N[0] - 1) + padPre[0])).ravel(),
                                                          (X + (0.5 * (N[1] - 1) + padPre[1])).ravel()],
                                             order=order,
                                             mode='nearest').reshape(N)  # Same as matlab interp2
    #imgTransformed = np.abs(imgTransformed)  # To remove random -0 values
    return imgTransformed


def fadeoutImageCosine(img, transitionLength=None, windowShift=None, fadeToVal=1):
    """Code converted from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/imageProcessing/cropPadWindow/fadeoutImage.m

    Main function is that calls this function is "fadeoutImage"


    Parameters
    -----------
    :param img : Numerical array image to fade out to a constant value
    :param transitionLength : Length of the transition to constant padding-value in pixels for the cosine-method.
                       Default --> transitionLength = np.ceil(np.mean(img.shape[])/8) is assigned

    :param windowShift : Amount of pixels [heigth, width] of which the fading window is shifted with respect to the
                  center of the input image. Default --> [0, 0]

    :param fadeToVal : Constant value to which the image is faded out. Default --> np.mean(img) is assigned

    Returns
    ----------
    :returns Faded image [N, M]
    """

    img = np.array(img)  # Convert image to array
    imageSize = np.shape(img)  # Size of the image

    # Transition length is taken as 1/8 of the image's aspect length if not assigned
    if not transitionLength:
        transitionLengthy = np.ceil(np.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLengthx = np.ceil(np.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLength = np.asarray([transitionLengthy, transitionLengthx])
    else:
        transitionLength = np.asarray([transitionLength[0], transitionLength[1]])

    if not windowShift:
        windowShift = np.array([0, 0])
    else:
        windowShift = np.array(windowShift)

    # Target value of the fadeout is taken as mean value of the image if not assigned
    if not fadeToVal:
        fadeToVal = img.mean()
        fadeToVal = fadeToVal.astype(np.float32)

    X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    # Shift center of the fading window
    Y = Y + windowShift[0]
    X = X - windowShift[1]

    eps = 0.000000000000001  # Some small number to prevent division by 0
    X = np.minimum(
        (np.pi / (transitionLength[1] + eps)) * np.maximum(np.abs(X) - (imageSize[1] / 2 - transitionLength[1]), 0),
        np.pi)
    Y = np.minimum(
        (np.pi / (transitionLength[0] + eps)) * np.maximum(np.abs(Y) - (imageSize[0] / 2 - transitionLength[0]), 0),
        np.pi)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    transitionMask = (0.25 * (1 + np.cos(X))) * (1 + np.cos(Y))

    # Superimpose constant and input images weighted with the transition mask.
    imgFaded = transitionMask * img + (fadeToVal * (1 - transitionMask)) * np.ones(imageSize)

    return imgFaded, transitionMask


def fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments, angularOffsetSegments,
                        fadeToVal, bottomApply):
    """ Code converted from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/imageProcessing/cropPadWindow/fadeoutImage.m

        Main function is that calls this function is "fadeoutImage"


        Parameters
        -----------
        :param img : Numerical array image to fade out to a constant value -- [N, M]

        :param fadeMethod : 'rectangle' or 'ellipse'

        :param ellipseSize : relative size of the ellipsoidal mask (half-axes along the different dimensions) used to
                              fade out the image in the ellipse-method. Default --> [0.8, 0.8]

        :param transitionLength : Length of the transition to constant padding-value in pixels for the cosine-method.
                           Default --> transitionLength = np.ceil(np.mean(img.shape[])/8) is assigned

        :param windowShift : Amount of pixels [heigth, width] of which the fading window is shifted with respect to the
                      center of the input image. Default = [0, 0]

        :param numSegments : Number of segments around the image

        :param angularOffsetSegments : Offset number for numSegments

        :param fadeToVal : Constant value to which the image is faded out. Default = np.mean(img) is assigned

        Returns
        ----------
        :returns Faded image [N, M]
        """

    img = np.array(img)
    imageSize = np.shape(img)

    # Transition length is taken as 1/8 of the image's aspect length if not assigned
    if not transitionLength:
        transitionLengthy = np.ceil(np.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLengthx = np.ceil(np.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLength = np.asarray([transitionLengthy, transitionLengthx])
    else:
        transitionLength = np.asarray([transitionLength[0], transitionLength[1]])

    if not windowShift:
        windowShift = np.array([0, 0])
    else:
        windowShift = np.array(windowShift)

    if not ellipseSize:
        ellipseSize = np.array([0.8, 0.8])
    else:
        ellipseSize = np.array(ellipseSize)

    if not numSegments:
        numSegments = 1

    if not angularOffsetSegments:
        angularOffsetSegments = 0

    # Initialize these so errors aren't thrown in if statements
    idxBoundary = None
    transitionMask = None

    if fadeMethod == 'ellipse':
        X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)

        # Shift center of the fading window
        Y = Y + windowShift[0]
        X = X - windowShift[1]

        # Radii in x and y in units of pixels
        ry = (ellipseSize[0] * imageSize[0] - 2) / 2
        rx = (ellipseSize[1] * imageSize[1] - 2) / 2

        # Indices of the image pixels lying within a centered ellipse
        idxEllipse = X ** 2 / rx ** 2 + Y ** 2 / ry ** 2 < 1

        # Extract pixels on the boundary of the ellipse to calculate mean values for fadeout
        rxInner = rx - transitionLength[1]
        ryInner = ry - transitionLength[0]

        idxBoundary = np.logical_xor(idxEllipse, (X ** 2 / rxInner ** 2 + Y ** 2 / ryInner ** 2 < 1))
        transitionMask = np.where(idxEllipse == True, 1, 0)

    if fadeMethod == 'rectangle':
        # Radii in x and y in units of pixels
        ry = int(np.ceil(ellipseSize[0] * imageSize[0]))
        rx = int(np.ceil(ellipseSize[1] * imageSize[1]))

        # Indices of the image pixels lying within a centered rectangle
        # idxRectangle = np.ones((ry, rx)).astype(int)
        idxRectangle = np.ones((ry, rx)).astype(int)
        idxRectangle = padToSize(idxRectangle, outputSize=imageSize, padMethod='constant', padType='both', padValue=0)
        idxRectangle = np.roll(idxRectangle, shift=windowShift*np.array([-1, 1]), axis=(0, 1))

        # extract pixels on the boundary of the rectangle to calculate mean values for fadeout
        rxInner = int(rx - transitionLength[1])
        ryInner = int(ry - transitionLength[0])
        idxInnerRectangle = np.ones((ryInner, rxInner)).astype(int)
        idxInnerRectangle = padToSize(idxInnerRectangle, outputSize=imageSize, padMethod='constant', padType='both',
                                      padValue=0)
        idxInnerRectangle = np.roll(idxInnerRectangle, shift=windowShift*np.array([-1, 1]), axis=(0, 1))

        idxBoundary = np.logical_xor(idxRectangle, idxInnerRectangle)
        transitionMask = np.where(idxRectangle == True, 1, 0)

    # Determine values in which image is faded out to

    # Case 0: Constant fadeout-value has been assigned
    if fadeToVal or fadeToVal == 0:
        fadeToVals = np.float32(fadeToVal)

    # Case 1: Compute fadeout-values in multiple angular segments
    elif numSegments > 1:
        X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)

        # Shift center of the fading window
        Y = Y - windowShift[0]
        X = X - windowShift[1]

        def cart2pol(x, y):
            """Converts cartesian coordinates to polar"""
            rho_ = np.sqrt(x ** 2 + y ** 2)
            theta_ = np.arctan2(y, x)
            return rho_, theta_

        rho, theta = cart2pol(X, Y)
        theta = np.mod(theta + angularOffsetSegments, 2 * np.pi)
        # This splits the whole image into slices like a pie for however many segments are chosen
        segmentIndices = np.minimum((np.floor((numSegments / (2 * np.pi)) * theta) + 1).astype(int), numSegments)
        imFilt = gaussian(img,
                          sigma=10 / 2.35,
                          truncate=2)  # Same as Matlab imgaussfilt, 2.35 comes from 2 * np.sqrt(2 * np.log(2))
        # Compute target values for the fade out in each angular segment
        fadeToVals = np.zeros_like(img)



        ################## NEW :  MIRRORS THE FADEOUT ACROSS THE Y AXIS SO IT IS SYMMETRIC ##################
        # Ensure left-right symmetry by finding correct mirrored segments across the Y-axis
        mirroredTheta = np.pi - theta  # Correct mirroring across Y-axis
        mirroredTheta = np.mod(mirroredTheta, 2 * np.pi)  # Ensure valid range [0, 2π]
        segmentFadeValues = np.zeros(numSegments)  # Store fade values for each segment
        mirroredSegmentIndices = (np.floor((numSegments / (2 * np.pi)) * mirroredTheta) + 1).astype(int)

        # Compute fade-out values ensuring left-right symmetry
        for segment in range(1, numSegments + 1):
            # Find pixels in this segment and its mirrored counterpart
            inSegment = np.where(segmentIndices == segment, True, False)
            inMirroredSegment = np.where(mirroredSegmentIndices == segment, True, False)  # Correct Y-axis mirroring

            # Compute a shared fade-out value for the mirrored segments
            meanFadeValue = np.mean(imFilt[idxBoundary & (inSegment | inMirroredSegment)])

            # Assign the same value to both mirrored segments
            segmentFadeValues[segment - 1] = meanFadeValue

        # Assign fade-out values to fadeToVals with enforced left-right symmetry
        for segment in range(1, numSegments + 1):
            inSegment = np.where(segmentIndices == segment, True, False)
            fadeToVals[inSegment] = segmentFadeValues[segment - 1]
         ################################################################### NEW ########################################


        # # ORIGINAL - ABOVE IS NEW - Comment below out if above is used
        # for segment in range(1, numSegments + 1):
        #     inSegment = np.where(segment == segmentIndices, True, False)
        #     fadeToVals[inSegment] = np.mean(imFilt[idxBoundary & inSegment])

        # Smear out the transition between the individual segments
        fadeToVals = gaussian(fadeToVals, sigma=(transitionLength[0] / 2.35, transitionLength[1] / 2.35),
                              truncate=2)  # Same as Matlab imgaussfilt, , 2.35 comes from 2 * np.sqrt(2 * np.log(2))

    # Case 2: Compute one fadeout-value globally
    else:
        fadeToVals = np.mean(img[idxBoundary])  # Only 1 segment

    if transitionLength[0] > 1 and transitionLength[1] > 1:
        transitionMask = np.float32(transitionMask)
        transitionMask = gaussian(transitionMask, sigma=(transitionLength[0] / 2.35, transitionLength[1] / 2.35),
                                  truncate=2)  # Same as Matlab imgaussfilt


    if bottomApply:
        # Create a mask for the bottom half of the image with smooth transition
        # transition_region_height = transitionLength[0].item() if isinstance(transitionLength[0], torch.Tensor) else \
        # transitionLength[0]
        transition_region_height = 2
        bottom_half_mask = np.zeros_like(img)
        bottom_half_mask[int((img.shape[0]) / 2) + transition_region_height:, :] = 1
        transition_region = np.linspace(0, 1, transition_region_height)
        for i in range(transition_region_height):
            bottom_half_mask[int((img.shape[0]) / 2) + i, :] = transition_region[i]
        imgFaded = img * (1 - bottom_half_mask) + (img * transitionMask + fadeToVals * (1 - transitionMask)) * bottom_half_mask
    else:
        # Convex combination of transition masks
        # Apply fade-out effect using fadeToVals as target
        imgFaded = img * transitionMask + fadeToVals * (1 - transitionMask)  # original

    # Convex combination of transition masks
    # imgFaded = img * transitionMask + fadeToVals * (1 - transitionMask)
    imgFaded[np.isnan(imgFaded)] = 1
    imgFaded[np.isinf(imgFaded)] = 1

    return imgFaded, transitionMask


def fadeoutImage(img, fadeMethod=None, fadeToVal=None, transitionLength=None, ellipseSize=None, numSegments=None,
                 angularOffsetSegments=None, windowShift=None, bottomApply=False):
    """Base function with base parameters for fadeoutImageCosine or fadeoutImageEllipse functions"""
    if fadeMethod is None:
        fadeMethod = 'cosine'
    if fadeToVal is None:
        fadeToVal = []
    if transitionLength is None:
        transitionLength = []
    if ellipseSize is None:
        ellipseSize = [0.8, 0.8]
    if numSegments is None:
        numSegments = 1
    if angularOffsetSegments is None:
        angularOffsetSegments = 0
    if windowShift is None:
        windowShift = [0, 0]
    if bottomApply is None:
        bottomApply = False

    if fadeMethod == 'cosine':
        imgFaded, window = fadeoutImageCosine(img, transitionLength, windowShift, fadeToVal)
        return imgFaded, window
    if fadeMethod == 'ellipse':
        imgFaded, window = fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments,
                                               angularOffsetSegments, fadeToVal, bottomApply)
        return imgFaded, window
    if fadeMethod == 'rectangle':
        imgFaded, window = fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments,
                                               angularOffsetSegments, fadeToVal, bottomApply)
        return imgFaded, window
    else:
        return TypeError('Invalid value for fadeMethod. Choices are cosine, ellipse, or rectangle.')


def angularAverage(im):
    """
    This function computes the average of a 2D- or 3D-image over concentric circles or concentric spherical shells
    around the image center. To get proper x coordinates to plot (radii), you need to multiply the result by
    dx (pixel size) and divide by the scaling you want (for example, um -- for microns)

    :param im: Input image. Assumes that the image is already centered in the window
    :return: radii: The radii of the concentric circles or spherical shells over which the averages were computed
             averages: Computed angular averages
    """

    # Image dimensions
    N = np.array(im.shape)
    ndim = np.ndim(N)

    # Image center
    center = np.floor(N / 2).astype(int) + 1

    # Compute distance of pixels to center
    x = 0
    y = 0
    for jj in range(ndim):
        x += (np.arange(1, N[jj] + 1) - center[jj]).flatten().reshape(N[1], 1)
        y += (np.arange(1, N[jj] + 1) - center[jj]).flatten().reshape(1, N[0])
    r = np.sqrt(x ** 2 + y ** 2)

    # Each point is contained in between two concentric circles or concentric spherical shells:
    # Compute index of these circles/shells and the distance from the larger of the two shells.
    lower_shell_idx = np.round(r + 0.5 - 1e-10).astype(int)
    upper_shell_idx = lower_shell_idx + 1
    dist_to_upper_shell = lower_shell_idx - r
    lower_shell_idx = np.maximum(lower_shell_idx, 1)

    # Number of pixels corresponding to the different radial circles or shells
    n_shells = np.bincount(upper_shell_idx.ravel(), weights=(1 - dist_to_upper_shell).flatten())
    n_shells += np.bincount(lower_shell_idx.ravel(), weights=dist_to_upper_shell.ravel(), minlength=len(n_shells))
    n_shells = n_shells[1:]

    # Compute average values over all circles or shells
    averages = (np.bincount(lower_shell_idx.ravel(), (dist_to_upper_shell * im).ravel(),
                            minlength=len(n_shells) + 1)[1:] +
                np.bincount(upper_shell_idx.ravel(), ((1 - dist_to_upper_shell) * im).ravel(),
                            minlength=len(n_shells))[1:]) / n_shells

    # Restrict to values on shells that are fully contained within the image
    num_shells = int(np.ceil(np.min(N) / 2))
    averages = averages[:num_shells]

    # Corresponding radial coordinates
    radii = np.arange(num_shells)

    return radii, averages


def smooth2D_3x3(img):
    """2D smoothing of an image --> 3x3 kernel"""
    Ny, Nx = np.shape(img)
    f = np.zeros((Ny, Nx))

    f[Ny//2, Nx//2] = 4
    f[Ny//2 - 1, Nx//2] = 1
    f[Ny//2 + 1, Nx//2] = 1
    f[Ny//2 - 1, Nx//2 - 1] = 1
    f[Ny//2, Nx//2 - 1] = 1
    f[Ny//2 + 1, Nx//2 - 1] = 1
    f[Ny//2 - 1, Nx//2 + 1] = 1
    f[Ny//2, Nx//2 + 1] = 1
    f[Ny//2 + 1, Nx//2 + 1] = 1

    # This is convolution, but in k-space it is multiplication. 1/12 factor normalizes the object to maximum 1 since
    # it all adds up to 12

    out = 1/12 * np.abs(IFFT(FFT(img) * FFT(f)))
    return out


def smooth2D_5x5(img):
    """ 2D smoothing of an image --> 5x5 kernel """

    Ny, Nx = np.shape(img)
    f = np.zeros((Ny, Nx), dtype=complex)

    f[Ny // 2, Nx // 2] = 4
    f[Ny // 2, Nx // 2 - 1] = 1
    f[Ny // 2, Nx // 2 - 2] = 1
    f[Ny // 2, Nx // 2 + 1] = 1
    f[Ny // 2, Nx // 2 + 2] = 1

    f[Ny // 2 - 1, Nx // 2] = 1
    f[Ny // 2 - 1, Nx // 2 - 1] = 1
    f[Ny // 2 - 1, Nx // 2 - 2] = 1
    f[Ny // 2 - 1, Nx // 2 + 1] = 1
    f[Ny // 2 - 1, Nx // 2 + 2] = 1

    f[Ny // 2 - 2, Nx // 2] = 1
    f[Ny // 2 - 2, Nx // 2 - 1] = 1
    f[Ny // 2 - 2, Nx // 2 - 2] = 1
    f[Ny // 2 - 2, Nx // 2 + 1] = 1
    f[Ny // 2 - 2, Nx // 2 + 2] = 1

    f[Ny // 2 + 1, Nx // 2] = 1
    f[Ny // 2 + 1, Nx // 2 - 1] = 1
    f[Ny // 2 + 1, Nx // 2 - 2] = 1
    f[Ny // 2 + 1, Nx // 2 + 1] = 1
    f[Ny // 2 + 1, Nx // 2 + 2] = 1

    f[Ny // 2 + 2, Nx // 2] = 1
    f[Ny // 2 + 2, Nx // 2 - 1] = 1
    f[Ny // 2 + 2, Nx // 2 - 2] = 1
    f[Ny // 2 + 2, Nx // 2 + 1] = 1
    f[Ny // 2 + 2, Nx // 2 + 2] = 1

    # This is convolution, but in k-space it is multiplication. 1/28 factor normalizes the object to maximum 1
    out = 1 / 28 * np.abs(IFFT(FFT(img) * FFT(f)))
    return out


def binArray(data, axis, binStep, binSize, func=np.nanmean):
    """

    From: https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730

    axis: Axis you want to bin
    binStep: Number of points between each bin (Allows for overlapping bins--keep it the same as binSize
             to prevent overlap)
    binSize: Size of each bin
    func: The function you want to apply to the bin
          (np.max for maxpooling, np.mean for an averaging pixels, np.sum for summing pixels, etc...)

    Usages for 2D: newImg = binArray(img, axis=0, binStep=binStep, binSize=binSize, func=np.mean)
                   NewImg = binArray(newImg, axis=1, binStep=binStep, binSize=binSize, func=np.mean)
    """

    data = np.array(data)  # Converts data to array
    dims = np.array(data.shape)  # shape of the data along both dimensions
    argDims = np.arange(data.ndim)  # gives range given the length of the dimensions of the data
    argDims[0], argDims[axis] = argDims[axis], argDims[0]
    data = data.transpose(argDims)  # argDims=[0, 1] keeps it the same, argDims = [1, 0] transposes it
    data = [func(np.take(data, np.arange(int(i * binStep), int(i * binStep + binSize)), 0), 0)
            for i in np.arange(dims[axis] // binStep)]
    data = np.array(data).transpose(argDims)
    return data


def rescaleImgToCustomCoord(img, imageSize, extentx, extenty, Nx, Ny, padVal):
    """

    img : imported image or array
    imageSize : Desired physical size of the object
    extentx : Physical length of the grid size in x
    extenty : Physical length of the grid size in y
    Nx : Number of pixels in x
    Ny : Number of pixels in y

    returns
    --------
    result : Image given a physical size based off created meshgrid
    """

    img = np.array(img, dtype=np.float32)
    numImgPixelsWidth, numImgPixelsHeight = img.shape
    print('Original image height and width in pixels:', (numImgPixelsHeight, numImgPixelsWidth))
    if imageSize is not None:
        # Below is equivalent to Nobj = a/extent * N, which is equivalent to Nobj = a/ps (or ps = a/Nobj)
        # Nobj is the number of object pixels in x and y, N is total number of pixels in x and y,
        # ps is object pixel size, extent is the physical grid size in x and y, and a is the physical object size
        newNumImgPixelsHeight, newNumImgPixelsWidth = int(np.round(imageSize / extenty * Ny)), \
                                                      int(np.round(imageSize / extentx * Nx))
        print('New height and width in pixels:', (newNumImgPixelsHeight, newNumImgPixelsWidth))
    else:
        # By default, the image fills the entire aperture plane
        newNumImgPixelsHeight, newNumImgPixelsWidth = Ny, Nx
        print('Height and width in pixels remains the same. The image will fill the entire aperture plane')

    newShape = (newNumImgPixelsWidth, newNumImgPixelsHeight)  # cv2 takes (width, height) so it must be in this order
    # img = cv2.resize(img, dsize=newShape, interpolation=cv2.INTER_AREA)  # Use if shrinking the image
    img = cv2.resize(img, dsize=newShape, interpolation=cv2.INTER_CUBIC)  # Use if enlarging the image
    result = padToSize(img, outputSize=[Ny, Nx], padMethod='constant', padType='both', padValue=padVal)
    return result


def signalToNoise(img, axis=0, ddof=0):
    """Calculates signal to noise ratio for a given input image

    Parameters
    -----------
    :param img : Input image to find signal to noise ratio [N, M]
    :param axis: Axis on which to calculate the signal to noise ratio -- takes either 0 or 1
    :param ddof: delta degrees of freedom -- keep this 0

    Returns
    -----------
    :returns Ratio of the mean to the standard deviation

    """
    img = np.asanyarray(img)
    mu = img.mean(axis)
    sd = img.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, mu / sd)


def Gaussian(Ny, Nx, A, sigma):
    """
    Function of a 2D gaussian -- see Wikipedia, apodization function for sampled signal.
    The 1D signal can be expressed as exp(-x**2/(2*sigma**2))

    :param Ny: Number of pixels in y
    :param Nx: Number of pixels in x
    :param A: Amplitude of the gaussian
    :param sigma: Standard deviation of the gaussian
    :return: Normalized Gaussian
    """
    gau = np.zeros((Ny, Nx))
    for i in range(0, Ny):
        for j in range(0, Nx):
            xx = i - Ny // 2
            yy = j - Nx // 2
            gau[i, j] = A * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    # Normalize Gaussian
    gau /= np.sum(np.sum(gau))
    return gau


def GaussianBeam(dim_x, dim_y, dx, dy, lam, prop_distance, beam_waist):
    """

    dim_x : Dimension of the object in x
    dim_y : Dimension of the object in y
    dx : Pixel size of the object in x. Setting this to 1 puts the beam size in pixels instead of an actual physical
         size
    dy : Pixel size of the object in y. Setting this to 1 puts the beam size in pixels instead of an actual physical
         size
    lam : Wavelength of the illuminating light
    prop_distance : Propagation distance
    beam_waist :  1/e-decay half-width of the field amplitude at prop_distance = 0

    returns
    --------
    result : Numerical array. Propagated wave field at distance, prop_distance

    """
    X, Y = centeredGrid([dim_y, dim_x], dx=[dy, dx], computeMeshgrid=True)
    k = 2*np.pi/lam
    rayleigh_length = np.pi * beam_waist**2 / lam

    beam_width = beam_waist * np.sqrt(1 + (prop_distance/rayleigh_length) ** 2)
    radius_curvature = prop_distance + rayleigh_length ** 2 / prop_distance
    phi = np.arctan2(prop_distance, rayleigh_length)

    if prop_distance == 0:
        psi = np.exp(-(X ** 2 + Y ** 2) / beam_waist ** 2)
    else:
        psi = beam_waist * np.exp(1j * k * prop_distance - 1j * phi) / beam_width * \
              np.exp(-(X ** 2 + Y ** 2) / beam_width ** 2 + 1j * k * (X ** 2 + Y ** 2) / (2 * radius_curvature))

    return psi


def showImg(img, title=None, xlabel=None, ylabel=None, cmap='Greys_r', clim=(None, None), darkBackground=False,
            drawLineouts=False, saveImg=False):
    """Display a single image

    Parameters
    -----------
    :param img : Input array [N, M]
    :param title : Image title
    :param xlabel : x axis label
    :param ylabel : y axis label
    :param cmap : Image colormap
    :param clim : Color limits of image
    :param darkBackground : Boolean -- Determines whether dark background or white background is used
    :param drawLineouts : True or False -- determines whether line outs are drawn on the figure
    :param saveImg : True or False -- determines whether a figure is saved or not

    Returns
    ----------
    :returns Plot of image

    """
    Ny, Nx = np.shape(img)
    if darkBackground:
        plt.style.use('dark_background')
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    if title is None:
        title = 'Image'
    fig1.suptitle(title)
    divider = make_axes_locatable(ax1)
    top_ax = divider.append_axes("top", 1.0, pad=0.2, sharex=ax1)  # Create top lineout
    right_ax = divider.append_axes("right", 1.0, pad=0.2, sharey=ax1)  # Create right lineout
    top_ax.xaxis.set_tick_params(labelbottom=False)  # Remove ticks from top lineout
    right_ax.yaxis.set_tick_params(labelleft=False)  # Remove ticks from right lineout
    if xlabel is None:
        xlabel = 'x'
    if ylabel is None:
        ylabel = 'y'
    ax1.set_xlabel(xlabel)  # Set x label
    ax1.set_ylabel(ylabel)  # Set y label
    top_ax.set_ylabel(title)
    right_ax.set_xlabel(title)
    img1 = ax1.imshow(img, cmap=cmap, clim=clim)
    cx = Nx // 2
    cy = Ny // 2
    if drawLineouts:
        cy, cx = np.round(ndimage.measurements.center_of_mass(img)).astype(int)
        print('Center of mass in y and x are:', cy, cx)
        ax1.axvline(x=cx, color='r')  # Plot vertical line out
        ax1.axhline(y=cy, color='g')  # Plot horizontal line out
    xvals = np.linspace(0, (Nx - 1), Nx)  # vals of 0, 1, 2, ... Nx - 1 -- total Nx values
    yvals = np.linspace(0, (Ny - 1), Ny)  # vals of 0, 1, 2, ... Ny - 1 -- total Ny values
    right_ax.plot(img[:, cx], yvals, 'red', lw=1)
    top_ax.plot(xvals, img[cy, :], 'green', lw=1)
    cax1 = divider.append_axes("left", size="5%", pad="20%")  # Location of colorbar
    fig1.colorbar(img1, cax=cax1, orientation='vertical')  # Add colorbar and change its orientation
    cax1.yaxis.set_ticks_position("left")  # Change ticks location on colorbar
    if saveImg:
        plt.savefig('/Users/danielhodge/Desktop/tempPlot', bbox_inches='tight', transparent=True)
    plt.show()

def compare_plots_lineout(img1, img2):
    """
    :param img1: Ground truth image
    :param img2: Calculated image
    :return: Lineout plots comparing two images across the center
    """
    Ny, Nx = img1.shape

    cx = Nx // 2
    cy = Ny // 2

    xvals = np.linspace(0, (Nx - 1), Nx)  # vals of 0, 1, 2, ... Nx - 1 -- total Nx values
    yvals = np.linspace(0, (Ny - 1), Ny)  # vals of 0, 1, 2, ... Ny - 1 -- total Ny values

    plt.figure()
    plt.title('Horizontal Lineout')
    plt.plot(yvals, img1[:, cx], 'black', lw=1)
    plt.plot(yvals, img2[:, cx], 'red', lw=1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title('Vertical Lineout')
    plt.plot(xvals, img1[cy, :], 'black', lw=1)
    plt.plot(xvals, img2[cy, :], 'red', lw=1)
    plt.grid()
    plt.show()

    return


def figureUpdate(img, fig, ax, title):
    """Update and re-render the figure with the given image and title"""
    fig.canvas.draw_idle()  # The next time you repaint the GUI window, re-render the figure first
    fig.canvas.start_event_loop(0.000001)
    # plt.gcf().canvas.draw_idle()  # The next time you repaint the GUI window, re-render the figure first
    # plt.gcf().canvas.start_event_loop(0.1)
    ax.imshow(img)
    ax.set_title(title)


def plotUpdate(data, fig, ax, title):
    """Update and re-render the plot with the given data and title"""
    fig.canvas.draw_idle()  # The next time you repaint the GUI window, re-render the figure first
    fig.canvas.start_event_loop(0.0001)
    # plt.gcf().canvas.draw_idle()  # The next time you repaint the GUI window, re-render the figure first
    # plt.gcf().canvas.start_event_loop(0.1)
    ax.plot(data)
    ax.set_title(title)


class MoviePlotter:
    """
    View a stack x (M, Ny, Nx) with slider + play/pause + prev/next.

    ROI controls (no zoom_center):
      - roi_px=(W, H): show a centered W×H window (always centered on the image)
      - roi_bounds=(x0, x1, y0, y1): explicit pixel bounds (x=cols, y=rows)
    """
    def __init__(
        self,
        x,
        cmap='gray',
        pause_ms=250,
        show_colorbar=False,
        roi_px=None,          # (W, H) centered
        roi_bounds=None       # (x0, x1, y0, y1)
    ):
        self.x = np.asarray(x)
        assert self.x.ndim == 3, "x must be (M, Ny, Nx)"
        self.M, self.Ny, self.Nx = self.x.shape

        self.cmap = cmap
        self.show_colorbar = show_colorbar
        self.pause_ms = int(pause_ms)
        self.playing = False
        self.idx = 0

        # ROI settings
        self.roi_px = None if roi_px is None else (int(roi_px[0]), int(roi_px[1]))
        self.roi_bounds = None if roi_bounds is None else tuple(map(int, roi_bounds))

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        plt.subplots_adjust(bottom=0.16)

        frame0 = self.x[0]
        self.im = self.ax.imshow(
            frame0, cmap=self.cmap,
            vmin=np.min(frame0), vmax=np.max(frame0),
            interpolation='nearest'
        )
        self.ax.set_axis_off()
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04) if show_colorbar else None

        # Slider
        ax_slider = self.fig.add_axes([0.12, 0.07, 0.76, 0.035])
        self.slider = Slider(ax_slider, "Frame", 0, self.M - 1, valinit=0, valstep=1, valfmt='%d')
        self.slider.on_changed(self._on_slider)

        # Buttons (ASCII labels to avoid glyph issues)
        ax_prev  = self.fig.add_axes([0.12, 0.015, 0.12, 0.04])
        ax_play  = self.fig.add_axes([0.27, 0.015, 0.12, 0.04])
        ax_next  = self.fig.add_axes([0.42, 0.015, 0.12, 0.04])
        ax_reset = self.fig.add_axes([0.57, 0.015, 0.12, 0.04])

        self.btn_prev  = Button(ax_prev,  "Prev")
        self.btn_play  = Button(ax_play,  "Play")
        self.btn_next  = Button(ax_next,  "Next")
        self.btn_reset = Button(ax_reset, "Reset")

        self.btn_prev.on_clicked(lambda e: self._step(-1))
        self.btn_next.on_clicked(lambda e: self._step(+1))
        self.btn_reset.on_clicked(lambda e: self._reset())
        self.btn_play.on_clicked(lambda e: self._toggle_play())

        # Timer for playback
        self.timer = self.fig.canvas.new_timer(interval=self.pause_ms)
        self.timer.add_callback(self._on_timer)

        # Title + initial view
        self.ax.set_title(f"Frame 0 / {self.M-1}")
        self._apply_view()

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()

    # ---------- Public ROI helpers ----------
    def set_roi_size(self, width, height):
        """Centered ROI size in pixels; overrides roi_bounds."""
        self.roi_px = (int(width), int(height))
        self.roi_bounds = None
        self._apply_view(); self.fig.canvas.draw_idle()

    def set_roi_bounds(self, x0, x1, y0, y1):
        """Explicit bounds (x0, x1, y0, y1) in pixels; overrides roi_px."""
        self.roi_bounds = (int(x0), int(x1), int(y0), int(y1))
        self.roi_px = None
        self._apply_view(); self.fig.canvas.draw_idle()

    def clear_roi(self):
        """Show full image."""
        self.roi_px = None
        self.roi_bounds = None
        self._apply_view(); self.fig.canvas.draw_idle()

    # ---------- Handlers ----------
    def _on_slider(self, val):
        self._update_frame(int(val))

    def _step(self, delta):
        self.idx = (self.idx + delta) % self.M
        self.slider.set_val(self.idx)

    def _reset(self):
        self.slider.set_val(0)

    def _toggle_play(self):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _on_timer(self):
        if self.playing:
            self._step(+1)

    def _on_key(self, event):
        if event.key == 'right':
            self._step(+1)
        elif event.key == 'left':
            self._step(-1)
        elif event.key == ' ':
            self._toggle_play()
        elif event.key in ('q', 'Q', 'escape'):
            self.playing = False
            self.timer.stop()
            plt.close(self.fig)

    # ---------- Rendering ----------
    def _update_frame(self, idx):
        self.idx = idx
        frame = self.x[idx]
        self.im.set_data(frame)
        self.im.set_clim(np.min(frame), np.max(frame))  # raw per-frame scaling
        if self.cbar is not None:
            self.cbar.update_normal(self.im)
        self.ax.set_title(f"Frame {idx} / {self.M-1}")
        self._apply_view()
        self.fig.canvas.draw_idle()

    def _apply_view(self):
        """
        View priority:
          1) roi_bounds (x0, x1, y0, y1)
          2) roi_px (W, H) centered on image center
          3) full image
        """
        if self.roi_bounds is not None:
            x0, x1, y0, y1 = self._clip_bounds(*self.roi_bounds)
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y1, y0)  # invert y for image coords
            return

        if self.roi_px is not None:
            W, H = self.roi_px
            cx = self.Nx // 2
            cy = self.Ny // 2
            x0 = cx - W // 2
            x1 = x0 + W
            y0 = cy - H // 2
            y1 = y0 + H
            x0, x1, y0, y1 = self._clip_bounds(x0, x1, y0, y1)
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y1, y0)
            return

        # Full image
        self.ax.set_xlim(-0.5, self.Nx - 0.5)
        self.ax.set_ylim(self.Ny - 0.5, -0.5)

    def _clip_bounds(self, x0, x1, y0, y1):
        """Clip bounds to image extents and enforce at least 1 pixel width/height."""
        x0 = int(np.clip(x0, -0.5, self.Nx - 0.5))
        x1 = int(np.clip(x1, -0.5, self.Nx - 0.5))
        y0 = int(np.clip(y0, -0.5, self.Ny - 0.5))
        y1 = int(np.clip(y1, -0.5, self.Ny - 0.5))
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        return x0, x1, y0, y1



# Old
# def phaseUnwrapping(img):
#     """Phase unwrapping function"""
#     Ny, Nx = np.shape(img)
#     f = np.zeros((Ny, Nx))
#     for ii in range(Ny):
#         for jj in range(Nx):
#             y = ii - Ny/2
#             x = jj - Nx/2
#             f[ii, jj] = x**2 + y**2
#     eps = 0.000001
#     a = IFFT(FFT(np.cos(img) * IFFT(FFT(np.sin(img)) * f)) / (f + eps))
#     b = IFFT(FFT(np.sin(img) * IFFT(FFT(np.cos(img)) * f)) / (f + eps))
#     outPhase = np.real(a - b)
#     return outPhase

def phaseUnwrapping(img):
    """Phase unwrapping function"""
    Ny, Nx = np.shape(img)
    y, x = np.mgrid[:Ny, :Nx]
    f = (x - Nx/2)**2 + (y - Ny/2)**2
    eps = 0.000001
    a = IFFT(FFT(np.cos(img) * IFFT(FFT(np.sin(img)) * f)) / (f + eps))
    b = IFFT(FFT(np.sin(img) * IFFT(FFT(np.cos(img)) * f)) / (f + eps))
    outPhase = np.real(a - b)
    return outPhase


def centerOfMass(img):
    """Finds the center of mass of an image"""
    img = np.array(img)
    N = np.shape(img)
    ndim = len(N)
    centOfMass = np.zeros((1, 2)).flatten()
    for iDim in reversed(range(ndim)):
        notDims = np.argwhere(np.array(range(ndim)) != iDim).flatten()
        imgSummed = img
        for notDim in notDims:
            imgSummed = np.expand_dims(np.sum(imgSummed, axis=notDim), axis=1)
        centOfMass[iDim] = np.array(np.matmul(np.expand_dims(np.arange(0, N[iDim]), axis=0), imgSummed)).flatten() \
                           / np.sum(imgSummed)
        centOfMass = np.round(centOfMass).astype(int)
    return centOfMass

def FWHM_2D(X, Y):
    """FWHM for 1D array"""
    half_max = np.max(Y)/2
    d = Y - half_max

    # This is 1/e^2 value not FWHM below, but called it the same name to switch fast
    # e = 2.7182818
    # half_max = np.max(Y) * 1/e**2  # For 1/e^2 fall off
    # d = Y - half_max

    indices = np.where(d > 0)[0]  # [0] Allows access to the first array, which we want
    return [np.abs(X[indices[-1]] - X[indices[0]]), half_max]

def calculate_fwhm(array_2d):
    """FWHM for 2D array"""
    # Determine the maximum value and its index
    max_value = np.max(array_2d)

    # Determine the half-maximum value
    half_max = max_value / 2

    # e = 2.7182818
    # #Determine the 1//e^2 value
    # half_max = max_value * 1 / e ** 2  # For 1/e^2 fall off

    # Find the indices where the array is greater than or equal to the half-maximum value
    indices = np.where(array_2d >= half_max)

    # Calculate the FWHM in each dimension
    fwhm_x = np.max(indices[1]) - np.min(indices[1])
    fwhm_y = np.max(indices[0]) - np.min(indices[0])

    return fwhm_x, fwhm_y

def SNR(input, axis=0, ddof=0):
    input = np.asanyarray(input)
    m = input.mean(axis)
    sd = input.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def estimate_noise(I):
    """Estimates the variance of additive zero mean Gaussian noise in an image
    Reference: "Fast Noise Variance Estimation" by JOHN IMMERKÆR

    The standard deviation (Eq. 3) is the square root of the variance (Eq. 2)

    :param: I: Image with Gaussian noise
    :returns: Scalar value of Gaussian noise contribution in an image

    """
    H, W = I.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))  # Eq. 3 in the paper

    return sigma

def voigt_2d(x, y, sigma_g, gamma_l):
    """
    Generate a 2D Voigt profile. A 2D Voigt function is a convolution of a 2D Gaussian function with a 2D Lorentzian
    function.

    Parameters:
    - x, y: 2D arrays of x and y coordinates.
    - sigma_g: Standard deviation of the Gaussian component.
    - gamma_l: Half-width at half-maximum (HWHM) of the Lorentzian component.

    Returns:
    - 2D array of the Voigt profile values.
    """

    r = np.sqrt(x ** 2 + y ** 2)  # Calculate the radius from the center
    r_flat = r.ravel()  # Convert radius to a 1D array for voigt_profile function
    voigt_1d = voigt_profile(r_flat, sigma_g, gamma_l)  # Calculate Voigt profile in 1D
    voigt_2d = voigt_1d.reshape(x.shape)
    voigt_normalized = voigt_2d / np.sum(np.sum(voigt_2d))

    return voigt_normalized


def PropagatorS(Nx, Ny, dx, dy, lam, zeff):
    # Recall k = 2*pi*f where f is a SPATIAL frequency
    k0 = 2 * np.pi / lam  # Wavevector
    dkx = 2 * np.pi / (Nx * dx)  # Pixel size (x-dimension) in k-space (spatial frequency)
    dky = 2 * np.pi / (Ny * dy)  # Pixel size (y-dimension) in k-space (spatial frequency)
    KX = np.arange(-Nx / 2, Nx / 2, 1) * dkx
    KY = np.arange(-Ny / 2, Ny / 2, 1) * dky
    kx, ky = np.meshgrid(KX, KY)
    arg = k0 ** 2 - kx ** 2 - ky ** 2
    arg[arg < 0] = 0
    kz = np.sqrt(np.abs(arg))
    p = np.exp(1j * zeff * kz)  # This is for forward propagation
    return p, kx, ky

def plot_longitudinal_profile_intensity(longitudinal_profile_E, X, dx, start_distance, end_distance, grid=False, xlim=None, ylim=None):
    """visualize the diffraction pattern longitudinal profile intensity with matplotlib"""

    plt.style.use("dark_background")

    I_long = np.abs(longitudinal_profile_E)**2
    I_long = I_long.transpose(1, 0)

    fig = plt.figure(figsize=(16 / 9 * 6, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    if xlim != None:
        ax1.set_xlim(np.array(xlim))

    if ylim != None:
        ax1.set_ylim(np.array(ylim))

    ax1.set_ylabel("Size of SiO2 Shell [m]")
    ax1.set_xlabel('Screen Distance [m]')
    ax1.set_title("Longitudinal Profile")
    if grid == True:
        ax1.grid(alpha=0.2)

    im = ax1.imshow(I_long, cmap='inferno', extent=[start_distance, end_distance, float(X[0]), float(X[-1] + dx)],
                    interpolation='spline36', aspect='auto')
    plt.savefig('/Users/danielhodge/Desktop/LongitudinalPlot.png')
    plt.show()

def get_longitudinal_profile(E, Ny, Nx, dx, dy, lam, start_distance, end_distance, steps):
    """
    Propagates the field at n steps equally spaced between start_distance and end_distance, and returns the field over
    the xz plane
    """

    zrange = np.linspace(start_distance, end_distance, steps)
    E0 = E.copy()
    longitudinal_profile_E = np.zeros((steps, Nx), dtype=complex)
    t0 = time.time()

    bar = progressbar.ProgressBar()
    for i in bar(range(steps)):
        prop, _, _ = PropagatorS(Nx=Nx, Ny=Ny, dx=dx, dy=dy, lam=lam, zeff=zrange[i])
        E = IFFT(FFT(E0) * prop)
        longitudinal_profile_E[i, :] = E[Ny // 2, :]

    print("Time it took in seconds: ", time.time() - t0)

    return longitudinal_profile_E


def create_circular_mask(size, percentage, smooth_pixels):
    """
    Create a circular mask of a certain percentage of the grid size with smoothed edges.

    size: int
        The length of a side of the square grid.
    percentage: float
        The percentage of the grid size that the circle should occupy (0 to 100).
    smooth_pixels: int
        The number of pixels over which the edges should be smoothed.

    Returns
    -------
    mask: 2D numpy array
        The circular mask with values smoothly transitioning from 1 inside the circle to 0 outside.
    """
    radius = percentage * (size / 2)  # Convert percentage to radius
    y, x = np.ogrid[:size, :size]
    center = (size // 2, size // 2)
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a circular mask with a smooth transition
    mask = np.ones((size, size))
    mask[distance_from_center > radius + smooth_pixels] = 0
    transition_zone = (distance_from_center >= radius) & (distance_from_center <= radius + smooth_pixels)
    mask[transition_zone] = np.cos((distance_from_center[transition_zone] - radius) * (np.pi / (2 * smooth_pixels)))

    return mask


def reflect_image_2d(image):
    # Original
    center = image

    # Reflections
    flip_ud = np.flipud(image)       # Up-Down flip
    flip_lr = np.fliplr(image)       # Left-Right flip
    flip_udlr = np.flipud(flip_lr)   # Both flips

    # Tile construction
    top_row = np.hstack([flip_udlr, flip_ud, flip_udlr])
    mid_row = np.hstack([flip_lr,    center, flip_lr])
    bot_row = np.hstack([flip_udlr, flip_ud, flip_udlr])

    tiled_image = np.vstack([top_row, mid_row, bot_row])
    return tiled_image

def add_gaussian_noise(x, mu, std):
    noise = np.random.normal(mu, std, size=x.shape)
    x_noisy = np.abs(x + noise)
    return x_noisy

# Self-made Voigt function
# def gaussian_profile(x, y, sigma):
#     """2D Gaussian function"""
#     return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#
#
# def lorentzian_profile(x, y, gamma):
#     """2D Lorentzian function."""
#     return gamma / (np.pi * (x ** 2 + y ** 2 + gamma ** 2))
#
#
# def voigt_2d(x, y, sigma_g, gamma_l, normalize=True):
#     """Generate a 2D Voigt profile by numerically convolving a Gaussian and Lorentzian."""
#
#     gaussian_grid = gaussian_profile(x, y, sigma_g)
#     lorentzian_grid = lorentzian_profile(x, y, gamma_l)
#
#     # Convolve Gaussian and Lorentzian profiles
#     voigt_grid = np.real(IFFT(FFT(gaussian_grid) * FFT(lorentzian_grid)))
#     #voigt_grid = fftconvolve(gaussian_grid, lorentzian_grid, mode='same')
#
#     voigt_grid = voigt_grid / np.sum(voigt_grid)
#
#     return voigt_grid


# from scipy.stats import poisson
# def add_poisson_noise(image, photons_per_pixel=1e12):
#     """Adds Poisson noise to image"""
#     image = np.array(image, dtype=np.float32)
#     poisson_factor = 1e-12 * photons_per_pixel
#     scaled_image = poisson_factor * image
#     noisy_image = (1.0 / poisson_factor) * poisson.rvs(scaled_image)
#
#     return noisy_image

# class CrossCorrelationLoss(nn.Module):
#     def __init__(self):
#         super(CrossCorrelationLoss, self).__init__()
#
#     def forward(self, x, y):
#         x_mean = torch.mean(x)
#         y_mean = torch.mean(y)
#         x = x - x_mean
#         y = y - y_mean
#         cc = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
#         return 1 - cc  # Higher correlation means lower loss

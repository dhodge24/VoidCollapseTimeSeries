"""
This file contains functions that are frequently used. PyTorch version for increased speed

Many functions here were converted to python from MATLAB and/or were modified from existing code from:
1) https://gitlab.gwdg.de/irp/holotomotoolbox
2) https://github.com/rafael-fuente/diffractsim

If a citation is missing, let me know

"""

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import numbers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FFT(img):
    """2D Fourier transform"""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))  # Provides correct magnitude and phase output


def IFFT(img):
    """2D inverse Fourier transform"""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(img)))  # Provides correct magnitude and phase output


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
    if torch.is_tensor(img):
        img = img
    else:
        img = torch.tensor(img, dtype=torch.float32)

    if torch.Tensor.dim(img) == 2:
        img = torch.unsqueeze(img, dim=0)

    imageSize = torch.tensor(img.size(), dtype=torch.float32)  # Image shape
    y, x = imageSize[-2], imageSize[-1]

    assert newSize[0] <= y, 'Output height is larger than the input.'
    assert newSize[1] <= x, 'Output width is larger than the input.'

    # Note: if N or M is even and N or M odd, then the (M/2)+1 element is the center, the same holds for N. This is the
    # location of the 0th order in a FFT of an image with even number of pixels.

    newSizey = newSize[0]
    newSizex = newSize[1]
    rawOffset = (torch.asarray([y, x]) - torch.asarray(newSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(torch.ceil(rawOffsety)) - 1
    offsetx = int(torch.ceil(rawOffsetx)) - 1
    offsetEndy = offsety + newSizey
    offsetEndx = offsetx + newSizex
    croppedImg = img[0, offsety:offsetEndy, offsetx: offsetEndx]

    return croppedImg


def croppedArray(img, crop_pre, crop_post=None):
    """Code converted from MATLAB to python from here:
    https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/auxiliary/croparray.m

    This function crops a given numerical array by a specified amount of rows/columns at the beginning and/or
    end of the array

    :param img: Input image to be cropped [N, M] or [N, N]
    :param crop_pre: Tuple of non-negative integers. Amount of rows/columns/etc to crop at the beginning of the array
    along the different dimensions
    :param crop_post: Tuple of non-negative integers, optional Amount rows/columns/etc to crop at the end of the array
    along the different dimensions. If not assigned, cropPost = cropPre (--> symmetric cropping) is assumed.

    :return: Cropped image
    """

    N = img.squeeze(dim=0).shape
    num_dimensions = len(N)

    if crop_post is None:
        crop_post = crop_pre

    # If crop_pre and/or crop_post have fewer entries than the number of dimensions of the array, fill with zeros
    crop_pre = torch.cat((crop_pre, torch.zeros(num_dimensions - len(crop_pre)))).to(torch.int)
    crop_post = torch.cat((crop_post, torch.zeros(num_dimensions - len(crop_post)))).to(torch.int)

    cropped_array = torch.clone(img).squeeze(dim=0)
    for dim in range(num_dimensions):
        if crop_pre[dim] > 0 or crop_post[dim] > 0:
            idx_start = crop_pre[dim]
            idx_end = N[dim] - crop_post[dim]
            idx = tuple(slice(None) if i != dim else slice(idx_start, idx_end) for i in range(num_dimensions))
            cropped_array = cropped_array[idx]
    return cropped_array


def padToSize(img, outputSize, padMethod, padType, padValue):
    """
    padToSize pads or crops an image to a given size

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

    if torch.is_tensor(img):
        img = img
    else:
        img = torch.tensor(img, dtype=torch.float32)

    if torch.Tensor.dim(img) == 2:  # must be shape (N, C, H, W) or (C, H, W)
        img = torch.unsqueeze(img, dim=0)

    imageSize = img.size()
    y, x = imageSize[-2], imageSize[-1]
    ynew, xnew = torch.tensor(outputSize)  # Desired output size
    ypad = ynew - y  # padding to add in y
    xpad = xnew - x  # padding to add in x

    if padValue is None:
        padValue = 0.0

    # Crop the image if padding becomes negative
    if torch.any(torch.tensor([ypad, xpad]) < 0):  # any is a logical OR operation
        imageSize = torch.minimum(torch.tensor([imageSize[-2], imageSize[-1]]), torch.tensor(outputSize))
        imgCropped = cropToCenter(img, imageSize)
        return imgCropped

    # Pad the image if padding is positive
    else:
        if padMethod == 'replicate' and padType == 'pre':
            paddedImg = F.pad(img, pad=(int(xpad), 0, int(ypad), 0), mode='replicate')
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'replicate' and padType == 'post':
            paddedImg = F.pad(img, pad=(0, int(xpad), 0, int(ypad)), mode='replicate')
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'replicate' and padType == 'both':
            pad_top = int(ypad // 2)
            pad_bottom = int(ypad // 2 + ypad % 2)
            pad_left = int(xpad // 2)
            pad_right = int(xpad // 2 + xpad % 2)
            paddedImg = F.pad(img, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'replicate' and padType == 'preandpost':
            pad_top = int(torch.ceil(ypad / 2))
            pad_bottom = int(torch.floor(ypad / 2))
            pad_left = int(torch.ceil(xpad / 2))
            pad_right = int(torch.floor(xpad / 2))
            paddedImg = F.pad(img, pad=(pad_left, 0, pad_top, 0), mode='replicate')
            paddedImg = F.pad(paddedImg, pad=(0, pad_right, 0, pad_bottom), mode='replicate')
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'replicate' and padType == 'postandpre':
            pad_top = int(torch.floor(ypad / 2))
            pad_bottom = int(torch.ceil(ypad / 2))
            pad_left = int(torch.floor(xpad / 2))
            pad_right = int(torch.ceil(xpad / 2))
            paddedImg = F.pad(img, pad=(0, pad_right, 0, pad_bottom), mode='replicate')
            paddedImg = F.pad(paddedImg, pad=(pad_left, 0, pad_top, 0), mode='replicate')
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg

        if padMethod == 'constant' and padType == 'pre':
            paddedImg = F.pad(img, pad=(int(xpad), 0, int(ypad), 0), mode='constant', value=padValue)
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'constant' and padType == 'post':
            paddedImg = F.pad(img, pad=(0, int(xpad), 0, int(ypad)), mode='constant', value=padValue)
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'constant' and padType == 'both':
            pad_top = int(ypad // 2)
            pad_bottom = int(ypad // 2 + ypad % 2)
            pad_left = int(xpad // 2)
            pad_right = int(xpad // 2 + xpad % 2)
            paddedImg = F.pad(img, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=padValue)
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'constant' and padType == 'preandpost':
            pad_top = int(torch.ceil(ypad / 2))
            pad_bottom = int(torch.floor(ypad / 2))
            pad_left = int(torch.ceil(xpad / 2))
            pad_right = int(torch.floor(xpad / 2))
            paddedImg = F.pad(img, pad=(pad_left, 0, pad_top, 0), mode='constant', value=padValue)
            paddedImg = F.pad(paddedImg, pad=(0, pad_right, 0, pad_bottom), mode='constant', value=padValue)
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg
        if padMethod == 'constant' and padType == 'postandpre':
            pad_top = int(torch.floor(ypad / 2))
            pad_bottom = int(torch.ceil(ypad / 2))
            pad_left = int(torch.floor(xpad / 2))
            pad_right = int(torch.ceil(xpad / 2))
            paddedImg = F.pad(img, pad=(0, pad_right, 0, pad_bottom), mode='constant', value=padValue)
            paddedImg = F.pad(paddedImg, pad=(pad_left, 0, pad_top, 0), mode='constant', value=padValue)
            paddedImg = paddedImg.squeeze(dim=0)
            return paddedImg


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
        dx = torch.tensor([1])  # Step size of 1
    else:
        dx = torch.tensor(dx[:], dtype=torch.float32)


    dx = torch.tensor(np.transpose(dx[:].numpy())) * torch.ones([1, len(N)]).flatten()
    ndim = len(N)

    # Optionally assemble meshgrid (same as MATLAB ndgrid if index is ij) (occupies more memory!)
    if computeMeshgrid:
        Y = torch.arange(-N[0] / 2 + 0.5, N[0] / 2, 1.0) * dx[0]
        X = torch.arange(-N[1] / 2 + 0.5, N[1] / 2, 1.0) * dx[1]
        xnew, ynew = torch.meshgrid(X, Y, indexing='ij')
        xnew = xnew.to(torch.float32)
        ynew = ynew.to(torch.float32)
        return xnew, ynew

    elif ndim > 1:
        Y = torch.arange(-N[0] / 2 + 0.5, N[0] / 2, 1).flatten().reshape(1, N[0]) * dx[0]
        X = torch.arange(-N[1] / 2 + 0.5, N[1] / 2, 1).flatten().reshape(N[1], 1) * dx[1]
        xnew = X.to(torch.float32)
        ynew = Y.to(torch.float32)
        return xnew, ynew

from torchvision.transforms import GaussianBlur
# import math
# def gaussian_kernel(size, sigma=2., dim=2, channels=1):
#     # The gaussian kernel is the product of the gaussian function of each dimension.
#     # kernel_size should be an odd number.
#
#     kernel_size = 2 * size + 1
#
#     kernel_size = [kernel_size] * dim
#     sigma = [sigma] * dim
#     kernel = 1
#     meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')
#
#     for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#         mean = (size - 1) / 2
#         kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
#
#     # Make sure sum of values in gaussian kernel equals 1.
#     kernel = kernel / torch.sum(kernel)
#
#     # Reshape to depthwise convolutional weight
#     kernel = kernel.view(1, 1, *kernel.size())
#     kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#
#     return kernel
#
#
# def _gaussian_blur(x, size=1, sigma=2, dim=2, channels=1):
#     kernel = gaussian_kernel(size=size, sigma=sigma, dim=dim, channels=channels)
#     kernel_size = 2 * size + 1
#
#     x = x[None, ...]
#     padding = int((kernel_size - 1) / 2)
#     x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
#     x = torch.squeeze(F.conv2d(x, kernel, groups=1))
#
#     return x


class GaussianSmoothing(nn.Module):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10

    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed separately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cpu'):
        super(GaussianSmoothing, self).__init__()
        self.device = device
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32, device=device)
                                    for size in kernel_size], indexing='ij')
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * torch.sqrt(torch.tensor(2 * torch.pi, device=device))) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input.to(torch.float32), weight=self.weight.to(torch.float32), groups=self.groups)


def fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments, angularOffsetSegments,
                        fadeToVal, bottom_apply):
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

    img = torch.tensor(img, dtype=torch.float32, device=device) if not torch.is_tensor(img) else img.to(device)
    imageSize = img.shape

    # Transition length is taken as 1/8 of the image's aspect length if not assigned
    if not transitionLength:
        transitionLengthy = torch.ceil(torch.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLengthx = torch.ceil(torch.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLength = [transitionLengthy, transitionLengthx]
    else:
        transitionLength = [transitionLength[0], transitionLength[1]]

    if not windowShift:
        windowShift = torch.tensor([0, 0])
    else:
        windowShift = torch.tensor(windowShift)

    if not ellipseSize:
        ellipseSize = torch.tensor([0.8, 0.8])
    else:
        ellipseSize = torch.tensor(ellipseSize)

    if not numSegments:
        numSegments = 1

    if not angularOffsetSegments:
        angularOffsetSegments = 0

    # Initialize these so errors aren't thrown in if statements
    idxBoundary = None
    transitionMask = None
    # fadeToVals = None

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

        idxBoundary = torch.logical_xor(idxEllipse, (X ** 2 / rxInner ** 2 + Y ** 2 / ryInner ** 2 < 1))
        transitionMask = torch.where(idxEllipse == True, torch.tensor(1), torch.tensor(0))

    if fadeMethod == 'rectangle':
        # Radii in x and y in units of pixels
        ry = (torch.ceil(ellipseSize[0] * imageSize[0])).to(torch.int)
        rx = (torch.ceil(ellipseSize[1] * imageSize[1])).to(torch.int)

        # Indices of the image pixels lying within a centered rectangle
        idxRectangle = torch.ones((ry, rx)).to(torch.int)
        idxRectangle = padToSize(idxRectangle, outputSize=imageSize, padMethod='constant', padType='both', padValue=0)
        idxRectangle = torch.roll(idxRectangle,
                                  shifts=(windowShift * torch.tensor([-1, 1])).tolist(),
                                  dims=(0, 1))

        # extract pixels on the boundary of the rectangle to calculate mean values for fadeout
        rxInner = (rx - transitionLength[1]).to(torch.int)
        ryInner = (ry - transitionLength[0]).to(torch.int)
        idxInnerRectangle = torch.ones((ryInner, rxInner)).to(torch.int)
        idxInnerRectangle = padToSize(idxInnerRectangle, outputSize=imageSize, padMethod='constant', padType='both',
                                      padValue=0)
        idxInnerRectangle = torch.roll(idxInnerRectangle,
                                       shifts=(windowShift * torch.tensor([-1, 1])).tolist(),
                                       dims=(0, 1))

        idxBoundary = torch.logical_xor(idxRectangle, idxInnerRectangle)
        transitionMask = torch.where(idxRectangle == True, torch.tensor(1), torch.tensor(0))

    # Determine values in which image is faded out to

    # Case 0: Constant fadeout-value has been assigned
    if fadeToVal:
        fadeToVals = torch.tensor(fadeToVal)

    # Case 1: Compute fadeout-values in multiple angular segments
    elif numSegments > 1:
        X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)

        # Shift center of the fading window
        Y = Y - windowShift[0]
        X = X - windowShift[1]

        def cart2pol(x, y):
            """Converts cartesian coordinates to polar"""
            rho_ = torch.sqrt(x ** 2 + y ** 2)
            theta_ = torch.arctan2(y, x)
            return rho_, theta_

        rho, theta = cart2pol(X, Y)
        theta = torch.remainder(theta + angularOffsetSegments, 2 * torch.pi)
        # This splits the whole image into slices like a pie for however many segments are chosen
        segmentIndices = torch.minimum((torch.floor((numSegments / (2 * torch.pi)) * theta) + 1),
                                       torch.tensor(numSegments)).to(torch.int)

        sigma = 10 / 2.35
        truncate = 2
        kernel_size = int(truncate * sigma * 2 + 1)
        gaussian_blur = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma)
        img = img.unsqueeze(dim=0)
        imFilt = gaussian_blur(img)
        imFilt = imFilt.squeeze(dim=0).to(device)
        imFilt = padToSize(img=imFilt,
                           outputSize=[img.shape[-2], img.shape[-1]],
                           padMethod='replicate',
                           padType='both',
                           padValue=0)
        img = img.squeeze(dim=0)

        # Compute target values for the fade out in each angular segment
        fadeToVals = torch.zeros_like(img)
        for segment in range(1, numSegments + 1):
            inSegment = torch.where(segment == segmentIndices, True, False)
            fadeToVals[inSegment] = torch.mean(imFilt[idxBoundary & inSegment]).to(device)

        sigma = transitionLength[0] / 2.35
        truncate = 2
        kernel_size = int(truncate * sigma * 2 + 1)
        gaussian_blur = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma)
        fadeToVals = fadeToVals.unsqueeze(dim=0)
        fadeToVals = gaussian_blur(fadeToVals)
        fadeToVals = fadeToVals.squeeze(dim=0)
        fadeToVals = padToSize(img=fadeToVals,
                               outputSize=[img.shape[-2], img.shape[-1]],
                               padMethod='replicate',
                               padType='both',
                               padValue=0)

    # Case 2: Compute one fadeout-value globally
    else:
        fadeToVals = torch.mean(img[idxBoundary])  # Only 1 segment

    if transitionLength[0] > 1 and transitionLength[1] > 1:
        transitionMask = transitionMask.to(torch.float32).to(device)

        sigma = transitionLength[0] / 2.35
        truncate = 2
        kernel_size = int(truncate * sigma * 2 + 1)
        gaussian_blur = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma)
        transitionMask = transitionMask.unsqueeze(dim=0)
        transitionMask = gaussian_blur(transitionMask)
        transitionMask = transitionMask.squeeze(dim=0)
        transitionMask = padToSize(img=transitionMask,
                                   outputSize=[img.shape[-2], img.shape[-1]],
                                   padMethod='replicate',
                                   padType='both',
                                   padValue=0)

    if bottom_apply:
        # Create a mask for the bottom half of the image with smooth transition
        # transition_region_height = transitionLength[0].item() if isinstance(transitionLength[0], torch.Tensor) else \
        # transitionLength[0]
        transition_region_height = torch.tensor(2)
        bottom_half_mask = torch.zeros_like(img).to(device)
        bottom_half_mask[(img.shape[0]) // 2 + transition_region_height:, :] = 1
        transition_region = torch.linspace(0, 1, transition_region_height)
        for i in range(transition_region_height):
            bottom_half_mask[(img.shape[0]) // 2 + i, :] = transition_region[i]
        imgFaded = (img * (1 - bottom_half_mask) +
                    (img * transitionMask + fadeToVals * (1 - transitionMask)) * bottom_half_mask)
    else:
        # Convex combination of transition masks
        # Apply fade-out effect using fadeToVals as target
        imgFaded = img * transitionMask + fadeToVals * (1 - transitionMask)  # original

    # if bottom_apply:
    #     # Create a mask for the bottom half of the image with a smooth transition
    #     transition_region_height = transitionLength[0].item() if isinstance(transitionLength[0], torch.Tensor) else \
    #     transitionLength[0]
    #     bottom_half_mask = torch.zeros_like(img).to(device)
    #     bottom_half_mask[(img.shape[0] // 2 + transition_region_height):, :] = 1
    #     transition_region = torch.linspace(0, 1, transition_region_height).to(device)
    #
    #     for i in range(transition_region_height):
    #         bottom_half_mask[(img.shape[0] // 2 + i), :] = transition_region[i]
    #
    #     # Create the faded image only for the bottom half
    #     img_bottom_faded = img * transitionMask + fadeToVals * (1 - transitionMask)
    #
    #     # Combine the original top half with the faded bottom half
    #     imgFaded = img.clone()
    #     imgFaded[(img.shape[0] // 2):, :] = img[(img.shape[0] // 2):, :] * (
    #                 1 - bottom_half_mask[(img.shape[0] // 2):, :]) + img_bottom_faded[(img.shape[0] // 2):,
    #                                                                  :] * bottom_half_mask[(img.shape[0] // 2):, :]
    # else:
    #     # Convex combination of transition masks
    #     # Apply fade-out effect using fadeToVals as target
    #     imgFaded = img * transitionMask + fadeToVals * (1 - transitionMask)

    imgFaded[torch.isnan(imgFaded)] = 1
    imgFaded[torch.isinf(imgFaded)] = 1

    return imgFaded, transitionMask


def fadeoutImage(img, fadeMethod=None, fadeToVal=None, transitionLength=None, ellipseSize=None, numSegments=None,
                 angularOffsetSegments=None, windowShift=None, bottom_apply=False):
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
    if bottom_apply is None:
        bottom_apply = False

    if fadeMethod == 'ellipse':
        imgFaded, window = fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments,
                                               angularOffsetSegments, fadeToVal, bottom_apply)
        return imgFaded, window
    if fadeMethod == 'rectangle':
        imgFaded, window = fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments,
                                               angularOffsetSegments, fadeToVal, bottom_apply)
        return imgFaded, window
    else:
        return TypeError('Invalid value for fadeMethod. Choices are cosine, ellipse, or rectangle.')


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

    if torch.Tensor.dim(img) == 3:
        img = torch.squeeze(img, dim=0)

    Ny, Nx = img.shape
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
        cy, cx = torch.round(ndimage.measurements.center_of_mass(img)).to(torch.int)
        print('Center of mass in y and x are:', cy, cx)
        ax1.axvline(x=cx, color='r')  # Plot vertical line out
        ax1.axhline(y=cy, color='g')  # Plot horizontal line out
    xvals = torch.linspace(0, (Nx - 1), Nx)  # vals of 0, 1, 2, ... Nx - 1 -- total Nx values
    yvals = torch.linspace(0, (Ny - 1), Ny)  # vals of 0, 1, 2, ... Ny - 1 -- total Ny values
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

    if torch.Tensor.dim(img1) == 3:
        img1 = torch.squeeze(img1, dim=0)
    if torch.Tensor.dim(img2) == 3:
        img2 = torch.squeeze(img2, dim=0)

    cx = Nx // 2
    cy = Ny // 2

    xvals = torch.linspace(0, (Nx - 1), Nx)  # vals of 0, 1, 2, ... Nx - 1 -- total Nx values
    yvals = torch.linspace(0, (Ny - 1), Ny)  # vals of 0, 1, 2, ... Ny - 1 -- total Ny values

    plt.figure()
    plt.title('Vertical Lineout')
    plt.plot(xvals, img1[:, cx], 'black', lw=1)
    plt.plot(xvals, img2[:, cx], 'red', lw=1)
    plt.grid()
    plt.savefig("/Users/danielhodge/Desktop/Compare_Plot_horz", bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

    plt.figure()
    plt.title('Horizontal Lineout')
    plt.plot(xvals, img1[cy, :], 'black', lw=1)
    plt.plot(xvals, img2[cy, :], 'red', lw=1)
    plt.grid()
    plt.savefig("/Users/danielhodge/Desktop/Compare_Plot_vert", bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

    return

def rescaleImgToCustomCoord(img, imageSize, extent_x, extent_y, Nx, Ny, padVal):
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

    if torch.is_tensor(img):
        img = img
    else:
        img = torch.tensor(img, dtype=torch.float32)

    img = img.unsqueeze(dim=0)
    _, numImgPixelsWidth, numImgPixelsHeight = img.shape
    print('Original image height and width in pixels:', (numImgPixelsHeight, numImgPixelsWidth))
    if imageSize is not None:
        # Below is equivalent to Nobj = a/extent * N, which is equivalent to Nobj = a/ps (or ps = a/Nobj)
        # Nobj is the number of object pixels in x and y, N is total number of pixels in x and y,
        # ps is object pixel size, extent is the physical grid size in x and y, and a is the physical object size
        newNumImgPixelsHeight, newNumImgPixelsWidth = int(torch.round(imageSize / extent_y * Ny)), \
                                                      int(torch.round(imageSize / extent_x * Nx))
        print('New height and width in pixels:', (newNumImgPixelsHeight, newNumImgPixelsWidth))
    else:
        # By default, the image fills the entire aperture plane
        newNumImgPixelsHeight, newNumImgPixelsWidth = Ny, Nx
        print('Height and width in pixels remains the same. The image will fill the entire aperture plane')

    newShape = (newNumImgPixelsWidth, newNumImgPixelsHeight)  # cv2 takes (width, height) so it must be in this order
    # img = cv2.resize(img, dsize=newShape, interpolation=cv2.INTER_AREA)  # Use if shrinking the image
    #img = cv2.resize(img, dsize=newShape, interpolation=cv2.INTER_CUBIC)  # Use if enlarging the image
    img = transforms.Resize(size=newShape, antialias=True)(img)
    result = padToSize(img[0, :, :], outputSize=[Ny, Nx], padMethod='constant', padType='both', padValue=padVal)
    return result


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
    X, Y = centeredGrid([dim_y, dim_x], dx=[dx, dy], computeMeshgrid=True)
    k = 2*torch.pi/lam
    rayleigh_length = torch.pi * beam_waist**2 / lam

    beam_width = beam_waist * torch.sqrt(1 + (prop_distance/rayleigh_length) ** 2)
    radius_curvature = prop_distance + rayleigh_length ** 2 / prop_distance
    phi = torch.atan(prop_distance / rayleigh_length)

    if prop_distance == 0:
        psi = torch.exp(-(X ** 2 + Y ** 2) / beam_waist ** 2)
    else:
        psi = beam_waist * torch.exp(1j * k * prop_distance - 1j * phi) / beam_width * \
              torch.exp(-(X ** 2 + Y ** 2) / beam_width ** 2 + 1j * k * (X ** 2 + Y ** 2) / (2 * radius_curvature))

    return psi


# import torch.fft as fft
#
# def phaseUnwrapping(img):
#     """Optimized phase unwrapping function"""
#     _, Ny, Nx = img.shape
#     y = torch.arange(Ny) - Ny / 2
#     x = torch.arange(Nx) - Nx / 2
#     yy, xx = torch.meshgrid(y, x, indexing='ij')
#     f = xx**2 + yy**2
#
#     eps = 1e-6
#     sin_img = torch.sin(img)
#     cos_img = torch.cos(img)
#
#     F_cos = fft.fftn(cos_img, dim=(-2, -1))
#     F_sin = fft.fftn(sin_img, dim=(-2, -1))
#
#     a = fft.ifftn(F_cos * fft.ifftn(F_sin * f, dim=(-2, -1)) / (f + eps), dim=(-2, -1))
#     b = fft.ifftn(F_sin * fft.ifftn(F_cos * f, dim=(-2, -1)) / (f + eps), dim=(-2, -1))
#
#     outPhase = torch.real(a - b)
#     return outPhase

def phaseUnwrapping(img):
    """Phase unwrapping function"""
    _, Ny, Nx = img.shape
    f = torch.zeros((Ny, Nx))
    for ii in range(Ny):
        for jj in range(Nx):
            y = ii - Ny/2
            x = jj - Nx/2
            f[ii, jj] = x**2 + y**2
    eps = 0.000001
    a = IFFT(FFT(torch.cos(img) * IFFT(FFT(torch.sin(img)) * f)) / (f + eps))
    b = IFFT(FFT(torch.sin(img) * IFFT(FFT(torch.cos(img)) * f)) / (f + eps))
    outPhase = torch.real(a - b)
    return outPhase


def ASM_propagator(x, lam, dx, dy, zeff, prop_method):
    """Transfer function (non-paraxial) in PyTorch

    Parameters
    -----------
    :param x : Original complex wavefront [N, M] as a torch.Tensor
    :param lam : Wavelength
    :param dx : Pixel size in x dimension
    :param dy : Pixel size in y dimension
    :param zeff : Propagation distance

    Returns
    -----------
    :returns wavefrontOut : Complex wavefront after propagation [N, M] as a torch.Tensor
    """

    Ny, Nx = x.shape

    k0 = 2 * torch.pi / lam
    KX = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(Nx, d=dx, dtype=torch.float64))  # Spacing in Fourier space
    KY = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(Ny, d=dy, dtype=torch.float64))  # Spacing in Fourier space
    kx, ky = torch.meshgrid(KX, KY, indexing='ij')
    arg = k0 ** 2 - kx ** 2 - ky ** 2
    arg[arg < 0] = 0
    kz = torch.sqrt(torch.real(arg))
    H = None
    if prop_method == 'fwd':
        H = torch.exp(1j * zeff * kz)  # forward propagation kernel
    elif prop_method == 'bwd':
        H = torch.exp(-1j * zeff * kz)  # backward propagation kernel

    # FFT and IFFT operations in PyTorch
    out = IFFT(FFT(x) * H)

    return out


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
    mask: 2D PyTorch tensor
        The circular mask with values smoothly transitioning from 1 inside the circle to 0 outside.
    """
    radius = percentage * (size / 2)  # Convert percentage to radius
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = (size // 2, size // 2)
    distance_from_center = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a circular mask with a smooth transition
    mask = torch.ones((size, size), dtype=torch.float32)
    mask[distance_from_center > radius + smooth_pixels] = 0
    transition_zone = (distance_from_center >= radius) & (distance_from_center <= radius + smooth_pixels)
    mask[transition_zone] = torch.cos((distance_from_center[transition_zone] - radius) * (np.pi / (2 * smooth_pixels)))

    return mask


# def gradient_2d_forward(image):
#     """Calculates the first-order forward finite-difference derivative along the horizontal (x-direction)
#     and vertical (y-direction) for a 2D image array."""
#     Dx = torch.diff(image, dim=1, append=image[:, -1:])
#     Dy = torch.diff(image, dim=0, append=image[-1:, :])
#     return Dx, Dy
#
#
# def gradient_2d_backward(image):
#     """Calculates the first-order backward finite-difference derivative along the horizontal (x-direction)
#     and vertical (y-direction) directions for a 2D image array
#     See: https://math.mit.edu/~gs/linearalgebra/ila5/TransposeDerivative01.pdf
#     """
#     Dx = -torch.diff(image, dim=1, prepend=image[:, :1])
#     Dy = -torch.diff(image, dim=0, prepend=image[:1, :])
#     return Dx, Dy
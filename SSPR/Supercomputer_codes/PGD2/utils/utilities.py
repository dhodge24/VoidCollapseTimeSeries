"""Commonly used functions"""

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
import cv2 as cv2

def FFT(img):
    """2D Fourier transform"""
    return cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(img)))  # Provides correct magnitude and phase output

def IFFT(img):
    """2D inverse Fourier transform"""
    return cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(img)))  # Provides correct magnitude and phase output

def cropToCenter(img, newSize):
    """Code converted from MATLAB to python from here: ...
    cropToCenter returns the central part of a 2d or 3d input array. ...

        Parameters
        -----------
        :param img : Numerical array to crop of size [N, M]
        :param newSize : Size in which to crop the numerical array [N, M]

        Returns
        -----------
        :returns croppedImg : Numerical array that is cropped
    """
    img = cp.array(img)
    imageSize = cp.array(cp.shape(img), dtype=cp.float32)
    y, x = imageSize[0], imageSize[1]

    assert newSize[0] <= y, 'Output height is larger than the input.'
    assert newSize[1] <= x, 'Output width is larger than the input.'

    newSizey = newSize[0]
    newSizex = newSize[1]
    rawOffset = (cp.asarray([y, x]) - cp.asarray(newSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(cp.ceil(rawOffsety)) - 1
    offsetx = int(cp.ceil(rawOffsetx)) - 1
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

    imageSize = cp.array(cp.shape(img))
    y, x = imageSize[0], imageSize[1]

    assert newSize[0] <= y, 'Output height is larger than the input.'
    assert newSize[1] <= x, 'Output width is larger than the input.'

    newSizey = newSize[0]
    newSizex = newSize[1]
    rawOffset = (cp.array([y, x]) - cp.array(newSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(cp.ceil(rawOffsety)) - 1
    offsetx = int(cp.ceil(rawOffsetx)) - 1

    shiftDisty = offsety - rawOffsety
    shiftDistx = offsetx - rawOffsetx

    if (cp.array([shiftDisty, shiftDistx])).any():
        img = shiftImage(img, -cp.array([shiftDisty, shiftDistx]))

    offsetEndy = offsety + newSizey
    offsetEndx = offsetx + newSizex

    croppedImg = img[offsety:offsetEndy, offsetx: offsetEndx]

    return croppedImg

def croppedArray(array, crop_pre, crop_post=None):
    """Code converted from MATLAB to python from here: ...
    This function crops a given numerical array by a specified amount of rows/columns at the beginning and/or
    end of the array

    :param array: Input image to be cropped [N, M] or [N, N]
    :param crop_pre: Tuple of non-negative integers. Amount of rows/columns/etc to crop at the beginning of the array
    along the different dimensions
    :param crop_post: Tuple of non-negative integers, optional Amount rows/columns/etc to crop at the end of the array
    along the different dimensions. If not assigned, cropPost = cropPre (--> symmetric cropping) is assumed.

    :return: Cropped image
    """

    N = cp.shape(array)
    num_dimensions = len(N)

    if crop_post is None:
        crop_post = crop_pre

    crop_pre = cp.concatenate((crop_pre, cp.zeros(num_dimensions - len(crop_pre)))).astype(int)
    crop_post = cp.concatenate((crop_post, cp.zeros(num_dimensions - len(crop_post)))).astype(int)

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

    largeSize = cp.shape(largeArray)
    smallSize = cp.shape(smallArray)

    assert largeSize[0] >= smallSize[0], 'Height is larger than the total.'
    assert largeSize[1] >= smallSize[1], 'Width is larger than the total.'

    rawOffset = (cp.array(largeSize) - cp.array(smallSize)) / 2 + 1
    rawOffsety = rawOffset[0]
    rawOffsetx = rawOffset[1]
    offsety = int(cp.floor(rawOffset[0]))
    offsetx = int(cp.floor(rawOffset[1]))

    shiftDisty = int(rawOffsety - offsety)
    shiftDistx = int(rawOffsetx - offsetx)
    shiftDist = cp.array([shiftDisty, shiftDistx])
    innerSize = (smallSize + cp.ceil(shiftDist)).astype(int)

    innerSizey = innerSize[0]
    innerSizex = innerSize[1]
    offsetEndy = int(offsety + innerSizey)
    offsetEndx = int(offsetx + innerSizex)

    if shiftDistx and shiftDisty is None:
        innerArray = smallArray
    else:
        innerArray = cp.zeros(innerSize, dtype=cp.float32)
        innerMask = cp.zeros(innerSize, dtype=cp.float32)
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

    img = cp.array(img)
    imageSize = cp.shape(img)
    y, x = imageSize[0], imageSize[1]
    ynew, xnew = outputSize  # Desired output size
    ypad = ynew - y  # padding to add in y
    xpad = xnew - x  # padding to add in x

    if padValue is None:
        padValue = 0.0

    # Crop the image if padding becomes negative
    if (cp.array([ypad, xpad]) < 0).any():  # any is a logical OR operation
        imageSize = cp.minimum(imageSize, outputSize)
        imgCropped = cropToCenter(img, imageSize)
        return imgCropped

    # Pad the image if padding is positive
    else:
        if padMethod == 'replicate' and padType == 'pre':
            paddedImg = cp.pad(img,
                               pad_width=[(int(ypad), 0), (int(xpad), 0)],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'post':
            paddedImg = cp.pad(img,
                               pad_width=[(0, int(ypad)), (0, int(xpad))],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'both':
            pad_top = int(ypad // 2)
            pad_bottom = int(ypad // 2 + ypad % 2)
            pad_left = int(xpad // 2)
            pad_right = int(xpad // 2 + xpad % 2)
            paddedImg = cp.pad(img,
                               pad_width=[(pad_top, pad_bottom), (pad_left, pad_right)],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'preandpost':
            pad_top = int(cp.ceil(ypad / 2))
            pad_bottom = int(cp.floor(ypad / 2))
            pad_left = int(cp.ceil(xpad / 2))
            pad_right = int(cp.floor(xpad / 2))
            paddedImg = cp.pad(img,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='edge')
            paddedImg = cp.pad(paddedImg,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='edge')
            return paddedImg
        if padMethod == 'replicate' and padType == 'postandpre':
            pad_top = int(cp.floor(ypad / 2))
            pad_bottom = int(cp.ceil(ypad / 2))
            pad_left = int(cp.floor(xpad / 2))
            pad_right = int(cp.ceil(xpad / 2))
            paddedImg = cp.pad(img,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='edge')
            paddedImg = cp.pad(paddedImg,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='edge')
            return paddedImg

        if padMethod == 'constant' and padType == 'pre':
            paddedImg = cp.pad(img,
                               pad_width=[(int(ypad), 0), (int(xpad), 0)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg
        if padMethod == 'constant' and padType == 'post':
            paddedImg = cp.pad(img,
                               pad_width=[(0, int(ypad)), (0, int(xpad))],
                               mode='constant',
                               constant_values=padValue)

            return paddedImg
        if padMethod == 'constant' and padType == 'both':
            pad_top = int(ypad // 2)
            pad_bottom = int(ypad // 2 + ypad % 2)
            pad_left = int(xpad // 2)
            pad_right = int(xpad // 2 + xpad % 2)
            paddedImg = cp.pad(img,
                               pad_width=[(pad_top, pad_bottom), (pad_left, pad_right)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg
        if padMethod == 'constant' and padType == 'preandpost':
            pad_top = int(cp.ceil(ypad / 2))
            pad_bottom = int(cp.floor(ypad / 2))
            pad_left = int(cp.ceil(xpad / 2))
            pad_right = int(cp.floor(xpad / 2))
            paddedImg = cp.pad(img,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='constant',
                               constant_values=padValue)
            paddedImg = cp.pad(paddedImg,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg
        if padMethod == 'constant' and padType == 'postandpre':
            pad_top = int(cp.floor(ypad / 2))
            pad_bottom = int(cp.ceil(ypad / 2))
            pad_left = int(cp.floor(xpad / 2))
            pad_right = int(cp.ceil(xpad / 2))
            paddedImg = cp.pad(img,
                               pad_width=[(0, pad_bottom), (0, pad_right)],
                               mode='constant',
                               constant_values=padValue)
            paddedImg = cp.pad(paddedImg,
                               pad_width=[(pad_top, 0), (pad_left, 0)],
                               mode='constant',
                               constant_values=padValue)
            return paddedImg

def centeredGrid(N, dx=None, computeMeshgrid=False):
    if dx is None:
        dx = [1]  # Step size of 1

    dx = cp.transpose(cp.array(dx[:])) * cp.ones([1, len(N)]).flatten()
    ndim = len(N)

    if computeMeshgrid:
        Y = cp.arange(-N[0] / 2 + 0.5, N[0] / 2, 1) * dx[0]
        X = cp.arange(-N[1] / 2 + 0.5, N[1] / 2, 1) * dx[1]
        xnew, ynew = cp.meshgrid(X, Y)
        xnew = xnew.astype(cp.float32)
        ynew = ynew.astype(cp.float32)
        return xnew, ynew

    elif ndim > 1:
        Y = cp.arange(-N[0] / 2 + 0.5, N[0] / 2, 1).flatten().reshape(1, N[0]) * dx[0]
        X = cp.arange(-N[1] / 2 + 0.5, N[1] / 2, 1).flatten().reshape(N[1], 1) * dx[1]
        xnew = X.astype(cp.float32)
        ynew = Y.astype(cp.float32)
        return xnew, ynew


def padFadeOut(img, outputSize, transitionLength=None, padValue=None, fadeoutParallel=False, computeMeshgrid=False):
    img = cp.array(img, dtype=cp.float32)
    imageSize = cp.asarray(cp.shape(img))
    outputSize = cp.asarray(outputSize)
    padPre = cp.ceil((cp.asarray(outputSize) - imageSize[0:2]) / 2)
    padPost = cp.floor((cp.asarray(outputSize) - imageSize[0:2]) / 2)

    if padValue is None:
        padValue = img.mean() + 0.0
    if transitionLength is None:
        transitionLengthy = int(padPre[0])
        transitionLengthx = int(padPre[1])
        transitionLength = cp.asarray([transitionLengthy, transitionLengthx])
    else:
        transitionLength = cp.asarray([transitionLength[0], transitionLength[1]])

    if fadeoutParallel:
        imgPadded = padToSize(img, outputSize=outputSize, padMethod='constant', padType='preandpost', padValue=padValue)
        imgPadRegion1 = imgPadded
        imgPadRegion2 = cp.transpose(imgPadded)
        imgPadRegion = cp.asarray([imgPadRegion1, imgPadRegion2])

        for dim in [0, 1]:
            otherDim = 2 - dim
            imgTemp = cp.fft.fft(imgPadRegion[dim], axis=0)
            if dim == 0:
                kernel = cp.real(cp.fft.fft(cp.hstack([1 / 3, 1 / 3, cp.zeros([outputSize[0] - 3]), 1 / 3]), axis=0))
            else:
                kernel = cp.real(cp.fft.fft(cp.hstack([1 / 3, 1 / 3, cp.zeros([outputSize[1] - 3]), 1 / 3]), axis=0))
            list1 = cp.flip(cp.arange(padPre[otherDim - 1]))
            list1 = [int(x) for x in list1]
            list2 = cp.flip(cp.arange(padPost[otherDim - 1]))
            list2 = [int(x) for x in list2]
            for jj in cp.flip(range(len(list1))):
                imgTemp[:, jj] = kernel * imgTemp[:, jj + 1]
            for jj in cp.flip(range(len(list2))):
                imgTemp[:, -1 - jj] = kernel * imgTemp[:, -1 - jj - 1]
            imgTemp = cp.fft.ifft(imgTemp, axis=0)
            if cp.isrealobj(imgTemp):
                imgTemp = cp.real(imgTemp)
            imgPadRegion[dim] = imgTemp

        imgPadded = imgPadRegion[0] + cp.transpose(imgPadRegion[1]) - imgPadded

    else:
        imgPadded = padToSize(img, outputSize=outputSize, padMethod='replicate', padType='preandpost', padValue=None)

    Y, X = centeredGrid(outputSize, dx=[1, 1], computeMeshgrid=computeMeshgrid)
    eps = 0.000000000000001
    X = cp.minimum((cp.pi / (transitionLength[1] + eps)) * cp.maximum(cp.abs(X) - imageSize[1] / 2, 0), cp.pi)
    Y = cp.minimum((cp.pi / (transitionLength[0] + eps)) * cp.maximum(cp.abs(Y) - imageSize[0] / 2, 0), cp.pi)
    X = cp.array(X, dtype=cp.float32)
    Y = cp.array(Y, dtype=cp.float32)
    transitionMask = (0.25 * (1 + cp.cos(X))) * (1 + cp.cos(Y))

    imgPadded = transitionMask * imgPadded + (1 - transitionMask) * padValue
    return imgPadded

def shiftGrid(X, Y, shifts):
    shifts = cp.array(shifts, dtype=cp.float32)
    Y = Y + shifts[0]
    X = X - shifts[1]
    return Y, X

def rotateGrid(X, Y, rotAngleDegree):
    rotAngleDegree = cp.array(rotAngleDegree, dtype=cp.float32)
    YTemp = cp.cos(cp.radians(rotAngleDegree)) * Y - cp.sin(cp.radians(rotAngleDegree)) * X
    XTemp = cp.sin(cp.radians(rotAngleDegree)) * Y + cp.cos(cp.radians(rotAngleDegree)) * X
    Y = YTemp
    X = XTemp
    return Y, X

def magnifyGrid(X, Y, magnify):
    magnify = cp.array([magnify], dtype=cp.float32)
    magnify = (magnify[:] * cp.ones((1, 2))).flatten()
    Y = magnify[0] * Y
    X = magnify[1] * X
    return Y, X

def shiftImage(img, shifts):
    img = cp.array(img, dtype=cp.float32)
    imgTransformed = shiftRotateMagnifyImage(img, shifts=shifts)
    return imgTransformed

def magnifyImage(img, magnify):
    img = cp.array(img, dtype=cp.float32)
    imgTransformed = shiftRotateMagnifyImage(img, magnify=magnify)
    return imgTransformed

def rotateImage(img, rotAngleDegree):
    img = cp.array(img, dtype=cp.float32)
    imgTransformed = shiftRotateMagnifyImage(img, rotAngleDegree=rotAngleDegree)
    return imgTransformed

def shiftRotateImage(img, shifts, rotAngleDegree):
    img = cp.array(img, dtype=cp.float32)
    imgTransformed = shiftRotateMagnifyImage(img, shifts=shifts, rotAngleDegree=rotAngleDegree)
    return imgTransformed

def shiftRotateMagnifyImage(img, magnify=None, rotAngleDegree=None, shifts=None, padMethod='replicate', order=3,
                            invertTransform=False):
    if magnify is None:
        magnify = [1, 1]
    if rotAngleDegree is None:
        rotAngleDegree = 0
    if shifts is None:
        shifts = [0, 0]

    N = cp.shape(img)
    X, Y = centeredGrid(N, dx=[1, 1], computeMeshgrid=True)
    if invertTransform:
        Y, X = magnifyGrid(X, Y, cp.array(magnify, dtype=cp.float32))
        Y, X = rotateGrid(X, Y, cp.array(rotAngleDegree, dtype=cp.float32))
        Y, X = shiftGrid(X, Y, cp.array(shifts, dtype=cp.float32))
    else:
        Y, X = shiftGrid(X, Y, cp.array([-1 * s for s in shifts], dtype=cp.float32))
        Y, X = rotateGrid(X, Y, -cp.array(rotAngleDegree, dtype=cp.float32))
        Y, X = magnifyGrid(X, Y, cp.array([1 / mag for mag in magnify], dtype=cp.float32))

    padPre = cp.maximum(0, [cp.ceil(-(N[0] - 1) / 2 - cp.min(Y[:])), cp.ceil(-(N[1] - 1) / 2 - cp.min(X[:]))])
    padPost = cp.maximum(0, [cp.ceil(cp.max(Y[:]) - (N[0] - 1) / 2), cp.ceil(cp.max(X[:]) - (N[1] - 1) / 2)])
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

    imgTransformed = map_coordinates(img,
                                     coordinates=[(Y + (0.5 * (N[0] - 1) + padPre[0])).ravel(),
                                                  (X + (0.5 * (N[1] - 1) + padPre[1])).ravel()],
                                     order=order,
                                     mode='nearest').reshape(N)

    return imgTransformed

def fadeoutImageCosine(img, transitionLength=None, windowShift=None, fadeToVal=1):
    #img = cp.array(img)  # Convert image to CuPy array
    imageSize = cp.shape(img)  # Size of the image

    # Transition length is taken as 1/8 of the image's aspect length if not assigned
    if transitionLength is None:
        transitionLengthy = cp.ceil(cp.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLengthx = cp.ceil(cp.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLength = cp.asarray([transitionLengthy, transitionLengthx])
    else:
        transitionLength = cp.asarray([transitionLength[0], transitionLength[1]])

    if windowShift is None:
        windowShift = cp.array([0, 0])
    else:
        windowShift = cp.array(windowShift)

    # Target value of the fadeout is taken as mean value of the image if not assigned
    if fadeToVal is None:
        fadeToVal = img.mean()
        fadeToVal = fadeToVal.astype(cp.float32)

    X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)
    X = cp.array(X, dtype=cp.float32)
    Y = cp.array(Y, dtype=cp.float32)

    # Shift center of the fading window
    Y = Y + windowShift[0]
    X = X - windowShift[1]

    eps = 0.000000000000001  # Some small number to prevent division by 0
    X = cp.minimum(
        (cp.pi / (transitionLength[1] + eps)) * cp.maximum(cp.abs(X) - (imageSize[1] / 2 - transitionLength[1]), 0),
        cp.pi)
    Y = cp.minimum(
        (cp.pi / (transitionLength[0] + eps)) * cp.maximum(cp.abs(Y) - (imageSize[0] / 2 - transitionLength[0]), 0),
        cp.pi)
    X = cp.array(X, dtype=cp.float32)
    Y = cp.array(Y, dtype=cp.float32)
    transitionMask = (0.25 * (1 + cp.cos(X))) * (1 + cp.cos(Y))

    # Superimpose constant and input images weighted with the transition mask.
    imgFaded = transitionMask * img + (fadeToVal * (1 - transitionMask)) * cp.ones(imageSize, dtype=cp.float32)

    return imgFaded, transitionMask

def fadeoutImageEllipse(img, fadeMethod, ellipseSize, transitionLength, windowShift, numSegments, angularOffsetSegments,
                        fadeToVal, bottomApply):
    #img = cp.array(img, dtype=cp.float32)
    imageSize = cp.shape(img)

    # Transition length is taken as 1/8 of the image's aspect length if not assigned
    if transitionLength is None:
        transitionLengthy = cp.ceil(cp.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLengthx = cp.ceil(cp.mean([img.shape[0], img.shape[1]]) / 8)
        transitionLength = cp.asarray([transitionLengthy, transitionLengthx])
    else:
        transitionLength = cp.asarray([transitionLength[0], transitionLength[1]])

    if windowShift is None:
        windowShift = cp.array([0, 0])
    else:
        windowShift = cp.array(windowShift)

    if ellipseSize is None:
        ellipseSize = cp.array([0.8, 0.8])
    else:
        ellipseSize = cp.array(ellipseSize)

    if numSegments is None:
        numSegments = 1

    if angularOffsetSegments is None:
        angularOffsetSegments = 0

    # Initialize these so errors aren't thrown in if statements
    idxBoundary = None
    transitionMask = None

    if fadeMethod == 'ellipse':
        X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)
        Y = Y + windowShift[0]
        X = X - windowShift[1]
        ry = (ellipseSize[0] * imageSize[0] - 2) / 2
        rx = (ellipseSize[1] * imageSize[1] - 2) / 2
        idxEllipse = (X ** 2 / rx ** 2 + Y ** 2 / ry ** 2 < 1)
        rxInner = rx - transitionLength[1]
        ryInner = ry - transitionLength[0]
        idxBoundary = cp.logical_xor(idxEllipse, (X ** 2 / rxInner ** 2 + Y ** 2 / ryInner ** 2 < 1))
        transitionMask = cp.where(idxEllipse == True, 1, 0)

    if fadeMethod == 'rectangle':
        ry = int(cp.ceil(ellipseSize[0] * imageSize[0]))
        rx = int(cp.ceil(ellipseSize[1] * imageSize[1]))
        idxRectangle = cp.ones((ry, rx)).astype(int)
        idxRectangle = padToSize(idxRectangle, outputSize=imageSize, padMethod='constant', padType='both', padValue=0)
        idxRectangle = cp.roll(idxRectangle, shift=windowShift * cp.array([-1, 1]), axis=(0, 1))
        rxInner = int(rx - transitionLength[1])
        ryInner = int(ry - transitionLength[0])
        idxInnerRectangle = cp.ones((ryInner, rxInner)).astype(int)
        idxInnerRectangle = padToSize(idxInnerRectangle, outputSize=imageSize, padMethod='constant', padType='both',
                                      padValue=0)
        idxInnerRectangle = cp.roll(idxInnerRectangle, shift=windowShift * cp.array([-1, 1]), axis=(0, 1))
        idxBoundary = cp.logical_xor(idxRectangle, idxInnerRectangle)
        transitionMask = cp.where(idxRectangle == True, 1, 0)

    if fadeToVal:
        fadeToVals = cp.array(fadeToVal, dtype=cp.float32)
    elif numSegments > 1:
        X, Y = centeredGrid(imageSize, dx=[1], computeMeshgrid=True)
        Y = Y - windowShift[0]
        X = X - windowShift[1]

        def cart2pol(x, y):
            rho_ = cp.sqrt(x ** 2 + y ** 2)
            theta_ = cp.arctan2(y, x)
            return rho_, theta_

        rho, theta = cart2pol(X, Y)
        theta = cp.mod(theta + angularOffsetSegments, 2 * cp.pi)
        segmentIndices = cp.minimum((cp.floor((numSegments / (2 * cp.pi)) * theta) + 1).astype(int), numSegments)
        imFilt = gaussian_filter(img, sigma=10 / 2.35, truncate=2)
        fadeToVals = cp.zeros_like(img)
        idxBoundary = idxBoundary
        for segment in range(1, numSegments + 1):
            inSegment = cp.where(segment == segmentIndices[:], True, False)
            fadeToVals[inSegment] = cp.mean(imFilt[idxBoundary & inSegment])
        fadeToVals = gaussian_filter(fadeToVals, sigma=(transitionLength[0] / 2.35, transitionLength[1] / 2.35),
                                      truncate=2)
    else: 
        fadeToVals = cp.mean(img[idxBoundary])

    if transitionLength[0] > 1 and transitionLength[1] > 1:
        transitionMask = cp.array(transitionMask, dtype=cp.float32)
        transitionMask = gaussian_filter(transitionMask, sigma=(transitionLength[0] / 2.35, transitionLength[1] / 2.35),
                                          truncate=2)

    # new
    if bottomApply:
        # Create a mask for the bottom half of the image with smooth transition
        # transition_region_height = transitionLength[0].item() if isinstance(transitionLength[0], torch.Tensor) else \
        # transitionLength[0]
        transition_region_height = 10
        bottom_half_mask = cp.zeros_like(img)
        bottom_half_mask[int((img.shape[0]) / 1.95) + transition_region_height:, :] = 1
        transition_region = cp.linspace(0, 1, transition_region_height)
        for i in range(transition_region_height):
            bottom_half_mask[int((img.shape[0]) / 1.95) + i, :] = transition_region[i]
        imgFaded = img * (1 - bottom_half_mask) + (img * transitionMask + fadeToVals * (1 - transitionMask)) * bottom_half_mask
    else:
        # Convex combination of transition masks
        # Apply fade-out effect using fadeToVals as target
        imgFaded = img * transitionMask + fadeToVals * (1 - transitionMask)  # original

    # original
    #imgFaded = img * transitionMask + fadeToVals * (1 - transitionMask)
    imgFaded[cp.isnan(imgFaded)] = 1
    imgFaded[cp.isinf(imgFaded)] = 1

    return imgFaded, transitionMask

def fadeoutImage(img, fadeMethod=None, fadeToVal=None, transitionLength=None, ellipseSize=None, numSegments=None,
                 angularOffsetSegments=None, windowShift=None, bottomApply=False):
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
        raise TypeError('Invalid value for fadeMethod. Choices are cosine, ellipse, or rectangle.')

def angularAverage(im):
    N = cp.array(im.shape)
    ndim = cp.ndim(N)
    center = cp.floor(N / 2).astype(int) + 1
    x = cp.zeros_like(im)
    y = cp.zeros_like(im)

    for jj in range(ndim):
        x += (cp.arange(1, N[jj] + 1) - center[jj]).flatten().reshape(N[1], 1)
        y += (cp.arange(1, N[jj] + 1) - center[jj]).flatten().reshape(1, N[0])

    r = cp.sqrt(x ** 2 + y ** 2)
    lower_shell_idx = cp.round(r + 0.5 - 1e-10).astype(int)
    upper_shell_idx = lower_shell_idx + 1
    dist_to_upper_shell = lower_shell_idx - r
    lower_shell_idx = cp.maximum(lower_shell_idx, 1)
    n_shells = cp.bincount(upper_shell_idx.ravel(), weights=(1 - dist_to_upper_shell).flatten())
    n_shells += cp.bincount(lower_shell_idx.ravel(), weights=dist_to_upper_shell.ravel(), minlength=len(n_shells))
    n_shells = n_shells[1:]

    averages = (cp.bincount(lower_shell_idx.ravel(), (dist_to_upper_shell * im).ravel(),
                            minlength=len(n_shells) + 1)[1:] +
                cp.bincount(upper_shell_idx.ravel(), ((1 - dist_to_upper_shell) * im).ravel(),
                            minlength=len(n_shells))[1:]) / n_shells

    num_shells = int(cp.ceil(cp.min(N) / 2))
    averages = averages[:num_shells]
    radii = cp.arange(num_shells)

    return radii, averages


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
    mask: 2D cupy array
        The circular mask with values smoothly transitioning from 1 inside the circle to 0 outside.
    """
    radius = percentage * (size / 2)  # Convert percentage to radius
    y, x = cp.ogrid[:size, :size]
    center = (size // 2, size // 2)
    distance_from_center = cp.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a circular mask with a smooth transition
    mask = cp.ones((size, size))
    mask[distance_from_center > radius + smooth_pixels] = 0
    transition_zone = (distance_from_center >= radius) & (distance_from_center <= radius + smooth_pixels)
    mask[transition_zone] = cp.cos((distance_from_center[transition_zone] - radius) * (cp.pi / (2 * smooth_pixels)))

    return mask


def smooth2D_3x3(img):
    Ny, Nx = img.shape
    f = cp.zeros((Ny, Nx))

    f[Ny//2, Nx//2] = 4
    f[Ny//2 - 1, Nx//2] = 1
    f[Ny//2 + 1, Nx//2] = 1
    f[Ny//2 - 1, Nx//2 - 1] = 1
    f[Ny//2, Nx//2 - 1] = 1
    f[Ny//2 + 1, Nx//2 - 1] = 1
    f[Ny//2 - 1, Nx//2 + 1] = 1
    f[Ny//2, Nx//2 + 1] = 1
    f[Ny//2 + 1, Nx//2 + 1] = 1

    out = 1/12 * cp.abs(cp.fft.ifft2(cp.fft.fft2(img) * cp.fft.fft2(f)))
    return out

def smooth2D_5x5(img):
    Ny, Nx = cp.shape(img)
    f = cp.zeros((Ny, Nx), dtype=complex)

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

    out = 1 / 28 * cp.abs(IFFT(FFT(img) * FFT(f)))
    return out

def binArray(data, axis, binStep, binSize, func=cp.nanmean):
    data = cp.array(data)
    dims = cp.array(data.shape)
    argDims = cp.arange(data.ndim)
    argDims[0], argDims[axis] = argDims[axis], argDims[0]
    data = data.transpose(argDims)
    data = [func(cp.take(data, cp.arange(int(i * binStep), int(i * binStep + binSize)), 0), 0)
            for i in cp.arange(dims[axis] // binStep)]
    data = cp.array(data).transpose(argDims)
    return data

def rescaleImgToCustomCoord(img, imageSize, extentx, extenty, Nx, Ny, padVal):
    img = cp.array(img, dtype=cp.float32)
    numImgPixelsWidth, numImgPixelsHeight = img.shape
    print('Original image height and width in pixels:', (numImgPixelsHeight, numImgPixelsWidth))
    if imageSize is not None:
        newNumImgPixelsHeight, newNumImgPixelsWidth = int(cp.round(imageSize / extenty * Ny)), \
                                                      int(cp.round(imageSize / extentx * Nx))
        print('New height and width in pixels:', (newNumImgPixelsHeight, newNumImgPixelsWidth))
    else:
        newNumImgPixelsHeight, newNumImgPixelsWidth = Ny, Nx
        print('Height and width in pixels remains the same. The image will fill the entire aperture plane')

    newShape = (newNumImgPixelsWidth, newNumImgPixelsHeight)
    img = cp.asarray(cp.asnumpy(img))  # Convert back to NumPy array for resizing
    img = cv2.resize(img, dsize=newShape, interpolation=cv2.INTER_CUBIC)
    result = padToSize(img, outputSize=[Ny, Nx], padMethod='constant', padType='both', padValue=padVal)
    return result

def signalToNoise(img, axis=0, ddof=0):
    img = cp.asanyarray(img)
    mu = img.mean(axis)
    sd = img.std(axis=axis, ddof=ddof)
    return cp.where(sd == 0, 0, mu / sd)

def Gaussian(Ny, Nx, A, sigma):
    gau = cp.zeros((Ny, Nx))
    for i in range(0, Ny):
        for j in range(0, Nx):
            xx = i - Ny // 2
            yy = j - Nx // 2
            gau[i, j] = A * cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return gau

def phaseUnwrapping(img):
    """Unwraps phase"""
    Ny, Nx = cp.shape(img)
    x = cp.arange(Ny) - Ny / 2
    y = cp.arange(Nx) - Nx / 2
    X, Y = cp.meshgrid(x, y, indexing='ij')
    f = X ** 2 + Y ** 2
    a = IFFT(FFT(cp.cos(img)*IFFT(FFT(cp.sin(img))*f))/(f+0.000001))
    b = IFFT(FFT(cp.sin(img)*IFFT(FFT(cp.cos(img))*f))/(f+0.000001))
    out = cp.real(a - b) # was np.real(a-b)
    return out


def extend_horizontally(image, mask, percentage_split=1):
    """
    The purpose of this function is to use a mask to find corresponding edges of a diffraction pattern
    confined to a circular aperture and extend the edges horizontally to fill the computational domain.
    The extension is applied only to the top percentage of the mask, based on the percentage_split parameter.

    :param image: Image to be extended horizontally such that it fills the computational domain
    :param mask: Circular mask used to find object edges confined to a circular aperture
    :param percentage_split: The percentage (0-1) of the mask height where the horizontal extension is applied (default 0.5)
    :return: Extended diffraction pattern
    """

    height, width = image.shape
    new_image = cp.copy(image)

    # Find the last row where the mask has non-zero values
    last_mask_row = cp.max(cp.where(cp.any(mask == 1, axis=1))[0])

    # Determine the row where the horizontal extension stops
    split_row = int(last_mask_row * percentage_split)

    # Find the leftmost non-zero pixel for each row in the top percentage of the mask
    left_indices = cp.argmax(mask[:split_row, :], axis=1)
    left_indices = cp.where(cp.any(mask[:split_row, :], axis=1), left_indices, 0)

    # Find the rightmost non-zero pixel for each row in the top percentage of the mask
    right_indices = width - cp.argmax(mask[:split_row, ::-1], axis=1) - 1
    right_indices = cp.where(cp.any(mask[:split_row, :], axis=1), right_indices, width - 1)

    # Calculate the average of the leftmost and rightmost values for each row in the top percentage
    left_values = new_image[cp.arange(split_row), left_indices]
    right_values = new_image[cp.arange(split_row), right_indices]
    avg_values = (left_values + right_values) / 2

    # Create a matrix with the averaged values propagated across the rows in the top percentage
    avg_propagated = cp.tile(avg_values, (width, 1)).T

    # Create masks for the areas to the left of the leftmost and to the right of the rightmost non-zero pixels in the top percentage
    left_mask = cp.arange(width) < left_indices[:, None]
    right_mask = cp.arange(width) > right_indices[:, None]

    # Update the top portion of the image symmetrically with the averaged values
    new_image[:split_row, :] = cp.where(left_mask | right_mask, avg_propagated, new_image[:split_row, :])

    return new_image

#def extend_horizontally(image, mask):
#    height, width = image.shape
#    new_image = cp.copy(image)
#
#    # Find the leftmost non-zero pixel for each row
#    left_indices = cp.argmax(mask, axis=1)
#    left_indices = cp.where(cp.any(mask, axis=1), left_indices, 0)  # Ensure rows with no non-zero values are handled
#
#    # Create an array of the leftmost values propagated to the left
#    left_propagated = cp.tile(new_image[cp.arange(height), left_indices], (width, 1)).T
#    left_mask = cp.arange(width) < left_indices[:, None]
#    new_image = cp.where(left_mask, left_propagated, new_image)
#
#    # Find the rightmost non-zero pixel for each row
#    right_indices = width - cp.argmax(mask[:, ::-1], axis=1) - 1
#    right_indices = cp.where(cp.any(mask, axis=1), right_indices, width - 1)  # Ensure rows with no non-zero values are handled
#
#    # Create an array of the rightmost values propagated to the right
#    right_propagated = cp.tile(new_image[cp.arange(height), right_indices], (width, 1)).T
#    right_mask = cp.arange(width) > right_indices[:, None]
#    new_image = cp.where(right_mask, right_propagated, new_image)
#
#    return new_image


#def phaseUnwrapping(img):
#    """Phase unwrapping function"""
#    Ny, Nx = cp.shape(img)
#    y, x = cp.mgrid[:Ny, :Nx]
#    f = (x - Nx/2)**2 + (y - Ny/2)**2
#    eps = 0.000001
#    a = IFFT(FFT(cp.cos(img) * IFFT(FFT(cp.sin(img)) * f)) / (f + eps))
#    b = IFFT(FFT(cp.sin(img) * IFFT(FFT(cp.cos(img)) * f)) / (f + eps))
#    outPhase = cp.real(a - b)
#    return outPhase

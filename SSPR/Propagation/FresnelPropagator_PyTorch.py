"""Class implementation for Fresnel-propagation and back-propagation of images"""

import torch
from SSPR.Propagation.fresnel_propagation_kernel_PyTorch import fresnel_propagation_kernel
from SSPR.utilities_PyTorch import FFT, IFFT, padToSize, croppedArray


class FresnelPropagator:
    """Fresnel propagation for forward and backward"""
    def __init__(self, size_in, fresnel_number, size_pad=None, size_out=None, pad_method='replicate',
                 prop_method='fourier'):

        self.device = torch.device("cpu")
        self.size_in = torch.tensor(size_in)
        self.fresnel_number = fresnel_number
        if isinstance(self.fresnel_number, (int, float)):
            self.fresnel_number = torch.tensor(self.fresnel_number, dtype=torch.float32).view(-1, 1)

        self.size_pad = torch.tensor(size_pad) if size_pad is not None else size_pad
        self.size_out = torch.tensor(size_out) if size_out is not None else size_out
        self.pad_method = pad_method
        self.prop_method = prop_method
        self.num_dims = len(self.size_in)  # self.num_dims = np.ndim(self.size_in)

        if self.size_pad is None:
            self.size_pad = self.size_in
        elif len(self.size_pad) != len(self.size_in) or any(self.size_pad < self.size_in):
            raise ValueError('size_pad must be of the same length as size_in AND size_pad >= size_in.')

        if self.size_out is None:
            self.size_out = self.size_pad
        elif len(self.size_out) != len(self.size_in) or any(self.size_out > self.size_pad):
            raise ValueError('size_out must be of the same length as size_in AND size_pad >= size_out.')

        if self.fresnel_number.shape[-1] > 1 and self.fresnel_number.shape[-1] != self.num_dims:
            raise ValueError('fresnel_numbers must be 1 or equal to the number of dimensions size_in')

        # Determine padding and cropping amounts, if any
        self.pad_pre = torch.ceil((self.size_pad - self.size_in) / 2).to(torch.int)
        self.pad_post = torch.floor((self.size_pad - self.size_in) / 2).to(torch.int)
        self.crop_pre = torch.ceil((self.size_pad - self.size_out) / 2).to(torch.int)
        self.crop_post = torch.floor((self.size_pad - self.size_out) / 2).to(torch.int)

        # Construct propagation kernel
        self.prop_kernel = fresnel_propagation_kernel(Ny=self.size_pad[-2],
                                                      Nx=self.size_pad[-1],
                                                      fresnel_number=self.fresnel_number,
                                                      prop_method=self.prop_method).to(self.device)

    def forward_propagate(self, im):
        """Propagate Fresnel-propagates a given image according to the internal parameters
        (Fresnel-numbers, padding, etc) of the Fresnel-propagator object.

        :param im: Real space object to forward propagate to detector plane
        :return: Forward propagated image
        """

        N = im.shape

        # Optional padding
        if any(self.size_pad != self.size_in):
            im = padToSize(im,
                           outputSize=[int(self.pad_pre[-2] + N[-2]),
                                       int(self.pad_pre[-1] + N[-1])],
                           padMethod=self.pad_method,
                           padType='pre',
                           padValue=None)

            im = padToSize(im,
                            outputSize=[int(self.pad_pre[-2] + self.pad_post[-2] + N[-2]),
                                        int(self.pad_pre[-1] + self.pad_post[-1] + N[-1])],
                            padMethod=self.pad_method,
                            padType='post',
                            padValue=None)

        # Forward propagation step
        im_forward = IFFT(FFT(im) * self.prop_kernel)

        # Optional cropping
        if any(self.size_pad != self.size_out):
            im_forward = croppedArray(im_forward, crop_pre=self.crop_pre, crop_post=self.crop_post)

        return im_forward

    def back_propagate(self, im):
        """
        :param im: Fourier space object to backward propagate to sample plane
        :return: Backward propagated image
        """

        N = im.shape

        if any(self.size_pad != self.size_out):
            im = padToSize(im,
                           outputSize=[int(self.crop_pre[-2] + N[-2]),
                                       int(self.crop_pre[-1] + N[-1])],
                           padMethod=self.pad_method,
                           padType='pre',
                           padValue=None)

            im = padToSize(im,
                           outputSize=[int(self.crop_pre[-2] + self.crop_post[-2] + N[-2]),
                                       int(self.crop_pre[-1] + self.crop_post[-1] + N[-1])],
                           padMethod=self.pad_method,
                           padType='post',
                           padValue=None)

        # Backward propagation step
        im_backward = IFFT(FFT(im) * torch.conj(self.prop_kernel))

        if any(self.size_pad != self.size_in):
            im_backward = croppedArray(im_backward, crop_pre=self.pad_pre, crop_post=self.pad_post)

        return im_backward
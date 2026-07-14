"""

The purpose of this code is to make the propagator kernel (exponential function) for propagating an object

Code converted and modified from:
https://gitlab.gwdg.de/irp/holotomotoolbox/-/blob/master/functions/fresnelPropagation/fresnelPropagationKernel.m?ref_type=heads

"""
import numpy as np
from SSPR.utilities import FFT


def fresnel_propagation_kernel_fourier(Ny, Nx, fresnel_number):
    """ This is the standard propagator variant

    See:
    1. "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences"
        by David G. Voelz
    2. "Computational Fourier Optics" by David G. Voelz. Applied here is the "transfer function approach (TF)" where
        Voelz writes it as H(Fx, Fy) = exp(ikz)exp(-i*pi*lam*z(Fx^2+Fy^2)). Equivalently, this can be written as
        H(kx, ky) = exp(ikz)exp(-i/(4*pi*Fr)*(kx^2+ky^2)). exp(ikz) is ignored since it is just some constant. kx and
        ky are defined here as 2*pi/Nx and 2*pi/Ny multiplied by the values at each point in the array. We divide by
        Δx^2 since it was not included in the spacing for the fftfreq function. Thus, we can rewrite the variables in
        terms of the Fresnel number. Fr is defined as Fr=Δx^2/(z*lam), but is modified if we are in cone beam geometry
        to Fr=Δx_eff^2/(z_eff*lam)

    The regime in which this propagator is valid depends on this condition: Δx >= z*lam / L --> Δx >= z*lam / (N*dx),
    where Δx is the pixel size, z is the propagation distance, and lam is the wavelength

    :param Nx: Length of the object in the x dimension
    :param Ny: Length of the object in the y dimension
    :param fresnel_number: Single Fresnel number used for propagation
    :return: Transfer function (TF) propagation kernel
    """

    condition = 1.0 / (Nx * fresnel_number)  # Fresnel number is in terms of the effective pixel size (smallest feature)

    if condition <= 1.0:
        # fftfreq automatically does an ifftshift, so take fftshift to put 0 in the center and take the normal FFT/IFFT.
        # Another option is to leave off the fftshift here and use the  normal numpy fft2 and ifft2 functions (might be
        # better computationally and saves time? For clarity, I will do the first option.)
        # fftfreq is the same as: 2*np.pi*np.fft.ifftshift(np.arange(np.ceil(-Nx/2), np.ceil(Ny/2), 1))/(N*dx)

        KX = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=1))  # Spacing in Fourier space
        KY = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=1))  # Spacing in Fourier space
        kx, ky = np.meshgrid(KX, KY)
        kernel = np.exp((-1j / (4 * np.pi * fresnel_number)) * (kx**2 + ky**2))
    else:
        raise ValueError("Not oversampled enough for this propagator. Use another method.")
    return kernel

def fresnel_propagation_kernel_chirp(Ny, Nx, fresnel_number):
    """This is a real space convolution-kernel

    See:
    1. "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences"
        by David G. Voelz
    2. "Computational Fourier Optics" by David G. Voelz. Applied here is the "Impulse Response approach (IR)" where
        Voelz writes it as h(x, y) = exp(ikz)*-i/(lam*z)*exp(i*k/(2*z)*(x^2+y^2)). Equivalently, this can be written as
        h(x, y) = exp(ikz)*-i/(lam*z)*exp(i*pi*Fr*(x^2+y^2)). exp(ikz) is ignored since it is just some constant. Δx^2
        comes out of the x^2 and y^2 since it was not included in the sample spacing in the fftfreq function. Thus, we
        can rewrite the variables in terms of the Fresnel number.

    The regime in which this propagator is valid depends on this condition: Δx <= z*lam / L ---> Δx <= z*lam / (N*dx),
    where Δx is the pixel size, z is the propagation distance, and lam is the wavelength

    :param Nx: Length of the object in the x dimension
    :param Ny: Length of the object in the y dimension
    :param fresnel_number: Single Fresnel number used for propagation
    :return: Impulse response (IR) propagation kernel (chirp)
    """

    condition = 1.0 / (Nx * fresnel_number)  # Fresnel number is in terms of the effective pixel size (smallest feature)

    if condition >= 1.0:
        # fftfreq automatically does an ifftshift, so take fftshift to put 0 in the center and take the normal FFT/IFFT.
        # Another option is to leave off the fftshift here and use the  normal numpy fft2 and ifft2 functions (might be
        # better computationally and saves time? For clarity, I will do the first option.)
        # fftfreq is the same as: 2*np.pi*np.fft.ifftshift(np.arange(np.ceil(-Nx/2), np.ceil(Ny/2), 1))/(N*dx)
        X = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=2 * np.pi / Nx))  # Spacing in real space
        Y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=2 * np.pi / Ny))  # Spacing in real space
        x, y = np.meshgrid(X, Y)
        pre_factor = -1j * np.abs(fresnel_number) * np.sign(fresnel_number)
        kernel = pre_factor * np.exp((1j * np.pi * fresnel_number) * (x ** 2 + y ** 2))
        kernel = FFT(kernel)
    else:
        raise ValueError("Not oversampled enough for this propagator. Use another method.")
    return kernel

def fresnel_propagation_kernel_chirpLimited(Ny, Nx, fresnel_number):
    """Same as chirp, but with an additional modification that limits the scattering angle of image features to
    a physically correct range defined by Abbe's criterion. In particular, this suppresses the artifacts that typically
    arise from the 'chirp' method at larger Fresnel numbers.


    See:
    1. "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria
    and consequences" by David G. Voelz
    2. "Computational Fourier Optics" by David G. Voelz
    3. "Time-resolved X-ray phase-contrast tomography" by Aike Ruhlandt (2017)

    OF = np.cos(np.pi / 2 * np.minimum(np.arctan2(np.sqrt(x**2+y**2), z) / np.arcsin(lam / (2*Δx)), 1)**2).
    The OF (obliquity factor) above is the equivalent statement to the one in this code, which is defined as
    OF = cos(pi/2 * minimum(4 * Fr^2 * (x^2 + y^2), 1). Not sure where this was obtained/derived, but it works.

    :param Nx: Length of the object in the x dimension
    :param Ny: Length of the object in the y dimension
    :param fresnel_number: Single Fresnel number used for propagation
    :return: ChirpLimited propagation kernel
    """
    # fftfreq automatically does an ifftshift, so take fftshift to put 0 in the center and take the normal FFT/IFFT.
    # Another option is to leave off the fftshift here and use the  normal numpy fft2 and ifft2 functions (might be
    # better computationally and saves time? For clarity, I will do the first option.)
    # fftfreq is the same as: 2*np.pi*np.fft.ifftshift(np.arange(np.ceil(-Nx/2), np.ceil(Ny/2), 1))/(N*dx)
    X = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=2 * np.pi / Nx))  # Spacing in real space
    Y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=2 * np.pi / Ny))  # Spacing in real space
    x, y = np.meshgrid(X, Y)
    pre_factor = -1j * np.abs(fresnel_number) * np.sign(fresnel_number)
    kernel = pre_factor * np.exp((1j * np.pi * fresnel_number) * (x ** 2 + y ** 2))
    # Multiply by obliquity factor as proposed in Aike Ruhlandts PhD-thesis (2017)
    theta_div_theta_max = 4 * fresnel_number ** 2 * (x**2 + y**2)
    obliquity_factor = np.cos((np.pi / 2) * np.minimum(theta_div_theta_max, 1))
    kernel = kernel * obliquity_factor
    kernel = FFT(kernel)
    return kernel

def fresnel_propagation_kernel(Ny, Nx, fresnel_number, prop_method):
    """Propagation methods"""

    if prop_method == 'fourier':
        return fresnel_propagation_kernel_fourier(Ny, Nx, fresnel_number)
    elif prop_method == 'chirp':
        return fresnel_propagation_kernel_chirp(Ny, Nx, fresnel_number)
    elif prop_method == 'chirpLimited':
        return fresnel_propagation_kernel_chirpLimited(Ny, Nx, fresnel_number)
    else:
        raise ValueError('Invalid value of the argument method. Admissible choices: '
                             '\'fourier\', \'chirp\', and \'chirpLimited\'.')

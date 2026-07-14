"""Simulate a hologram for x-ray imaging"""

import torch
from SSPR.Propagation.FresnelPropagator_PyTorch import FresnelPropagator


def simulate_hologram(fresnel_number, phase_image, beam=None, absorption_image=None, size_pad=None, size_out=None,
                      beta_over_delta_ratio=0, pad_method='replicate', prop_method='fourier', simulate_linear=False):
    """

    beta_delta_ratio : Default = 0
    Fixed ratio between absorption and phase shifts.

    phase_image : Numerical array
    Numerical array (2D image) used to simulate the projected phase shift of an object defined as the negative integral
    over the phase shifting component delta of the refractive index along the direction of propagation through the
    object; only 2D arrays with negative values are accepted

    absorption_image : Default = None
    Numerical array (2D image) used to simulate the projected absorption of an object defined as the negative integral
    over the absorption component beta of the refractive index along the direction of propagation through the object.
    Only 2D arrays with positive values are accepted. If left empty, the projected absorption is calculated by
    beta_delta_ratio * phase_image.

    simulate_linear : Default = False
    Simulate using the linear or nonlinear model of x-ray phase-contrast imaging
"""

    fresnel_number = fresnel_number.view(-1, 1)

    if beam is None:
        beam = torch.ones_like(phase_image)

    # Make sure that phaseImage only contains negative values
    if len(phase_image[phase_image > 0]) > 0:
        raise ValueError('Positive values exist. Please use only images containing negative values for phase_image.')

    # Construct mixed phase- and absorption-image
    if absorption_image is None:
        phase_abs_image = (beta_over_delta_ratio + 1j) * phase_image
    else:
        # Make sure that absorption_image only contains positive values
        if len(absorption_image[absorption_image < 0]) > 0:
            raise ValueError('Negative values exist. Please use only images containing positive values for '
                             'absorption_image.')

        phase_abs_image = 1j * phase_image - absorption_image

    # Set up propagator
    fp = FresnelPropagator(size_in=phase_image.shape,
                           fresnel_number=fresnel_number,
                           size_pad=size_pad,
                           size_out=size_out,
                           pad_method=pad_method,
                           prop_method=prop_method)

    # Simulate hologram using the linear or nonlinear model of X-ray phase contrast
    if simulate_linear:
        hologram = 1 + 2 * torch.real(fp.forward_propagate(beam * phase_abs_image))
    else:
        hologram = torch.abs(fp.forward_propagate(beam * torch.exp(phase_abs_image))) ** 2

    return hologram, phase_abs_image

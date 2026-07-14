
import cupy as cp
from typing import List

from parameters.Measurement import Measurement
from models.ConeBeam import ConeBeam


class FresnelPropagator():

    def __init__(self, measurements: List[Measurement], beam_setup, data_shape, running_device):
        self.buffer = cp.ones(data_shape, dtype=cp.complex64)

        self.num_distances = len(measurements)
        self.fresnel_numbers = [ConeBeam.get_fr(beam_setup, measurements[distance]) for distance in
                                range(self.num_distances)]

        sample_grid = cp.meshgrid(cp.fft.fftfreq(data_shape[0]), cp.fft.fftfreq(data_shape[1]), indexing='ij')

        xi, eta = sample_grid

        kernel_func = lambda distance: (
            cp.exp((-1j * cp.pi) / self.fresnel_numbers[distance] * (xi * xi + eta * eta)).astype(cp.complex64))
        kernel_func_conj = lambda distance: (
            cp.exp((-1j * cp.pi) / (-self.fresnel_numbers[distance]) * (xi * xi + eta * eta)).astype(cp.complex64))

        self.fresnel_kernels = [kernel_func(distance) for distance in range(self.num_distances)]
        self.fresnel_kernels_conj = [kernel_func_conj(distance) for distance in range(self.num_distances)]


    def propagate_forward(self, x, distance):
        self.buffer = cp.array(x)
        x = cp.fft.ifft2(cp.fft.fft2(self.buffer) * self.fresnel_kernels[distance])
        return x

    def propagate_forward_all(self, x):
        return [self.propagate_forward(x, distance) for distance in range(self.num_distances)]

    def propagate_back(self, x, distance):
        self.buffer = cp.array(x)
        x = cp.fft.ifft2(cp.fft.fft2(self.buffer) * self.fresnel_kernels_conj[distance])
        return x

    def propagate_back_all(self, x):
        propagated = [self.propagate_back(x, distance) for distance in range(self.num_distances)]
        return propagated

    def get_measurements(self, x, distance):
        return cp.abs(self.propagate_forward(x, distance))

    def get_measurements_from_propagated_all(self, x):
        return [cp.abs(x[distance]).astype(x[distance].dtype) for distance in range(self.num_distances)]

    def get_measurements_all(self, x):
        return [self.get_measurements(x, distance) for distance in range(self.num_distances)]

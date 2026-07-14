
import cupy as cp
from typing import List
from copy import deepcopy

from parameters.BeamSetup import BeamSetup
from parameters.DataDimensions import DataDimensions
from parameters.Measurement import Measurement
from parameters.Padding import Padding
from preprocessing.process_data_dimensions import process_data_dimensions
from preprocessing.process_image import process_image


def blackman(x,width):
    return 0.42 - 0.5 * cp.cos(x * 2 * cp.pi / (width - 1)) + 0.08 * cp.cos(x * 4 * cp.pi / (width - 1))


def process_padding_options(measurements:List[Measurement], beam_setup:BeamSetup, data_dimensions:DataDimensions, padding_options:Padding):
    measurements = deepcopy(measurements)
    padding_options = deepcopy(padding_options)
    beam_setup = deepcopy(beam_setup)
    data_dimensions = deepcopy(data_dimensions)

    beam_setup = process_beam_setup(beam_setup,padding_options)
    data_dimensions = process_data_dimensions(data_dimensions, padding_options)

    for measurement in measurements:
        measurement.data = process_image(measurement.data, padding_options, data_dimensions)

    return measurements, beam_setup, data_dimensions

def process_beam_setup(beam_setup:BeamSetup, padding_options:Padding):
    beam_setup.px_size = beam_setup.px_size * padding_options.down_sampling_factor
    return beam_setup

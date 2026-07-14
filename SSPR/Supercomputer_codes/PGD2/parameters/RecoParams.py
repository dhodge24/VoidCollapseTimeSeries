
from typing import List

from parameters.BeamSetup import BeamSetup
from parameters.Measurement import Measurement
from parameters.Options import Options
from parameters.DataDimensions import DataDimensions


class RecoParams():
    def __init__(self, beam_setup: BeamSetup, measurements: List[Measurement],
                 reco_options: List[Options], data_dimensions: DataDimensions, output_path: str):
        self.beam_setup = beam_setup
        self.measurements = list(measurements)
        self.reco_options = list(reco_options)
        self.data_dimensions = data_dimensions
        self.output_path = output_path

    @property
    def beam_setup(self):
        return self._beam_setup

    @property
    def measurements(self):
        return self._measurements

    @property
    def reco_options(self):
        return self._reco_options

    @property
    def data_dimensions(self):
        return self._data_dimensions

    @property
    def output_path(self):
        return self._output_path

    @beam_setup.setter
    def beam_setup(self, beam_setup):
        if type(beam_setup) is BeamSetup:
            self._beam_setup = beam_setup
        else:
            raise TypeError("Expected BeamSetup but got ", type(beam_setup))

    @measurements.setter
    def measurements(self, measurements):
        if type(measurements) is list and len(measurements) > 0 and type(measurements[0]) is Measurement:
            self._measurements = measurements
        else:
            raise TypeError("Expected list[Measurements] but got ", type(measurements))

    @reco_options.setter
    def reco_options(self, reco_options):
        if type(reco_options) is list and len(reco_options) > 0 and type(reco_options[0]) is Options:
            self._reco_options = reco_options
        else:
            raise TypeError("Expected list[Options] but got ", type(reco_options))

    @data_dimensions.setter
    def data_dimensions(self, data_dimensions):
        if type(data_dimensions) is DataDimensions:
            self._data_dimensions = data_dimensions
        else:
            raise TypeError("Expected DataDimensions but got ", type(data_dimensions))

    @output_path.setter
    def output_path(self, output_path):
        if type(output_path) is str:
            self._output_path = output_path
        else:
            raise TypeError("Expected str but got ", type(output_path))

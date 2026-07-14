from parameters.BeamSetup import BeamSetup
from parameters.Measurement import Measurement


class ConeBeam(BeamSetup):
    def __init__(self):
        None

    @staticmethod
    def z12(setup:BeamSetup, measurement:Measurement):
        return setup.z02 - measurement.z01

    @staticmethod
    def get_fr(setup:BeamSetup, measurement:Measurement):
        z12 = ConeBeam.z12(setup, measurement)

        lam = 1240 / setup.energy * 1e-9
        M = (z12 + measurement.z01) / measurement.z01
        dx_eff = setup.px_size / M
        z_eff = z12 / M
        fr_eff = dx_eff ** 2 / lam / z_eff
        return fr_eff

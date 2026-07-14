
import cupy as cp


class BeamSetup():
    def __init__(self, energy, px_size, z02):
        self.energy = energy
        self.px_size = px_size
        self.z02 = z02

    @property
    def energy(self):
        return self._energy

    @property
    def px_size(self):
        return self._px_size

    @property
    def z02(self):
        return self._z02

    @energy.setter
    def energy(self, energy):
        if type(energy) is float or type(energy) is cp.float64:
            self._energy = energy
        elif type(energy) is int:
            self._energy = float(energy)
        else:
            raise TypeError("Expected float or int for energy but got ", type(energy))

    @px_size.setter
    def px_size(self,px_size):
        if type(px_size) is float or type(px_size) is cp.float64:
            self._px_size = px_size
        elif type(px_size):
            self._px_size = float(px_size)
        else:
            raise TypeError("Expected float or int for px_size but got ", type(px_size))

    @z02.setter
    def z02(self, z02):
        if type(z02) is float or type(z02) is cp.float64:
            self._z02 = z02
        elif type(z02) is int:
            self._z02 = float(z02)
        else:
            raise TypeError("Expected float or int for z02 but got ", type(z02))

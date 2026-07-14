
import cupy as cp

class Measurement():
    def __init__(self,
                 z01,
                 data_path,
                 data=None,):
        self.z01 = z01
        self.data = data
        self.data_path = data_path

    @property
    def data(self):
        return self._data

    @property
    def data_path(self):
        return self._data_path

    @property
    def z01(self):
        return self._z01

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
            return
        elif type(data) is cp.ndarray:
            self._data = data
        else:
            raise TypeError("Expected numpy array but got ", type(data))

    @data_path.setter
    def data_path(self, data_path):
        if type(data_path) is str:
            self._data_path = data_path
        else:
            raise TypeError("Expected string for data_path but got ", type(data_path))

    @z01.setter
    def z01(self, z01):
        if type(z01) is float or type(z01) is cp.float64:
            self._z01 = z01
        elif type(z01) is int:
            self._z01 = float(z01)
        else:
            raise TypeError("Expected float or int but got ", type(z01))

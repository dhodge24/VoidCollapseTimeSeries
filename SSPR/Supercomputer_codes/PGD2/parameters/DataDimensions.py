import cupy as cp

class DataDimensions():
    def __init__(self, total_size, fov_size, window_type, window=None):
        self.total_size = total_size
        self.fov_size = fov_size
        self.fov_range = [(0,fov_size[0]),(0,fov_size[1])]
        self.window_type = window_type
        self.window = window

    @property
    def total_size(self):
        return self._total_size

    @property
    def fov_size(self):
        return self._fov_size

    @property
    def fov_range_raw(self):
        return [(self._fov_range[0][0], self._fov_range[0][1]), (self._fov_range[1][0], self._fov_range[1][1])]

    @property
    def fov_range(self):
        return [slice(self._fov_range[0][0], self._fov_range[0][1]), slice(self._fov_range[1][0], self._fov_range[1][1])]

    @property
    def window_type(self):
        return self._window_type

    @property
    def window(self):
        return self._window

    @total_size.setter
    def total_size(self, total_size):
        if type(total_size) is tuple:
            self._total_size = total_size
        else:
            raise TypeError("Expected tuple for total_size, but got", type(total_size))

    @fov_size.setter
    def fov_size(self, fov_size):
        if type(fov_size) is tuple:
            self._fov_size = fov_size
        else:
            raise TypeError("Expected tuple for fov_size, but got", type(fov_size))

    @fov_range.setter
    def fov_range(self, fov_range):
        if (type(fov_range) is list
                and len(fov_range) == 2
                and type(fov_range[0]) is tuple and type(fov_range[1]) is tuple
                and len(fov_range[0]) == 2 and len(fov_range[1]) == 2):
            self._fov_range = fov_range
        elif type(fov_range) is tuple \
                and len(fov_range) == 2 \
                and type(fov_range[0]) is tuple and type(fov_range[1]) is tuple \
                and len(fov_range[0]) == 2 and len(fov_range[1]) == 2:
            self._fov_range = list(fov_range)
        else:
            print(fov_range)
            raise TypeError("Expected support in format of tuple list, e.g. [(1,2),(3,4)]")

    @window_type.setter
    def window_type(self, window_type):
        if type(window_type) is str:
            self._window_type = window_type
        else:
            raise TypeError("Expected string for window_type but got ", type(window_type))

    def window(self, window):
        if(window == None):
            self._window = None
        elif type(window) is cp.ndarray:
            self._window = window
        else:
            raise TypeError("Expected torch tensor for window but got ", type(window))


from enum import Enum

class Padding():
    class PaddingMode(Enum):
        UNDEFINED = 0
        NORMAL = 1
        MIRROR_LEFT = 2
        MIRROR_HORIZONTAL = 3
        MIRROR_ALL = 4
        REPETITION_LEFT = 5
        REPETITION_ALL = 6
        REPETITION_HORIZONTAL = 7

    def __init__(self,
                 padding_mode=PaddingMode.NORMAL,
                 padding_factor=2,
                 down_sampling_factor=1,
                 cutting_band=0,
                 i0=1.0,
                 prototype_field=None
                 ):
        self.padding_mode = padding_mode
        self.padding_factor = padding_factor
        self.down_sampling_factor = down_sampling_factor
        self.cutting_band = cutting_band
        self.i0 = i0
        self.prototype_field = prototype_field

    @property
    def padding_mode(self):
        return Padding.PaddingMode[self._padding_mode]

    @property
    def padding_factor(self):
        return self._padding_factor

    @property
    def down_sampling_factor(self):
        return self._down_sampling_factor

    @property
    def cutting_band(self):
        return self._cutting_band

    @property
    def i0(self):
        return self._i0

    @property
    def prototype_field(self):
        return self._prototype_field

    @padding_mode.setter
    def padding_mode(self, padding_mode):
        if type(padding_mode) is Padding.PaddingMode:
            self._padding_mode = padding_mode.name
        elif type(padding_mode) is int:
            self._padding_mode = Padding.PaddingMode(padding_mode).name
        elif type(padding_mode) is str:
            self._padding_mode = Padding.PaddingMode(padding_mode).name
        else:
            raise TypeError("Expected PaddingMode or int but got ", type(padding_mode))

    @padding_factor.setter
    def padding_factor(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected int or float but got {type(value)}")
        self._padding_factor = value

    @down_sampling_factor.setter
    def down_sampling_factor(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected int but got {type(value)}")
        self._down_sampling_factor = value

    @cutting_band.setter
    def cutting_band(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected int but got {type(value)}")
        self._cutting_band = value

    @i0.setter
    def i0(self, value):
        if not isinstance(value, float):
            raise TypeError(f"Expected float but got {type(value)}")
        self._i0 = value

    @prototype_field.setter
    def prototype_field(self, prototype_field):
        self._prototype_field = prototype_field

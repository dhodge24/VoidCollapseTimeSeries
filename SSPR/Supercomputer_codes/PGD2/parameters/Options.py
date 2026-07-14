
from parameters.Padding import Padding


class Options():
    def __init__(self,
                 iterations,
                 update_rate=0.0,
                 nesterov_momentum=0.0,
                 l2_weight_absorption=0.0,
                 omega_f_fwhm_phaseshift=0.0,
                 omega_f_fwhm_absorption=0.0,
                 omega_ne_fwhm=None,
                 padding=Padding(),
                 verbose_interval=100,
                 prototype_field=None
                 ):
        self.iterations = iterations
        self.update_rate = update_rate
        self.nesterov_momentum = nesterov_momentum
        self.l2_weight_absorption = l2_weight_absorption
        self.omega_f_fwhm_phaseshift = omega_f_fwhm_phaseshift
        self.omega_f_fwhm_absorption = omega_f_fwhm_absorption
        self.omega_ne_fwhm = omega_ne_fwhm
        self.padding = padding
        self.verbose_interval = verbose_interval
        self.prototype_field = prototype_field

    @property
    def iterations(self):
        return self._iterations

    @property
    def update_rate(self):
        return self._update_rate

    @property
    def nesterov_momentum(self):
        return self._nesterov_momentum

    @property
    def l2_weight_absorption(self):
        return self._l2_weight_absorption

    @property
    def omega_f_fwhm_phaseshift(self):
        return self._omega_f_fwhm_phaseshift

    @property
    def omega_f_fwhm_absorption(self):
        return self._omega_f_fwhm_absorption

    @property
    def omega_ne_fwhm(self):
        return self._omega_ne_fwhm

    @property
    def padding(self):
        return self._padding

    @property
    def verbose_interval(self):
        return self._verbose_interval

    @property
    def prototype_field(self):
        return self._prototype_field

    @iterations.setter
    def iterations(self, iterations):
        if type(iterations) is int:
            self._iterations = iterations
        else:
            raise TypeError("Expected int for iterations but got ", type(iterations))

    @update_rate.setter
    def update_rate(self, update_rate) -> None:
        if type(update_rate) is float or type(update_rate) is int:
            self._update_rate = update_rate
        else:
            raise TypeError("Expected float or int for update_rate but got ", type(update_rate))

    @nesterov_momentum.setter
    def nesterov_momentum(self, nesterov_momentum) -> None:
        if type(nesterov_momentum) is float or type(nesterov_momentum) is int:
            self._nesterov_momentum = nesterov_momentum
        else:
            raise TypeError("Expected float or int for nesterov_momentum but got ", type(nesterov_momentum))

    @l2_weight_absorption.setter
    def l2_weight_absorption(self, l2_weight_absorption):
        if type(l2_weight_absorption) is float or type(l2_weight_absorption) is int:
            self._l2_weight_absorption = l2_weight_absorption
        else:
            raise TypeError("Expected float or int for l2_weight_absorption but got ", type(l2_weight_absorption))

    @omega_f_fwhm_phaseshift.setter
    def omega_f_fwhm_phaseshift(self, omega_f_fwhm_phaseshift):
        if type(omega_f_fwhm_phaseshift) is float or type(omega_f_fwhm_phaseshift) is int:
            self._omega_f_fwhm_phaseshift = omega_f_fwhm_phaseshift
        else:
            raise TypeError("Expected float or int but got ", type(omega_f_fwhm_phaseshift))

    @omega_f_fwhm_absorption.setter
    def omega_f_fwhm_absorption(self, omega_f_fwhm_absorption):
        if type(omega_f_fwhm_absorption) is float or type(omega_f_fwhm_absorption) is int:
            self._omega_f_fwhm_absorption = omega_f_fwhm_absorption
        else:
            raise TypeError("Expected float or int for gauss_fwhm but got ", type(omega_f_fwhm_absorption))

    @omega_ne_fwhm.setter
    def omega_ne_fwhm(self, omega_ne_fwhm):
        if omega_ne_fwhm == None or type(omega_ne_fwhm) is int:
            if omega_ne_fwhm == 0:
                self._omega_ne_fwhm = None
            else:
                self._omega_ne_fwhm = omega_ne_fwhm
        else:
            raise TypeError("Expected int or None but got ", type(omega_ne_fwhm))

    @padding.setter
    def padding(self, padding):
        if isinstance(padding, Padding):
            self._padding = padding
        else:
            raise TypeError("Expected Padding instance but got ", type(padding))

    @verbose_interval.setter
    def verbose_interval(self, verbose_interval):
        if isinstance(verbose_interval, int) or verbose_interval is None:
            self._verbose_interval = verbose_interval
        else:
            raise TypeError("Expected integer but got ", type(verbose_interval))

    @prototype_field.setter
    def prototype_field(self, prototype_field):
        self._prototype_field = prototype_field

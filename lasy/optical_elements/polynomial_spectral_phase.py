from lasy.backend import xp

from .optical_element import OpticalElement
# Define symbol regardless, then overwrite if cupy
_exp_i_phase_fused = None

if xp.__name__ == "cupy":
    import cupy as cp

    @cp.fuse()
    def _exp_i_phase_fused(omega, omega0, delay, gdd, tod, fod):
        dw = omega - omega0
        phase = (((fod/24.0)*dw + (tod/6.0))*dw + (gdd/2.0))*dw + delay
        phase = phase * dw
        return cp.exp(1j * phase)

class PolynomialSpectralPhase(OpticalElement):
    r"""
    Class for an optical element that adds spectral phase (e.g. a dazzler).

    The amplitude multiplier corresponds to:

    .. math::

        T(\omega) = \exp(i(\phi(\omega)))

    where :math:`\phi(\omega)` is the spectral phase given by:

    .. math::

        \phi(\omega) = \text{delay} (\omega - \omega_0) + \frac{\text{GDD}}{2!} (\omega - \omega_0)^2 + \frac{\text{TOD}}{3!} (\omega - \omega_0)^3 + \frac{\text{FOD}}{4!} (\omega - \omega_0)^4

    The other parameters in this formula are defined below.

    Parameters
    ----------
    omega0 : float (in rad/s)
        Central angular frequency about which the polynomial is expanded
    delay : float (in s), optional
        Group delay (by default: ``delay=0``). Positive value delays the pulse, i.e. it arrives at a later time
    gdd : float (in s^2), optional
        Group Delay Dispersion (by default: ``gdd=0``). ``gdd > 0`` corresponds to a positive
        chirp, i.e. the low-frequency part of the spectrum arriving earlier than the
        high-frequency part of the spectrum.
    tod : float (in s^3), optional
        Third-order Dispersion (by default: ``tod=0``). For a Gaussian pulse, adding a positive
        TOD (``tod > 0``) results in the apparition of post-pulses, i.e. lower intensity pulses
        arriving after the main pulse.
    fod : float (in s^4), optional
        Fourth-order Dispersion (by default: ``fod=0``).
    """

    def __init__(self, omega0, delay=0, gdd=0, tod=0, fod=0):
        self.omega0 = omega0
        self.delay = delay
        self.gdd = gdd
        self.tod = tod
        self.fod = fod

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.


        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        if xp.__name__ == "cupy":
            dtype = omega.dtype
            return _exp_i_phase_fused(
                omega,
                dtype.type(self.omega0),
                dtype.type(self.delay),
                dtype.type(self.gdd),
                dtype.type(self.tod),
                dtype.type(self.fod),
                )    
        else:
               dw = omega - self.omega0
               phase = (((self.fod/24.0)*dw + (self.tod/6.0))*dw + (self.gdd/2.0))*dw + self.delay
               phase = phase * dw
               return xp.exp(1j * phase)

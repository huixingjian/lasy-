from lasy.backend import xp

from .transverse_profile import TransverseProfile


class SuperGaussianTransverseProfile(TransverseProfile):
    r"""
    Class for the analytic profile of a super-Gaussian laser pulse.

    More precisely, the transverse envelope corresponds to:

    .. math::

        \mathcal{T}(x, y) = \exp\left( -\left({\frac{{x^2 + y^2}}{w_0^2}}\right)^{\dfrac{n}{2}} \right)

    Parameters
    ----------
    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

    n_order : float (in meter)
        The shape parameter of the super-gaussian function, i.e. :math:`n` in the above formula.
        If :math:`n=2` the super-Gaussian becomes a standard Gaussian function.
        If :math:`n=1` the super-Gaussian becomes a Laplace function.
    """

    def __init__(self, w0, n_order):
        super().__init__()
        self.w0 = w0
        self.n_order = n_order

    def _evaluate(self, x, y):
        """
        Return the transverse envelope.

        Parameters
        ----------
        x, y : ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        p = (self.n_order) * (0.5)
        
        # allocate output (or reuse a preallocated buffer)
        envelope = xp.empty_like(x, dtype=x.dtype)          # float32 if x is float32
        
        xp.multiply(x, x, out=envelope)                     # envelope = x*x
        tmp = xp.empty_like(y)                              # one extra buffer
        xp.multiply(y, y, out=tmp)                          # tmp = y*y
        xp.add(envelope, tmp, out=envelope)                 # envelope = x^2 + y^2
        xp.divide(envelope, w0_2, out=envelope)             # envelope /= w0^2
        xp.power(envelope, p, out=envelope)                 # envelope = ( ... )^p
        xp.negative(envelope, out=envelope)                 # envelope = -envelope
        xp.exp(envelope, out=envelope)                      # envelope = exp(envelope)

        return envelope

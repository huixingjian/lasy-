from scipy.constants import c
from lasy.backend import xp
from .optical_element import OpticalElement

# Fused exp for CuPy
_exp_r2_k_fused = None
if xp.__name__ == "cupy":
    import cupy as cp

    @cp.fuse()
    def _exp_r2_k_fused(r2_3d, k_3d):
        return cp.exp((-1j) * r2_3d * k_3d)


class ChromaticLens(OpticalElement):
    def __init__(self, R1, R2, d, n_func):
        self.R1 = R1
        self.R2 = R2
        self.d = d
        self.n_func = n_func

    def amplitude_multiplier(self, x, y, omega):
        # x,y,omega are always 3D with shape (Nx, Ny, Nw)

        # 2D radius^2 from a single frequency slice (saves huge memory)
        x2d = x[:, :, 0]
        y2d = y[:, :, 0]
        r2 = x2d * x2d + y2d * y2d              # (Nx,Ny)

        # 1D omega axis from one spatial point
        omega_1d = omega[0, 0, :]               # (Nw,)

        # 1D chromatic focal length f(omega)
        lam = 2 * xp.pi * c / omega_1d * 1e6    # (Nw,) microns
        n = self.n_func(lam)                    # (Nw,)

        f = 1.0 / (
            (n - 1.0)
            * (1.0 / self.R1 - 1.0 / self.R2 + (n - 1.0) * self.d / (n * self.R1 * self.R2))
        )                                        # (Nw,)

        komega = omega_1d / (2.0 * c * f)        # (Nw,)

        # Broadcast to 3D only at the final exp
        r2_3d = r2[:, :, None]                   # (Nx,Ny,1)
        k_3d  = komega[None, None, :]            # (1,1,Nw)

        if xp.__name__ == "cupy":
            return _exp_r2_k_fused(r2_3d, k_3d)

        return xp.exp((-1j) * r2_3d * k_3d)

from scipy.constants import c
from lasy.backend import xp
from .optical_element import OpticalElement

# Optional: fuse exp for CuPy
_exp_r2_k_fused = None
if xp.__name__ == "cupy":
    import cupy as cp

    @cp.fuse()
    def _exp_r2_k_fused(r2, komega):  # r2: (Nx,Ny,1)  komega: (1,1,Nw)
        return cp.exp((-1j) * r2 * komega)


class ChromaticLens(OpticalElement):
    def __init__(self, R1, R2, d, n_func):
        self.R1 = R1
        self.R2 = R2
        self.d = d
        self.n_func = n_func

    def amplitude_multiplier(self, x, y, omega):
        """
        Memory-optimized:
        - expects x,y as 2D grids OR 1D axes (see below)
        - expects omega as 1D
        """

        # --- 1) Make r^2 as a 2D array only ---
        # If x,y are 1D axes, use broadcasting to build r2 without 3D meshgrids
        # x: (Nx,)  y: (Ny,)  -> r2: (Nx,Ny)
        if x.ndim == 1 and y.ndim == 1:
            r2 = x[:, None] * x[:, None] + y[None, :] * y[None, :]
        else:
            # x,y already 2D grids (Nx,Ny)
            r2 = x * x + y * y

        # --- 2) Compute f(omega) as 1D only ---
        # omega MUST be 1D here for the big memory win
        if omega.ndim != 1:
            raise ValueError("For low memory, pass omega as 1D (Nw,), not a 3D meshgrid.")

        lam = 2 * xp.pi * c / omega * 1e6  # (Nw,) microns
        n = self.n_func(lam)               # (Nw,) refractive index

        f = 1.0 / (
            (n - 1.0)
            * (1.0 / self.R1 - 1.0 / self.R2 + (n - 1.0) * self.d / (n * self.R1 * self.R2))
        )                                  # (Nw,)

        komega = omega / (2.0 * c * f)     # (Nw,)

        # --- 3) Broadcast to 3D only at the final exp ---
        r2_3d = r2[:, :, None]             # (Nx,Ny,1)
        k_3d  = komega[None, None, :]      # (1,1,Nw)

        if xp.__name__ == "cupy":
            return _exp_r2_k_fused(r2_3d, k_3d)

        return xp.exp((-1j) * r2_3d * k_3d)

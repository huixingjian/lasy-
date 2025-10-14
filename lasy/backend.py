try:
    import cupy as xp

    use_cupy = True

    def to_cpu(arr):
        """Convert array from cupy to numpy"""
        if isinstance(arr, xp.ndarray):
            return xp.asnumpy(arr)
        else:
            return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy"""
        if not isinstance(arr, xp.ndarray):
            return xp.asarray(arr)
        else:
            return arr

except ImportError:
    import numpy as xp

    use_cupy = False

    def to_cpu(arr):
        """Convert array from cupy to numpy"""
        return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy"""
        return arr


__all__ = ["use_cupy", "xp", "to_cpu", "to_gpu"]

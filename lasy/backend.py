try:
    import cupy as xp

    use_cupy = True

    def to_cpu(arr):
        return xp.as_numpy(arr)

except ImportError:
    import numpy as xp

    use_cupy = False

    def to_cpu(arr):
        return arr

__all__ = ["use_cupy", "xp", "to_cpu"]

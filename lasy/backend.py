try:
    import cupy as xp

    use_cupy = True

    def to_cpu(arr):
        if isinstance(arr, xp.ndarray):
            return xp.asnumpy(arr)
        else:
            return arr

    def to_gpu(arr):
        if not isinstance(arr, xp.ndarray):
            return xp.asarray(arr)
        else:
            return arr

except ImportError:
    import numpy as xp

    use_cupy = False

    def to_cpu(arr):
        return arr

    def to_gpu(arr):
        return arr


__all__ = ["use_cupy", "xp", "to_cpu", "to_gpu"]

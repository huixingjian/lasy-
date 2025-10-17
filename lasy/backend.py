try:
    import cupy as xp
    from cupyx import scipy as xp_sci

    use_cupy = True

    def to_cpu(arr):
        """Convert array from cupy to numpy."""
        if isinstance(arr, xp.ndarray):
            return xp.asnumpy(arr)
        elif isinstance(arr, list):
            return [to_cpu(a) for a in arr]
        elif isinstance(arr, tuple):
            return (to_cpu(a) for a in arr)
        else:
            assert False, f"type is {type(arr)}"
            return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy."""
        if not isinstance(arr, xp.ndarray):
            return xp.asarray(arr)
        else:
            return arr

except ImportError:
    import numpy as xp
    import scipy as xp_sci

    use_cupy = False

    def to_cpu(arr):
        """Convert array from cupy to numpy."""
        return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy."""
        return arr

# assert use_cupy==False
# print("use_cupy:", use_cupy)

__all__ = ["use_cupy", "xp", "xp_sci", "to_cpu", "to_gpu"]

# -----------------------------------------------------------------------------
# From https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension:
#
# In fractal geometry, the Minkowski–Bouligand dimension, also known as
# Minkowski dimension or box-counting dimension, is a way of determining the
# fractal dimension of a set S in a Euclidean space Rn, or more generally in a
# metric space (X, d).
# -----------------------------------------------------------------------------
# code taken from https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0
import numpy as np


def fractal_dimension(b_image):
    """
    Calculates the fractal dimension of the given binary image Z
    :param b_image: a binary 2d image
    :return: the Minkowski–Bouligand dimension of the image
    """

    # Only for 2d image
    assert(len(b_image.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Minimal dimension of image
    p = min(b_image.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(b_image, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# I = scipy.misc.imread("sierpinski.png")/256.0
# print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(I))
# print("Haussdorf dimension (theoretical):        ", (np.log(3)/np.log(2)))

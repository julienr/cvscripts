# Copyright (2012) Julien Rebetez <julien@fhtagn.net>
import numpy as np
def to_homg(x):
    """
    Transform x to homogeneous coordinates
    If X is MxN, returns an (M+1)xN array with ones on the last row

    >>> to_homg(np.array([[1, 2, 3], [1, 2, 3]], dtype=float))
    array([[ 1.,  2.,  3.],
           [ 1.,  2.,  3.],
           [ 1.,  1.,  1.]])
    >>> to_homg(np.array([[1], [2]]))
    array([[ 1.],
           [ 2.],
           [ 1.]])
    >>> to_homg([1, 2])
    array([1, 2, 1])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        return np.r_[x, np.ones((1,x.shape[1]))]
    else:
        return np.r_[x, 1]


def from_homg(x):
    """
    Transform homogeneous x to non-homogeneous coordinates
    If X is MxN, returns an (M-1)xN array that will contain nan when for
    columns where the last row was 0

    >>> from_homg(np.array([[1, 2, 3],
    ...                     [4, 5, 0]], dtype=float))
    array([[ 0.25,  0.4 ,   nan]])
    >>> from_homg(np.array([1, 5], dtype=float))
    array([ 0.2])
    >>> from_homg([1, 5, 0])
    array([ nan,  nan])
    >>> from_homg((1, 4, 0.5))
    array([ 2.,  8.])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        #valid = np.nonzero(x[-1,:])
        valid = x[-1,:] != 0
        result = np.empty((x.shape[0]-1, x.shape[1]), dtype=float)
        result[:,valid] = x[:-1,valid] / x[-1, valid]
        result[:,~valid] = np.nan
        return result
    else:
        if x[-1] == 0:
            result = np.empty(len(x)-1, dtype=float)
            result[:] = np.nan
            return result
        else:
            return np.array(x[:-1]) / x[-1]

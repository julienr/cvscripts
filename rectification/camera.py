import numpy as np
import numpy.linalg as la

def positive_qr(Z):
    """
    Compute QR decomposition such that R has nonnegative diagonal elements
    http://www.mathworks.com/matlabcentral/answers/6659-changes-to-qr-factorization-qr
    """
    Q, R = la.qr(Z)
    D = np.diag(np.sign(np.diag(R)))
    Q = Q.dot(D)
    R = D.dot(R)
    return Q, R

def positive_rq(S):
    """
    Taken from vgg_rq. Computes RQ decomposition (like qr but other way around)
    If [R,Q] = rq(X), then R is upper-triangular, Q is orthogonal and X=R*Q
    Moreover, if S is a real matrix, then det(Q) > 0
    >>> K = np.array([[50, 0, 20],
    ...               [0, 50, 30],
    ...               [0, 0, 1]])
    >>> R = np.array([[np.cos(np.pi/7), -np.sin(np.pi/7), 0],
    ...               [np.sin(np.pi/7), np.cos(np.pi/7), 0],
    ...               [0, 0, 1]])
    >>> S = K.dot(R)
    >>> rK, rR = positive_rq(S)
    >>> np.allclose(R, rR)
    True
    >>> np.allclose(K, rK)
    True
    """
    Q, R = positive_qr(np.flipud(S).T)
    R = np.flipud(R.T)
    Q = np.flipud(Q.T)
    return R[:,::-1], Q


def KR_from_P(P):
    """
    Extract K, R from camera matrix P = K[R | t]
    K is scaled so that K[3,3] = 1
    Inspired by VGG_KR_FROM_P
    """
    K, R = positive_rq(P[:,:3])
    K = K/K[2,2]
    return K, R


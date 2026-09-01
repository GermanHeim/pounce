"""Small numerical helpers shared by the statistics in this package.

`pounce.sensitivity` and `pounce.curve_fit` both project a covariance
onto the subspace an active set leaves free, and both used to compute
the basis for that subspace with their own copy of the same SVD. One
copy, so a change to the rank tolerance reaches both.

They do NOT share a covariance routine, and should not: `curve_fit`
reads `inv(H_S)` straight off the factor because its parameters ARE the
decision variables and `K = H_S`, while `pounce.sensitivity` recovers a
tangent map because its fitted block sits inside a larger model with
equalities. Same word, different computation.
"""
import numpy as np


def nullspace(A):
    """Orthonormal basis of the null space of `A`'s rows, as columns.

    The projection basis `Z`: with `A` the active constraint normals,
    the covariance lives in `span(Z)` and is zero along everything else.
    Rank is read off the singular values at the standard relative
    tolerance, so a redundantly listed constraint costs a dimension
    once rather than twice.
    """
    A = np.atleast_2d(A)
    if A.shape[0] == 0:
        return np.eye(A.shape[1])
    _, sv, vh = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.finfo(float).eps * (sv[0] if sv.size else 1.0)
    rank = int(np.sum(sv > tol))
    return vh[rank:].T

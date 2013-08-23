# encoding: utf-8
"""
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
from collections import defaultdict

import numpy as np

cimport numpy as np
cimport cython

np.import_array()


def find_pivot(nodes, long[:] degrees, long lim):
    cdef int pivot = -1
    cdef int pivot_degree = -1
    cdef int degree
    cdef int n
    for n in nodes:
        degree = degrees[n]
        if degree > pivot_degree:
            pivot = n
            pivot_degree = degree
        if degree == lim:
            break
    return pivot, pivot_degree

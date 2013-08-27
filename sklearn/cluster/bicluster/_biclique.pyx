# encoding: utf-8
# cython: profile=True
from collections import defaultdict

import numpy as np

cimport numpy as np
cimport cython

np.import_array()

from libc.stdlib cimport malloc, free


cdef class HashSet:
    cdef long[:] data  # holds elements
    cdef long capacity  # 2^bits
    cdef long bits
    cdef long cardinality

    cdef long min_bits
    cdef float upper
    cdef float lower
    cdef long empty
    cdef long removed

    cdef long iter_posn

    def __cinit__(self):
        self.min_bits = 3
        self.bits = self.min_bits
        self.capacity = 2 ** self.bits
        self.upper = 0.7
        self.lower = 0.1
        self.empty = (2 ** 32) - 1
        self.removed = (2 ** 32) - 2
        self.iter_posn = 0
        self.cardinality = 0
        self.data = np.empty(self.capacity, dtype=np.int)
        self.data[:] = self.empty

    def __len__(self):
        return self.cardinality

    def __iter__(self):
        self.iter_posn = 0
        return self

    def __next__(self):
        if self.iter_posn >= self.capacity:
            raise StopIteration
        cdef long val = self.data[self.iter_posn]
        while val == self.empty or \
              val == self.removed:
            self.iter_posn += 1
            if self.iter_posn >= self.capacity:
                raise StopIteration
            val = self.data[self.iter_posn]
        self.iter_posn += 1
        return val

    def __bool__(self):
        return len(self) > 0

    def _hash(self, long value):
        """Knuth's multiplicative method"""
        return (value * 2654435761) % self.capacity

    def _find_posn(self, long value):
        cdef long posn = self._hash(value)
        cdef long entry = self.data[posn]
        while (entry != self.empty):
            if entry == value:
                break
            posn = (posn + 1) % self.capacity
            entry = self.data[posn]
        return posn

    def add(self, long value, bint resize=1):
        cdef long posn = self._find_posn(value)
        if (self.data[posn] != self.empty) and (self.data[posn] != self.removed) and (self.data[posn] != value):
            raise Exception()
        if self.data[posn] != value:
            self.data[posn] = value
            self.cardinality += 1
            if resize:
                self._resize()

    def update(self, long[:] values):
        cdef long val
        for val in values:
            self.add(val)

    def remove(self, long value):
        cdef long posn = self._find_posn(value)
        if self.data[posn] == value:
            self.data[posn] = self.removed
            self.cardinality -= 1
            self._resize()

    def pop(self):
        if not self:
            raise IndexError('pop from empty set')
        cdef long value
        for value in self.data:
            if value != self.empty and value != self.removed:
                break
        self.remove(value)
        return value

    def contains(self, long value):
        cdef long posn = self._find_posn(value)
        return self.data[posn] == value

    def copy(self):
        cdef HashSet result = HashSet()
        cdef long val
        for val in self:
            result.add(val)
        return result

    def _resize(self):
        if self.cardinality > <long>(self.capacity * self.upper):
            self.bits += 1
        elif self.cardinality < <long>(self.capacity * self.lower):
            if self.bits <= self.min_bits:
                return
            self.bits -= 1
        else:
            return
        cdef long[:] old_data = self.data
        cdef long old_capacity = self.capacity

        self.capacity = 2 ** self.bits
        self.data = np.empty(self.capacity, dtype=np.int)
        self.data[:] = self.empty

        self.cardinality = 0
        cdef long old_val
        for val in old_data:
            if val != self.empty and val != self.removed:
                self.add(val, 0)

    def intersection(self, HashSet other):
        cdef long val
        cdef HashSet result = HashSet()
        if self.capacity < other.capacity:
            smaller = self
            larger = other
        else:
            smaller = other
            larger = self
        for val in smaller:
            if larger.contains(val):
                result.add(val)
        return result

    def intersection_array(self, long[:] arr):
        cdef long val
        cdef HashSet result = HashSet()
        for val in arr:
            if self.contains(val):
                result.add(val)
        return result

    def difference(self, HashSet other):
        cdef long val
        cdef HashSet result = HashSet()
        for val in self:
            if not other.contains(val):
                result.add(val)
        return result

    def difference_array(self, long[:] other):
        cdef long val
        cdef HashSet result = self.copy()
        for val in other:
            result.remove(val)
        return result

    def __repr__(self):
        return "HashSet({})".format([i for i in self])

cpdef precompute_neighbors(long[:, :] X):
    cdef long n_rows
    cdef long n_cols
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    cdef long row
    cdef long col
    cdef dict neighbors
    neighbors = {}
    for row in range(n_rows):
        neighbors[row] = set(np.nonzero(X[row])[0] + n_rows)
    for col in range(n_cols):
        neighbors[col + n_rows] = set(np.nonzero(X[:, col])[0])

    second_neighbors = defaultdict(set)
    for row in range(n_rows):
        for col in neighbors[row]:
            second_neighbors[row].update(neighbors[col])
    for col in range(n_cols):
        col = col + n_rows
        for row in neighbors[col]:
            second_neighbors[col].update(neighbors[row])

    cdef long node
    cdef dict all_neighbors
    all_neighbors = {}
    for node in neighbors:
        all_neighbors[node] = neighbors[node] | second_neighbors[node]
        all_neighbors[node].discard(node)

    for node in all_neighbors:
        all_neighbors[node] = np.array(list(all_neighbors[node]), dtype=np.int)
    return all_neighbors


cpdef find_pivot(HashSet nodes, long[:] degrees, long lim):
    cdef long pivot = -1
    cdef long pivot_degree = -1
    cdef long degree
    cdef long n
    for n in nodes:
        degree = degrees[n]
        if degree > pivot_degree:
            pivot = n
            pivot_degree = degree
        if degree == lim:
            break
    return pivot, pivot_degree


def find_bicliques(long[:, :] X):
    """Find all bicliques in a bipartite graph formed from array X.

    The graph has m+n nodes, where X has shape (m, n). If X[i, j]
    is nonzero, there is an edge between i and j.

    The bicliques are enumerated by a modification of the
    Bron-Kerbosch algorithm. This function is based on networkx's
    implementation.

    """
    neighbors = precompute_neighbors(X)
    degrees = np.zeros(len(neighbors), dtype=np.long)
    for n in neighbors:
        degrees[n] = neighbors[n].shape[0]
    pivot = np.argmax(degrees)

    candidates = HashSet()
    candidates.update(np.arange(X.shape[0] + X.shape[1]))
    small_candidates = candidates.difference_array(neighbors[pivot])
    done = HashSet()
    stack = []
    biclique_so_far = []

    while small_candidates or stack:
        if small_candidates:
            node = small_candidates.pop()
        else:
            # go to next in stack
            candidates, done, small_candidates = stack.pop()
            biclique_so_far.pop()
            continue

        # add next node to biclique
        biclique_so_far.append(node)
        candidates.remove(node)
        done.add(node)
        nn = neighbors[node]
        new_candidates = candidates.intersection_array(nn)
        new_done = done.intersection_array(nn)

        # check if we have more to search
        if not new_candidates:
            if not new_done:
                result = biclique_so_far[:]
                if len(result) > 1:
                    # found a biclique
                    yield result
            biclique_so_far.pop()
            continue

        n_new_candidates = len(new_candidates)
        pivot_done, degree_done = find_pivot(new_done, degrees,
                                             n_new_candidates)
        if degree_done == n_new_candidates:
            # shortcut: this part of tree already searched
            biclique_so_far.pop()
            continue
        pivot_cand, degree_cand = find_pivot(new_candidates, degrees,
                                             n_new_candidates - 1)
        if degree_cand > degree_done:
            pivot = pivot_cand
        else:
            pivot = pivot_done

        # save search tree for later backout
        stack.append((candidates, done, small_candidates))
        candidates = new_candidates
        done = new_done
        small_candidates = candidates.difference_array(neighbors[pivot])

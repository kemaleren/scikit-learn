"""Implements the Cheng and Church biclustering algorithm.

Authors : Kemal Eren
License: BSD 3 clause

"""
from abc import ABCMeta

import numpy as np

from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.externals import six

from sklearn.utils.validation import check_arrays
from sklearn.utils.validation import check_random_state

from .utils import check_array_ndim
from .utils import get_submatrix
from .utils import get_indicators


class EmptyBiclusterException(Exception):
    pass


class ChengChurch(six.with_metaclass(ABCMeta, BaseEstimator,
                                     BiclusterMixin)):
    def __init__(self, n_clusters=3, max_msr=1, deletion_threshold=1,
                 row_deletion_cutoff=100, column_deletion_cutoff=100,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_msr = max_msr
        self.deletion_threshold = deletion_threshold
        self.row_deletion_cutoff = row_deletion_cutoff
        self.column_deletion_cutoff = column_deletion_cutoff
        self.random_state = random_state

    def _check_parameters(self):
        if self.n_clusters < 1:
            raise ValueError("'n_clusters' must be > 0, but its value"
                             " is {}".format(self.n_clusters))
        if self.max_msr < 0:
            raise ValueError("'max_msr' must be > 0.0, but its value"
                             " is {}".format(self.max_msr))
        if self.deletion_threshold < 1:
            raise ValueError("'deletion_threshold' must be >= 1.0, but its"
                             " value is {}".format(self.deletion_threshold))
        if self.row_deletion_cutoff < 1:
            raise ValueError("'row_deletion_cutoff' must be >= 1, but its"
                             " value is {}".format(self.row_deletion_cutoff))
        if self.column_deletion_cutoff < 1:
            raise ValueError("'column_deletion_cutoff' must be >= 1, but its"
                             " value is {}".format(
                                 self.column_deletion_cutoff))

    def _precompute(self, rows, cols, X):
        submatrix = get_submatrix(rows, cols, X)
        row_mean = submatrix.mean(axis=1)
        col_mean = submatrix.mean(axis=0)
        mean = submatrix.mean()
        return submatrix, row_mean, col_mean, mean

    def _compute_msr(self, rows, cols, X):
        submatrix, row_mean, col_mean, mean = \
            self._precompute(rows, cols, X)
        n_rows, n_cols = submatrix.shape
        if n_rows == 0 or n_cols == 0:
            raise EmptyBiclusterException()
        msr_array = submatrix - row_mean[:, np.newaxis] - col_mean + mean
        msr = np.power(msr_array, 2).sum() / (n_rows * n_cols)
        row_msr = np.power(msr_array, 2).sum(axis=1) / n_rows
        col_msr = np.power(msr_array, 2).sum(axis=0) / n_cols
        return msr, row_msr, col_msr

    def _compute_inverse_row_msr(self, rows, cols, X):
        submatrix, row_mean, col_mean, mean = \
            self._precompute(rows, cols, X)
        inverse_row_msr = -submatrix + row_mean - col_mean + mean
        inverse_row_msr = np.power(inverse_row_msr, 2).sum(axis=1)
        return inverse_row_msr / len(cols)

    def _node_deletion(self, rows, cols, X):
        msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
        n_rows = len(rows)
        n_cols = len(cols)
        while msr > self.max_msr:
            row_id = np.argmax(row_msr)
            col_id = np.argmax(col_msr)
            if row_msr[row_id] > col_msr[col_id]:
                rows = np.setdiff1d(rows, [rows[row_id]])
            else:
                cols = np.setdiff1d(cols, [cols[col_id]])
            if n_rows == len(rows) and n_cols == len(cols):
                break
            else:
                n_rows = len(rows)
                n_cols = len(cols)
            msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
        return rows, cols

    def _multiple_node_deletion(self, rows, cols, X):
        msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
        n_rows = len(rows)
        n_cols = len(cols)
        while msr > self.max_msr:
            if n_rows >= self.row_deletion_cutoff:
                to_remove = row_msr > self.deletion_threshold * msr
                rows = np.setdiff1d(rows, rows[to_remove])

            if n_cols >= self.column_deletion_cutoff:
                to_remove = col_msr > self.deletion_threshold * msr
                rows = np.setdiff1d(cols, cols[to_remove])

            if n_rows == len(rows) and n_cols == len(cols):
                rows, cols = self._node_deletion(rows, cols, X)
                break
            else:
                n_rows = len(rows)
                n_cols = len(cols)
            msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
        return rows, cols

    def _node_addition(self, rows, cols, X):
        msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
        n_rows = len(rows)
        n_cols = len(cols)
        while msr > self.max_msr:
            to_add = row_msr < msr
            rows = np.setunion1d(rows, rows[to_add])

            msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
            to_add = col_msr < msr
            rows = np.setunion1d(cols, cols[to_add])

            if self.inverse_rows:
                inverse_row_msr = self._compute_inverse_row_msr(rows, cols, X)
                to_add = inverse_row_msr < msr
                rows = np.setunion1d(rows, rows[to_add])

            if n_rows == len(rows) and n_cols == len(cols):
                break
            else:
                n_rows = len(rows)
                n_cols = len(cols)
            msr, row_msr, col_msr = self._compute_msr(rows, cols, X)
        return rows, cols

    def _mask(self, X, rows, cols, generator, minval, maxval):
        mask_vals = generator.uniform(minval, maxval, (len(rows), len(cols)))
        X[rows[:, np.newaxis], cols] = mask_vals

    def fit(self, X):
        X = X.copy()  # need to modify it in-place
        self._check_parameters()
        X, = check_arrays(X, dtype=np.float64)
        check_array_ndim(X)
        minval, maxval = X.min(), X.max()
        n_rows, n_cols = X.shape

        generator = check_random_state(self.random_state)
        results = []

        for i in range(self.n_clusters):
            try:
                rows = np.arange(n_rows)
                cols = np.arange(n_cols)
                rows, cols = self._multiple_node_deletion(rows, cols, X)
                rows, cols = self._node_addition(rows, cols, X)
                self._mask(X, rows, cols, generator, minval, maxval)
                if len(rows) == 0 or len(cols) == 0:
                    break
                results.append((rows, cols))
            except EmptyBiclusterException:
                break
        if results:
            # TODO: move this to utils
            indicators = (get_indicators(r, c, X.shape) for r, c in results)
            rows, cols = zip(*indicators)
            self.rows_ = np.vstack(rows)
            self.columns_ = np.vstack(cols)
        else:
            self.rows_ = np.array([])
            self.columns_ = np.array([])

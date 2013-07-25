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
from .utils import get_indicators


class EmptyBiclusterException(Exception):
    pass


class ChengChurch(six.with_metaclass(ABCMeta, BaseEstimator,
                                     BiclusterMixin)):
    def __init__(self, n_clusters=3, max_msr=1, deletion_threshold=1,
                 row_deletion_cutoff=100, column_deletion_cutoff=100,
                 inverse_rows=True, random_state=None):
        self.n_clusters = n_clusters
        self.max_msr = max_msr
        self.deletion_threshold = deletion_threshold
        self.row_deletion_cutoff = row_deletion_cutoff
        self.column_deletion_cutoff = column_deletion_cutoff
        self.inverse_rows = inverse_rows
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

    def _sr_array(self, X):
        arr = X - X.mean(axis=1)[:, np.newaxis] - X.mean(axis=0) + X.mean()
        return np.power(arr, 2)

    def _msr(self, X):
        return self._sr_array(X).mean()

    def _row_msr(self, X):
        return self._sr_array(X).mean(axis=1)

    def _col_msr(self, X):
        return self._sr_array(X).mean(axis=0)

    def _sr_array_add(self, rows, cols, X):
        rows = rows[:, np.newaxis]
        arr = (X - X[:, cols].mean(axis=1)[:, np.newaxis] -
               X[rows, :].mean(axis=0) + X[rows, cols].mean())
        return np.power(arr, 2)

    def _row_msr_add(self, rows, cols, X):
        return self._sr_array_add(rows, cols, X).mean(axis=1)

    def _col_msr_add(self, rows, cols, X):
        return self._sr_array_add(rows, cols, X).mean(axis=0)

    def _row_msr_inverse_add(self, rows, cols, X):
        rows = rows[:, np.newaxis]
        arr = (-X + X[:, cols].mean(axis=1)[:, np.newaxis] -
               X[rows, :].mean(axis=0) + X.mean())
        return np.power(arr, 2).mean(axis=1)

    def _node_deletion(self, rows, cols, X):
        while self._msr(X[rows, cols]) > self.max_msr:
            n_rows, n_cols = len(rows), len(cols)
            row_msr = self._row_msr(X[rows, cols])
            col_msr = self._col_msr(X[rows, cols])
            row_id = np.argmax(row_msr)
            col_id = np.argmax(col_msr)
            if row_msr[row_id] > col_msr[col_id]:
                rows = rows.ravel()
                rows = np.setdiff1d(rows, [rows[row_id]])
                rows = rows[:, np.newaxis]
            else:
                cols = np.setdiff1d(cols, [cols[col_id]])
            if n_rows == len(rows) and n_cols == len(cols):
                break
        return rows, cols

    def _multiple_node_deletion(self, rows, cols, X):
        while self._msr(X[rows, cols]) > self.max_msr:
            n_rows, n_cols = len(rows), len(cols)
            row_msr = self._row_msr(X[rows, cols])
            if n_rows >= self.row_deletion_cutoff:
                to_remove = row_msr > (self.deletion_threshold *
                                       self._msr(X[rows, cols]))
                rows = rows.ravel()
                rows = np.setdiff1d(rows, rows[to_remove])
                rows = rows[:, np.newaxis]

            col_msr = self._col_msr(X[rows, cols])
            if n_cols >= self.column_deletion_cutoff:
                to_remove = col_msr > (self.deletion_threshold *
                                       self._msr(X[rows, cols]))
                cols = np.setdiff1d(cols, cols[to_remove])

            if n_rows == len(rows) and n_cols == len(cols):
                rows, cols = self._node_deletion(rows, cols, X)
                break
        return rows, cols

    def _node_addition(self, rows, cols, X):
        while True:
            n_rows, n_cols = len(rows), len(cols)
            to_add = (self._col_msr_add(rows, cols, X) <
                      self._msr(X[rows, cols]))[0]
            to_add = np.nonzero(to_add)[0]
            cols = np.union1d(cols, to_add)

            old_rows = rows.copy()
            to_add = (self._row_msr_add(rows, cols, X) <
                      self._msr(X[rows, cols]))
            to_add = np.nonzero(to_add)[0]
            rows = np.union1d(rows.ravel(), to_add)[:, np.newaxis]

            if self.inverse_rows:
                to_add = (self._row_msr_inverse_add(old_rows, cols, X) <
                          self._msr(X[rows, cols]))
                to_add = np.nonzero(to_add)[0]
                rows = np.union1d(rows.ravel(), to_add)[:, np.newaxis]

            if n_rows == len(rows) and n_cols == len(cols):
                break
        return rows, cols

    def _mask(self, X, rows, cols, generator, minval, maxval):
        mask_vals = generator.uniform(minval, maxval, (len(rows), len(cols)))
        X[rows, cols] = mask_vals

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
                rows = np.arange(n_rows)[:, np.newaxis]
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

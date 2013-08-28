"""Implements the BiMax biclustering algorithm.

Authors : Kemal Eren
License: BSD 3 clause

"""
from abc import ABCMeta

import numpy as np

from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.externals import six

from .utils import get_indicators


class BiMax(six.with_metaclass(ABCMeta, BaseEstimator,
                               BiclusterMixin)):
    """Method to find all maximal biclusters in a boolean array."""

    def __init__(self):
        pass

    def _conquer(self, data, rows, cols, col_sets):
        if np.all(data[np.array(list(rows))[:, np.newaxis], list(cols)]):
            return [(rows, cols)]
        rows_u, rows_v, rows_w, cols_u, cols_v = \
            self._divide(data, rows, cols, col_sets)
        results_u = []
        results_v = []
        if rows_u:
            results_u = self._conquer(data, rows_u.union(rows_v),
                                      cols_u, col_sets)
        if rows_v and rows_w:
            results_v = self._conquer(data, rows_v, cols_v, col_sets)
        elif rows_w:
            new_col_sets = col_sets[:]
            new_col_sets.append(cols_v)
            results_v = self._conquer(data, rows_w.union(rows_v),
                                      cols_u.union(cols_v), new_col_sets)
        return results_u + results_v

    def _divide(self, data, rows, cols, col_sets):
        new_rows = self._reduce(data, rows, cols, col_sets)
        row_cands = list(r for r in new_rows
                         if 0 < data[r, list(cols)].sum() < len(cols))
        try:
            r = row_cands[0]
            cols_u = set(c for c in cols if data[r, c])
        except IndexError:
            cols_u = cols
        cols_v = cols.difference(cols_u)
        rows_u = set()
        rows_v = set()
        rows_w = set()
        for r in new_rows:
            incl_cols = set(c for c in cols if data[r, c])
            if incl_cols.issubset(cols_u):
                rows_u.add(r)
            elif incl_cols.issubset(cols_v):
                rows_v.add(r)
            else:
                rows_w.add(r)
        return rows_u, rows_v, rows_w, cols_u, cols_v

    def _reduce(self, data, rows, cols, col_sets):
        result = set()
        for r in rows:
            incl_cols = set(c for c in cols if data[r, c])
            if incl_cols and all(cset.intersection(incl_cols)
                                 for cset in col_sets):
                result.add(r)
        return result

    def fit(self, X):
        """Creates a biclustering for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        """
        n_rows, n_cols = X.shape
        result = self._conquer(X, set(range(n_rows)),
                               set(range(n_cols)), [])
        row_ind = []
        col_ind = []
        for rows, cols in result:
            ri, ci = get_indicators(rows, cols, X.shape)
            row_ind.append(ri)
            col_ind.append(ci)
        self.rows_ = np.vstack(row_ind)
        self.columns_ = np.vstack(col_ind)

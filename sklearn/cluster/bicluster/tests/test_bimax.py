"""Testing for BiMax."""

import numpy as np
from sklearn.cluster.bicluster import BiMax


def test_divide():
    data = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],

                     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
                     [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],

                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]])

    rows = set(range(data.shape[0]))
    cols = set(range(data.shape[0]))
    model = BiMax()
    rows_all, rows_none, rows_some, cols_all, cols_none = \
        model._divide(data, rows, cols, [])
    assert rows_all == set(range(6))
    assert rows_some == set([6, 7, 8])
    assert rows_none == set([9, 10, 11])
    assert cols_all == set(range(5))
    assert cols_none == set(range(5, 12))


def test_reduce():
    data = np.array([[1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 1, 0],
                     [1, 1, 1, 0, 0, 1],
                     [1, 1, 1, 0, 0, 0]])
    rows = set(range(4))
    cols = set(range(6))
    col_sets = [set([3, 4, 5])]
    model = BiMax()
    new_rows, nz_cols = model._reduce(data, rows, cols, col_sets)
    assert new_rows == set(range(3))
    assert nz_cols == {0: set([0, 1, 2, 3]),
                       1: set([0, 1, 2, 4]),
                       2: set([0, 1, 2, 5]),
                       3: set([0, 1, 2])}


def test_bimax():
    data = np.zeros((20, 20), dtype=np.int8)
    data[0:10, 0:10] = 1
    model = BiMax()
    model.fit(data)
    assert len(model.rows_) == 1
    assert len(model.columns_) == 1
    rows, cols = model.get_indices(0)
    assert set(rows) == set(range(10))
    assert set(cols) == set(range(10))

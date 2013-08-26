"""Testing for BiMax."""

import numpy as np
from sklearn.cluster.bicluster import BiMax
from sklearn.cluster.bicluster._biclique import precompute_neighbors, find_pivot, HashSet


def test_hashset():
    s = HashSet()
    s.add(3)
    s.add(4)
    assert s.contains(3)
    s.remove(3)
    assert not s.contains(3)

    s1 = HashSet()
    s2 = HashSet()
    s1.add(1)
    s1.add(2)
    s2.add(2)
    s2.add(3)

    r = s1.intersection(s2)
    assert r.contains(2)
    assert not r.contains(1)
    assert not r.contains(3)

    r = s1.difference(s2)
    assert r.contains(1)
    assert not r.contains(2)
    assert not r.contains(3)

    s1.update(np.array([3, 4, 5], dtype=np.int))
    assert s1.contains(3)
    assert s1.contains(4)
    assert s1.contains(5)
    assert not s1.contains(6)

    assert len(s1) == 5
    val = s1.pop()
    assert len(s1) == 4
    assert not s1.contains(val)

    for node in s1:
        assert s1.contains(node)

    e = HashSet()
    if e:
        assert False
    e.add(1)
    if e:
        assert True
    if not e:
        assert False


def test_neighbors():
    data = np.zeros((20, 20), dtype=np.int)
    data[0:10, 0:10] = 1
    neighbors = precompute_neighbors(data)
    for i in range(40):
        if (0 <= i < 10) or (20 <= i < 30):
            assert len(neighbors[i]) == 19
        else:
            assert len(neighbors[i]) == 0


def test_find_pivot():
    nodes = HashSet()
    nodes.update(np.arange(10, dtype=np.int))
    degrees = np.zeros((10,), dtype=np.int)
    degrees[3] = 5
    assert find_pivot(nodes, degrees, 10) == (3, 5)


def test_bimax():
    data = np.zeros((20, 20), dtype=np.int)
    data[0:10, 0:10] = 1
    model = BiMax()
    model.fit(data)
    assert len(model.rows_) == 1
    assert len(model.columns_) == 1
    rows, cols = model.get_indices(0)
    assert set(rows) == set(range(10))
    assert set(cols) == set(range(10))

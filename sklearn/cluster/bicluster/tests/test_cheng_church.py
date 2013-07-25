"""Testing for Spectral Biclustering methods"""

import numpy as np
from sklearn.utils.testing import assert_equal
from sklearn.cluster.bicluster import ChengChurch
from sklearn.metrics import consensus_score


def test_cheng_church():
    """Test Cheng and Church algorithm on a simple problem."""
    generator = np.random.RandomState(0)
    data = generator.uniform(0, 100, (30, 30))
    data[:10, :10] = 20
    data[10:20, 10:20] = 50
    data[20:30, 20:300] = 80
    model = ChengChurch(n_clusters=3, max_msr=10, random_state=0)
    model.fit(data)

    rows = np.zeros((3, 30), dtype=np.bool)
    cols = np.zeros((3, 30), dtype=np.bool)

    rows[0, 0:10] = True
    rows[1, 10:20] = True
    rows[2, 20:30] = True

    cols[0, 0:10] = True
    cols[1, 10:20] = True
    cols[2, 20:30] = True

    assert_equal(consensus_score((rows, cols), model.biclusters_), 1.0)

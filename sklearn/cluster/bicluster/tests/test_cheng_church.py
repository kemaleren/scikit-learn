"""Testing for Spectral Biclustering methods"""

import numpy as np
from sklearn.utils.testing import assert_equal
from sklearn.cluster.bicluster import ChengChurch
from sklearn.metrics import consensus_score


def test_cheng_church():
    """Test Cheng and Church algorithm on a simple problem."""
    generator = np.random.RandomState(0)
    data = generator.uniform(0, 100, (150, 150))
    data[:50, :50] = 20
    data[50:100, 50:100] = 50
    data[100:150, 100:150] = 80
    model = ChengChurch(n_clusters=3, max_msr=10, random_state=0)
    model.fit(data)

    rows = np.zeros((3, 150), dtype=np.bool)
    cols = np.zeros((3, 150), dtype=np.bool)

    rows[0, 0:50] = True
    rows[1, 50:100] = True
    rows[2, 100:1500] = True

    cols[0, 0:50] = True
    cols[1, 50:100] = True
    cols[2, 100:200] = True

    assert_equal(consensus_score((rows, cols), model.biclusters_), 1.0)

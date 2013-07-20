"""Testing for Spectral Biclustering methods"""

import numpy as np
from sklearn.utils.testing import assert_array_equal
from sklearn.cluster.bicluster import ChengChurch


def test_cheng_church():
    """Test Cheng and Church algorithm on a simple problem."""
    generator = np.random.RandomState(0)
    data = generator.normal(0, 5, (50, 50))
    data[:10, :10] = 10
    model = ChengChurch(n_clusters=1, max_msr=10)
    model.fit(data)

    assert_array_equal(np.nonzero(model.rows_[0])[0],
                       np.arange(10))
    assert_array_equal(np.nonzero(model.columns_[0])[0],
                       np.arange(10))

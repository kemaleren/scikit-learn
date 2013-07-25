"""Testing for Spectral Biclustering methods"""

from sklearn.utils.testing import assert_equal
from sklearn.cluster.bicluster import ChengChurch
from sklearn.metrics import consensus_score
from sklearn.datasets import make_msr


def test_cheng_church():
    """Test Cheng and Church algorithm on a simple problem."""
    data, rows, cols = make_msr((150, 150), 3, random_state=0)
    model = ChengChurch(n_clusters=3, max_msr=10, random_state=0)
    model.fit(data)
    assert_equal(consensus_score((rows, cols), model.biclusters_), 1.0)

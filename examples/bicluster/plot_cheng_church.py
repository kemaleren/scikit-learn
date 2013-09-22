"""
========================================
A demo of the Cheng and Church algorithm
========================================

This example demonstrates how to generate a dataset and bicluster it
using Cheng and Church.

The data is generated with the ``make_msr_biclusters`` function, then
shuffled and passed to Cheng and Church. The rows and columns of the
shuffled matrix are rearranged to show the biclusters found by the
algorithm.

"""

print(__doc__)

# Author: Kemal Eren <kemal@kemaleren.com>
# License: BSD 3 clause

from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import make_msr_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import ChengChurch
from sklearn.metrics import consensus_score

data, rows, columns = make_msr_biclusters(shape=(100, 100),
                                          n_clusters=3, noise=10,
                                          shuffle=False,
                                          random_state=0)

data, row_idx, col_idx = sg._shuffle(data, random_state=0)

# Fit the biclustering model
model = ChengChurch(n_clusters=3, max_msr=100, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print "consensus score: {:.1f}".format(score)

# Plot the affinity matrix

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

row_order = np.argsort(np.argmax(model.rows_, axis=0), kind='mergesort')
column_order = np.argsort(np.argmax(model.columns_, axis=0), kind='mergesort')
plt.matshow(data[row_order].T[column_order].T[:300], cmap=plt.cm.Blues)
plt.title("Reordered dataset using biclustering")

# Plot profiles
plt.figure()
n_cols = data.shape[1]
for row in data:
    plt.plot(range(n_cols), row)
plt.title('Parallel coordinates of shuffled data')
plt.xlabel('column numbers')
plt.ylabel('value')


plt.figure()
bicluster = model.get_submatrix(0, data)
n_cols = bicluster.shape[1]
for row in bicluster:
    plt.plot(range(n_cols), row)
plt.title('Parallel coordinates of first bicluster')
plt.xlabel('column numbers')
plt.ylabel('value')
plt.xlim(0, n_cols)

plt.show()

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import load_digits

from sklearn.metrics import accuracy_score

from sklearn.metrics.pairwise import pairwise_distances
import kmedoids

import numpy as np

def run_single_fold(X, y, train_index, valid_index):
    D_matrix_train = X[np.ix_(train_index, train_index)]
    D_matrix_valid = X[np.ix_(np.concatenate((train_index, valid_index), axis=0), np.concatenate((train_index, valid_index), axis=0))]

    km = kmedoids.KMedoids(n_clusters = len(np.unique(y)), metric="precomputed", init = "build", random_state = 42) # why init = 'build' ??
    km.fit(D_matrix_train)
    km.dict_ = {i: y[train_index][id] for i, id in enumerate(km.medoid_indices_)}

    predicted_train_labels = [km.dict_[id] for id in km.predict(D_matrix_train)]
    train_acc = accuracy_score(predicted_train_labels, y[train_index])

    predicted_valid_labels = [km.dict_[id] for id in km.predict(D_matrix_valid)][len(train_index):] # we only look at the labels for the new data
    valid_acc = accuracy_score(predicted_valid_labels, y[valid_index]) 

    return train_acc, valid_acc
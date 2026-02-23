"""
utils.py
Common utilities for the HAR KNN assignments with tie-breaking.
"""

import os
import jax 
import jax.numpy as jnp
import functools
import numpy as np
from typing import Tuple, List


def load_har_dataset(base_path: str = "UCI HAR Dataset") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load the HAR dataset from the UCI archive."""
    train_x = os.path.join(base_path, "train", "X_train.txt")
    train_y = os.path.join(base_path, "train", "y_train.txt")
    test_x = os.path.join(base_path, "test", "X_test.txt")
    test_y = os.path.join(base_path, "test", "y_test.txt")

    if not (os.path.exists(train_x) and os.path.exists(train_y) and os.path.exists(test_x) and os.path.exists(test_y)):
        raise FileNotFoundError(
            f"Dataset files not found under {base_path}. Ensure UCI HAR Dataset is in current directory."
        )

    X_train = jnp.asarray(np.loadtxt(train_x), dtype=jnp.float32)
    y_train = jnp.asarray(np.loadtxt(train_y, dtype=np.int32).ravel(), dtype=jnp.int32)
    X_test = jnp.asarray(np.loadtxt(test_x), dtype=jnp.float32)
    y_test = jnp.asarray(np.loadtxt(test_y, dtype=np.int32).ravel(), dtype=jnp.int32)
    return X_train, y_train, X_test, y_test


def euclidean_distances(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute squared Euclidean distances between rows of a and rows of b."""
    a_sq = jnp.sum(a**2, axis=1, keepdims=True)

    b_sq = jnp.sum(b**2, axis=1)
 
    dists_sq = a_sq + b_sq - 2 * (a @ b.T)

    return dists_sq


def majority_vote(neighbor_labels: jnp.ndarray, neighbor_distances: jnp.ndarray) -> int:
    """
    neighbor_labels: array of labels of k nearest neighbors
    neighbor_distances: array of distances corresponding to neighbor_labels
    Returns the majority label. In case of tie, pick the label whose closest
    neighbor among the tied labels is nearest.
    """

    order = jnp.argsort(neighbor_distances)
    sorted_labels= neighbor_labels[order]

    counts = jnp.bincount(sorted_labels - 1, length=6)

    max_vote = jnp.max(counts)
    tied = (counts == max_vote)

    pos = tied[sorted_labels - 1]
    winner_idx = jnp.argmax(pos)
    return sorted_labels[winner_idx]
    
    

def confusion_matrix_multiclass(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the confusion matrix
    Rows = true classes, Columns = predicted classes.
    """
    classes = jnp.unique(jnp.concatenate((y_true, y_pred)))
    num_classes = len(classes)

    true_idx = jnp.searchsorted(classes, y_true)
    pred_idx = jnp.searchsorted(classes, y_pred)

    confusion_matrix = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    confusion_matrix = confusion_matrix.at[true_idx, pred_idx].add(1)

    return confusion_matrix

def display_confusion_matrix_and_accuracy(k: int, y_true: jnp.ndarray, y_pred: jnp.ndarray):
    """
    Display a readable confusion matrix table.
    """
    acc = accuracy(y_true, y_pred)
    print(f"k = {k}  accuracy={acc*100:5.2f}%")
    cm = confusion_matrix_multiclass(y_true, y_pred)
    classes = jnp.unique(jnp.concatenate((y_true, y_pred)))

    print("Confusion Matrix:")
    print("True\\Pred", end="\t")
    for c in classes:
        print(f"{c}", end="\t")
    print()

    for i, c in enumerate(classes):
        print(f"{c}", end="\t\t")
        for j in range(len(classes)):
            print(f"{cm[i, j]}", end="\t")
        print()

def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    return jnp.mean(y_true == y_pred)


@functools.partial(jax.jit, static_argnames=("k",))
def predict_knn_jit(X_test: jnp.ndarray, X_train: jnp.ndarray, y_train: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    JIT-compiled KNN prediction helper.

    Students: implement this function using JAX operations only.
    Requirements (enforced by the assignment writeup):
      - call euclidean_distances(X_test, X_train)
      - select k nearest neighbors using jnp.argpartition or jnp.argsort
      - gather neighbor labels and distances using indexing / jnp.take_along_axis
      - vote using jax.vmap(majority_vote)(...)
    Returns:
      y_pred: shape (n_test,), dtype int32
    """

    dist = euclidean_distances(X_test, X_train)
    knn = jnp.argpartition(dist, k - 1, axis=1)[:, :k]
    neighbor_dist = jnp.take_along_axis(dist, knn, axis=1)
    neighbor_labels = y_train[knn]

    y_pred = jax.vmap(majority_vote, in_axes=(0, 0))(neighbor_labels, neighbor_dist)

    return jnp.asarray(y_pred, dtype=jnp.int32)



class ScratchKNN:
    """KNN classifier from scratch with nearest-neighbor tie-breaking."""

    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test: jnp.ndarray) -> jnp.ndarray:
        """
        Predict labels for test samples.
        Uses full sorting to get k nearest neighbors.
        """
        y_pred = predict_knn_jit(X_test, self.X_train, self.y_train, self.k)
        return jnp.asarray(y_pred, dtype=jnp.int32)

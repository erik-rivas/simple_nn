# Let's first create the data
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder


def spiral_data(samples=100, classes=3):
    features = np.zeros((samples * classes, 2))
    labels = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, samples)
            + np.random.randn(samples) * 0.2
        )
        features[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        labels[ix] = class_number
    return features, labels


import numpy as np


def train_test_split(*arrays, test_size=0.25, random_state=None, shuffle=True):
    """
    Split arrays into random train and test subsets using NumPy.

    Parameters:
        *arrays : sequence of array-like objects
            Arrays to be split. They must have the same length along the first dimension.
        test_size : float, optional (default=0.25)
            Represents the proportion of the dataset to include in the test split.
        random_state : int or RandomState, optional (default=None)
            Seed or random number generator object for reproducible output.
        shuffle : bool, optional (default=True)
            Whether to shuffle the arrays before splitting.

    Returns:
        split_arrays : tuple
            Tuple containing the split arrays: (train_X, test_X, train_y, test_y)
    """
    if len(arrays) < 2:
        raise ValueError("At least two input arrays are required.")

    # Check if the lengths of input arrays are consistent
    lengths = np.array([len(arr) for arr in arrays])
    if not np.all(lengths == lengths[0]):
        raise ValueError("All input arrays must have the same length.")

    n_samples = lengths[0]
    n_test = int(n_samples * test_size)

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_indices = indices[:-n_test]
    test_indices = indices[-n_test:]

    list_result = list()

    for arr in arrays:
        if type(arr) == np.ndarray:
            list_result.append(arr[train_indices])
            list_result.append(arr[test_indices])
        elif type(arr) == pd.DataFrame:
            list_result.append(arr.iloc[train_indices])
            list_result.append(arr.iloc[test_indices])

    split_arrays = tuple(list_result)
    return split_arrays


def generate_data(n_classes=2, n_features=2, n_samples=20):
    # Generate a binary classification dataset

    data = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        random_state=101,
    )
    X = data[0]
    y = data[1]

    return X, y


if __name__ == "__main__":
    X = np.arange(20).reshape(10, 2)
    y = np.array([sum(x) for x in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

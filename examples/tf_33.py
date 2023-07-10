import numpy as np
import pandas as pd

from libs.helpers import train_test_split
from neural_network.estimators.linear_classifier import LinearClassifier


def run():
    diabetes = pd.read_csv("./resources/pima-indians-diabetes.csv")
    diabetes.head()
    cols_to_norm = [
        "BMI",
        "Blood_pressure",
        "Glucose_concentration",
        "Insulin",
        "Number_pregnant",
        "Pedigree",
        "Triceps",
    ]

    diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    X = diabetes[cols_to_norm]
    y = diabetes["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=False
    )

    nn = LinearClassifier(n_features=len(X.columns))
    loss = nn.train(
        x_data=X_train.to_numpy(),
        y_data=y_train.to_numpy(),
        bach_size=10,
        epochs=1000,
    )

    print(nn.layer_dense.weights)
    print(nn.layer_dense.biases)

    print(loss)

    # X_test.i
    print(X.iloc[0:5])
    print(y.iloc[0:5])
    y_pred = nn.forward(X_test.iloc[0:5])

    print(y.iloc[:5])
    print(np.apply_along_axis(lambda x: [0] if x < 0.5 else [1], 1, y_pred))

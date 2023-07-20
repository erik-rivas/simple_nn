import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from models.mnist import MnistModel
from models.mnist_conv import MnistModelConv


class MnistSimpleClassifier:
    def fetch_data(self, images_to_read=1000):
        X_train, y_train, X_test, y_test = MnistModel.get_mnist(
            items_to_read=images_to_read
        )

        # One-hot encode the labels
        one_hot = OneHotEncoder(sparse_output=False)
        y_train_onehot = one_hot.fit_transform(y_train.reshape(-1, 1))
        y_test_onehot = one_hot.fit_transform(y_test.reshape(-1, 1))

        self.X_train = (X_train / 255.0).reshape(-1, 1, 28, 28)
        self.X_test = (X_test / 255.0).reshape(-1, 1, 28, 28)
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_onehot = y_train_onehot
        self.y_test_onehot = y_test_onehot

    def generate_network(
        self, epochs, batch_size, iter_per_batch, learning_rate, verbose
    ):
        # Create a model
        network = MnistModelConv(random_state=101)
        self.network = network

        network.train(
            self.X_train,
            self.y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            iter_per_batch=iter_per_batch,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        accuracy, precision, recall, f1_score = network.evaluate(
            self.X_test, self.y_test_onehot
        )
        print(
            f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}"
        )
        self.y_pred = self.network.predict(self.X_test)

        return network

    def plot_results(self):
        # get random images from test data set
        n_samples = 50
        sample_indexes = np.random.choice(
            self.X_test.shape[0], n_samples, replace=False
        )
        sample_images = self.X_test[sample_indexes]
        sample_labels = self.y_test[sample_indexes]
        # sample_labels_onehot = self.y_test_onehot[sample_indexes]

        # predict for sample images
        predictions = self.network.predict(sample_images)
        predicted_labels = np.argmax(predictions, axis=1)
        # Plot the n_samples images
        n_cols = 10
        n_rows = n_samples // n_cols
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))

        for i in range(n_samples):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(sample_images[i].reshape(28, 28), cmap="gray")
            title = f"{sample_labels[i]} :: {predicted_labels[i]}"
            ax.set(title=title, aspect=1, xticks=[], yticks=[])

        # Plotting
        plt.figure(figsize=(10, 7))

        # plt.subplot(2, 1, 2)
        self.network.plot_history(show=False)

        plt.show()

    def run(
        self, images_to_read, epochs, batch_size, iter_per_batch, learning_rate, verbose
    ):
        self.fetch_data(images_to_read=images_to_read)
        self.generate_network(
            epochs=epochs,
            batch_size=batch_size,
            iter_per_batch=iter_per_batch,
            learning_rate=learning_rate,
            verbose=verbose,
        )
        self.plot_results()


def run():
    simple_classifier = MnistSimpleClassifier()
    simple_classifier.run(
        images_to_read=None,
        epochs=5,
        batch_size=8,
        iter_per_batch=2,
        learning_rate=0.01,
        verbose=1000,
    )

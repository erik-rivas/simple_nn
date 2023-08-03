import numpy as np
import matplotlib.pyplot as plt

from models.lstm import LSTM

def run():
    # Generate a sine wave
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    sine_wave = np.sin(t)

    # Hyperparameters
    input_size = 5
    hidden_size = 10
    learning_rate = 0.1
    epochs = 200
    sequence_length = input_size
    batch_size = 1

    # LSTM instance
    lstm = LSTM(input_size, hidden_size)

    # Training loop
    for epoch in range(epochs):
        lstm.hidden_state = np.zeros((1, hidden_size))
        lstm.cell_state = np.zeros((1, hidden_size))
        for i in range(len(sine_wave) - sequence_length - 1):
            input_sequence = sine_wave[i:i+sequence_length].reshape((batch_size, sequence_length))
            target = sine_wave[i+sequence_length]
            output = lstm.forward(input_sequence)
            loss = 0.5 * (output - target) ** 2
            gradient = (output - target)
            d_inputs = lstm.backward(gradient)
            lstm.update(learning_rate)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {np.mean(loss)}")

    # Generate predictions
    predictions = []
    lstm.hidden_state = np.zeros((1, hidden_size))
    lstm.cell_state = np.zeros((1, hidden_size))
    for i in range(len(sine_wave) - sequence_length - 1):
        input_sequence = sine_wave[i:i+sequence_length].reshape((batch_size, sequence_length))
        predictions.append(lstm.forward(input_sequence))

    predictions = np.array(predictions).squeeze()

    # Plot the actual sine wave and the LSTM's predictions
    plt.plot(sine_wave[sequence_length+1:], label='Actual', c='r')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Sine Wave Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

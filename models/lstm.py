
import numpy as np
from libs.helpers import sigmoid, tanh
from neural_network.layers.layer_dense import Layer_Dense


class LSTM:
    def __init__(self, input_size, hidden_size):
        # Forget Gate
        self.forget_gate = Layer_Dense(input_size + hidden_size, hidden_size)
        # Input Gate
        self.input_gate = Layer_Dense(input_size + hidden_size, hidden_size)
        # Candidate layer
        self.candidate_layer = Layer_Dense(input_size + hidden_size, hidden_size)
        # Output Gate
        self.output_gate = Layer_Dense(input_size + hidden_size, hidden_size)

        # Initialize cell state and hidden state
        self.cell_state = np.zeros((1, hidden_size))
        self.hidden_state = np.zeros((1, hidden_size))

    def forward(self, x):
        # Concatenate input with previous hidden state
        self.combined_input = np.hstack((x, self.hidden_state))

        # Forget Gate
        forget_gate_output = sigmoid(self.forget_gate.forward(self.combined_input))

        # Input Gate
        input_gate_output = sigmoid(self.input_gate.forward(self.combined_input))

        # Candidate layer (new memory)
        candidate_memory = tanh(self.candidate_layer.forward(self.combined_input))

        # Update Cell State
        self.cell_state = forget_gate_output * self.cell_state + input_gate_output * candidate_memory

        # Output Gate
        output_gate_output = sigmoid(self.output_gate.forward(self.combined_input))

        # Update Hidden State
        self.hidden_state = output_gate_output * tanh(self.cell_state)

        return self.hidden_state

    def backward(self, gradient):
        # Gradients for Output Gate
        d_output_gate = gradient * tanh(self.cell_state)
        d_output_gate_input = d_output_gate * sigmoid(self.output_gate.forward(self.combined_input)) * (1 - sigmoid(self.output_gate.forward(self.combined_input)))
        d_output_gate_input = self.output_gate.backward(d_output_gate_input)

        # Gradients for Cell State
        d_cell_state = gradient * self.output_gate.forward(self.combined_input) * (1 - tanh(self.cell_state) ** 2)

        # Gradients for Input Gate
        d_input_gate = d_cell_state * self.candidate_layer.forward(self.combined_input)
        d_input_gate_input = d_input_gate * sigmoid(self.input_gate.forward(self.combined_input)) * (1 - sigmoid(self.input_gate.forward(self.combined_input)))
        d_input_gate_input = self.input_gate.backward(d_input_gate_input)

        # Gradients for Candidate Layer
        d_candidate = d_cell_state * sigmoid(self.input_gate.forward(self.combined_input))
        d_candidate_input = d_candidate * (1 - tanh(self.candidate_layer.forward(self.combined_input)) ** 2)
        d_candidate_input = self.candidate_layer.backward(d_candidate_input)

        # Gradients for Forget Gate
        d_forget_gate = d_cell_state * self.cell_state
        d_forget_gate_input = d_forget_gate * sigmoid(self.forget_gate.forward(self.combined_input)) * (1 - sigmoid(self.forget_gate.forward(self.combined_input)))
        d_forget_gate_input = self.forget_gate.backward(d_forget_gate_input)

        # Compute the gradient with respect to the input
        d_input = d_output_gate_input + d_input_gate_input + d_candidate_input + d_forget_gate_input

        return d_input

    def update(self, lr):
        # Update weights and biases for all gates
        self.forget_gate.update(lr)
        self.input_gate.update(lr)
        self.candidate_layer.update(lr)
        self.output_gate.update(lr)

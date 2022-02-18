import numpy as np
import utils
import typing
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test

class SoftmaxModel(SoftmaxModel):
    
    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for idx, size in enumerate(self.neurons_per_layer):
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                fan_in = self.neurons_per_layer[idx-1] if idx else self.I
                w = np.random.normal(0, 1/np.sqrt(fan_in), size=w_shape)
            else:
                w = np.random.uniform(-1, 1, size=w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        self.a_layers = []
        self.z_layers = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        def sigmoid(x):
            return 1 / (1 + np.exp(-1 * x))

        def softmax(x):
            X = np.exp(x)
            total = np.sum(X, axis=1, keepdims=True)
            return X / total


        a = X.copy()

        last_idx = len(self.ws) - 1
        self.a_layers = [X.copy()]
        self.z_layers = []
        for idx, w in enumerate(self.ws):
            z = np.dot(a, w)
            self.z_layers.append(z)
            
            if idx < last_idx:
                a = sigmoid(z) if not self.use_improved_sigmoid else 1.7159 * np.tanh(2*z / 3)
                self.a_layers.append(a)
            else:
                a = softmax(z)

            
            
        return a

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        # self.grads = []

        def sigmoid(x):
            return 1 / (1 + np.exp(-1 * x))

        def sigmoid_deriv(z):
            return sigmoid(z) * (1 - sigmoid(z))

        def improved_sigmoid_deriv(z):
            return (2*1.7159) / (3 *(np.square(np.cosh(2*z/3))))

        # Backprop for last layer first
        delta_k = -(targets - outputs)
        bs = X.shape[0]

        a_layers = self.a_layers
        self.grads[-1] = np.dot(a_layers[-1].T, delta_k) / bs
        
        delta = delta_k

        # Backprop for all other layers
        for layer_num in range(len(self.grads) - 2, -1, -1):
            z = self.z_layers[layer_num]
            f_prime = improved_sigmoid_deriv(z) if self.use_improved_sigmoid else sigmoid_deriv(z)
            w = self.ws[layer_num + 1]
            sum_wjj = np.dot(delta, w.T)
            delta = f_prime * sum_wjj
            self.grads[layer_num] = np.dot(self.a_layers[layer_num].T, delta) / bs
        
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."



    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Modify your network here
    neurons_per_layer = [64, 64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)

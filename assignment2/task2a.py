import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    batch_size = X.shape[0]
    mean = np.mean(X)
    std = np.std(X)
    X = (X-mean)/std
    ones = np.ones((batch_size, 1))
    X = np.hstack((X,ones))
    print(f"Mean: {mean}\nStandard deviation: {std}")
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 3a)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    batch_size = np.size(targets, 0)
    num_classes = np.size(targets, 1)
    C = 0
    for n in range(batch_size):
        Cn = 0
        for k in range(num_classes):
            Cn -= targets[n,k] * np.log(outputs[n,k])

        C += Cn 
    return C / batch_size


class SoftmaxModel:

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
        ws = []
        fan_in = self.I 
        
        for n in neurons_per_layer:
            if use_improved_weight_init:
                w = np.random.normal(0,1/np.sqrt(fan_in), (fan_in, n))
            else: 
                w = np.random.uniform(-1,1,(fan_in, n))
            fan_in = n + 1
            ws.append(w)  
        self.ws = ws
        self.grads = [None for i in range(len(self.ws))]

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

        a_prev = X
        batch_size = X.shape[0]
        zs = []
        for i, w in enumerate(self.ws):
            z = np.matmul(a_prev,w)
            if i == len(self.ws) -1:
                exp_zk = np.exp(z)
                exp_sum = exp_zk.sum(axis=1, keepdims=True)
                a = exp_zk / exp_sum
            else:
                zs.append(z)
                if self.use_improved_sigmoid:
                    a = 1.7159 * np.tanh(2/3 * z)
                else:
                    a = 1/(1 + np.exp(-z))
            a_prev = np.hstack((a, np.ones((batch_size ,1))))
        
        self.hidden_layer_output = zs
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

        delta = -(targets - outputs)
        batch_size = X.shape[0]
        zs = self.hidden_layer_output
        grads = []
        for i, w in reversed(list(enumerate(self.ws))):
            if i == 0:
                hidden_layer_with_bias = X
            else:
                z = zs[i - 1]
                if self.use_improved_sigmoid:
                    #TODO: Improved sigmoid does not work
                    a = 1.7159 * np.tanh(2/3 * z)
                    sigmoid_derivative = 1.14383/((np.cosh(z))**2)
                else: 
                    a = 1/(1 + np.exp(-z))
                    sigmoid_derivative =a * (1- a)
                hidden_layer_with_bias = np.hstack((a, np.ones((batch_size,1))))

            grad = 1/np.size(targets, 0) * np.matmul(hidden_layer_with_bias.T, delta)
            if i != 0:
                delta = sigmoid_derivative * np.matmul(delta, w[:-1,:].T)
            
            grads.append(grad)

        self.grads = list(reversed(grads))

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO implement this function (Task 3a)
    num_examples = np.size(Y,0)
    Y_new = np.zeros((num_examples, num_classes))
    for i, example in enumerate(Y):
        Y_new[i, example[0]] = 1
    return Y_new



def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)

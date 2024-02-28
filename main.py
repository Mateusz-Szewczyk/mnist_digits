import mnist
import numpy as np
from scipy.special import expit

X_test = mnist.test_images().astype(np.float16)
y_test = mnist.test_labels().astype(np.float16)

X_train = mnist.train_images().astype(np.float16)
y_train = mnist.train_labels().astype(np.float16)


def vectorized(labels) -> np.array:
    arr = []
    for label in labels:
        e = np.zeros((10, 1))
        e[int(label)] = 1.0
        arr.append(e)
    return np.array(arr)


def squeezed(data) -> np.array:
    return np.array([x.reshape(-1, 1) for x in data])


training_data = [[X, y] for X, y in zip(squeezed(X_train), vectorized(y_train))]
test_data = [[X, y] for X, y in zip(squeezed(X_test), vectorized(y_test))]


def sigmoid(z):
    return expit(z)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, sizes, lmbda=0.1):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.lmbda = lmbda
        self.biases = [np.random.randn(y, 1) / np.sqrt(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * self.lmbda / len(mini_batch)) * w - (eta / len(mini_batch)) * nw for w, nw in
                        zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y



def main():
    net = NeuralNetwork([784, 100, 10])
    net.SGD(training_data, 50, 30, 0.001, test_data=test_data)


if __name__ == "__main__":
    main()

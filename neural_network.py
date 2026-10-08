import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1 #количество слоёв с весами

        self.params = {}

        for l in range(1, self.num_layers + 1):
            fan_in = layer_sizes[l - 1]
            fan_out = layer_sizes[l]

            self.params[f'W{l}'] = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in) #используем инициализацию Хе
            self.params[f'b{l}'] = np.zeros((fan_out, 1))

        print(f"Нейросеть создана с архитектурой: {layer_sizes}")

    def count_parameters(self):
        total = 0

        for l in range(1, self.num_layers + 1):
            total += self.params[f'W{l}'].size + self.params[f'b{l}'].size
        return total

    def relu(self, z): #функция активации 
        return np.maximum(0, z)

    def relu_derivative(self, z): #обратное распространение
        return (z > 0).astype(float)

    def softmax(self, z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp = np.exp(z_shifted)

        return exp / np.sum(exp, axis=0, keepdims=True)

    def forward(self, x):
        cache = {}
        A = x

        for l in range(1, self.num_layers):
            cache[f'A{l-1}'] = A
            z = self.params[f'W{l}'] @ A + self.params[f'b{l}']
            cache[f'z{l}'] = z

            A = self.relu(z)
            cache[f'A{l}'] = A

        l = self.num_layers
        cache[f'A{l-1}'] = A

        z = self.params[f'W{l}'] @ A + self.params[f'b{l}']
        cache[f'z{l}'] = z

        A_last = self.softmax(z)
        cache[f'A{l}'] = A_last

        return A_last, cache  

    def predict(self, x):
        probabilities, _ = self.forward(x)

        predictions = np.argmax(probabilities, axis=0)

        return predictions, probabilities

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[1]

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m

        return loss

    def backward(self, y_true, cache):
        m = y_true.shape[1]
        grads = {}
        L = self.num_layers

        dz_current = cache[f'A{L}'] - y_true

        grads[f'dW{L}'] = (1/m) * dz_current @ cache[f'A{L-1}'].T
        grads[f'db{L}'] = (1/m) * np.sum(dz_current, axis=1, keepdims=True)

        for l in range(L - 1, 0, -1):
            dz_current = self.params[f'W{l+1}'].T @ dz_current * self.relu_derivative(cache[f'z{l}'])

            grads[f'dW{l}'] = (1/m) * dz_current @ cache[f'A{l-1}'].T
            grads[f'db{l}'] = (1/m) * np.sum(dz_current, axis=1, keepdims=True)

        return grads
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.num_layers + 1):
            self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']


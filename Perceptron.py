import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        # Inicializa os pesos aleatoriamente e define taxa de aprendizado e número de épocas
        self.weights = np.random.rand(input_size + 1)  # Adiciona +1 para o bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        # Função de ativação degrau
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Adiciona o bias (valor constante 1) às entradas
        inputs = np.append(inputs, 1)  # Adiciona o bias
        # Calcula a soma ponderada
        weighted_sum = np.dot(inputs, self.weights)
        # Aplica a função de ativação
        return self.activation(weighted_sum)

    def train(self, training_inputs, labels):
        # Treina o perceptron para ajustar os pesos
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Atualiza os pesos com base no erro
                error = label - prediction
                self.weights[:-1] += self.learning_rate * error * inputs  # Atualiza pesos
                self.weights[-1] += self.learning_rate * error  # Atualiza bias

# Exemplo de uso:
# Dados de entrada (OR lógico)
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Labels correspondentes (OR lógico)
labels = np.array([0, 1, 1, 1])

# Inicializa o perceptron
perceptron = Perceptron(input_size=2)

# Treina o perceptron
perceptron.train(training_inputs, labels)

# Testa o perceptron
for inputs in training_inputs:
    print(f"Entrada: {inputs} -> Predição: {perceptron.predict(inputs)}")

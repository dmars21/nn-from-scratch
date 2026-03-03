import math
import random

# -----------------------
# Funzione sigmoid
# -----------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s) 

# -----------------------
# Classe neurone singolo
# -----------------------

class BinaryNeuron:

    def __init__(self, lr=0.1):
        self.w1 = random.random()
        self.w2 = random.random()
        self.b = random.random()
        self.learning_rate = lr

    # Forward propagation
    def forward(self, x1, x2):
        self.z = self.w1 * x1 + self.w2 * x2 + self.b
        self.output = sigmoid(self.z)
        return self.output

    # Backward propagation
    def backward(self, x1, x2, target):
        #Loss utilizzata -> Mean Squared Error
        self.loss = (self.output - target) ** 2

        #Derivata loss rispetto output
        self.d_loss_d_output = 2 * (self.output - target)

        #Derivata output rispetto z
        d_output_d_z = sigmoid_derivative(self.z)

        #Chain rule
        d_loss_d_z = self.d_loss_d_output * d_output_d_z

        #Derivate rispetto ai pesi
        d_loss_d_w1 = d_loss_d_z * x1
        d_loss_d_w2 = d_loss_d_z * x2
        d_loss_d_b = d_loss_d_z

        # Aggiornamento pesi
        self.w1 -= self.learning_rate * d_loss_d_w1
        self.w2 -= self.learning_rate * d_loss_d_w2
        self.b -= self.learning_rate * d_loss_d_b

        return self.loss

    # Training
    def train(self, data, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for x1, x2, target in data:
                self.forward(x1, x2)
                self.backward(x1, x2, target)
                total_loss += self.loss

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - Loss: {total_loss}")

# -----------------------
# Dataset funzione logica AND
# -----------------------

training_data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]

# -----------------------
# Creazione e training
# -----------------------

neuron = BinaryNeuron()
neuron.train(training_data, epochs=10000)

# -----------------------
# Inference
# -----------------------

print("\nInference:")
for x1, x2, _ in training_data:
    output = neuron.forward(x1, x2)
    output_bin = round(output)
    print(f"{x1}, {x2} -> {output_bin} [{output:.4f}]")
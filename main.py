import math
import random

# -----------------------
# Softmax
# -----------------------

def softmax(z):
    exp_values = [math.exp(i) for i in z]
    total = sum(exp_values)
    return [i / total for i in exp_values]

# -----------------------
# Classe classificatore
# -----------------------

class MulticlassNeuron:

    def __init__(self, lr=0.1):
        # 3 neuroni di output, ognuno con 2 pesi
        self.weights = [
            [random.random(), random.random()],
            [random.random(), random.random()],
            [random.random(), random.random()]
        ]
        self.biases = [random.random(), random.random(), random.random()]
        self.learning_rate = lr

    # Forward
    def forward(self, x1, x2):
        self.inputs = [x1, x2]

        self.z = []
        for i in range(3):
            z_i = (
                self.weights[i][0] * x1 +
                self.weights[i][1] * x2 +
                self.biases[i]
            )
            self.z.append(z_i)

        self.output = softmax(self.z)
        return self.output

    # Backward (cross-entropy + softmax)
    def backward(self, target):
        # target è one-hot, es: [1,0,0]

        self.loss = 0
        for i in range(3):
            # Cross-entropy loss
            self.loss += target[i] * math.log(self.output[i] + 1e-9)

            # Derivata combinata softmax + cross entropy
            d_loss_d_z = self.output[i] - target[i]

            # Aggiornamento pesi
            self.weights[i][0] -= self.learning_rate * d_loss_d_z * self.inputs[0]
            self.weights[i][1] -= self.learning_rate * d_loss_d_z * self.inputs[1]

            # Aggiornamento bias
            self.biases[i] -= self.learning_rate * d_loss_d_z
        
        self.loss *= -1

    # Training
    def train(self, data, epochs):
        for epoch in range(epochs):
            total_loss = 0

            for x1, x2, target in data:
                self.forward(x1, x2)
                self.backward(target)
                
                total_loss += self.loss
                
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - Loss: {total_loss:.4f}")

# -----------------------
# Dataset esempio (3 classi)
# -----------------------
# Classe 0 → vicino (0,0)
# Classe 1 → vicino (1,0)
# Classe 2 → vicino (0,1)

training_data = [
    (0, 0, [1,0,0]),
    (0.1, 0.2, [1,0,0]),

    (1, 0, [0,1,0]),
    (0.9, 0.2, [0,1,0]),

    (0, 1, [0,0,1]),
    (0.2, 0.9, [0,0,1]),
]

# -----------------------
# Training
# -----------------------

model = MulticlassNeuron()
model.train(training_data, epochs=10000)

# -----------------------
# Inference
# -----------------------

print("\nInference:")
for x1, x2, target in training_data:
    probs = model.forward(x1, x2)
    predicted_class = probs.index(max(probs))
    print(f"{x1}, {x2} -> Classe {predicted_class}  Prob: {probs}")
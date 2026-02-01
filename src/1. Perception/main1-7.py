import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Simple 2-layer network: 2 inputs -> 2 hidden -> 1 output
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 2)  # Hidden layer with 2 neurons
        self.output = nn.Linear(2, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


# Initialize network, loss, and optimizer
# Weights are chosen randomly, does not always converge - networks with more
# hidden units will converge more often
model = XORNet()

# Print the random initial weights
print("Initial weights:")
print("Hidden layer:", model.hidden.weight.data)
print("Hidden bias:", model.hidden.bias.data)
print("Output layer:", model.output.weight.data)
print("Output bias:", model.output.bias.data)

# Same squared error that Widrow and Hoff used, just taking the average across all 4 examples
# This is known as "batch" or "minibatch" gradient descent.
criterion = nn.MSELoss()

# Using the Adam optimizer here instead of vanilla SGD, SGD gets stuck when model is small.
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the network
print("\nResults:")
with torch.no_grad():
    for i in range(len(X)):
        output = model(X[i : i + 1])
        print(
            f"Input: {X[i].numpy()}, Target: {y[i].item()}, Output: {output.item():.4f}"
        )

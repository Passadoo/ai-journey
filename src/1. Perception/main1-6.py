import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Linearly Separable case
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([[-1], [1], [1], [1]])
w0_range = np.arange(-1.9, 2.0, 0.2)
w1_range = np.arange(-1.9, 2.0, 0.2)
b = 1  # Bias term

# Initialize lists to store results
w0_points = []
w1_points = []
error_points = []

# Compute error for each weight combination
for w0 in w0_range:
    for w1 in w1_range:
        yhat = X[:, 0] * w0 + X[:, 1] * w1 + b  # Compute all 4 yhats at once
        error = np.mean((y.ravel() - yhat) ** 2)  # Mean squared error

        # Store the results
        w0_points.append(w0)
        w1_points.append(w1)
        error_points.append(error)

# Convert to numpy arrays
w0_points = np.array(w0_points)
w1_points = np.array(w1_points)
error_points = np.array(error_points)
# Create 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Create scatter plot with color mapping
scatter = ax.scatter(
    w0_points, w1_points, error_points, c=error_points, cmap="viridis", alpha=0.6, s=20
)

# Add labels and title
ax.set_xlabel("Weight w0")
ax.set_ylabel("Weight w1")
ax.set_zlabel("Mean Squared Error")
ax.set_title("Perceptron Error Surface\n(Linearly Separable Case)")

plt.show()

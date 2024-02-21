# Understanding-Activation-functions-in-Neural-Networks
### **1. Introduction to Activation Functions**

Activation functions determine the output of a neural network, its accuracy, and the computational efficiency of training a model. Their main purpose is to introduce non-linearity into the output of a neuron.

### **2. Types of Activation Functions**

#### **2.1 Linear Activation Function**
- **Formula**: \(f(x) = x\)
- It's a straight line that essentially means the output is proportional to the input.
- Rarely used in hidden layers of deep learning models because they don't introduce non-linearity.

```
def linear_activation(x):
    return x

# Test
inputs = [-2, -1, 0, 1, 2]
outputs = [linear_activation(i) for i in inputs]

print(outputs)  # Expected output: [-2, -1, 0, 1, 2]
```

```
import numpy as np
import matplotlib.pyplot as plt

# Define the linear activation function
def linear_activation(x):
    return x

# Generate a range of values
x = np.linspace(-10, 10, 400)

# Apply the activation function
y = linear_activation(x)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Linear Activation", color="blue")
plt.title("Linear Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()
```


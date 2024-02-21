# Understanding-Activation-functions-in-Neural-Networks

![newContent12](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/4ed358a3-ab2f-4154-b03d-406a9a49c30c)

### **1. Introduction to Activation Functions**

Activation functions determine the output of a neural network, its accuracy, and the computational efficiency of training a model. Their main purpose is to introduce non-linearity into the output of a neuron.

### What Are Activation Functions?
Activation functions are an integral building block of neural networks that enable them to learn complex patterns in data. They transform the input signal of a node in a neural network into an output signal that is then passed on to the next layer. Without activation functions, neural networks would be restricted to modeling only linear relationships between inputs and outputs.

### Why Are Activation Functions Essential?
Without activation functions, neural networks would just consist of linear operations like matrix multiplication. All layers would perform linear transformations of the input, and no non-linearities would be introduced.

Most real-world data is non-linear. For example, relationships between house prices and size, income, and purchases, etc., are non-linear. If neural networks had no activation functions, they would fail to learn the complex non-linear patterns that exist in real-world data.

Activation functions enable neural networks to learn these non-linear relationships by introducing non-linear behaviors through activation functions. This greatly increases the flexibility and power of neural networks to model complex and nuanced data.

![37914Screenshot (45)](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/47dbbd31-54e8-4d2f-8eeb-21ad31456de1)

### Activation Functions and their Derivatives
![94131Screenshot (43)](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/0bcb4fd7-8294-4713-849f-1786fc657afe)

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
![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/b6c3a8d2-898b-49c1-a5eb-7b0d8fa9a2d4)

#### **2.2 Sigmoid Activation Function**
- It compresses the output between 0 and 1.
- Problems: Vanishing gradient problem especially in deep networks.

```
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Test
x_values = np.linspace(-10, 10, 100)  # create 100 points between -10 and 10
y_values = sigmoid(x_values)

# Plotting
import matplotlib.pyplot as plt

plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
```
![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/701217bb-d877-43f4-ad8a-456daf251cc4)

#### **2.3 Hyperbolic Tangent (tanh) Function**
- It compresses the output between -1 and 1.
- Also suffers from the vanishing gradient problem, but less so than the sigmoid.

```
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the tanh function
def tanh_function(x):
    return np.tanh(x)

# Generate an array of values from -10 to 10 to represent our x-axis
x = np.linspace(-10, 10, 400)

# Compute tanh values for each x
y = tanh_function(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='tanh(x)', color='blue')
plt.title('Hyperbolic Tangent Function (tanh)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Setting the x and y axis limits
plt.axhline(y=0, color='black',linewidth=0.5)
plt.axvline(x=0, color='black',linewidth=0.5)
plt.legend()
plt.show()
```
![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/05ac666f-e20e-413b-89c8-ac609ec9a920)

#### **2.4 Rectified Linear Unit (ReLU)**
- **Formula**: \(f(x) = max(0, x)\)
- Popular and widely used.
- Introduces non-linearity.
- Problems: Dying ReLU problem where neurons can sometimes get stuck during training and not activate at all.

```
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate data
x = np.linspace(-10, 10, 400)
y = relu(x)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ReLU Function', color='blue')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid()
plt.legend()
plt.show()
```
![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/90a20353-d8bd-4225-968b-386ff7baa402)


#### **2.5 Leaky ReLU**
- Variation of ReLU.
- Tries to fix the Dying ReLU problem by allowing small negative values.

```
import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate data
x = np.linspace(-10, 10, 400)
y = leaky_relu(x)

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Leaky ReLU', color='blue')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid()
plt.legend()
plt.show()
```
![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/436c3857-a365-4464-b93e-00371bfb930c)

#### **2.6 Exponential Linear Unit (ELU)**
- Tries to make the mean activations closer to zero which speeds up training.
- Mitigates the dying ReLU problem.

```
#Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
#Step 2: Define the ELU function
def elu(x, alpha=1.0):
    """ELU activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

#Step 3: Visualize the ELU function
# Generate data
x = np.linspace(-10, 10, 400)
y = elu(x)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='ELU function', color='blue')
plt.title('Exponential Linear Unit (ELU) Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()
```

![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/98b539fa-bd52-4ba5-92bd-c53a8ae24f57)

#### **2.7 Softmax Activation Function**
- Used in the output layer of the classifier where we are trying to attain the probabilities to define the class of each input.

```
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example input
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

labels = ['x1', 'x2', 'x3']
plt.bar(labels, outputs)
plt.ylabel('Probability')
plt.title('Softmax Activation Output')
plt.show()
```
![download](https://github.com/alsani-ipe/Understanding-Activation-functions-in-Neural-Networks/assets/77545137/1fcfc60f-dee3-4350-a8d7-4ad99010592b)

### **3. How to Choose an Activation Function**

1. **Understand the Purpose**:
   - **Classification**: For binary classification, sigmoid is common in the output layer, while softmax is standard for multi-class tasks. For hidden layers, ReLU and its variants often work well.
   - **Regression**: Linear or identity functions are typical for the output layer, with ReLU-based functions in hidden layers.

2. **Avoid Saturating Activations for Deep Networks**:
   - Functions like the sigmoid and tanh can saturate (output values close to their min/max) causing vanishing gradient problems, especially in deep networks.
   - ReLU and its variants are often preferred due to their non-saturating nature.

3. **Address Dying Units with Modified ReLUs**:
   - Vanilla ReLU can cause dying units (neurons that stop learning) because they output zero for negative inputs. 
   - Variants such as Leaky ReLU, Parametric ReLU, and Exponential Linear Unit (ELU) can alleviate this by allowing small negative outputs.

4. **Consider Computational Efficiency**:
   - The complexity of some functions (e.g., sigmoid or tanh) might not be suitable for real-time applications or when computational resources are limited.
   - ReLU and its variants offer computational advantages due to their simplicity.

5. **Factor in Initialization**:
   - Some activation functions, like Sigmoid or tanh, benefit from Xavier/Glorot initialization.
   - ReLU-based functions often work well with He initialization.

6. **Understand Task Specificity**:
   - **Time-series**: Activation functions that preserve certain properties of the input (like tanh, which preserves sign) might be beneficial.
   - **Computer vision**: ReLU and its variants have shown strong performance due to their capability to handle non-linearities.

7. **Consider Gradient Stability**:
   - It's vital to maintain a stable gradient flow, especially in deep networks.
   - Avoid functions that might result in exploding or vanishing gradients. For example, if using tanh or sigmoid, be wary of the network's depth and other hyperparameters.

8. **Custom or Novel Activations**:
   - Sometimes, a problem may require a custom activation function. Experimentation can help uncover if a tailored function provides benefits.
   - Stay updated with recent literature, as newer activations like Swish and Mish have emerged and shown promise.

9. **Safety with Sparsity**:
   - If sparsity (having more zeros in the output) is desired, functions like ReLU naturally induce it.
   - However, too much sparsity might not be always beneficial. Balance is key.

10. **Regularization Interplay**:
   - Consider how the activation function interacts with regularization techniques like dropout. 
   - Some combinations might work synergistically, while others could hinder learning.

11. **Batch Normalization Impact**:
   - Using Batch Normalization can alleviate some issues tied to activation functions by normalizing layer outputs.
   - This can make the choice of activation function less critical, but it's still an essential consideration.

12. **Empirical Testing**:
   - Theoretical insights are helpful, but empirical testing on validation data is crucial.
   - Always benchmark different activation functions under the same conditions to determine the best fit.

13. **Understand the Output Range**:
   - Recognize the range of your desired output. For instance, using a ReLU in the output layer of a regression task might not be ideal if negative outputs are possible.

14. **Problem Constraints**:
   - In certain applications, the interpretability of a model might be vital. Some activation functions can make models more interpretable than others.

15. **Research and Community Consensus**:
   - Often, the broader machine learning community converges on certain best practices for specific tasks. Stay updated with the latest research.


# Conclusion
**Please feel free to ask in the comment section if you have any confusion or questions.**

**Here are some of the contributions I've made on Kaggle:**
1. [Pie Charts in Python](https://www.kaggle.com/code/alsaniipe/pie-charts-in-python)
1. [Scatter plots with Plotly Express](https://www.kaggle.com/code/alsaniipe/scatter-plots-with-plotly-express)
1. [X-ray Image Classification using Transfer Learning](https://www.kaggle.com/code/alsaniipe/x-ray-image-classification-using-transfer-learning)
1. [Flowers Classification by Using VGG16 Model 🎉🎉](https://www.kaggle.com/code/alsaniipe/flowers-classification-by-using-vgg16-model)
1. [Car Brand Prediction's by Using ResNet50 Model](https://www.kaggle.com/code/alsaniipe/car-brand-prediction-s-by-using-resnet50-model)
1. [Image Preprocessing-Morpological Analysis & Kernel](https://www.kaggle.com/code/alsaniipe/image-preprocessing-morpological-analysis-kernel)
1. [Image Similarity Index (SSIM analysis )](https://www.kaggle.com/code/alsaniipe/image-similarity-index-ssim-analysis)
1. [Image Preprocessing- Image Transformation & OpenCV](https://www.kaggle.com/code/alsaniipe/image-preprocessing-image-transformation-opencv)



# All Content Credit:
1. https://github.com/BytesOfIntelligences
2. https://www.datacamp.com/tutorial/introduction-to-activation-functions-in-neural-networks
3. https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/









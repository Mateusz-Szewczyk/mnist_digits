# Neural Network for Handwritten Digit Recognition
This repository contains a neural network implementation for handwritten digit recognition using the MNIST dataset. The network is implemented from scratch using Python and NumPy.

## Dataset
The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. It consists of 60,000 training images and 10,000 testing images, each of which is a grayscale image of size 28x28 pixels.

## Neural Network Architecture
The neural network architecture used in this implementation consists of an input layer with 784 neurons (28x28 pixels), 
followed by a hidden layer with 100 neurons and an output layer with 10 neurons (corresponding to digits 0-9). 
The activation function used in the hidden layer is the sigmoid function.

## Training
The neural network is trained using stochastic gradient descent (SGD) with mini-batch optimization. 
The training parameters include the number of epochs, mini-batch size, learning rate (eta), and regularization parameter (lambda).

## Learning and Improvement
Throughout this project, I embarked on a journey of learning and discovery in the field of neural networks. 
The hands-on experience provided by implementing this neural network for handwritten digit recognition allowed me to deepen my understanding of several key concepts.

### Concepts Explored
* **SGD Implementation**: Developing SGD from scratch enhanced my understanding of optimizing neural network parameters efficiently.

* **Gradients**: Grasping how gradients guide the learning process elucidated how networks adjust weights and biases to minimize loss.

* **Regularization Techniques**: Exploring L2 regularization expanded my knowledge of preventing overfitting and enhancing generalization.

* **Learning Rate Exploration**: Experimenting with different learning rates highlighted the delicate balance between convergence speed and training stability.

* **Fundamentals of Neural Networks**: Understanding feedforward and backpropagation provided a solid foundation for further learning.

### Future Goals
I aim to achieve a ***98% accuracy rate***, which is a challenging goal. Reaching this milestone will deepen my understanding of neural networks. 
It's not just about enhancing the model's performance; it's about grasping how these networks operate, paving the way for tackling more complex tasks in the future.

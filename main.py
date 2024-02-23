import mnist
import numpy as np
import matplotlib.pyplot as plt

X_test = mnist.test_images().astype(np.float32)
y_test = mnist.test_labels().astype(np.float32)

X_train = mnist.train_images().astype(np.float32)
y_train = mnist.train_labels().astype(np.float32)

plt.imshow(X_train[0], cmap="binary")
plt.show()

print(y_train[0])

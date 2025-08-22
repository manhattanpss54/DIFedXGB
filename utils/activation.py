import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return e_z / e_z.sum(axis=1, keepdims=True)
import numpy as np
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

expected_outputs = np.array([[0,1,1,0]]).T

with open('training.pickle', 'rb') as f:
    synaptic_weights = pickle.load(f)

output = sigmoid(np.dot(training_inputs, synaptic_weights))


print("The expected results are: ")
print(output)


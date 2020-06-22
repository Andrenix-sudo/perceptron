import numpy as np
import pickle


# activation function - converts the output from the neural network to a sigmod between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# learning rate function - used to make adjustments to the Error
def sigmoid_deriviative(x):
    return x * (1-x)

# array of training inputs
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])


# array of expected results that the AI must get
training_outputs = np.array([[0,1,1,0]]).T


# initilze random weights for each input
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synpatics weights: ")
print(synaptic_weights)

# loops 1000000 times using the training inputs
for iteration in range(100000):

    input_layer = training_inputs

    # multiplies the synpatic weights by the training inputs then calls the sigmoid function to get a result between 0 and 1
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # calculates the error by delta of the expected vs actual outputs
    error = training_outputs - outputs

    # calculates adjustments by multiplying the error and the learning rate with outputs 
    adjustments = error * sigmoid_deriviative(outputs)

    # makes adjustments to the synpatic weights
    synaptic_weights += np.dot(input_layer.T, adjustments)


print("Synaptic weights after training: ")
print(synaptic_weights)

with open('training.pickle', 'wb') as f:
    pickle.dump(synaptic_weights, f)

print("Outputs after training: ")    
print(outputs)

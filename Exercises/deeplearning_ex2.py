import torch

torch.manual_seed(2023)


def activation_func(x):
    #TODO Implement one of these following activation function: sigmoid, tanh, ReLU, leaky ReLU
    epsilon = 0.01   # Only use this variable if you choose Leaky ReLU
    result = None
    return result

def softmax(x):
    # TODO Implement softmax function here
    result = None
    return result


# Define the size of each layer in the network
num_input = 784  # Number of node in input layer (28x28)
num_hidden_1 = 128  # Number of nodes in hidden layer 1
num_hidden_2 = 256  # Number of nodes in hidden layer 2
num_hidden_3 = 128  # Number of nodes in hidden layer 3
num_classes = 10  # Number of nodes in output layer

# Random input
input_data = torch.randn((1, num_input))
# Weights for inputs to hidden layer 1
W1 = torch.randn(num_input, num_hidden_1)
# Weights for hidden layer 1 to hidden layer 2
W2 = torch.randn(num_hidden_1, num_hidden_2)
# Weights for hidden layer 2 to hidden layer 3
W3 = torch.randn(num_hidden_2, num_hidden_3)
# Weights for hidden layer 3 to output layer
W4 = torch.randn(num_hidden_3, num_classes)

# and bias terms for hidden and output layers
B1 = torch.randn((1, num_hidden_1))
B2 = torch.randn((1, num_hidden_2))
B3 = torch.randn((1, num_hidden_3))
B4 = torch.randn((1, num_classes))

#TODO Calculate forward pass of the network here. Result should have the shape of [1,10]
# Dont forget to check if sum of result = 1.0
result = None
print(result)

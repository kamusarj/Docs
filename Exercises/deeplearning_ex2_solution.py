import torch

torch.manual_seed(2023)


def activation_func(x):
    return 1 / (1 + torch.exp(-x))

def softmax(x):
    # Explanation: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    max_value, _ = torch.max(x, dim=1)
    norm = torch.exp(x - max_value)
    return norm / norm.sum()


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

result = softmax(activation_func(torch.matmul(activation_func(
    torch.matmul(activation_func(torch.matmul(activation_func(torch.matmul(input_data, W1) + B1), W2) + B2), W3) + B3),
                                      W4) + B4))
print(result)

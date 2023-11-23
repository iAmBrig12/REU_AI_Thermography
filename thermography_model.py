import torch.nn as nn

# define neural network
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(Net, self).__init__()
        
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

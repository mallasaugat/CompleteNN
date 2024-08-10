import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Define convolutional layers and fully connected layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        # Define the forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create an instance of the network
model = Network()

# Load the pre-trained weights
model.load_state_dict(torch.load("./models/model.pth"))

# Set the model to evaluation mode
model.eval()

# Define a dummy input with appropriate dimensions (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model to ONNX format
onnx_filename = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_filename, verbose=True)

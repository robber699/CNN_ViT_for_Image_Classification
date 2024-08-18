import torch
import torch.nn as nn

class YourCNN(nn.Module):
    def __init__(self, num_classes):
        super(YourCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # Changed in_channels to 3
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #dropout
        self.dropout1 = nn.Dropout(p=0.5)

        # Fully Connected Layer
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.dropout2 = nn.Dropout(p=0.5)
        # Second Convolutional Layer
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)  # Adjusted input size


        
    def forward(self, x):
        # Forward pass through the first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Forward pass through the second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten the output for the fully connected layer
        x=torch.flatten(x,1)

        x = self.dropout1(x)
        # Forward pass through the fully connected layer
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)    
        x = self.fc2(x)
        
        return x

# Instantiate the model
model = YourCNN(num_classes=10)

# Print the model architecture
#print(model)

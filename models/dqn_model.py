import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQNModel, self).__init__()
        
        # Input shape: (N, channels, height, width)
        # channels = 6 (own body, opponent body, food, walls, power-ups, portals)
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the flattened size
        self.flatten_size = 64 * input_shape[1] * input_shape[2]
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.flatten_size)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class DuelingDQNModel(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DuelingDQNModel, self).__init__()
        
        # Shared feature extraction
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the flattened size
        self.flatten_size = 64 * input_shape[1] * input_shape[2]
        
        # Value stream
        self.value_fc = nn.Linear(self.flatten_size, 256)
        self.value = nn.Linear(256, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(self.flatten_size, 256)
        self.advantage = nn.Linear(256, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.flatten_size)
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Subtracting the mean helps with stability
        return value + advantage - advantage.mean(dim=1, keepdim=True)
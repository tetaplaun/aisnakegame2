import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SharedFeatures(nn.Module):
    def __init__(self, input_shape):
        super(SharedFeatures, self).__init__()

        # Input shape: (N, channels, height, width)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate flattened size
        self.flatten_size = 64 * input_shape[1] * input_shape[2]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x.view(-1, self.flatten_size)


class Actor(nn.Module):
    def __init__(self, input_shape, action_size):
        super(Actor, self).__init__()

        # Feature extraction
        self.features = SharedFeatures(input_shape)

        # Policy head
        self.fc1 = nn.Linear(self.features.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, action_size)

    def forward(self, x):
        features = self.features(x)

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        # Policy output (action probabilities)
        policy = F.softmax(self.policy_head(x), dim=1)

        return policy

    def get_action(self, state, device):
        # Check if state is already a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        elif state.dim() == 3:  # If it's a tensor but needs to be unsqueezed
            state = state.unsqueeze(0).to(device)
        else:
            # Ensure state is on the correct device
            state = state.to(device)

        policy = self.forward(state)

        # Create a distribution and sample from it
        m = Categorical(policy)
        action = m.sample()

        return action.item(), m.log_prob(action)


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()

        # Feature extraction
        self.features = SharedFeatures(input_shape)

        # Value head
        self.fc1 = nn.Linear(self.features.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.features(x)

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        # Value output (state value estimation)
        value = self.value_head(x)

        return value


class RecurrentActorCritic(nn.Module):
    def __init__(self, input_shape, action_size, hidden_size=256):
        super(RecurrentActorCritic, self).__init__()

        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate flattened size
        self.flatten_size = 64 * input_shape[1] * input_shape[2]

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(self.flatten_size, hidden_size, batch_first=True)

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initial hidden and cell states
        self.hidden = None

    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state
        return (torch.zeros(1, batch_size, 256).to(device),
                torch.zeros(1, batch_size, 256).to(device))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten spatial features
        x = x.view(batch_size, 1, self.flatten_size)  # Add time dimension for LSTM

        # Process through LSTM
        if hidden is None:
            self.hidden = self.init_hidden(batch_size, x.device)

        lstm_out, self.hidden = self.lstm(x, hidden if hidden else self.hidden)
        lstm_out = lstm_out.view(batch_size, -1)

        # Get policy and value
        policy = F.softmax(self.actor(lstm_out), dim=1)
        value = self.critic(lstm_out)

        return policy, value, self.hidden

    def get_action(self, state, device):
        # Check if state is already a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        elif state.dim() == 3:  # If it's a tensor but needs to be unsqueezed
            state = state.unsqueeze(0).to(device)
        else:
            # Ensure state is on the correct device
            state = state.to(device)

        policy, value, _ = self.forward(state)

        # Create a distribution and sample from it
        m = Categorical(policy)
        action = m.sample()

        return action.item(), m.log_prob(action), value
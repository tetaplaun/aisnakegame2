import numpy as np
import random
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from environment import DualSnakeEnv
from models.dqn_model import DQNModel, DuelingDQNModel


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        snake_id,
        input_shape,
        action_size,
        dueling=True,
        gamma=0.99,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        min_replay_size=1000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=10000,
        double_dqn=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.snake_id = snake_id
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.double_dqn = double_dqn
        self.device = device
        
        # Initialize epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize networks
        if dueling:
            self.q_network = DuelingDQNModel(input_shape, action_size).to(device)
            self.target_network = DuelingDQNModel(input_shape, action_size).to(device)
        else:
            self.q_network = DQNModel(input_shape, action_size).to(device)
            self.target_network = DQNModel(input_shape, action_size).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_step = 0
    
    def get_action(self, state, training=True):
        # Implement epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.action_size - 1)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def update_epsilon(self):
        # Decay epsilon over time
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.training_step / self.epsilon_decay)
    
    def train(self, env, num_episodes=1000, max_steps=1000, save_freq=100, render=False, opponent_action_func=None):
        """
        Train the DQN agent.
        
        Args:
            env: The environment to train on
            num_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            save_freq: How often to save model checkpoints
            render: Whether to render the environment during training
            opponent_action_func: Function to get opponent's action; if None, random actions are used
        """
        # Create save directory if it doesn't exist
        save_dir = f'./saved_models/dqn_snake{self.snake_id}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        win_count = 0
        lose_count = 0
        
        # Main training loop
        for episode in range(num_episodes):
            # Reset environment and get initial state
            observations = env.reset()
            state = observations[f'snake{self.snake_id}']
            
            episode_reward = 0
            steps = 0
            done = False
            
            # Episode loop
            while not done and steps < max_steps:
                # Choose action for the agent's snake
                action = self.get_action(state)
                
                # Choose action for the opponent's snake
                if opponent_action_func is None:
                    # Random opponent if no function provided
                    opponent_action = random.randint(0, self.action_size - 1)
                else:
                    # Use provided function for opponent's action
                    opponent_id = 2 if self.snake_id == 1 else 1
                    opponent_state = observations[f'snake{opponent_id}']
                    opponent_action = opponent_action_func(opponent_state)
                
                # Create action dictionary based on snake ID
                if self.snake_id == 1:
                    actions = {'snake1': action, 'snake2': opponent_action}
                else:
                    actions = {'snake1': opponent_action, 'snake2': action}
                
                # Take step in environment
                next_observations, rewards, done, info = env.step(actions)
                next_state = next_observations[f'snake{self.snake_id}']
                reward = rewards[f'snake{self.snake_id}']
                
                # Render if enabled
                if render and episode % 100 == 0:  # Render every 100 episodes
                    env.render()
                    time.sleep(0.01)
                
                # Store experience in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Train if enough samples in replay buffer
                if len(self.replay_buffer) > self.min_replay_size:
                    self._train_step()
                    self.update_epsilon()
                
                # Update target network periodically
                if self.training_step % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Update episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Update win/loss counts
            if 'winner' in info and info['winner'] is not None:
                if info['winner'] == self.snake_id:
                    win_count += 1
                else:
                    lose_count += 1
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                win_rate = win_count / (win_count + lose_count) if (win_count + lose_count) > 0 else 0
                
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Win Rate: {win_rate:.2f}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                save_path = os.path.join(save_dir, f'model_episode_{episode+1}.pth')
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.q_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'training_step': self.training_step
                }, save_path)
                print(f"Model saved to {save_path}")
        
        # Save final model
        final_save_path = os.path.join(save_dir, 'model_final.pth')
        torch.save({
            'episode': num_episodes,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, final_save_path)
        print(f"Final model saved to {final_save_path}")
        
        # Return training metrics
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'win_count': win_count,
            'lose_count': lose_count
        }
    
    def _train_step(self):
        """Perform a single training step."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Compute next Q values using target network
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Use online network to select actions, target network to evaluate them
                next_actions = self.q_network(next_state_batch).max(1)[1].unsqueeze(1)
                next_q_values = self.target_network(next_state_batch).gather(1, next_actions)
            else:
                # Regular DQN
                next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            
            # Compute target Q values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update training step
        self.training_step += 1
        
        return loss.item()
    
    def load(self, model_path):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example training script
    env = DualSnakeEnv(grid_size=20, max_steps=1000)
    observations = env.reset()
    
    # Get observation shape and action size
    input_shape = observations['snake1'].shape  # (channels, height, width)
    action_size = 3  # 0: continue, 1: turn left, 2: turn right
    
    # Create DQN agent
    agent = DQNAgent(
        snake_id=1,
        input_shape=input_shape,
        action_size=action_size,
        dueling=True,
        double_dqn=True
    )
    
    # Train the agent
    training_metrics = agent.train(
        env=env,
        num_episodes=1000,
        max_steps=1000,
        save_freq=100,
        render=False
    )
    
    # Close environment
    env.close()
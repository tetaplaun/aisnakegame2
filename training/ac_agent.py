import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from collections import deque
from environment import DualSnakeEnv
from models.ac_model import Actor, Critic, RecurrentActorCritic


class RolloutBuffer:
    def __init__(self, capacity):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.capacity = capacity
        self.size = 0
        
    def push(self, state, action, reward, next_state, done, log_prob=None, value=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if log_prob is not None:
            self.log_probs.append(log_prob)
        
        if value is not None:
            self.values.append(value)
            
        self.size += 1
        
        if self.size > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            
            if log_prob is not None and len(self.log_probs) > 0:
                self.log_probs.pop(0)
                
            if value is not None and len(self.values) > 0:
                self.values.pop(0)
                
            self.size = self.capacity
            
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.size = 0
        
    def get_batch(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
            self.values
        )


class A2CAgent:
    def __init__(
        self,
        snake_id,
        input_shape,
        action_size,
        gamma=0.99,
        learning_rate=3e-4,
        entropy_weight=0.01,
        value_loss_weight=0.5,
        recurrent=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.snake_id = snake_id
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.device = device
        self.recurrent = recurrent
        
        # Initialize networks
        if recurrent:
            self.network = RecurrentActorCritic(input_shape, action_size).to(device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        else:
            self.actor = Actor(input_shape, action_size).to(device)
            self.critic = Critic(input_shape).to(device)
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=learning_rate
            )
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(capacity=10000)  # Different from DQN, it uses rollout trajectory
        
        # Training metrics
        self.training_step = 0
    
    def get_action(self, state, training=True):
        """Get action based on current policy."""
        with torch.no_grad():
            if self.recurrent:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = self.network.get_action(state_tensor, self.device)
                return action, log_prob, value
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob = self.actor.get_action(state_tensor, self.device)
                value = self.critic(state_tensor)
                return action, log_prob, value.item()
    
    def train(self, env, num_episodes=1000, max_steps=1000, save_freq=100, render=False, opponent_action_func=None):
        """
        Train the A2C agent.
        
        Args:
            env: The environment to train on
            num_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            save_freq: How often to save model checkpoints
            render: Whether to render the environment during training
            opponent_action_func: Function to get opponent's action; if None, random actions are used
        """
        # Create save directory if it doesn't exist
        save_dir = f'./saved_models/a2c_snake{self.snake_id}'
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
            
            # Reset recurrent network hidden state if using recurrent model
            if self.recurrent:
                self.network.hidden = None
            
            episode_reward = 0
            steps = 0
            done = False
            
            # Lists to store episode data
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            
            # Episode loop
            while not done and steps < max_steps:
                # Choose action for the agent's snake
                action, log_prob, value = self.get_action(state)
                
                # Store data for training
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                
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
                    actions_dict = {'snake1': action, 'snake2': opponent_action}
                else:
                    actions_dict = {'snake1': opponent_action, 'snake2': action}
                
                # Take step in environment
                next_observations, rewards_dict, done, info = env.step(actions_dict)
                next_state = next_observations[f'snake{self.snake_id}']
                reward = rewards_dict[f'snake{self.snake_id}']
                
                # Store reward
                rewards.append(reward)
                
                # Render if enabled
                if render and episode % 100 == 0:  # Render every 100 episodes
                    env.render()
                    time.sleep(0.01)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Compute returns and advantages for the episode
            returns = []
            advantages = []
            
            if done:
                R = 0  # No value for terminal state
            else:
                # Bootstrap with value of the last state
                with torch.no_grad():
                    if self.recurrent:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        _, value, _ = self.network.forward(state_tensor)
                        R = value.item()
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        R = self.critic(state_tensor).item()
            
            # Compute returns and advantages
            for step in reversed(range(len(rewards))):
                R = rewards[step] + self.gamma * R * (1 - done)
                advantage = R - values[step]
                
                returns.insert(0, R)
                advantages.insert(0, advantage)
            
            # Convert lists to tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            log_probs_tensor = torch.cat(log_probs).to(self.device)
            
            # Train the network(s)
            if self.recurrent:
                # Reset hidden state for training
                self.network.hidden = None
                
                # Forward pass
                policy, values_pred, _ = self.network.forward(states_tensor)
                
                # Calculate log probabilities of taken actions
                dist = torch.distributions.Categorical(policy)
                new_log_probs = dist.log_prob(actions_tensor)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Calculate actor loss
                actor_loss = -(new_log_probs * advantages_tensor).mean()
                
                # Calculate critic loss
                values_pred = values_pred.squeeze()
                critic_loss = F.mse_loss(values_pred, returns_tensor)
                
                # Calculate total loss
                loss = actor_loss + self.value_loss_weight * critic_loss - self.entropy_weight * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
            else:
                # Actor network training
                probs = self.actor(states_tensor)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()
                
                actor_loss = -(new_log_probs * advantages_tensor).mean()
                
                # Critic network training
                values_pred = self.critic(states_tensor).squeeze()
                critic_loss = F.mse_loss(values_pred, returns_tensor)
                
                # Calculate total loss
                loss = actor_loss + self.value_loss_weight * critic_loss - self.entropy_weight * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()
            
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
                      f"Win Rate: {win_rate:.2f}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                if self.recurrent:
                    save_path = os.path.join(save_dir, f'model_episode_{episode+1}.pth')
                    torch.save({
                        'episode': episode,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
                else:
                    save_path = os.path.join(save_dir, f'model_episode_{episode+1}.pth')
                    torch.save({
                        'episode': episode,
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
                print(f"Model saved to {save_path}")
        
        # Save final model
        if self.recurrent:
            final_save_path = os.path.join(save_dir, 'model_final.pth')
            torch.save({
                'episode': num_episodes,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, final_save_path)
        else:
            final_save_path = os.path.join(save_dir, 'model_final.pth')
            torch.save({
                'episode': num_episodes,
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, final_save_path)
        print(f"Final model saved to {final_save_path}")
        
        # Return training metrics
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'win_count': win_count,
            'lose_count': lose_count
        }
    
    def load(self, model_path):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.recurrent:
            self.network.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example training script
    env = DualSnakeEnv(grid_size=20, max_steps=1000)
    observations = env.reset()
    
    # Get observation shape and action size
    input_shape = observations['snake2'].shape  # (channels, height, width)
    action_size = 3  # 0: continue, 1: turn left, 2: turn right
    
    # Create A2C agent
    agent = A2CAgent(
        snake_id=2,
        input_shape=input_shape,
        action_size=action_size,
        recurrent=True  # Use recurrent model
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
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import SimplifiedDualSnakeEnv
from training.dqn_agent import DQNAgent
from training.ac_agent import A2CAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Train snake agents in simplified environment')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true', help='Render training episodes')
    parser.add_argument('--save-freq', type=int, default=500, help='Frequency to save model checkpoints')
    parser.add_argument('--grid-size', type=int, default=20, help='Size of the game grid')
    parser.add_argument('--wall-count', type=int, default=3, help='Number of walls in simplified environment')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run training on (cuda/cpu)')
    
    # Add arguments for continuing training from existing models
    parser.add_argument('--load-dqn', type=str, help='Path to existing DQN model to continue training')
    parser.add_argument('--load-a2c', type=str, help='Path to existing A2C model to continue training')
    return parser.parse_args()

def plot_training_metrics(metrics, agent_type, save_dir='./plots'):
    """Plot and save training metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['episode_rewards'], label=f'{agent_type} Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title(f'{agent_type} Training Rewards (Simplified Environment)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{agent_type.lower()}_rewards_simplified.png'))
    
    # Plot episode lengths
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['episode_lengths'], label=f'{agent_type} Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title(f'{agent_type} Episode Lengths (Simplified Environment)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{agent_type.lower()}_lengths_simplified.png'))
    
    # Plot win rate if available
    if 'win_count' in metrics and 'lose_count' in metrics:
        win_rate = metrics['win_count'] / (metrics['win_count'] + metrics['lose_count']) if (metrics['win_count'] + metrics['lose_count']) > 0 else 0
        plt.figure(figsize=(8, 6))
        plt.bar(['Win', 'Lose'], [metrics['win_count'], metrics['lose_count']])
        plt.ylabel('Count')
        plt.title(f'{agent_type} Win/Loss (Win Rate: {win_rate:.2f})')
        plt.grid(axis='y')
        plt.savefig(os.path.join(save_dir, f'{agent_type.lower()}_winrate_simplified.png'))

def train_dqn_agent(env, args):
    """Train a DQN agent in the simplified environment."""
    # Get observation shape and action size
    observations = env.reset()
    input_shape = observations['snake1'].shape
    action_size = 3  # 0: continue, 1: turn left, 2: turn right
    
    print(f"Training DQN agent on device: {args.device}")
    print(f"Input shape: {input_shape}, Action size: {action_size}")
    
    # Create agent
    agent = DQNAgent(
        snake_id=1,
        input_shape=input_shape,
        action_size=action_size,
        dueling=True,
        double_dqn=True,
        device=args.device
    )
    
    # Load model if specified
    if args.load_dqn:
        print(f"Loading DQN model from {args.load_dqn}")
        agent.load(args.load_dqn)
    
    # Train the agent
    save_dir = './saved_models/dqn_simplified'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Training DQN agent for {args.episodes} episodes...")
    metrics = agent.train(
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        render=args.render
    )
    
    # Plot and save metrics
    plot_training_metrics(metrics, 'DQN')
    
    return agent

def train_a2c_agent(env, args):
    """Train an A2C agent in the simplified environment."""
    # Get observation shape and action size
    observations = env.reset()
    input_shape = observations['snake2'].shape
    action_size = 3  # 0: continue, 1: turn left, 2: turn right
    
    print(f"Training A2C agent on device: {args.device}")
    print(f"Input shape: {input_shape}, Action size: {action_size}")
    
    # Create agent
    agent = A2CAgent(
        snake_id=2,
        input_shape=input_shape,
        action_size=action_size,
        recurrent=True,
        device=args.device
    )
    
    # Load model if specified
    if args.load_a2c:
        print(f"Loading A2C model from {args.load_a2c}")
        agent.load(args.load_a2c)
    
    # Train the agent
    save_dir = './saved_models/a2c_simplified'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Training A2C agent for {args.episodes} episodes...")
    metrics = agent.train(
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        render=args.render
    )
    
    # Plot and save metrics
    plot_training_metrics(metrics, 'A2C')
    
    return agent

def evaluate_agent(agent, env, num_episodes=10, agent_type='DQN', snake_id=1):
    """Evaluate a trained agent against random opponent."""
    wins = 0
    losses = 0
    total_steps = 0
    total_score = 0
    
    print(f"Evaluating {agent_type} agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        steps = 0
        
        while not done:
            # Get action for our agent
            state = observations[f'snake{snake_id}']
            if agent_type == 'DQN':
                action = agent.get_action(state, training=False)
            else:  # A2C
                action, _, _ = agent.get_action(state, training=False)
            
            # Random action for opponent
            opponent_id = 2 if snake_id == 1 else 1
            opponent_action = np.random.randint(0, 3)
            
            # Create action dictionary
            actions = {}
            actions[f'snake{snake_id}'] = action
            actions[f'snake{opponent_id}'] = opponent_action
            
            # Take step
            observations, _, done, info = env.step(actions)
            steps += 1
            
            # Render if enabled
            if env.render_mode == 'human':
                import time
                time.sleep(0.05)
        
        # Update stats
        total_steps += steps
        total_score += info[f'score{snake_id}']
        
        if info['winner'] == snake_id:
            wins += 1
            result = "Win"
        else:
            losses += 1
            result = "Loss"
            
        print(f"Episode {episode+1}/{num_episodes}: {result} after {steps} steps. "
              f"Score: {info[f'score{snake_id}']}")
    
    # Print summary
    win_rate = wins / num_episodes
    avg_steps = total_steps / num_episodes
    avg_score = total_score / num_episodes
    
    print(f"\n===== {agent_type} Agent Evaluation Summary =====")
    print(f"Win Rate: {win_rate:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Score: {avg_score:.2f}")
    
    return win_rate, avg_steps, avg_score

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create directories
    os.makedirs('./saved_models/dqn_simplified', exist_ok=True)
    os.makedirs('./saved_models/a2c_simplified', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Create simplified environment
    env = SimplifiedDualSnakeEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode='human' if args.render else None,
        wall_count=args.wall_count
    )
    
    # Train DQN agent
    dqn_agent = train_dqn_agent(env, args)
    
    # Create new environment instance for A2C training
    env = SimplifiedDualSnakeEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode='human' if args.render else None,
        wall_count=args.wall_count
    )
    
    # Train A2C agent
    a2c_agent = train_a2c_agent(env, args)
    
    # Evaluation environment with rendering
    eval_env = SimplifiedDualSnakeEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode='human',
        wall_count=args.wall_count
    )
    
    # Evaluate agents
    print("\nEvaluating trained agents...")
    dqn_results = evaluate_agent(dqn_agent, eval_env, num_episodes=5, agent_type='DQN', snake_id=1)
    a2c_results = evaluate_agent(a2c_agent, eval_env, num_episodes=5, agent_type='A2C', snake_id=2)
    
    # Close environments
    env.close()
    eval_env.close()
    
    print("\nTraining and evaluation complete!")
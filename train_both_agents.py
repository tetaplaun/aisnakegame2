import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import DualSnakeEnv
from training.dqn_agent import DQNAgent
from training.ac_agent import A2CAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Train two snake agents with different RL approaches')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true', help='Render training episodes')
    parser.add_argument('--save-freq', type=int, default=100, help='Frequency to save model checkpoints')
    parser.add_argument('--grid-size', type=int, default=20, help='Size of the game grid')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run training on (cuda/cpu)')
    
    # Add arguments for continuing training from existing models
    parser.add_argument('--load-dqn', type=str, help='Path to existing DQN model to continue training')
    parser.add_argument('--load-a2c', type=str, help='Path to existing A2C model to continue training')
    return parser.parse_args()

def plot_training_metrics(metrics1, metrics2, save_dir='./plots'):
    """Plot and save training metrics for comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(metrics1['episode_rewards'], label='DQN Agent (Snake 1)')
    plt.plot(metrics2['episode_rewards'], label='A2C Agent (Snake 2)')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'rewards.png'))
    
    # Plot episode lengths
    plt.figure(figsize=(12, 6))
    plt.plot(metrics1['episode_lengths'], label='DQN Agent (Snake 1)')
    plt.plot(metrics2['episode_lengths'], label='A2C Agent (Snake 2)')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'episode_lengths.png'))
    
    # Plot win rates
    win_rates1 = []
    win_rates2 = []
    window_size = 100
    
    snake1_wins = np.zeros(len(metrics1['episode_rewards']))
    snake2_wins = np.zeros(len(metrics2['episode_rewards']))
    
    # Indicator for wins (1 for win, 0 for loss)
    for i in range(min(len(metrics1['episode_rewards']), len(metrics2['episode_rewards']))):
        if i < len(metrics1['episode_rewards']) and metrics1.get('win_episodes', []):
            snake1_wins[i] = 1 if i in metrics1['win_episodes'] else 0
        if i < len(metrics2['episode_rewards']) and metrics2.get('win_episodes', []):
            snake2_wins[i] = 1 if i in metrics2['win_episodes'] else 0
    
    # Calculate rolling win rates
    for i in range(window_size, len(snake1_wins)):
        win_rates1.append(np.mean(snake1_wins[i-window_size:i]))
    for i in range(window_size, len(snake2_wins)):
        win_rates2.append(np.mean(snake2_wins[i-window_size:i]))
    
    plt.figure(figsize=(12, 6))
    if win_rates1:
        plt.plot(range(window_size, window_size + len(win_rates1)), win_rates1, label='DQN Agent (Snake 1)')
    if win_rates2:
        plt.plot(range(window_size, window_size + len(win_rates2)), win_rates2, label='A2C Agent (Snake 2)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (over last 100 episodes)')
    plt.title('Win Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'win_rates.png'))
    
    # Create bar chart for final win counts
    plt.figure(figsize=(8, 6))
    plt.bar(['DQN Agent (Snake 1)', 'A2C Agent (Snake 2)'], 
            [metrics1['win_count'], metrics2['win_count']])
    plt.ylabel('Number of Wins')
    plt.title('Total Wins Comparison')
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'total_wins.png'))

def train_separate():
    """Train agents separately against random opponents."""
    args = parse_args()
    
    # Create environment
    env = DualSnakeEnv(grid_size=args.grid_size, max_steps=args.max_steps)
    observations = env.reset()
    
    # Get observation shape and action size
    input_shape = observations['snake1'].shape
    action_size = 3  # 0: continue, 1: turn left, 2: turn right
    
    print(f"Training on device: {args.device}")
    print(f"Input shape: {input_shape}, Action size: {action_size}")
    
    # Create DQN agent for Snake 1
    print("Creating DQN agent for Snake 1...")
    dqn_agent = DQNAgent(
        snake_id=1,
        input_shape=input_shape,
        action_size=action_size,
        dueling=True,
        double_dqn=True,
        device=args.device
    )
    
    # Load DQN model if specified
    if args.load_dqn:
        print(f"Loading DQN model from {args.load_dqn}")
        dqn_agent.load(args.load_dqn)
    
    # Create A2C agent for Snake 2
    print("Creating A2C agent for Snake 2...")
    a2c_agent = A2CAgent(
        snake_id=2,
        input_shape=input_shape,
        action_size=action_size,
        recurrent=True,
        device=args.device
    )
    
    # Load A2C model if specified
    if args.load_a2c:
        print(f"Loading A2C model from {args.load_a2c}")
        a2c_agent.load(args.load_a2c)
    
    # Train DQN agent
    print("\nTraining DQN agent (Snake 1)...")
    dqn_metrics = dqn_agent.train(
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        render=args.render
    )
    
    # Train A2C agent
    print("\nTraining A2C agent (Snake 2)...")
    a2c_metrics = a2c_agent.train(
        env=env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        render=args.render
    )
    
    # Plot training metrics
    plot_training_metrics(dqn_metrics, a2c_metrics)
    
    # Close environment
    env.close()
    
    return dqn_agent, a2c_agent

def evaluate_agents(dqn_agent, a2c_agent, num_episodes=100, render=True):
    """Pit trained agents against each other and evaluate performance."""
    env = DualSnakeEnv(grid_size=20, max_steps=1000, render_mode='human' if render else None)
    
    dqn_wins = 0
    a2c_wins = 0
    ties = 0
    
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        step = 0
        
        while not done:
            # Get DQN action for Snake 1
            dqn_state = observations['snake1']
            dqn_action = dqn_agent.get_action(dqn_state, training=False)
            
            # Get A2C action for Snake 2
            a2c_state = observations['snake2']
            a2c_action, _, _ = a2c_agent.get_action(a2c_state, training=False)
            
            # Take step in environment
            actions = {'snake1': dqn_action, 'snake2': a2c_action}
            observations, _, done, info = env.step(actions)
            step += 1
            
            # Render if enabled
            if render:
                import time
                time.sleep(0.05)  # Slow down for visualization
        
        # Update win counts
        if info['winner'] == 1:
            dqn_wins += 1
            result = "DQN wins"
        elif info['winner'] == 2:
            a2c_wins += 1
            result = "A2C wins"
        else:
            ties += 1
            result = "Tie"
            
        print(f"Episode {episode+1}/{num_episodes}: {result} after {step} steps. "
              f"Scores - DQN: {info['score1']}, A2C: {info['score2']}")
    
    # Summary
    print("\n===== Final Results =====")
    print(f"DQN wins: {dqn_wins} ({dqn_wins/num_episodes*100:.1f}%)")
    print(f"A2C wins: {a2c_wins} ({a2c_wins/num_episodes*100:.1f}%)")
    print(f"Ties: {ties} ({ties/num_episodes*100:.1f}%)")
    
    # Create a plot
    labels = ['DQN (Snake 1)', 'A2C (Snake 2)', 'Ties']
    sizes = [dqn_wins, a2c_wins, ties]
    colors = ['#ff9999','#66b3ff','#c2c2f0']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Agent Performance Comparison')
    plt.savefig('./plots/performance_comparison.png')
    
    env.close()

if __name__ == "__main__":
    # Create save directories
    os.makedirs('./saved_models/dqn_snake1', exist_ok=True)
    os.makedirs('./saved_models/a2c_snake2', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Train agents
    dqn_agent, a2c_agent = train_separate()
    
    # Evaluate trained agents against each other
    evaluate_agents(dqn_agent, a2c_agent, num_episodes=10, render=True)
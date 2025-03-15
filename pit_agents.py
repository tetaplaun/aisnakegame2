import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from environment import DualSnakeEnv
from training.dqn_agent import DQNAgent
from training.ac_agent import A2CAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Pit trained snake agents against each other')
    parser.add_argument('--dqn-model', type=str, required=True, help='Path to the DQN model file')
    parser.add_argument('--ac-model', type=str, required=True, help='Path to the A2C model file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between frames when rendering')
    parser.add_argument('--grid-size', type=int, default=20, help='Size of the game grid')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run evaluation on (cuda/cpu)')
    return parser.parse_args()

def evaluate_agents(dqn_agent, a2c_agent, num_episodes=100, render=True, delay=0.05, grid_size=20, max_steps=1000):
    """Pit trained agents against each other and evaluate performance."""
    env = DualSnakeEnv(grid_size=grid_size, max_steps=max_steps, render_mode='human' if render else None)
    
    dqn_wins = 0
    a2c_wins = 0
    ties = 0
    
    dqn_scores = []
    a2c_scores = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        step = 0
        
        # Reset recurrent network hidden state if applicable
        if hasattr(a2c_agent, 'recurrent') and a2c_agent.recurrent:
            a2c_agent.network.hidden = None
        
        while not done and step < max_steps:
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
                time.sleep(delay)  # Slow down for visualization
        
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
        
        # Store scores and episode length
        dqn_scores.append(info['score1'])
        a2c_scores.append(info['score2'])
        episode_lengths.append(step)
        
        print(f"Episode {episode+1}/{num_episodes}: {result} after {step} steps. "
              f"Scores - DQN: {info['score1']}, A2C: {info['score2']}")
    
    # Summary
    print("\n===== Final Results =====")
    print(f"DQN wins: {dqn_wins} ({dqn_wins/num_episodes*100:.1f}%)")
    print(f"A2C wins: {a2c_wins} ({a2c_wins/num_episodes*100:.1f}%)")
    print(f"Ties: {ties} ({ties/num_episodes*100:.1f}%)")
    print(f"Average scores - DQN: {np.mean(dqn_scores):.2f}, A2C: {np.mean(a2c_scores):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
    
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    # Create a pie chart for win distribution
    labels = ['DQN (Snake 1)', 'A2C (Snake 2)', 'Ties']
    sizes = [dqn_wins, a2c_wins, ties]
    colors = ['#ff9999','#66b3ff','#c2c2f0']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Agent Performance Comparison')
    plt.savefig('./plots/performance_comparison.png')
    
    # Create a bar chart for average scores
    plt.figure(figsize=(8, 6))
    plt.bar(['DQN (Snake 1)', 'A2C (Snake 2)'], 
            [np.mean(dqn_scores), np.mean(a2c_scores)])
    plt.ylabel('Average Score')
    plt.title('Score Comparison')
    plt.grid(axis='y')
    plt.savefig('./plots/score_comparison.png')
    
    env.close()
    
    return {
        'dqn_wins': dqn_wins,
        'a2c_wins': a2c_wins,
        'ties': ties,
        'dqn_scores': dqn_scores,
        'a2c_scores': a2c_scores,
        'episode_lengths': episode_lengths
    }

def main():
    args = parse_args()
    
    # Create environment to get input shape and action size
    env = DualSnakeEnv(grid_size=args.grid_size)
    observations = env.reset()
    input_shape = observations['snake1'].shape
    action_size = 3  # 0: continue, 1: turn left, 2: turn right
    env.close()
    
    print(f"Running on device: {args.device}")
    print(f"Input shape: {input_shape}, Action size: {action_size}")
    
    # Create and load DQN agent
    print(f"Loading DQN agent from {args.dqn_model}...")
    dqn_agent = DQNAgent(
        snake_id=1,
        input_shape=input_shape,
        action_size=action_size,
        dueling=True,
        double_dqn=True,
        device=args.device
    )
    dqn_agent.load(args.dqn_model)
    
    # Create and load A2C agent
    print(f"Loading A2C agent from {args.ac_model}...")
    a2c_agent = A2CAgent(
        snake_id=2,
        input_shape=input_shape,
        action_size=action_size,
        recurrent=True,  # Assuming the saved model is recurrent
        device=args.device
    )
    a2c_agent.load(args.ac_model)
    
    # Evaluate agents
    print(f"\nEvaluating agents over {args.episodes} episodes...")
    results = evaluate_agents(
        dqn_agent=dqn_agent,
        a2c_agent=a2c_agent,
        num_episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay,
        grid_size=args.grid_size,
        max_steps=args.max_steps
    )

if __name__ == "__main__":
    main()
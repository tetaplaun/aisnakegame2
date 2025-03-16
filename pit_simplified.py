import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import SimplifiedDualSnakeEnv
from models.dqn_model import DuelingDQNModel
from models.ac_model import RecurrentActorCritic

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained agents against each other in the simplified environment')
    parser.add_argument('--dqn-model', type=str, required=True, help='Path to trained DQN model')
    parser.add_argument('--ac-model', type=str, required=True, help='Path to trained AC model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--grid-size', type=int, default=20, help='Size of the game grid')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--wall-count', type=int, default=3, help='Number of walls in simplified environment')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between frames when rendering')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='./evaluation_results', help='Directory to save evaluation results')
    return parser.parse_args()

def load_dqn_model(model_path, device):
    """Load trained DQN model."""
    # Get model architecture from checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Determine if it's a dueling model based on state_dict keys
    is_dueling = any('advantage' in key for key in checkpoint['model_state_dict'].keys())

    # Create model with the appropriate architecture
    if is_dueling:
        # Infer input shape from checkpoint
        for key, value in checkpoint['model_state_dict'].items():
            if 'conv' in key and 'weight' in key:
                input_channels = value.shape[1]
                break

        # Get action size from advantage layer
        for key, value in checkpoint['model_state_dict'].items():
            if key == 'advantage.weight':
                # The action size is the first dimension of the advantage weight matrix
                action_size = value.shape[0]
                break

        # Create model with inferred shapes
        model = DuelingDQNModel((input_channels, 20, 20), action_size).to(device)
    else:
        # Handle regular DQN (not implemented here as we're using dueling)
        raise NotImplementedError("Regular DQN loading not implemented")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def load_ac_model(model_path, device):
    """Load trained AC model."""
    # Get model architecture from checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Determine if it's a recurrent model based on state_dict keys
    is_recurrent = any('lstm' in key for key in checkpoint['model_state_dict'].keys())

    # Infer input shape and action size from checkpoint
    # First find the first convolutional layer's weight shape
    for key, value in checkpoint['model_state_dict'].items():
        if 'conv' in key and 'weight' in key:
            input_channels = value.shape[1]
            break

    # Find the actor's output layer's weight shape for action size
    for key, value in checkpoint['model_state_dict'].items():
        if 'actor' in key and 'weight' in key and '.2.' in key:  # Last layer in the actor sequential
            action_size = value.shape[0]
            break

    # Create model with inferred shapes
    if is_recurrent:
        model = RecurrentActorCritic((input_channels, 20, 20), action_size).to(device)
    else:
        # Handle non-recurrent AC (not implemented here as we're using recurrent)
        raise NotImplementedError("Non-recurrent AC loading not implemented")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def evaluate_agents(dqn_model, ac_model, args):
    """Evaluate trained DQN and AC agents against each other."""
    # Create environment
    env = SimplifiedDualSnakeEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode=None if args.no_render else 'human',
        wall_count=args.wall_count
    )

    # Statistics
    dqn_wins = 0
    ac_wins = 0
    ties = 0
    total_steps = 0
    dqn_scores = []
    ac_scores = []
    episode_lengths = []

    # Hidden state for recurrent model
    ac_hidden = None

    # Run evaluation episodes
    for episode in range(args.episodes):
        # Reset environment
        observations = env.reset()
        done = False
        steps = 0

        # Reset hidden state for AC model
        ac_hidden = (
            torch.zeros(1, 1, 256).to(args.device),
            torch.zeros(1, 1, 256).to(args.device)
        )

        # Run episode
        while not done:
            # Get DQN action
            dqn_state = torch.FloatTensor(observations['snake1']).unsqueeze(0).to(args.device)
            with torch.no_grad():
                dqn_q_values = dqn_model(dqn_state)
                dqn_action = torch.argmax(dqn_q_values).item()

            # Get AC action
            ac_state = torch.FloatTensor(observations['snake2']).unsqueeze(0).to(args.device)
            with torch.no_grad():
                ac_action_probs, _, ac_hidden = ac_model(ac_state, ac_hidden)
                ac_action = torch.argmax(ac_action_probs).item()

            # Take step
            actions = {'snake1': dqn_action, 'snake2': ac_action}
            observations, _, done, info = env.step(actions)
            steps += 1

            # Add delay if rendering
            if not args.no_render:
                time.sleep(args.delay)

        # Update statistics
        if info['winner'] == 1:
            dqn_wins += 1
            result = "DQN wins"
        elif info['winner'] == 2:
            ac_wins += 1
            result = "A2C wins"
        else:
            ties += 1
            result = "Tie"

        total_steps += steps
        dqn_scores.append(info['score1'])
        ac_scores.append(info['score2'])
        episode_lengths.append(steps)

        # Print episode result
        print(f"Episode {episode+1}/{args.episodes}: {result} after {steps} steps. "
              f"Scores - DQN: {info['score1']}, A2C: {info['score2']}")

    # Calculate statistics
    dqn_win_rate = dqn_wins / args.episodes
    ac_win_rate = ac_wins / args.episodes
    tie_rate = ties / args.episodes
    avg_episode_length = total_steps / args.episodes
    avg_dqn_score = sum(dqn_scores) / args.episodes
    avg_ac_score = sum(ac_scores) / args.episodes

    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"DQN Wins: {dqn_wins} ({dqn_win_rate:.2%})")
    print(f"A2C Wins: {ac_wins} ({ac_win_rate:.2%})")
    print(f"Ties: {ties} ({tie_rate:.2%})")
    print(f"Average Episode Length: {avg_episode_length:.2f} steps")
    print(f"Average DQN Score: {avg_dqn_score:.2f}")
    print(f"Average A2C Score: {avg_ac_score:.2f}")

    # Save results
    save_results(dqn_wins, ac_wins, ties, dqn_scores, ac_scores, episode_lengths, args)

    # Close environment
    env.close()

    return dqn_win_rate, ac_win_rate, tie_rate

def save_results(dqn_wins, ac_wins, ties, dqn_scores, ac_scores, episode_lengths, args):
    """Save evaluation results and plots."""
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Calculate win rates
    dqn_win_rate = dqn_wins / args.episodes
    ac_win_rate = ac_wins / args.episodes
    tie_rate = ties / args.episodes

    # Save results to text file
    with open(os.path.join(args.save_dir, 'simplified_evaluation_results.txt'), 'w') as f:
        f.write("===== Simplified Environment Evaluation Results =====\n")
        f.write(f"Environment: Simplified (Wall Count: {args.wall_count})\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"DQN Model: {args.dqn_model}\n")
        f.write(f"A2C Model: {args.ac_model}\n\n")

        f.write(f"DQN Wins: {dqn_wins} ({dqn_win_rate:.2%})\n")
        f.write(f"A2C Wins: {ac_wins} ({ac_win_rate:.2%})\n")
        f.write(f"Ties: {ties} ({tie_rate:.2%})\n\n")

        f.write(f"Average DQN Score: {sum(dqn_scores) / args.episodes:.2f}\n")
        f.write(f"Average A2C Score: {sum(ac_scores) / args.episodes:.2f}\n")
        f.write(f"Average Episode Length: {sum(episode_lengths) / args.episodes:.2f} steps\n")

    # Create pie chart of win rates
    plt.figure(figsize=(8, 6))
    labels = ['DQN', 'A2C', 'Ties']
    sizes = [dqn_win_rate, ac_win_rate, tie_rate]
    colors = ['#ff9999', '#66b3ff', '#c2c2f0']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Agent Win Rates (Simplified Environment)')
    plt.savefig(os.path.join(args.save_dir, 'simplified_win_rates.png'))

    # Create bar chart of scores
    plt.figure(figsize=(10, 6))
    x = np.arange(min(len(dqn_scores), len(ac_scores)))
    plt.bar(x - 0.2, dqn_scores[:len(x)], 0.4, label='DQN')
    plt.bar(x + 0.2, ac_scores[:len(x)], 0.4, label='A2C')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Agent Scores (Simplified Environment)')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'simplified_scores.png'))

    # Create line chart of episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths (Simplified Environment)')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'simplified_episode_lengths.png'))

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load models
    dqn_model = load_dqn_model(args.dqn_model, args.device)
    ac_model = load_ac_model(args.ac_model, args.device)

    # Evaluate agents
    print(f"Evaluating agents for {args.episodes} episodes...")
    evaluate_agents(dqn_model, ac_model, args)
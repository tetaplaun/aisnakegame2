# Autonomous Snake Game with Reinforcement Learning

This project implements a competitive snake game environment where two snakes compete against each other. Each snake is controlled by a distinct reinforcement learning algorithm, and the project allows training and evaluating these algorithms against each other.

## Game Features

- Two-snake competitive environment
- Novel game elements:
  - Power-ups (temporary invulnerability against opponent's body)
  - Speed boosts (temporarily doubles snake speed)
  - Teleportation portals (instantly transport between two locations)
  - Obstacles/walls (instant death upon collision)
- Scoring system based on food eaten
- Variable snake speeds
- Definitive win/loss conditions (no draws)

## Reinforcement Learning Approaches

The project implements two different RL approaches:

1. **Snake 1: Deep Q-Network (DQN) with Dueling Architecture**
   - Value-based reinforcement learning
   - Double DQN for more stable learning
   - Dueling architecture to separately estimate state value and action advantages
   - Epsilon-greedy exploration strategy
   - Experience replay for more efficient learning

2. **Snake 2: Actor-Critic with Recurrent Network**
   - Policy-based reinforcement learning
   - LSTM-based recurrent network to leverage temporal information
   - Combined actor-critic architecture for improved stability
   - Entropy regularization to encourage exploration
   - Shared convolutional feature extractor

## Project Structure

- `environment.py`: The game environment implementing the dual snake game
- `run_game.py`: Script to run the game with random actions (for testing)
- `models/`: Neural network model definitions
  - `dqn_model.py`: DQN and Dueling DQN model architectures
  - `ac_model.py`: Actor-Critic and Recurrent Actor-Critic model architectures
- `training/`: Agent implementations and training code
  - `dqn_agent.py`: DQN agent implementation
  - `ac_agent.py`: Actor-Critic agent implementation
- `train_both_agents.py`: Script to train both agents and save models
- `pit_agents.py`: Script to evaluate trained agents against each other
- `saved_models/`: Directory for saved model checkpoints
- `plots/`: Directory for training and evaluation plots

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pygame (for rendering)

## Getting Started

1. Install the required packages:
   ```
   pip install torch numpy matplotlib pygame
   ```

2. Run the game with random actions to test the environment:
   ```
   python run_game.py
   ```

3. Train both agents:
   ```
   python train_both_agents.py --episodes 1000 --grid-size 20
   ```
   Additional options:
   - `--render`: Render training episodes
   - `--save-freq`: Frequency to save model checkpoints
   - `--max-steps`: Maximum steps per episode
   - `--device`: Device to run training on (cuda/cpu)
   - `--load-dqn`: Path to existing DQN model to continue training
   - `--load-a2c`: Path to existing A2C model to continue training
   
   To continue training from saved models:
   ```
   python train_both_agents.py --episodes 500 --load-dqn saved_models/dqn_snake1/model_episode_1000.pth --load-a2c saved_models/a2c_snake2/model_episode_1000.pth
   ```

4. Evaluate trained agents against each other:
   ```
   python pit_agents.py --dqn-model saved_models/dqn_snake1/model_final.pth --ac-model saved_models/a2c_snake2/model_final.pth --episodes 100
   ```
   Additional options:
   - `--no-render`: Disable rendering
   - `--delay`: Delay between frames when rendering
   - `--grid-size`: Size of the game grid
   - `--max-steps`: Maximum steps per episode

## Game Controls

- The game is fully autonomous - both snakes are controlled by the RL agents.
- No user input is required during gameplay.

## Extending the Project

- Try implementing different RL algorithms (PPO, SAC, etc.)
- Modify the environment to add new game elements
- Implement curriculum learning to gradually increase difficulty
- Add human-playable mode to compete against trained agents
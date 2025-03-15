import time
import random
from environment import DualSnakeEnv

def main():
    # Initialize environment with human rendering
    env = DualSnakeEnv(grid_size=20, max_steps=1000, render_mode='human')
    
    # Reset environment
    env.reset()
    
    # Main game loop
    done = False
    while not done:
        # Random actions for now (will be replaced by RL agents)
        actions = {
            'snake1': random.randint(0, 2),  # 0: forward, 1: left, 2: right
            'snake2': random.randint(0, 2)
        }
        
        # Step environment
        _, _, done, info = env.step(actions)
        
        # Small delay for visualization
        time.sleep(0.1)
        
    # Display final results
    print(f"Game over! Snake {info['winner']} wins!")
    print(f"Final scores - Snake 1: {info['score1']}, Snake 2: {info['score2']}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
import time
import random
import numpy as np
from environment import DualSnakeEnv, Direction

def simple_food_seeking_agent(observation, snake_id, env):
    """A simple agent that tries to move toward food"""
    # Get the snake's head position
    if snake_id == 1:
        snake = env.snake1
        current_direction = env.direction1
    else:
        snake = env.snake2
        current_direction = env.direction2
    
    head_x, head_y = snake[0]
    food_x, food_y = env.food
    
    # Determine the direction to the food (accounting for grid boundaries)
    dx = food_x - head_x
    dy = food_y - head_y
    
    # Simple logic to decide which way to turn
    if current_direction == Direction.UP:
        if dx > 0:
            return 2  # Turn right
        elif dx < 0:
            return 1  # Turn left
        else:
            return 0  # Keep going
    elif current_direction == Direction.RIGHT:
        if dy > 0:
            return 2  # Turn right
        elif dy < 0:
            return 1  # Turn left
        else:
            return 0  # Keep going
    elif current_direction == Direction.DOWN:
        if dx < 0:
            return 2  # Turn right
        elif dx > 0:
            return 1  # Turn left
        else:
            return 0  # Keep going
    elif current_direction == Direction.LEFT:
        if dy < 0:
            return 2  # Turn right
        elif dy > 0:
            return 1  # Turn left
        else:
            return 0  # Keep going
    
    # Fallback to random action if something goes wrong
    return random.randint(0, 2)

def avoid_wall_ahead(observation, snake_id, env):
    """Check if there's a wall ahead and avoid it"""
    if snake_id == 1:
        snake = env.snake1
        current_direction = env.direction1
    else:
        snake = env.snake2
        current_direction = env.direction2
    
    head_x, head_y = snake[0]
    
    # Get position in front
    if current_direction == Direction.UP:
        front_x, front_y = head_x, head_y - 1
    elif current_direction == Direction.RIGHT:
        front_x, front_y = head_x + 1, head_y
    elif current_direction == Direction.DOWN:
        front_x, front_y = head_x, head_y + 1
    elif current_direction == Direction.LEFT:
        front_x, front_y = head_x - 1, head_y
    
    # Check if out of bounds
    if (front_x < 0 or front_x >= env.grid_size or 
        front_y < 0 or front_y >= env.grid_size):
        return random.choice([1, 2])  # Turn randomly
    
    # Check if wall ahead
    if env.grid[front_y, front_x] == 7:  # Wall value
        return random.choice([1, 2])  # Turn randomly
    
    # Check if snake body ahead
    for s in [env.snake1, env.snake2]:
        if (front_x, front_y) in s:
            return random.choice([1, 2])  # Turn randomly
    
    return None  # No need to avoid anything

def main():
    # Initialize environment with human rendering
    env = DualSnakeEnv(grid_size=20, max_steps=1000, render_mode='human')
    
    # Reset environment
    obs = env.reset()
    
    # Main game loop
    done = False
    while not done:
        actions = {}
        
        # For each snake, decide action
        for snake_id in [1, 2]:
            # First check if we need to avoid a wall or snake
            avoidance_action = avoid_wall_ahead(obs, snake_id, env)
            
            if avoidance_action is not None:
                # Need to avoid something
                actions[f'snake{snake_id}'] = avoidance_action
            else:
                # Head toward food
                if random.random() < 0.8:  # 80% chance to use food-seeking behavior
                    actions[f'snake{snake_id}'] = simple_food_seeking_agent(obs, snake_id, env)
                else:
                    # 20% chance to take random action (for exploration)
                    actions[f'snake{snake_id}'] = random.randint(0, 2)
        
        # Step environment
        obs, _, done, info = env.step(actions)
        
        # Small delay for visualization
        time.sleep(0.1)
        
    # Display final results
    print(f"Game over! Snake {info['winner']} wins!")
    print(f"Final scores - Snake 1: {info['score1']}, Snake 2: {info['score2']}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
import numpy as np
import random
import time
import pygame
from enum import Enum
from collections import deque


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GameElement(Enum):
    EMPTY = 0
    SNAKE1 = 1
    SNAKE2 = 2
    FOOD = 3
    POWER_UP = 4
    SPEED_BOOST = 5
    PORTAL = 6
    WALL = 7
    SNAKE1_HEAD = 8
    SNAKE2_HEAD = 9


class DualSnakeEnv:
    def __init__(self, grid_size=20, max_steps=1000, render_mode=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.reset()
        
        # Pygame setup for rendering
        if self.render_mode == 'human':
            pygame.init()
            self.cell_size = 30
            self.screen_width = self.grid_size * self.cell_size
            self.screen_height = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Autonomous Snake Battle')
            self.clock = pygame.time.Clock()
            
            # Define colors
            self.colors = {
                GameElement.EMPTY: (0, 0, 0),
                GameElement.SNAKE1: (0, 255, 0),
                GameElement.SNAKE2: (0, 0, 255),
                GameElement.FOOD: (255, 0, 0),
                GameElement.POWER_UP: (255, 255, 0),
                GameElement.SPEED_BOOST: (255, 165, 0),
                GameElement.PORTAL: (128, 0, 128),
                GameElement.WALL: (128, 128, 128),
                GameElement.SNAKE1_HEAD: (0, 200, 0),
                GameElement.SNAKE2_HEAD: (0, 0, 200)
            }
    
    def _create_random_obstacles(self):
        # Add some walls
        for _ in range(self.grid_size // 2):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                
                # Don't place walls at initial snake positions or too close to them
                if (x, y) in self.snake1 or (x, y) in self.snake2 or \
                   abs(x - self.snake1[0][0]) < 3 or abs(y - self.snake1[0][1]) < 3 or \
                   abs(x - self.snake2[0][0]) < 3 or abs(y - self.snake2[0][1]) < 3:
                    continue
                    
                self.grid[y, x] = GameElement.WALL.value
                break
                
        # Create portals (they come in pairs)
        for _ in range(1):  # One pair of portals
            portals = []
            for _ in range(2):
                while True:
                    x = random.randint(0, self.grid_size - 1)
                    y = random.randint(0, self.grid_size - 1)
                    
                    if (x, y) in self.snake1 or (x, y) in self.snake2 or \
                       self.grid[y, x] != GameElement.EMPTY.value:
                        continue
                        
                    portals.append((x, y))
                    self.grid[y, x] = GameElement.PORTAL.value
                    break
            
            self.portals.append(tuple(portals))
    
    def _place_element(self, element_type):
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            
            if self.grid[y, x] == GameElement.EMPTY.value:
                self.grid[y, x] = element_type.value
                return (x, y)
    
    def reset(self):
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Initialize snakes in opposite corners with different orientations
        self.snake1 = deque([(self.grid_size // 4, self.grid_size // 4)])
        self.snake2 = deque([(3 * self.grid_size // 4, 3 * self.grid_size // 4)])
        
        self.grid[self.snake1[0][1], self.snake1[0][0]] = GameElement.SNAKE1_HEAD.value
        self.grid[self.snake2[0][1], self.snake2[0][0]] = GameElement.SNAKE2_HEAD.value
        
        # Initialize snake directions
        self.direction1 = Direction.RIGHT
        self.direction2 = Direction.LEFT
        
        # Initialize game elements
        self.food = self._place_element(GameElement.FOOD)
        self.power_up = None
        self.speed_boost = None
        self.portals = []
        
        # Keep track of steps and scores
        self.steps = 0
        self.score1 = 0
        self.score2 = 0
        
        # Snake attributes
        self.snake1_length = 1
        self.snake2_length = 1
        self.snake1_speed = 1  # Normal speed
        self.snake2_speed = 1
        self.snake1_power_up = False
        self.snake2_power_up = False
        self.snake1_steps_to_move = 1
        self.snake2_steps_to_move = 1
        
        # Create random obstacles
        self._create_random_obstacles()
        
        # Game state
        self.done = False
        self.winner = None
        
        # Special element spawn timers
        self.power_up_timer = random.randint(20, 40)
        self.speed_boost_timer = random.randint(30, 50)
        
        # Initialize distance tracking for reward calculation
        food_x, food_y = self.food
        head1_x, head1_y = self.snake1[0]
        head2_x, head2_y = self.snake2[0]
        
        # Calculate initial distances using wrap-around grid logic
        dx1 = min(abs(food_x - head1_x), self.grid_size - abs(food_x - head1_x))
        dy1 = min(abs(food_y - head1_y), self.grid_size - abs(food_y - head1_y))
        self.prev_food_distance_snake1 = dx1 + dy1
        
        dx2 = min(abs(food_x - head2_x), self.grid_size - abs(food_x - head2_x))
        dy2 = min(abs(food_y - head2_y), self.grid_size - abs(food_y - head2_y))
        self.prev_food_distance_snake2 = dx2 + dy2
        
        # Return observation
        return self._get_observation()
    
    def _get_observation(self):
        """Create observation space for each agent."""
        # For each snake, we'll create a set of channels:
        # 1. Snake's own body
        # 2. Opponent snake's body
        # 3. Food
        # 4. Walls and obstacles
        # 5. Power-ups and speed boosts
        # 6. Portals
        
        obs1 = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)
        obs2 = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Fill channels for snake 1
        for x, y in self.snake1:
            obs1[0, y, x] = 1.0  # Own body
        
        for x, y in self.snake2:
            obs1[1, y, x] = 1.0  # Opponent body
        
        # Fill channels for snake 2
        for x, y in self.snake2:
            obs2[0, y, x] = 1.0  # Own body
            
        for x, y in self.snake1:
            obs2[1, y, x] = 1.0  # Opponent body
        
        # Food (same for both)
        x, y = self.food
        obs1[2, y, x] = 1.0
        obs2[2, y, x] = 1.0
        
        # Walls and obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == GameElement.WALL.value:
                    obs1[3, y, x] = 1.0
                    obs2[3, y, x] = 1.0
        
        # Power-ups and speed boosts
        if self.power_up:
            x, y = self.power_up
            obs1[4, y, x] = 1.0
            obs2[4, y, x] = 1.0
            
        if self.speed_boost:
            x, y = self.speed_boost
            obs1[4, y, x] = 0.5  # Different value to distinguish
            obs2[4, y, x] = 0.5
        
        # Portals
        for portal_pair in self.portals:
            for x, y in portal_pair:
                obs1[5, y, x] = 1.0
                obs2[5, y, x] = 1.0
        
        return {"snake1": obs1, "snake2": obs2}
    
    def _get_new_head_position(self, head, direction):
        x, y = head
        
        if direction == Direction.UP:
            return (x, (y - 1) % self.grid_size)
        elif direction == Direction.RIGHT:
            return ((x + 1) % self.grid_size, y)
        elif direction == Direction.DOWN:
            return (x, (y + 1) % self.grid_size)
        elif direction == Direction.LEFT:
            return ((x - 1) % self.grid_size, y)
    
    def _check_collision(self, head):
        x, y = head
        
        # Check if hit wall
        if self.grid[y, x] == GameElement.WALL.value:
            return True
            
        # Check if hit snake1 (excluding head for self-collision check)
        if head != self.snake1[0] and (x, y) in self.snake1:
            return True
            
        # Check if hit snake2 (excluding head for self-collision check)
        if head != self.snake2[0] and (x, y) in self.snake2:
            return True
            
        return False
    
    def _check_portal_teleport(self, head):
        x, y = head
        
        for portal_pair in self.portals:
            if (x, y) == portal_pair[0]:
                return portal_pair[1]
            elif (x, y) == portal_pair[1]:
                return portal_pair[0]
                
        return None
    
    def _move_snake(self, snake_id, action):
        """Move snake according to action and game rules."""
        if snake_id == 1:
            snake = self.snake1
            direction = self.direction1
            head = snake[0]
            power_up_active = self.snake1_power_up
            steps_to_move = self.snake1_steps_to_move
        else:
            snake = self.snake2
            direction = self.direction2
            head = snake[0]
            power_up_active = self.snake2_power_up
            steps_to_move = self.snake2_steps_to_move
        
        # Update direction based on action (0: no change, 1: turn left, 2: turn right)
        if action == 1:  # Turn left
            direction = Direction((direction.value - 1) % 4)
        elif action == 2:  # Turn right
            direction = Direction((direction.value + 1) % 4)
        
        # Update snake's direction
        if snake_id == 1:
            self.direction1 = direction
        else:
            self.direction2 = direction
        
        # Get new head position
        new_head = self._get_new_head_position(head, direction)
        
        # Check if teleport through portal
        portal_exit = self._check_portal_teleport(new_head)
        if portal_exit:
            new_head = portal_exit
        
        # Check collision
        collision = self._check_collision(new_head)
        
        # If collision and no power-up, game over for this snake
        if collision and not power_up_active:
            if snake_id == 1:
                self.winner = 2
            else:
                self.winner = 1
            self.done = True
            
            # Large penalty for dying, scaled by snake length to make losing a longer snake more painful
            # This encourages the agent to prioritize survival especially as it grows
            snake_length = self.snake1_length if snake_id == 1 else self.snake2_length
            death_penalty = -20 - (snake_length * 2)  # Base penalty (-20) plus length-based component
            return death_penalty
        # If collision but power-up active, ignore collision (except with walls)
        elif collision and power_up_active:
            if self.grid[new_head[1], new_head[0]] == GameElement.WALL.value:
                # Still can't go through walls even with power-up
                if snake_id == 1:
                    self.winner = 2
                else:
                    self.winner = 1
                self.done = True
                
                # Apply similar death penalty for colliding with walls even with power-up
                snake_length = self.snake1_length if snake_id == 1 else self.snake2_length
                death_penalty = -20 - (snake_length * 2)
                return death_penalty
        
        # Update grid and snake
        x, y = head
        if snake_id == 1:
            # Update old head to body
            if len(snake) > 1:  # Only if there's a body beyond head
                self.grid[y, x] = GameElement.SNAKE1.value
            else:
                self.grid[y, x] = GameElement.EMPTY.value
        else:
            # Update old head to body
            if len(snake) > 1:  # Only if there's a body beyond head
                self.grid[y, x] = GameElement.SNAKE2.value
            else:
                self.grid[y, x] = GameElement.EMPTY.value
                
        # Add new head
        snake.appendleft(new_head)
        x, y = new_head
        
        # Set head in grid
        if snake_id == 1:
            self.grid[y, x] = GameElement.SNAKE1_HEAD.value
        else:
            self.grid[y, x] = GameElement.SNAKE2_HEAD.value
            
        reward = 0
        
        # Check if snake ate food
        if new_head == self.food:
            # Increase snake length and score
            if snake_id == 1:
                self.snake1_length += 1
                # Progressive scoring that increases with snake length
                food_score = 10 + int(self.snake1_length * 1.5)
                self.score1 += food_score
            else:
                self.snake2_length += 1
                # Progressive scoring that increases with snake length
                food_score = 10 + int(self.snake2_length * 1.5)
                self.score2 += food_score
                
            # Place new food
            self.food = self._place_element(GameElement.FOOD)
            
            # Progressive reward for eating food
            # Base reward (10) + length-based bonus that scales quadratically
            snake_length = self.snake1_length if snake_id == 1 else self.snake2_length
            length_bonus = 2 * (snake_length ** 0.5)  # Square root scaling for balanced progression
            reward += food_score + length_bonus
        else:
            # Remove tail if didn't eat
            tail = snake.pop()
            x, y = tail
            self.grid[y, x] = GameElement.EMPTY.value
            
        # Check if snake got power-up
        if self.power_up and new_head == self.power_up:
            if snake_id == 1:
                self.snake1_power_up = True
                # Power-up lasts for 20 steps
                self.snake1_power_up_timer = 20
            else:
                self.snake2_power_up = True
                self.snake2_power_up_timer = 20
                
            self.power_up = None
            reward += 5
            
        # Check if snake got speed boost
        if self.speed_boost and new_head == self.speed_boost:
            if snake_id == 1:
                self.snake1_speed = 2  # Double speed
                # Speed boost lasts for 30 steps
                self.snake1_speed_timer = 30
            else:
                self.snake2_speed = 2
                self.snake2_speed_timer = 30
                
            self.speed_boost = None
            reward += 5
            
        # Set steps to move based on speed
        if snake_id == 1:
            self.snake1_steps_to_move = 1 / self.snake1_speed
        else:
            self.snake2_steps_to_move = 1 / self.snake2_speed
            
        return reward
    
    def step(self, actions):
        """
        Step the environment by one time step with actions for both snakes.
        actions: dict with keys 'snake1' and 'snake2', values are integers:
                0: continue in same direction
                1: turn left
                2: turn right
        """
        if self.done:
            return self._get_observation(), {"snake1": 0, "snake2": 0}, self.done, {"winner": self.winner, "score1": self.score1, "score2": self.score2}
        
        rewards = {"snake1": 0, "snake2": 0}
        
        # Move snakes based on their speed
        self.snake1_steps_to_move -= 1
        self.snake2_steps_to_move -= 1
        
        if self.snake1_steps_to_move <= 0:
            rewards["snake1"] += self._move_snake(1, actions["snake1"])
        
        if self.snake2_steps_to_move <= 0:
            rewards["snake2"] += self._move_snake(2, actions["snake2"])
        
        # Process other game elements
        self.steps += 1
        
        # Advanced reward system that considers distance to food
        for snake_id in [1, 2]:
            snake = self.snake1 if snake_id == 1 else self.snake2
            if len(snake) > 0:  # Make sure snake still exists
                head = snake[0]
                # Calculate Manhattan distance to food
                food_x, food_y = self.food
                head_x, head_y = head
                
                # Calculate distance using modulo for wrap-around grid (shortest path)
                dx = min(abs(food_x - head_x), self.grid_size - abs(food_x - head_x))
                dy = min(abs(food_y - head_y), self.grid_size - abs(food_y - head_y))
                current_distance = dx + dy
                
                # Store distance for next step comparison
                if snake_id == 1:
                    if hasattr(self, 'prev_food_distance_snake1'):
                        # Reward for getting closer to food (or penalize for moving away)
                        distance_change = self.prev_food_distance_snake1 - current_distance
                        if distance_change > 0:  # Got closer to food
                            rewards[f"snake{snake_id}"] += 0.05 * distance_change
                        elif distance_change < 0:  # Moved away from food
                            rewards[f"snake{snake_id}"] -= 0.02 * abs(distance_change)
                    
                    # Update the stored distance
                    self.prev_food_distance_snake1 = current_distance
                else:
                    if hasattr(self, 'prev_food_distance_snake2'):
                        # Reward for getting closer to food (or penalize for moving away)
                        distance_change = self.prev_food_distance_snake2 - current_distance
                        if distance_change > 0:  # Got closer to food
                            rewards[f"snake{snake_id}"] += 0.05 * distance_change
                        elif distance_change < 0:  # Moved away from food
                            rewards[f"snake{snake_id}"] -= 0.02 * abs(distance_change)
                    
                    # Update the stored distance
                    self.prev_food_distance_snake2 = current_distance
            
            # Reward/penalty system that scales with snake length
            snake_length = self.snake1_length if snake_id == 1 else self.snake2_length
            
            # Small step penalty to encourage efficiency, diminishing with length
            # (less penalty for longer snakes to encourage growth over quick paths)
            length_factor = max(0.05, 1.0 / (1 + 0.1 * snake_length))  # Reduces penalty as snake grows
            rewards[f"snake{snake_id}"] -= 0.01 * length_factor
            
            # Small survival reward that increases with snake length
            # This encourages the agent to grow and stay alive longer
            survival_bonus = 0.005 * (1 + 0.05 * snake_length)
            rewards[f"snake{snake_id}"] += survival_bonus
        
        # Check for max steps reached
        if self.steps >= self.max_steps:
            self.done = True
            # Winner is the snake with the highest score
            if self.score1 > self.score2:
                self.winner = 1
            elif self.score2 > self.score1:
                self.winner = 2
            else:
                # In case of tie, winner is the snake with the longest length
                if self.snake1_length > self.snake2_length:
                    self.winner = 1
                else:
                    self.winner = 2
        
        # Spawn power-ups and speed boosts periodically
        self.power_up_timer -= 1
        self.speed_boost_timer -= 1
        
        if self.power_up_timer <= 0 and not self.power_up:
            self.power_up = self._place_element(GameElement.POWER_UP)
            self.power_up_timer = random.randint(20, 40)
            
        if self.speed_boost_timer <= 0 and not self.speed_boost:
            self.speed_boost = self._place_element(GameElement.SPEED_BOOST)
            self.speed_boost_timer = random.randint(30, 50)
            
        # Update power-up and speed boost timers
        if self.snake1_power_up:
            self.snake1_power_up_timer -= 1
            if self.snake1_power_up_timer <= 0:
                self.snake1_power_up = False
                
        if self.snake2_power_up:
            self.snake2_power_up_timer -= 1
            if self.snake2_power_up_timer <= 0:
                self.snake2_power_up = False
                
        if self.snake1_speed > 1:
            self.snake1_speed_timer -= 1
            if self.snake1_speed_timer <= 0:
                self.snake1_speed = 1
                
        if self.snake2_speed > 1:
            self.snake2_speed_timer -= 1
            if self.snake2_speed_timer <= 0:
                self.snake2_speed = 1
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        return self._get_observation(), rewards, self.done, {"winner": self.winner, "score1": self.score1, "score2": self.score2}
    
    def render(self):
        if self.render_mode != 'human':
            return
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                # Get element and draw with appropriate color
                element = GameElement(self.grid[y, x])
                pygame.draw.rect(self.screen, self.colors[element], rect)
                
                # Draw border for clarity
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        # Draw food
        x, y = self.food
        food_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                              self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors[GameElement.FOOD], food_rect)
        
        # Draw power-up if exists
        if self.power_up:
            x, y = self.power_up
            power_up_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                      self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors[GameElement.POWER_UP], power_up_rect)
            
        # Draw speed boost if exists
        if self.speed_boost:
            x, y = self.speed_boost
            speed_boost_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                        self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors[GameElement.SPEED_BOOST], speed_boost_rect)
        
        # Display scores
        font = pygame.font.SysFont(None, 24)
        score_text = f"Snake 1: {self.score1}  Snake 2: {self.score2}"
        text_surface = font.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        # If game is done, display winner
        if self.done and self.winner:
            winner_text = f"Snake {self.winner} wins!"
            text_surface = font.render(winner_text, True, (255, 255, 0))
            text_rect = text_surface.get_rect(center=(self.screen_width//2, self.screen_height//2))
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    
    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
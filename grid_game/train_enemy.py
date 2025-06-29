import random
from game import Game, ROWS, COLS, TILE_SIZE
import pygame

def move_player_randomly(game):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dx, dy = random.choice(directions)
    
    new_x = game.player_pos[0] + dx
    new_y = game.player_pos[1] + dy

    if 0 <= new_x < ROWS and 0 <= new_y < COLS and game.grid[new_x][new_y] == 0:
        game.player_pos = [new_x, new_y]
        game.player_pixel_pos = [new_y * TILE_SIZE, new_x * TILE_SIZE]

def train_enemy_ai(episodes=10000):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    game = Game(screen)
    total_steps = 0

    for episode in range(episodes):
        game.reset()
        step = 0
        done = False

        while not done and step < 200:
            move_player_randomly(game)
            game.update_enemies(train=True)
            game.animate_enemies()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            step += 1
            total_steps += 1

            if total_steps % 20 == 0 and len(game.enemy_agent.memory) > 64:
                game.enemy_agent.replay(64)

        print(f"Episode {episode + 1}/{episodes} finished.")

    # Save the learned model
    game.enemy_agent.save("enemy_model.pth")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_enemy_ai()

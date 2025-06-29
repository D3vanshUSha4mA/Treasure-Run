import pygame
from game import Game

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Maze Chase")
    clock = pygame.time.Clock()

    game = Game(screen)
    game.enemy_agent.load("enemy_model.pth")

    running = True
    episode = 0

    while running:
        clock.tick(60)  # Limit to 60 frames per second
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.update()  # Handles player + enemy + rendering

        # Stop game after one episode (win or loss)
        if game.game_over or game.won:
            print(f"Episode {episode} ended | {'Won' if game.won else 'Lost'} | Epsilon: {round(game.enemy_agent.epsilon, 2)}")
            running = False  # End the main loop (do not reset)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()


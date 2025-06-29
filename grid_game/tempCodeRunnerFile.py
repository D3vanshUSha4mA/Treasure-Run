import pygame
from game import Game

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Maze Chase")
    clock = pygame.time.Clock()

    game = Game(screen)

    running = True
    while running:
        clock.tick(60)  # 60 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.update()
        if game.game_over:
            print("Game Over!")
            pygame.quit()
            
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

from GameObject import GameObject
import pygame


class Laser(GameObject):
    IMG = pygame.image.load("assets/laser.png")
    IMG_MASK = pygame.mask.from_surface(IMG)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw(self, screen):
        super().draw(screen)

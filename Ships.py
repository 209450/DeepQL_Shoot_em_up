import abc
from enum import auto

import pygame
from GameObject import GameObject, MoveDirection
from Laser import Laser


class ShipPossibleActions:
    LEFT = auto()
    RIGHT = auto()
    SHOOT = auto()


class Ship(GameObject):
    SHOOT_COOLDOWN = 30

    @abc.abstractmethod
    def __init__(self, x, y, health=1):
        super().__init__(x, y)
        self.health = 1
        self.lasers = []
        self.cooldown_counter = 0
        self.health = health

    @abc.abstractmethod
    def draw(self, screen):
        super().draw(screen)

    @abc.abstractmethod
    def shoot(self):
        pass


class Player(Ship):
    IMG = pygame.image.load("assets/ship.png")
    IMG_MASK = pygame.mask.from_surface(IMG)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print(self.__dict__)

    def draw(self, screen):
        super().draw(screen)

        for laser in self.lasers:
            laser.draw(screen)

    def shoot(self):
        super().shoot()
        img_width, img_height = self.IMG_MASK.get_size()

        laser = Laser(self.x, self.y - img_height)
        self.lasers.append(laser)

    def move_lasers(self, step):
        for laser in self.lasers:
            laser.move(step, MoveDirection.UP)


class Enemy(Ship):

    @abc.abstractmethod
    def __init__(self, is_alive=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_alive = is_alive

    @abc.abstractmethod
    def draw(self, screen):
        super().draw(screen)

    @abc.abstractmethod
    def shoot(self):
        pass


class Star(Enemy):
    IMG = pygame.image.load("assets/star.png")
    IMG_MASK = pygame.mask.from_surface(IMG)

    def __init__(self, *args, **kwargs):
        super(Ship, self).__init__(*args, **kwargs)

    def draw(self, screen):
        super().draw(screen)

    def shoot(self):
        pass

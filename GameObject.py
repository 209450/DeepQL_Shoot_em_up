import abc
from enum import Enum, auto

import pygame


class MoveDirection(Enum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()


class GameObject:
    IMG = None
    IMG_MASK = None

    @abc.abstractmethod
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @abc.abstractmethod
    def draw(self, screen):
        screen.blit(self.IMG, (self.x, self.y))

    def detect_collision(self, game_object):
        offset_x = self.x - game_object.x
        offset_y = self.y - game_object.y
        return self.IMG_MASK.overlap(game_object.IMG_MASK, (offset_x, offset_y)) is not None

    def move(self, step, direction):
        if direction is MoveDirection.UP:
            self.y -= step
        elif direction is MoveDirection.RIGHT:
            self.x += step
        elif direction is MoveDirection.DOWN:
            self.y += step
        elif direction is MoveDirection.LEFT:
            self.x -= step

    def __str__(self):
        return f"x:{self.x}, y:{self.y}"



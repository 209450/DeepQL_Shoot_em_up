from pathlib import Path

import numpy as np
import pygame
from pygame.locals import *

from DeepQLearning import train_deep_q_learning_agent
from FAlearning import *
from QLearning import DQLearningAgent, play_and_train, QLearningAgent
from Ships import ShipPossibleActions
from ShootEmUp import ShootEmUp

DEFAULT_TILE_SIZE = (32, 32)


class Game:
    def __init__(self, model, board):
        self.model = model
        self.running = True
        self.display_mode_on = True

        pygame.init()
        tile_width, tile_height = DEFAULT_TILE_SIZE
        self.screen = pygame.display.set_mode(
            (len(board[0]) * tile_width, (len(board) * tile_height)))

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False

    def on_render(self):
        if self.display_mode_on:
            model = self.model
            screen = self.screen

            self.screen.fill((0, 0, 0))  # reload background

            for enemy in model.enemies:
                enemy.draw(screen)
            model.player.draw(screen)

            pygame.display.update()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self.on_cleanup()

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True


# board = [
#     "   * ",
#     "  *  ",
#     " *   ",
#     "     ",
#     "     ",
#     "  p  "]


# board = [
#     "   * ",
#     "  *  ",
#     "    *",
#     "     ",
#     "     ",
#     "  p  "]

# board = [
#     "      * ",
#     "        ",
#     "  ***   ",
#     "     *  ",
#     "        ",
#     "      p "]


board = [
    "*  *  ",
    "      ",
    "  *   ",
    "    * ",
    "      ",
    "      ",
    "      ",
    "      ",
    "      ",
    "      ",
    "     p"]


def train_qlearning_agent(env):
    agent = QLearningAgent(alpha=0.1, epsilon=0.1, discount=0.99, get_legal_actions=env.get_possible_actions)

    for iteration in range(1000):
        play_and_train(env, agent)
        if iteration % 100 is 0:
            print(f"learning iteration:{iteration}")

    print("Learning completed")
    agent.turn_off_learning()
    return agent


def train_fa_learning_agent(env):
    tile_width, tile_height = DEFAULT_TILE_SIZE
    board_size = (len(board[0]) * tile_width, len(board) * tile_height)

    features = [ShotEnemies(env.get_next_state, DEFAULT_TILE_SIZE, board_size),
                CollisionWithPlayer(env.get_next_state, DEFAULT_TILE_SIZE, board_size),
                EnemyNearTheEndOfScreen(env.get_next_state, DEFAULT_TILE_SIZE, board_size),
                AimEnemy(env.get_next_state, DEFAULT_TILE_SIZE, board_size),
                ShootFarEnemies(env.get_next_state, DEFAULT_TILE_SIZE, board_size)]
    agent = FALearningAgent(alpha=0.1, epsilon=0.1, discount=0.99,
                            get_legal_actions=env.get_possible_actions,
                            features=features)

    for iteration in range(1000):
        play_and_train(env, agent)
        if iteration % 100 is 0:
            print(f"learning iteration:{iteration}")
            # print(agent.weights)

    print("Learning completed")
    agent.turn_off_learning()
    return agent


if __name__ == "__main__":
    clock = pygame.time.Clock()
    # game_model = ShootEmUp(board, DEFAULT_TILE_SIZE, player_control=True)
    game_model = ShootEmUp(board, DEFAULT_TILE_SIZE)
    game = Game(game_model, board)

    game.turn_off_display()
    agent = train_qlearning_agent(game_model)
    # agent = train_fa_learning_agent(game_model)
    game.turn_on_display()

    # agent = train_deep_q_learning_agent(game_model)
    while True:
        game_model.reset()
        state = game.model.get_state()
        done = False
        while game.running and not done:
            for event in pygame.event.get():
                game.on_event(event)

            next_action = agent.get_action(state)
            state, reward, done, score = game.model.step(next_action)

            print(score)
            game.on_render()
            clock.tick(10)

        if not game.running:
            break
    game.on_cleanup()

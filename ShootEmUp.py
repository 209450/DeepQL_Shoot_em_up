import random
from copy import copy, deepcopy

import pygame

from GameObject import GameObject, MoveDirection
from Ships import Player, Star, ShipPossibleActions

PLAYER = 'p'
STAR = '*'


class ShootEmUp:
    PLAYER_MOVE_STEP = 12
    LASER_MOVE_STEP = 6
    ENEMY_MOVE_STEP = 3

    def __init__(self, board, tile_size, player_control=False, endless_mode=False):
        self.board = board
        self.tile_size = tile_size
        self.player_control = player_control
        self.endless_mode = endless_mode
        self.endless_mode_counter = 0

        tile_width, tile_height = tile_size
        self.board_width, self.board_height = len(board[0]) * tile_width, len(board) * tile_height

        self.player = None
        self.enemies = []
        self.enemies_to_dispose = []
        for y, board_row in enumerate(board):
            for x, cell in enumerate(board_row):
                x_cords, y_cords = x * tile_width, y * tile_height
                if cell is PLAYER:
                    self.player = Player(x_cords, y_cords)
                if cell is STAR:
                    self.enemies.append(Star(x_cords, y_cords))

        self.init_player = deepcopy(self.player)
        self.init_enemies = deepcopy(self.enemies)
        self.init_enemies_to_dispose = []
        self.score = 0

    def reset(self):
        self.player = deepcopy(self.init_player)
        self.enemies = deepcopy(self.init_enemies)
        self.enemies_to_dispose = deepcopy(self.init_enemies_to_dispose)
        self.score = 0
        return self.get_state()

    def get_state(self):
        return ShootEmUpState(self.player, self.enemies, self.enemies_to_dispose)

    def is_terminal(self, state):
        if state.player.health <= 0:
            return True

        if len(state.enemies) is 0:
            return True

        return False

    def get_possible_actions(self, state):
        possible_actions = []

        player = state.player
        player_move_step = self.PLAYER_MOVE_STEP
        tile_width, tile_height = self.tile_size

        if player.x - player_move_step > 0:  # left
            possible_actions.append(ShipPossibleActions.LEFT)

        if player.x + player_move_step + tile_width < self.board_width:  # right
            possible_actions.append(ShipPossibleActions.RIGHT)

        possible_actions.append(ShipPossibleActions.SHOOT)

        return tuple(possible_actions)

    def get_reward(self, state, action, next_state):
        player = next_state.player

        if player.health <= 0:
            return -1000

        reward = -1
        if action is ShipPossibleActions.SHOOT:
            reward -= 10
        if action is ShipPossibleActions.LEFT or ShipPossibleActions.RIGHT:
            reward -= 1

        # check enemies hit by lasers
        for enemy in next_state.enemies_to_dispose:
            reward += 100

        if len(next_state.enemies) is 0:
            reward += 1000

        return reward

    def step(self, action):
        player = self.player
        player_move_step = self.PLAYER_MOVE_STEP
        tile_width, tile_height = self.tile_size
        prev_state = self.get_state()

        if self.endless_mode:
            self.endless_mode_counter += tile_height // self.ENEMY_MOVE_STEP
            if self.endless_mode_counter % tile_width is 0:
                x = random.randint(0, self.board_width - tile_width)
                self.enemies.append(Star(x, 0))

        # player control
        if self.player_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and player.x - player_move_step > 0:  # left
                player.move(player_move_step, MoveDirection.LEFT)
            elif keys[pygame.K_RIGHT] and player.x + player_move_step + tile_width < self.board_width:  # right
                player.move(player_move_step, MoveDirection.RIGHT)
            elif keys[pygame.K_SPACE]:
                player.shoot()
        # AI control
        else:
            if action is ShipPossibleActions.LEFT:
                self.player.move(self.PLAYER_MOVE_STEP, MoveDirection.LEFT)
            elif action is ShipPossibleActions.RIGHT:
                self.player.move(self.PLAYER_MOVE_STEP, MoveDirection.RIGHT)
            elif action is ShipPossibleActions.SHOOT:
                self.player.shoot()

        # lasers
        player.move_lasers(self.LASER_MOVE_STEP)

        # enemies
        self.enemies_to_dispose = []
        for enemy in self.enemies:
            enemy.move(self.ENEMY_MOVE_STEP, MoveDirection.DOWN)

            # collision with player
            if enemy.detect_collision(player):
                player.health -= 1
                self.enemies_to_dispose.append(enemy)
                self.enemies.remove(enemy)
                continue

            # out of screen
            if enemy.y + tile_height > self.board_height:
                player.health -= 1
                self.enemies_to_dispose.append(enemy)
                self.enemies.remove(enemy)
                continue

            # collision with laser
            for laser in player.lasers:
                if enemy.detect_collision(laser):
                    self.enemies_to_dispose.append(enemy)
                    self.enemies.remove(enemy)
                    self.player.lasers.remove(laser)
                    break

        next_state = self.get_state()
        reward = self.get_reward(prev_state, action, next_state)
        self.score += reward
        return next_state, reward, self.is_terminal(next_state), self.score

    def get_next_state(self, state, action):
        # state = deepcopy(state)

        if action is ShipPossibleActions.LEFT:
            state.player.move(self.PLAYER_MOVE_STEP, MoveDirection.LEFT)
        elif action is ShipPossibleActions.RIGHT:
            state.player.move(self.PLAYER_MOVE_STEP, MoveDirection.RIGHT)
        elif action is ShipPossibleActions.SHOOT:
            state.player.shoot()

        state.player.move_lasers(self.LASER_MOVE_STEP)

        state.enemies_to_dispose = []
        tile_width, tile_height = self.tile_size
        for enemy in state.enemies:
            enemy.move(self.ENEMY_MOVE_STEP, MoveDirection.DOWN)

            # collision with player
            if enemy.detect_collision(state.player):
                state.player.health -= 1
                state.enemies_to_dispose.append(enemy)
                state.enemies.remove(enemy)
                continue

            # out of screen
            if enemy.y + tile_height > self.board_height:
                state.player.health -= 1
                state.enemies_to_dispose.append(enemy)
                state.enemies.remove(enemy)
                continue

            # collision with laser
            for laser in state.player_lasers:
                if enemy.detect_collision(laser):
                    state.enemies_to_dispose.append(enemy)
                    state.enemies.remove(enemy)
                    state.player.lasers.remove(laser)
                    break

        return state



class ShootEmUpState:
    def __init__(self, player, enemies, enemies_to_dispose):
        self.player = deepcopy(player)
        self.enemies = deepcopy(enemies)
        self.enemies_to_dispose = deepcopy(enemies_to_dispose)
        self.player_lasers = self.player.lasers


    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        player = f"Player:[{self.player}], "
        enemies = f"Enemies:[{[str(enemy) for enemy in self.enemies]}], "
        enemies_to_dispose = f"Enemies_to_destroy:[{[str(enemy) for enemy in self.enemies_to_dispose]}], "
        player_lasers = f"Lasers:{[str(laser) for laser in self.player_lasers]}"

        return player + enemies + enemies_to_dispose + player_lasers

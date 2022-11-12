from abc import abstractmethod
from collections import defaultdict
import random

from Ships import ShipPossibleActions


class FALearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, features):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

        self.features = features
        self.weights = [random.random() for i in range(len(features))]

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._get_qvalue(state, action)

    # ---------------------START OF YOUR CODE---------------------#

    def _get_qvalue(self, state, action):

        qvalue = 0
        for feature, weight in zip(self.features, self.weights):
            qvalue += feature.function_approximation_value(state, action) * weight

        return qvalue

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #

        action_values = [self.get_qvalue(state, action) for action in possible_actions]
        max_value = max(action_values)

        return max_value

    def update(self, state, action, reward, next_state):
        """
        Aktualizacja wag:
           wi = wi + learning_rate * delta * feature_function(s,a)
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha  # stopień akceptacji nowej wartości

        #
        # INSERT CODE HERE to update value for the given state and action
        #

        # Błąd tymczasowy
        delta = (reward + gamma * self.get_value(next_state)) - self.get_qvalue(state, action)

        new_weights = []
        for feature, weight in zip(self.features, self.weights):
            new_weight = weight + learning_rate * delta * feature.function_approximation_value(state, action)
            new_weights.append(new_weight)

            # print(new_weights)
        self.weights = new_weights

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #

        actions = {}
        for action in possible_actions:
            actions[action] = self.get_qvalue(state, action)

        key_of_max_value = max(actions, key=actions.get)
        max_value = actions[key_of_max_value]

        keys_of_max_value = [key for key, value in actions.items() if value == max_value]
        best_action = random.choice(keys_of_max_value)

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #        

        best_action = self.get_best_action(state)
        chosen_action = best_action

        probability_of_random_action = random.uniform(0, 1)
        if probability_of_random_action < epsilon:  # epsilon - prawdopodobieństwo na wybor losowego
            chosen_action = random.choice(possible_actions)

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0


class Feature:
    def __init__(self, next_state_callback, tile_size, board_size):
        self.next_state_callback = next_state_callback
        self.tile_size = tile_size
        self.board_size = board_size
        self.next_state = None

    @abstractmethod
    def function_approximation_value(self, state, action):
        self.next_state = self.next_state_callback(state, action)


class ShotEnemies(Feature):
    def function_approximation_value(self, state, action):
        super().function_approximation_value(state, action)

        enemies = self.next_state.enemies_to_dispose
        if len(enemies) > 0:
            return 1
        else:
            return 0


class CollisionWithPlayer(Feature):
    def function_approximation_value(self, state, action):
        super().function_approximation_value(state, action)

        for enemy in self.next_state.enemies:
            if enemy.detect_collision(self.next_state.player):
                return 1

        return 0


class EnemyNearTheEndOfScreen(Feature):
    def function_approximation_value(self, state, action):
        super().function_approximation_value(state, action)

        enemies = self.next_state.enemies
        if len(enemies) > 0:
            board_height = self.board_size[1]
            tile_height = self.tile_size[1]

            distances = []
            for enemy in enemies:
                distance = board_height - (enemy.y + tile_height)
                distances.append(distance)

            if min(distances) > tile_height * 3:
                return 1 / len(distances)
            else:
                return 1
        else:
            return 0


class AimEnemy(Feature):
    def function_approximation_value(self, state, action):
        super().function_approximation_value(state, action)

        enemies = self.next_state.enemies
        if action is ShipPossibleActions.SHOOT and len(enemies) > 0:
            tile_width = self.tile_size[0]
            player = self.next_state.player

            is_aiming_enemies = []
            for enemy in enemies:
                enemy_right_corner = enemy.x + tile_width

                is_aiming_enemy = True if enemy.x <= player.x <= enemy_right_corner else False
                is_aiming_enemies.append(is_aiming_enemy)

            if True in is_aiming_enemies:
                return 1
            else:
                return 1 / len(is_aiming_enemies)
        else:
            return 0


class ShootFarEnemies(Feature):
    def function_approximation_value(self, state, action):
        super().function_approximation_value(state, action)

        enemies = self.next_state.enemies
        if len(enemies) > 0 and action is ShipPossibleActions.SHOOT:
            board_height = self.board_size[1]
            tile_height = self.tile_size[1]

            distances = []
            for enemy in enemies:
                distance = board_height - (enemy.y + tile_height)
                distances.append(distance)

            if min(distances) > tile_height * 3:
                return 1
            else:
                return 1 / len(distances)
        else:
            return 0

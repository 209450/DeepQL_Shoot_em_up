import random
from collections import defaultdict
import copy


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
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
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    # ---------------------START OF YOUR CODE---------------------#

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
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha  # stopień akceptacji nowej wartości

        #
        # INSERT CODE HERE to update value for the given state and action
        #

        current_qvalue = self.get_qvalue(state, action)
        next_state_qvalue = self.get_value(next_state)

        qvalue = (1 - learning_rate) * current_qvalue + learning_rate * (reward + gamma * next_state_qvalue)

        self.set_qvalue(state, action, qvalue)

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


class DQLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Double Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)
        """

        self.get_legal_actions = get_legal_actions
        self._qvaluesA = defaultdict(lambda: defaultdict(lambda: 0))
        self._qvaluesB = defaultdict(lambda: defaultdict(lambda: 0))
        self._functions = {'A': self._qvaluesA, 'B': self._qvaluesB}

        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action, function_key):
        """ Returns Q(state,action) """
        if function_key in self._functions.keys():
            return self._functions[function_key][state][action]
        else:
            return self._qvaluesA[state][action] + self._qvaluesB[state][action]

            # ---------------------START OF YOUR CODE---------------------#

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value in the state for the action
        #

        function_keys = copy.deepcopy(list(self._functions.keys()))

        first_function_key = random.choice(function_keys)
        function_keys.remove(first_function_key)
        second_function_key = function_keys.pop()

        first_qvalues = self._functions[first_function_key]
        second_qvalues = self._functions[second_function_key]

        best_next_action = self.get_best_action(next_state, first_function_key)

        tmp = reward + gamma * second_qvalues[next_state][best_next_action] - first_qvalues[state][action]
        qvalue = first_qvalues[state][action] + learning_rate * tmp

        self._functions[first_function_key][state][action] = qvalue

    def get_best_action(self, state, function_key):
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
            actions[action] = self.get_qvalue(state, action, function_key)

        key_of_max_value = max(actions, key=actions.get)
        max_value = actions[key_of_max_value]

        keys_of_max_value = [key for key, value in actions.items() if value == max_value]
        best_action = random.choice(keys_of_max_value)

        return best_action

    def get_action(self, state):
        """
        Choose action with epsilon greedy policy for Q1 + Q2
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

        best_action = self.get_best_action(state, None)
        chosen_action = best_action

        probability_of_random_action = random.uniform(0, 1)
        if probability_of_random_action < epsilon:  # epsilon - prawdopodobieństwo na wybor losowego
            chosen_action = random.choice(possible_actions)

        return chosen_action

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0


def play_and_train(env, agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False

    while not done:
        # get agent to pick action given state state.
        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)

        #
        # INSERT CODE HERE to train (update) agent for state
        #

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward
import random
from collections import deque

import numpy as np

from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

class DQNAgent:
    def __init__(self, action_size, learning_rate, model, get_legal_actions, epsilon_decay=0.999):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model = model
        self.get_legal_actions = get_legal_actions

    def remember(self, state, action, reward, next_state, done):
        # Function adds information to the memory about last action and its results
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon
        best_action = self.get_best_action(state)
        chosen_action = best_action

        if random.uniform(0, 1) < epsilon:
            random_actions = possible_actions.copy()
            random_actions.remove(best_action)
            chosen_action = random.choice(random_actions if random_actions else [best_action])

        return chosen_action

    def get_best_action(self, state):
        """
        Compute the best action to take in a state.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #

        return np.argmax(self.model.predict(state))

    def replay(self, batch_size):
        """
        Function learn network using randomly selected actions from the memory.
        First calculates Q value for the next state and choose action with the biggest value.
        Target value is calculated according to:
                Q(s,a) := (r + gamma * max_a(Q(s', a)))
        except the situation when the next action is the last action, in such case Q(s, a) := r.
        In order to change only those weights responsible for chosing given action, the rest values should be those
        returned by the network for state state.
        The network should be trained on batch_size samples.
        """
        #
        # INSERT CODE HERE to train network
        #

        if len(self.memory) < batch_size:
            return

        states = []
        targets = []
        next_states = []

        random_memories = random.sample(self.memory, batch_size)
        for random_memory in random_memories:
            state, action, reward, next_state, done = random_memory
            states.append(state.flatten())
            next_states.append(next_state.flatten())

        state_targets = self.model.predict_on_batch(np.array(states))  # list of actions
        next_state_targets = self.model.predict_on_batch(np.array(next_states))

        gamma = self.gamma
        for i, memory in enumerate(random_memories):
            _, action, reward, _, done = memory
            if done:
                state_targets[i][action] = reward
            else:
                Q_next_action = max(next_state_targets[i])
                state_targets[i][action] = reward + gamma * Q_next_action

                # if done:
        #     target[action] = reward
        # else:
        #     Q_next_action = max(self.model.predict(next_state)[0])
        #     target[action] = reward + gamma * Q_next_action
        # targets.append(target.flatten())

        states = np.array(states)
        targets = np.array(state_targets)
        self.model.train_on_batch(states, targets)

    def update_epsilon_value(self):
        # Every each epoch epsilon value should be updated according to equation:
        # self.epsilon *= self.epsilon_decay, but the updated value shouldn't be lower then epsilon_min value

        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon >= self.epsilon_min:
            self.epsilon = new_epsilon


def train_deep_q_learning_agent(env):
    board = env.board

    state_size = np.array(pygame.surfarray.array3d(pygame.display.get_surface())).shape
    action_size = 3
    learning_rate = 0.001

    model = Sequential()
    model.add(Dense(128, input_dim=state_size, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(action_size))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))

    agent = DQNAgent(action_size, learning_rate, model, env.get_possible_actions, epsilon_decay=0.95)
    agent.epsilon = 0.75

    done = False
    batch_size = 64
    EPISODES = 10000
    counter = 0
    for e in range(EPISODES):

        summary = []
        for i in range(100):
            total_reward = 0
            env_state = env.reset()
            # print(env_state)

            # state = np.array([to_categorical(env_state, num_classes=state_size)])
            state = env_state
            for time in range(1000):
                action = agent.get_action(state)
                next_state_env, reward, done, _ = env.step(action)
                total_reward += reward


                next_state = np.array([to_categorical(next_state_env, num_classes=state_size)])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break


            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # print(total_reward)
            # if i%20==0:
            #     print(i)
            summary.append(total_reward)

        agent.update_epsilon_value()
        print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(e, np.mean(summary), agent.epsilon))
        if np.mean(summary) > 0.9:
            print("You Win!")
            return agent


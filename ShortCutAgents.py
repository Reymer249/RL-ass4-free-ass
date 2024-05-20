import random
import numpy as np


class QLearningAgent(object):

    def __init__(self, env, epsilon, alpha):
        # make use of env to get the number of actions and states
        self.n_actions = env.action_size()
        self.n_states = env.state_size()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = 1  # gamma is 1 based on description of the task
        self.Q = np.zeros((env.state_size(), env.action_size()))

    def select_action(self, state):
        a = np.argmax(self.Q[state])  # greedy action
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def update(self, state, action, reward, state_prime):  # TODO: delete state_prime??
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
                                (reward + self.gamma * np.max(self.Q[state_prime]) - self.Q[state, action])


class SARSAAgent(object):

    def __init__(self, env, epsilon, alpha):
        # make use of env to get the number of actions and states
        self.n_actions = env.action_size()
        self.n_states = env.state_size()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = 1  # gamma is 1 based on description of the task
        self.Q = np.zeros((env.state_size(), env.action_size()))

    def select_action(self, state):
        a = np.argmax(self.Q[state])  # greedy action
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def update(self, state, action, reward, state_prime):
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * self.Q[state_prime, self.select_action(state_prime)] - self.Q[state, action])


class ExpectedSARSAAgent(object):

    def __init__(self, env, epsilon, alpha):
        # make use of env to get the number of actions and states
        self.n_actions = env.action_size()
        self.n_states = env.state_size()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = 1  # gamma is 1 based on description of the task
        self.Q = np.zeros((env.state_size(), env.action_size()))

    def select_action(self, state):
        a = np.argmax(self.Q[state])  # greedy action
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def update(self, state, action, reward, state_prime):
        Q_expected = np.sum(self.Q[state_prime] * (self.epsilon / self.n_actions)) + \
            np.max(self.Q[state_prime]) * (1 - self.epsilon)  # calculate expected Q value
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * Q_expected - self.Q[state, action])

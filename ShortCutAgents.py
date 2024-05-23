import random
import numpy as np
from collections import deque


class RollingBuffer:
    def __init__(self, n):
        self.n = n
        self.buffer = deque(maxlen=n)

    def add(self, number):
        self.buffer.append(number)

    def get_buffer(self):
        return np.array(self.buffer)


class BaseTDAgent(object):

    def __init__(self, env, epsilon, alpha, td: int = 1, gamma: float = 1):
        # make use of env to get the number of actions and states
        self.n_actions = env.action_size()
        self.n_states = env.state_size()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma  # gamma initially is 1 based on description of the task
        self.Q = np.zeros((env.state_size(), env.action_size()))
        self.td = td
        self.previous_states = RollingBuffer(n=self.td)
        self.previous_actions = RollingBuffer(n=self.td)
        self.previous_rewards = RollingBuffer(n=self.td)

    def select_action(self, state):
        """
        E-gredy actions selection
        :param state: from which state (number)?
        :return: the action to take (number)
        """
        a = np.argmax(self.Q[state])  # greedy action
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def write_history(self, prev_state, prev_action, reward):
        self.previous_states.add(prev_state)
        self.previous_actions.add(prev_action)
        self.previous_rewards.add(reward)

    def clear_history(self):
        self.previous_states = RollingBuffer(n=self.td)
        self.previous_actions = RollingBuffer(n=self.td)
        self.previous_rewards = RollingBuffer(n=self.td)


class QLearningAgent(BaseTDAgent):

    def update(self, state_prime):
        rewards = self.previous_rewards.get_buffer()
        states = self.previous_states.get_buffer()
        actions = self.previous_actions.get_buffer()
        if len(rewards) == self.td:
            G = 0
            for i in range(self.td):
                G += (self.gamma ** i) * rewards[i]

            # DIFFERENCE
            G += (self.gamma ** self.td) * np.max(self.Q[state_prime])

            self.Q[states[0], actions[0]] = \
                self.Q[states[0], actions[0]] + self.alpha * (G - self.Q[states[0], actions[0]])


class SARSAAgent(BaseTDAgent):

    def update(self, state_prime):
        rewards = self.previous_rewards.get_buffer()
        states = self.previous_states.get_buffer()
        actions = self.previous_actions.get_buffer()
        if len(rewards) == self.td:
            G = 0
            for i in range(self.td):
                G += (self.gamma ** i) * rewards[i]

            # DIFFERENCE
            G += (self.gamma ** self.td) * self.Q[state_prime, self.select_action(state_prime)]

            self.Q[states[0], actions[0]] = \
                self.Q[states[0], actions[0]] + self.alpha * (G - self.Q[states[0], actions[0]])


class ExpectedSARSAAgent(BaseTDAgent):

    def update(self, state, action, reward, state_prime):
        rewards = self.previous_rewards.get_buffer()
        states = self.previous_states.get_buffer()
        actions = self.previous_actions.get_buffer()
        if len(rewards) == self.td:
            G = 0
            for i in range(self.td):
                G += (self.gamma ** i) * rewards[i]

            # DIFFERENCE
            Q_expected = np.sum(self.Q[state_prime] * (self.epsilon / self.n_actions)) + \
                     np.max(self.Q[state_prime]) * (1 - self.epsilon)  # calculate expected Q value
            G += (self.gamma ** self.td) * Q_expected

            self.Q[states[0], actions[0]] = \
                self.Q[states[0], actions[0]] + self.alpha * (G - self.Q[states[0], actions[0]])

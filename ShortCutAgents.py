import random
import numpy as np
from collections import deque


class RollingBuffer:
    """
    A RollingBuffer is a class that implements a fixed-size buffer using a deque (double-ended queue).
    This buffer is used to store a limited history of values which can be any numeric data such as states,
    actions, or rewards in reinforcement learning contexts. The buffer automatically discards the oldest
    data as new data is added once it reaches its maximum capacity.

    Attributes:
        n (int): The maximum number of elements that can be stored in the buffer.
        buffer (deque): The deque that stores the elements of the buffer.

    Methods:
        __init__(self, n): Initializes the RollingBuffer with a specified size.
        add(self, number): Adds a new element to the buffer. If the buffer is full, the oldest element is removed.
        get_buffer(self): Returns a NumPy array containing the current elements of the buffer.
    """
    def __init__(self, n):
        self.n = n
        self.buffer = deque(maxlen=n)

    def add(self, number):
        self.buffer.append(number)

    def get_buffer(self):
        return np.array(self.buffer)


class BaseTDAgent(object):
    """
    BaseTDAgent is a base class for temporal difference learning agents in a reinforcement learning context.
    This class provides the basic functionalities for agents that learn policies based on the temporal difference (TD)
    method, such as Q-learning and SARSA.

    Attributes:
        env (Environment): The environment in which the agent operates. Must provide action_size() and state_size() methods.
        epsilon (float): The probability of choosing a random action. This facilitates exploration.
        alpha (float): The learning rate which determines to what extent newly acquired information overrides old information.
        td (int): The number of steps to consider for updating the value estimates.
        gamma (float): The discount factor which quantifies how much importance is given to future rewards.
        Q (numpy.ndarray): The table used to store the estimated values of state-action pairs.
        previous_states (RollingBuffer): A rolling buffer of previous states encountered by the agent.
        previous_actions (RollingBuffer): A rolling buffer of previous actions taken by the agent.
        previous_rewards (RollingBuffer): A rolling buffer of rewards received by the agent.

    Methods:
        __init__(self, env, epsilon, alpha, td, gamma=1.0):
            Initializes a new agent instance with the specified environment, exploration rate, learning rate,
            temporal difference steps, and discount factor.
        select_action(self, state):
            Decides an action based on the current state using an epsilon-greedy strategy.
        write_history(self, prev_state, prev_action, reward):
            Records the latest state, action, and reward to their respective rolling buffers.
        clear_history(self):
            Resets the history buffers for states, actions, and rewards.
    """
    def __init__(self, env, epsilon: float, alpha: float, td: int, gamma: float = 1):
        """
        Initializes a new instance of the BaseTDAgent class.

        This constructor sets up the agent with the necessary parameters and initializes the Q-table to zeros.
        It also creates rolling buffers to keep track of the agent's recent history of states, actions, and rewards.

        Parameters:
            env (Environment): The environment in which the agent will operate. It should provide methods like
                               action_size() and state_size() to get the respective sizes.
            epsilon (float): The exploration rate, which is the probability of choosing a random action. This is
                             used to balance exploration and exploitation.
            alpha (float): The learning rate, which determines how quickly the agent learns. It controls the rate
                           at which new information affects the old information in the Q-table.
            td (int): The number of steps to consider for updating the value estimates. This is used to initialize
                      the size of the rolling buffers.
            gamma (float): The discount factor, which quantifies the importance of future rewards. A value of 1.0
                           means future rewards are considered as important as immediate rewards.

        Attributes:
            n_actions (int): The number of possible actions in the environment, derived from env.action_size().
            n_states (int): The number of possible states in the environment, derived from env.state_size().
            Q (numpy.ndarray): A table with dimensions (state_size, action_size) initialized to zero, used to store
                               the estimated values of state-action pairs.
            previous_states (RollingBuffer): A rolling buffer that stores the history of states encountered by the agent.
            previous_actions (RollingBuffer): A rolling buffer that stores the history of actions taken by the agent.
            previous_rewards (RollingBuffer): A rolling buffer that stores the history of rewards received by the agent.
        """
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

    def select_action(self, state: int):
        """
        E-gredy actions selection
        :param state: from which state (number)?
        :return: the action to take (number)
        """
        a = np.argmax(self.Q[state])  # greedy action
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def write_history(self, prev_state: int, prev_action: int, reward: int):
        """
        Records the history of the agent's interactions with the environment.

        This method logs the previous state, action taken, and the reward received into rolling buffers. These buffers
        are used to store a fixed number of recent interactions, which can be utilized for learning updates in temporal
        difference learning methods.

        Parameters:
            prev_state (int): The state from which the action was taken.
            prev_action (int): The action that was executed.
            reward (int): The reward received after taking the action.

        The buffers are implemented as instances of the RollingBuffer class, which maintains a fixed-size sequence of
        elements. When the buffer exceeds its capacity, the oldest element is removed to make room for the new one.
        This method ensures that only the most recent interactions are stored, which are the most relevant for the
        agent's immediate learning needs.
        """
        self.previous_states.add(prev_state)
        self.previous_actions.add(prev_action)
        self.previous_rewards.add(reward)

    def clear_history(self):
        """
        Clears the history of the agent's interactions.

        This method resets the rolling buffers for states, actions, and rewards. It is typically called at the beginning
        of a new episode or after an update to the agent's policy, ensuring that the history does not contain irrelevant
        data from previous episodes or different policy phases.

        By reinitializing the rolling buffers, the agent starts with a clean slate for accumulating new and relevant
        interaction data that will be used for future updates. This is crucial for the correctness of temporal difference
        learning methods, which rely heavily on the sequence and recency of interaction data.
        """
        self.previous_states = RollingBuffer(n=self.td)
        self.previous_actions = RollingBuffer(n=self.td)
        self.previous_rewards = RollingBuffer(n=self.td)


class QLearningAgent(BaseTDAgent):
    """
    QLearningAgent is a subclass of BaseTDAgent that implements the Q-learning algorithm.

    Attributes:
        Q (numpy.ndarray): The Q-table which stores the value for each state-action pair.
        previous_rewards (RollingBuffer): A buffer that stores the most recent rewards received.
        previous_states (RollingBuffer): A buffer that stores the most recent states visited.
        previous_actions (RollingBuffer): A buffer that stores the most recent actions taken.
        td (int): The number of time steps to consider for updating the Q-values.
        gamma (float): The discount factor used in the calculation of the total discounted reward.
        alpha (float): The learning rate which determines the extent to which newly acquired information overrides old information.
        epsilon (float): The exploration rate which dictates how often the agent will choose a random action, facilitating exploration of the action space.

    Methods:
        update(state_prime: int):
            Updates the Q-values based on the agent's experience. This method uses the reward and state information stored in the rolling buffers
            to compute the discounted future reward and updates the Q-values for the state-action pairs accordingly.
    """
    def update(self, state_prime: int):
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
    """
    SARSAAgent is a subclass of BaseTDAgent that implements the SARSA (State-Action-Reward-State-Action) learning algorithm.

    Attributes:
        Q (numpy.ndarray): The Q-table which stores the value for each state-action pair.
        previous_rewards (RollingBuffer): A buffer that stores the most recent rewards received.
        previous_states (RollingBuffer): A buffer that stores the most recent states visited.
        previous_actions (RollingBuffer): A buffer that stores the most recent actions taken.
        td (int): The number of time steps to consider for updating the Q-values.
        gamma (float): The discount factor used in the calculation of the total discounted reward.
        alpha (float): The learning rate which determines the extent to which newly acquired information overrides old information.
        epsilon (float): The exploration rate which dictates how often the agent will choose a random action, facilitating exploration of the action space.

    Methods:
        update(state_prime: int):
            Updates the Q-values based on the agent's experience using the SARSA algorithm. This method uses the reward and state information stored in the rolling buffers
            to compute the discounted future reward and updates the Q-values for the state-action pairs accordingly. The key difference in SARSA compared to Q-learning is that
            it uses the action that was actually taken in the next state (not necessarily the best possible action), which makes it an on-policy method.
    """
    def update(self, state_prime: int):
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
    """
    ExpectedSARSAAgent is a subclass of BaseTDAgent that implements the Expected SARSA learning algorithm.

    Attributes:
        Q (numpy.ndarray): The Q-table which stores the value for each state-action pair.
        previous_rewards (RollingBuffer): A buffer that stores the most recent rewards received.
        previous_states (RollingBuffer): A buffer that stores the most recent states visited.
        previous_actions (RollingBuffer): A buffer that stores the most recent actions taken.
        td (int): The number of time steps to consider for updating the Q-values.
        gamma (float): The discount factor used in the calculation of the total discounted reward.
        alpha (float): The learning rate which determines the extent to which newly acquired information overrides old information.
        epsilon (float): The exploration rate which dictates how often the agent will choose a random action, facilitating exploration of the action space.
        n_actions (int): The number of possible actions in the action space.

    Methods:
        update(state_prime: int):
            Updates the Q-values based on the agent's experience using the Expected SARSA algorithm. This method uses the reward and state information stored in the rolling buffers
            to compute the discounted future reward and updates the Q-values for the state-action pairs accordingly. Expected SARSA differs from SARSA by using the expected value of the
            next state's Q-values, taking into account the policy's action probabilities, rather than the Q-value of the actually taken action in the next state.
    """
    def update(self, state_prime: int):
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

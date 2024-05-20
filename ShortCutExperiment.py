# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from typing import Union
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# Helper classes and functions
class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        """
        y: vector of average reward results
        label: string to appear as label in plot legend
        """
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def save(self, name='test.png'):
        """
        name: string for filename of saved figure
        """
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


class ComparisonPlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Parameter (exploration)')
        self.ax.set_ylabel('Average reward')
        self.ax.set_xscale('log')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, x, y, label=None):
        """
        x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x
        label: string to appear as label in plot legend
        """
        if label is not None:
            self.ax.plot(x, y, label=label)
        else:
            self.ax.plot(x, y)

    def save(self, name='test.png'):
        """
        :param: name - string for filename of saved figure
        """
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


def smooth(y, window, poly=1):
    """
    y: vector to be smoothed
    window: size of the smoothing window
    """
    return savgol_filter(y, window, poly)


def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12, 12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions == 0] = '^'
    print_string[greedy_actions == 1] = 'v'
    print_string[greedy_actions == 2] = '<'
    print_string[greedy_actions == 3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12)) == 0] = '0'
    line_breaks = np.zeros((12, 1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8'))


# Experiment functions
def run_repetition(
        env: Union[ShortcutEnvironment, WindyShortcutEnvironment],
        agent: Union[QLearningAgent, SARSAAgent, ExpectedSARSAAgent],
        n_episodes: int,
        n_steps: int
) -> np.array:
    """
    Runs a single repetition of an experiment.

    This function simulates a series of episodes in the given environment using the specified agent. For each episode,
    it simulates actions up to a maximum number of steps or until the environment signals completion. The cumulative
    reward for each episode is recorded.

    :param env: The environment in which the agent operates. Must be an instance of ShortcutEnvironment or WindyShortcutEnvironment.
    :param agent: The agent that will interact with the environment. Must be an instance of QLearningAgent, SARSAAgent, or ExpectedSARSAAgent.
    :param n_episodes: The number of episodes to simulate.
    :param n_steps: The maximum number of steps to simulate for each episode.
    :return: A numpy array containing the cumulative reward for each episode.
    """
    episodes_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):  # for each episode
        cum_reward = 0
        for _ in range(n_steps):  # simulate one run (episode) with max of n_steps steps
            state = env.state()
            action = agent.select_action(state)
            reward = env.step(action)
            cum_reward += reward
            state_prime = env.state()
            agent.update(state, action, reward, state_prime)
            episodes_rewards[i] = cum_reward
            if env.done():
                break
        env.reset()

    return episodes_rewards


def run_repetitions(
        env: ShortcutEnvironment,
        agent_type: str,
        n_repetitions: int,
        n_episodes: int,
        n_steps: int,
        **kwargs
) -> np.array:
    """
    Runs multiple repetitions of an experiment.

    This function simulates multiple repetitions of an experiment in the given environment using agents of the specified type. 
    For each repetition, it runs a series of episodes, simulating actions up to a maximum number of steps or until the environment 
    signals completion. The average cumulative reward across all repetitions for each episode is calculated and returned.

    :param env: The environment in which the agents operate. Must be an instance of ShortcutEnvironment.
    :param agent_type: The type of agent to use for the experiment. Must be one of "q-learning", "SARSA", or "ExpectedSARSA".
    :param n_repetitions: The number of repetitions of the experiment to run.
    :param n_episodes: The number of episodes to simulate in each repetition.
    :param n_steps: The maximum number of steps to simulate for each episode.
    :param kwargs: Additional keyword arguments to pass to the agent constructor. Typically includes learning rate (alpha) and exploration rate (epsilon).
    :return: A numpy array containing the average cumulative reward for each episode across all repetitions.
    """
    curve = np.zeros(n_episodes)
    for _ in range(n_repetitions):
        if agent_type == "q-learning":
            agent = QLearningAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"])
        elif agent_type == "SARSA":
             agent = SARSAAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"])
        else:
             agent = ExpectedSARSAAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"])
        repetition_curve = run_repetition(env, agent, n_episodes=n_episodes, n_steps=n_steps)
        env.reset()  # just double-check, it should be also done in the run_repetitions
        curve += repetition_curve
    return curve / n_repetitions


def experiment(
        env: ShortcutEnvironment,
        windy_env: WindyShortcutEnvironment,
        n_repetitions: int,
        n_episodes: int,
        n_episodes_single: int,
        n_steps: int,
        smoothing_window: int,
        epsilon: float,
        alphas: list,
):
    """
    Conducts experiments comparing Q-learning and SARSA algorithms.

    This function runs experiments to compare the performance of Q-learning and SARSA algorithms in a given environment. 
    It performs experiments with a single fixed alpha value and multiple alpha values to observe the impact of the learning 
    rate on the algorithms' performance. The function plots learning curves for each alpha value and identifies the best 
    alpha based on the area under the curve (AUC) metric. The experiments are conducted in both a standard environment and 
    a windy environment to observe the algorithms' robustness under different conditions.

    :param env: The standard environment in which the agents operate. Must be an instance of ShortcutEnvironment.
    :param windy_env: The windy environment in which the agents operate. Must be an instance of WindyShortcutEnvironment.
    :param n_repetitions: The number of repetitions of the experiment to run for averaging results.
    :param n_episodes: The number of episodes to simulate in each repetition for the multi-alpha experiment.
    :param n_episodes_single: The number of episodes to simulate in the single-alpha experiment.
    :param n_steps: The maximum number of steps to simulate for each episode.
    :param smoothing_window: The window size for smoothing the learning curves.
    :param epsilon: The exploration rate for the agents.
    :param alphas: A list of learning rates (alpha) to test in the experiments.
    """
    # Assignment 1 - Q-learning

    # single experiment
    alpha = 0.1
    q_agent = QLearningAgent(env, epsilon=epsilon, alpha=alpha)
    run_repetition(env, q_agent, n_episodes=n_episodes_single, n_steps=n_steps)
    print("\nQ-LEARNING GREEDY POLICY: \n")
    print_greedy_actions(q_agent.Q)

    # different alpha values
    q_agent_max_auc = -np.inf
    q_agent_best_alpha = alphas[0]
    QLearn_plot = LearningCurvePlot(title="Q-learning learning curves based on alpha hyperparam.")
    for i in range(len(alphas)):
        alpha = alphas[i]
        curve = run_repetitions(
            env=env,
            agent_type="q-learning",
            n_repetitions=n_repetitions,
            n_episodes=n_episodes,
            n_steps=n_steps,
            epsilon=epsilon,
            alpha=alpha
        )
        QLearn_plot.add_curve(smooth(curve, smoothing_window), f"alpha={alpha}")
        curve_auc = np.trapz(curve)  # We get the AUC characteristic (approximation)
        # to compare the curves and select the best for the inter-model comparison
        if curve_auc > q_agent_max_auc:  # We maximize the curve as the reward and the AUC are always negative
            q_agent_max_auc = curve_auc
            q_agent_best_alpha = alpha
    QLearn_plot.save("Q-Learning_plot")

    # Assignment 2 - SARSA

    # single experiment
    alpha = 0.1
    sarsa_agent = SARSAAgent(env, epsilon=epsilon, alpha=alpha)
    run_repetition(env, sarsa_agent, n_episodes=n_episodes_single, n_steps=n_steps)
    print("\nSARSA GREEDY POLICY: \n")
    print_greedy_actions(sarsa_agent.Q)

    # different alpha values
    sarsa_agent_max_auc = -np.inf
    sarsa_agent_best_alpha = alphas[0]
    SARSA_plot = LearningCurvePlot(title="SARSA learning curves based on alpha hyperparam.")
    for i in range(len(alphas)):
        alpha = alphas[i]
        curve = run_repetitions(
            env=env,
            agent_type="SARSA",
            n_repetitions=n_repetitions,
            n_episodes=n_episodes,
            n_steps=n_steps,
            epsilon=epsilon,
            alpha=alpha
        )
        SARSA_plot.add_curve(smooth(curve, smoothing_window), f"alpha={alpha}")
        curve_auc = np.trapz(curve)  # We get the AUC characteristic (approximation)
        # to compare the curves and select the best for the inter-model comparison
        if curve_auc > sarsa_agent_max_auc:  # We maximize the curve as the reward is always negative
            sarsa_agent_max_auc = curve_auc
            sarsa_agent_best_alpha = alpha
    SARSA_plot.save("SARSA_plot")

    # Assignment 3 - Windy weather!
    alpha = 0.1

    q_agent_wind = QLearningAgent(windy_env, epsilon=epsilon, alpha=alpha)
    run_repetition(windy_env, q_agent_wind, n_episodes=n_episodes_single, n_steps=n_steps)
    sarsa_agent_wind = SARSAAgent(windy_env, epsilon=epsilon, alpha=alpha)
    run_repetition(windy_env, sarsa_agent_wind, n_episodes=n_episodes_single, n_steps=n_steps)

    print("\n~~~ WINDY WEATHER ~~~\n")

    print("\nQ-LEARNING GREEDY POLICY WITH WIND:\n")
    print_greedy_actions(q_agent_wind.Q)
    print("\nSARSA GREEDY POLICY WITH WIND:\n")
    print_greedy_actions(sarsa_agent_wind.Q)

    # Assignment 4 - Expected SARSA

    # single experiment
    alpha = 0.1
    exp_sarsa_agent = ExpectedSARSAAgent(env, epsilon=epsilon, alpha=alpha)
    run_repetition(env, exp_sarsa_agent, n_episodes=n_episodes_single, n_steps=n_steps)
    print("\nEXPECTED SARSA GREEDY POLICY: \n")
    print_greedy_actions(exp_sarsa_agent.Q)

    # different alpha values
    exp_sarsa_agent_max_auc = -np.inf
    exp_sarsa_agent_best_alpha = alphas[-1]
    Ex_SARSA_plot = LearningCurvePlot(title="Expected SARSA learning curves based on alpha hyperparam.")
    for i in range(len(alphas)):
        alpha = alphas[i]
        curve = run_repetitions(
            env=env,
            agent_type="expected_SARSA",
            n_repetitions=n_repetitions,
            n_episodes=n_episodes,
            n_steps=n_steps,
            epsilon=epsilon,
            alpha=alpha
        )
        Ex_SARSA_plot.add_curve(smooth(curve, smoothing_window), f"alpha={alpha}")
        curve_auc = np.trapz(curve)  # We get the AUC characteristic (approximation)
        # to compare the curves and select the best for the inter-model comparison
        if curve_auc > exp_sarsa_agent_max_auc:  # We maximize the curve as the reward is always negative
            exp_sarsa_agent_max_auc = curve_auc
            exp_sarsa_agent_best_alpha = alpha
    Ex_SARSA_plot.save("Exp_SARSA_plot")

    # Comparison of the methods
    comparison_plot = LearningCurvePlot(title="Comparison plot of the tuned models")
    q_curve = run_repetitions(
        env=env,
        agent_type="q-learning",
        n_repetitions=n_repetitions,
        n_episodes=n_episodes,
        n_steps=n_steps,
        epsilon=epsilon,
        alpha=q_agent_best_alpha
    )
    comparison_plot.add_curve(
        smooth(q_curve, smoothing_window), f"Q_learning (alpha={q_agent_best_alpha})"
    )
    sarsa_curve = run_repetitions(
        env=env,
        agent_type="SARSA",
        n_repetitions=n_repetitions,
        n_episodes=n_episodes,
        n_steps=n_steps,
        epsilon=epsilon,
        alpha=sarsa_agent_best_alpha
    )
    comparison_plot.add_curve(
        smooth(sarsa_curve, smoothing_window), f"SARSA (alpha={sarsa_agent_best_alpha})"
    )
    exp_sarsa_curve = run_repetitions(
        env=env,
        agent_type="expected_SARSA",
        n_repetitions=n_repetitions,
        n_episodes=n_episodes,
        n_steps=n_steps,
        epsilon=epsilon,
        alpha=exp_sarsa_agent_best_alpha
    )
    comparison_plot.add_curve(
        smooth(exp_sarsa_curve, smoothing_window), f"Exp. SARSA (alpha={exp_sarsa_agent_best_alpha})"
    )
    comparison_plot.save("Comparison")


if __name__ == "__main__":
    # Setting up the parameters of the experiment
    environment = ShortcutEnvironment()
    windy_environment = WindyShortcutEnvironment()
    n_repetitions = 100
    n_episodes = 1_000
    n_episodes_single = 10_000
    n_steps = 100
    alphas = [0.01, 0.1, 0.5, 0.9]
    epsilon = 0.1
    smoothing_window = 31
    # The experiment itself
    experiment(
        env=environment,
        windy_env=windy_environment,
        n_repetitions=n_repetitions,
        n_episodes=n_episodes,
        n_episodes_single=n_episodes_single,
        n_steps=n_steps,
        alphas=alphas,
        epsilon=epsilon,
        smoothing_window=smoothing_window
    )

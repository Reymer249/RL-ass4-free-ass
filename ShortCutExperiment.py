# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from typing import Union
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from Helpers import *
# additional import
from tqdm import tqdm


def run_repetition(
        env: Union[ShortcutEnvironment, WindyShortcutEnvironment],
        agent: Union[QLearningAgent, SARSAAgent, ExpectedSARSAAgent],
        n_episodes: int,
        max_n_steps: int
) -> np.array:
    """
    Runs a single repetition of an experiment.

    This function simulates a series of episodes in the given environment using the specified agent. For each episode,
    it simulates actions up to a maximum number of steps or until the environment signals completion. The cumulative
    reward for each episode is recorded.

    :param env: The environment in which the agent operates. Must be an instance of ShortcutEnvironment or WindyShortcutEnvironment.
    :param agent: The agent that will interact with the environment. Must be an instance of QLearningAgent, SARSAAgent, or ExpectedSARSAAgent.
    :param n_episodes: The number of episodes to simulate.
    :param max_n_steps: The maximum number of steps to simulate for each episode.
    :return: A numpy array containing the cumulative reward for each episode.
    """
    episodes_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):  # for each episode
        cum_reward = 0
        for _ in range(max_n_steps):  # simulate one run (episode) with max of n_steps steps
            state = env.state()
            action = agent.select_action(state)
            # making the step and writing down the reward (so that we can compute n-step TD)
            reward = env.step(action)
            agent.write_history(prev_state=state, prev_action=action, reward=reward)
            cum_reward += reward
            state_prime = env.state()
            agent.update(state_prime)
            episodes_rewards[i] = cum_reward
            if env.done():
                break
        env.reset()
        agent.clear_history()

    return episodes_rewards


def run_repetitions(
        env: ShortcutEnvironment,
        agent_type: str,
        n_repetitions: int,
        n_episodes: int,
        max_n_steps: int,
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
    :param max_n_steps: The maximum number of steps to simulate for each episode.
    :param kwargs: Additional keyword arguments to pass to the agent constructor. Typically includes learning rate (alpha) and exploration rate (epsilon).
    :return: A numpy array containing the average cumulative reward for each episode across all repetitions.
    """
    curve = np.zeros(n_episodes)
    for _ in range(n_repetitions):
        if agent_type == "Q-learning":
            agent = QLearningAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"], td=kwargs["td"])
        elif agent_type == "SARSA":
            agent = SARSAAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"], td=kwargs["td"])
        elif agent_type == "Exp. SARSA":
            agent = ExpectedSARSAAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"], td=kwargs["td"])
        else:
            raise Exception("The agent type specified is not supported. We are sorry :(")
        repetition_curve = run_repetition(env, agent, n_episodes=n_episodes, max_n_steps=max_n_steps)
        env.reset()  # just double-check, it should be also done in the run_repetitions
        curve += repetition_curve
    return curve / n_repetitions


def experiment(
        envs: list,
        n_repetitions: int,
        n_episodes: int,
        max_n_steps: int,
        smoothing_window: int,
        epsilon: float,
        td_steps: list,
        best_alphas_dict: dict,
        alphas: list,
        use_best_alphas: bool
):
    """
    Conducts experiments comparing Q-learning and SARSA algorithms.

    This function runs experiments to compare the performance of Q-learning and SARSA algorithms in a given environment.
    It performs experiments with a single fixed alpha value and multiple alpha values to observe the impact of the learning
    rate on the algorithms' performance. The function plots learning curves for each alpha value and identifies the best
    alpha based on the area under the curve (AUC) metric. The experiments are conducted in both a standard environment and
    a windy environment to observe the algorithms' robustness under different conditions.

    :param envs: the list with the environments
    :param n_repetitions: number of repetitions for every "curve" (every combination of params). The "curve" will be
    averaged over this number of "curves"
    :param n_episodes: number of episodes in the experiment. Defines the "length" of the curve
    :param max_n_steps: max number of steps in each episode
    :param smoothing_window:
    :param epsilon: exploration parameter
    :param td_steps: number of the time steps in the TD algs.
    :param best_alphas_dict: The dictionary with the best alphas for every algorithm
    :param alphas: The list of alphas we want to make experiment with (for every curve)
    :param use_best_alphas: Whether to ignore the alphas parameter and use the best alpha for the algorithm
    :return: none
    """

    algs = {
        0: "Q-learning",
        1: "SARSA",
        2: "Exp. SARSA"
    }

    if use_best_alphas:  # here we use the best alphas from the results of the ass. 2
        # to query the results after completion, we have to run results[alg_number, td_number, env_number]
        results = np.zeros((3, len(td_steps), 2, n_episodes))
        for alg_number, alg_name in tqdm(algs.items(), desc=f"running different algorithms"):
            for env_number in tqdm(range(len(envs)), desc=f"running {alg_name} with diff. envs"):
                env = envs[env_number]
                for td_number in tqdm(range(len(td_steps)), desc=f"running {alg_name} with diff. TD steps"):
                    td_value = td_steps[td_number]
                    curve = run_repetitions(
                        env=env,
                        agent_type=alg_name,
                        n_repetitions=n_repetitions,
                        n_episodes=n_episodes,
                        max_n_steps=max_n_steps,
                        epsilon=epsilon,
                        alpha=best_alphas_dict[alg_name],
                        td=td_value
                    )
                    curve = smooth(curve, smoothing_window)
                    results[alg_number, td_number, env_number] = curve
                    with open('results_best_alphas.npy', 'wb') as f:
                        np.save(f, results)
    else:  # here we go over all possible values defined in the provided list
        # to query the results after completion, we have to run 
        # results[alg_number, td_number, alpha_number, env_number]
        results = np.zeros((3, len(td_steps), len(alphas), 2, n_episodes))
        for alg_number, alg_name in tqdm(algs.items(), desc=f"running different algorithms"):
            for env_number in tqdm(range(len(envs)), desc=f"running {alg_name} with diff. envs"):
                env = envs[env_number]
                for alpha_number in tqdm(range(len(alphas)), desc=f"running {alg_name} with diff alpha values"):
                    alpha_value = alphas[alpha_number]
                    for td_number in tqdm(range(len(td_steps)), desc=f"running {alg_name} with diff. TD steps"):
                        td_value = td_steps[td_number]
                        curve = run_repetitions(
                            env=env,
                            agent_type=alg_name,
                            n_repetitions=n_repetitions,
                            n_episodes=n_episodes,
                            max_n_steps=max_n_steps,
                            epsilon=epsilon,
                            alpha=alpha_value,
                            td=td_value
                        )
                        curve = smooth(curve, smoothing_window)
                        results[alg_number, td_number, alpha_number, env_number] = curve
                        with open('results.npy', 'wb') as f:
                            np.save(f, results)


if __name__ == "__main__":
    # Setting up the parameters of the experiment
    environment = ShortcutEnvironment()
    windy_environment = WindyShortcutEnvironment()
    n_repetitions = 100
    n_episodes = 1_000
    max_n_steps = 100
    td_steps = [1, 2, 4, 8, 16, 32, 64]
    alphas = [0.2, 0.4, 0.6, 0.8, 1]
    epsilon = 0.1
    smoothing_window = 31
    best_alphas_dict = {
        "Q-learning": 0.9,
        "SARSA": 0.5,
        "Exp. SARSA": 0.9
    }
    # The experiment itself
    experiment(
        envs=[environment, windy_environment],
        n_repetitions=n_repetitions,
        n_episodes=n_episodes,
        max_n_steps=max_n_steps,
        td_steps=td_steps,
        epsilon=epsilon,
        smoothing_window=smoothing_window,
        best_alphas_dict=best_alphas_dict,
        alphas=alphas,
        use_best_alphas=False
    )

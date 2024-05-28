from typing import Union
import numpy as np
from Helpers import *

def plot(
        smoothing_window: int,
        td_steps: list,
        alphas: list,
        optimal_val: int
):
    """
    Plots results, achieved from conducted experiment using results.npy file.

    This function creates plots of 3 experiments both for deterministic and windy environments. The first experiment shows
    RMS error for each considered td_steps value and alpha value. It is done for each of the 3 agents: Q-Learning, SARSA and
    Expected SARSA; and for each of the 2 environments. The second experiment plots learning curves of the 3 agents for considered
    td_steps values and best alpha for each of them (based on RMS error). The last experiment takes best values of td_steps and
    alpha for each agent and plots learning curves for them.

    :param smoothing_window: size of the smoothing window
    :param td_steps: number of the time steps in the TD algs.
    :param alphas: The list of alphas we want to make experiment with (for every curve)
    :param optimal_val: The optimal theoretical value for the environment
    :return: none
    """

    results = np.load('results.npy')
    algs = {
        0: "Q-learning",
        1: "SARSA",
        2: "Exp. SARSA"
    }

    for env in range(2):
        # Best alpha values for optimal learning curves of each td_steps and agent
        best_alphas_QLearn_dict = {}
        best_alphas_SARSA_dict = {}
        best_alphas_ExpSARSA_dict = {}

        ### EXPERIMENT 1###
        # Plots comparing average RMS errors based on td_steps(i) and alpha(j) for each agent
        QLearn_plot = ComparisonPlot(title="Average RMS errors of Q-learning based on td_steps and alpha.")
        SARSA_plot = ComparisonPlot(title="Average RMS errors of SARSA based on td_steps and alpha.")
        ExpSARSA_plot = ComparisonPlot(title="Average RMS errors of Expected SARSA based on td_steps and alpha.")
        for i in range(len(td_steps)):
            QLearn_rms = np.zeros(len(alphas))
            SARSA_rms = np.zeros(len(alphas))
            ExpSARSA_rms = np.zeros(len(alphas))
            for j in range(len(alphas)):
                # Average RMS error of a run of each agent, based on td_steps and alpha
                QLearn_rms[j] = np.sqrt(np.mean((results[0, i, j, env] - optimal_val) ** 2))
                SARSA_rms[j] = np.sqrt(np.mean((results[1, i, j, env] - optimal_val) ** 2))
                ExpSARSA_rms[j] = np.sqrt(np.mean((results[2, i, j, env] - optimal_val) ** 2))
            # Saving indexes of best alphas for each td_steps based on the environment for later learning curves
            best_alphas_QLearn_dict[td_steps[i]] = np.argmin(QLearn_rms)
            best_alphas_SARSA_dict[td_steps[i]] = np.argmin(SARSA_rms)
            best_alphas_ExpSARSA_dict[td_steps[i]] = np.argmin(ExpSARSA_rms)
            # Adding comparison curves
            QLearn_plot.add_curve(alphas, QLearn_rms, f"n={td_steps[i]}")
            SARSA_plot.add_curve(alphas, SARSA_rms, f"n={td_steps[i]}")
            ExpSARSA_plot.add_curve(alphas, ExpSARSA_rms, f"n={td_steps[i]}")
        if env == 0:
            QLearn_plot.save("Q-Learning_comparison.png")
            SARSA_plot.save("SARSA_comparison.png")
            ExpSARSA_plot.save("ExpSARSA_comparison.png")
        else:
            QLearn_plot.save("Q-Learning_comparison_wind.png")
            SARSA_plot.save("SARSA_comparison_wind.png")
            ExpSARSA_plot.save("ExpSARSA_comparison_wind.png")

        ### EXPERIMENT 2 ###
        # Learning curves comparing td_steps using best alphas for each td_step and agent
        QLearn_plot = LearningCurvePlot(title="Q-learning learning curves based on td_steps hyperparam.")
        SARSA_plot = LearningCurvePlot(title="SARSA learning curves based on td_steps hyperparam.")
        ExpSARSA_plot = LearningCurvePlot(title="Expected SARSA learning curves based on td_steps hyperparam.")
        for i in range(len(td_steps)):
            # Experiment 1 - Q-learning agent
            curve = results[0, i, best_alphas_QLearn_dict[td_steps[i]], env]
            QLearn_plot.add_curve(smooth(curve, smoothing_window), f"n={td_steps[i]} (alpha = {alphas[best_alphas_QLearn_dict[td_steps[i]]]})")

            # Experiment 2 - SARSA agent
            curve = results[1, i, best_alphas_SARSA_dict[td_steps[i]], env]
            SARSA_plot.add_curve(smooth(curve, smoothing_window), f"n={td_steps[i]} (alpha = {alphas[best_alphas_SARSA_dict[td_steps[i]]]})")

            # Experiment 3 - Expected SARSA agent
            curve = results[2, i, best_alphas_ExpSARSA_dict[td_steps[i]], env]
            ExpSARSA_plot.add_curve(smooth(curve, smoothing_window), f"n={td_steps[i]} (alpha = {alphas[best_alphas_ExpSARSA_dict[td_steps[i]]]})")
        if env == 0:
            QLearn_plot.save("Q-Learning_plot")
            SARSA_plot.save("SARSA_plot")
            ExpSARSA_plot.save("ExpSARSA_plot")
        else:
            QLearn_plot.save("Q-Learning_wind_plot")
            SARSA_plot.save("SARSA_wind_plot")
            ExpSARSA_plot.save("ExpSARSA_wind_plot")

        ### EXPERIMENT 3 ###
        # Comparing best learning curves of each agent based on td_steps, alpha
        comparison_plot = LearningCurvePlot(title="Learning curves of algs. with optimal hyperparameters")
        # Q-learning agent
        n = 1
        curve = results[0, n, best_alphas_QLearn_dict[td_steps[n]], env]
        comparison_plot.add_curve(smooth(curve, smoothing_window), f"{algs[0]} (n={td_steps[n]}, alpha={alphas[best_alphas_QLearn_dict[td_steps[n]]]})")
        # SARSA agent
        n = 1 if env==0 else 0
        curve = results[1, n, best_alphas_SARSA_dict[td_steps[n]], env]
        comparison_plot.add_curve(smooth(curve, smoothing_window), f"{algs[1]} (n={td_steps[n]}, alpha={alphas[best_alphas_SARSA_dict[td_steps[n]]]})")
        # Expected SARSA agent
        n = 1 if env==0 else 0
        curve = results[2, n, best_alphas_ExpSARSA_dict[td_steps[n]], env]
        comparison_plot.add_curve(smooth(curve, smoothing_window), f"{algs[2]} (n={td_steps[n]}, alpha={alphas[best_alphas_ExpSARSA_dict[td_steps[n]]]})")
        if env==0:
            comparison_plot.save("comparison_plot")
        else:
            comparison_plot.save("comparison_wind_plot")
    


if __name__ == "__main__":
    # Setting up the parameters of the plots
    td_steps = [1, 2, 4, 8, 16, 32, 64]
    alphas = [0.2, 0.4, 0.6, 0.8, 1]
    smoothing_window = 31
    optimal_val = -7
    # The experiment itself
    plot(
        td_steps=td_steps,
        smoothing_window=smoothing_window,
        alphas=alphas,
        optimal_val=optimal_val
    )

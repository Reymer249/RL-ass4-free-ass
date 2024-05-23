import numpy as np
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

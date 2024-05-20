import random
import numpy as np


class Environment(object):

    def __init__(self):
        pass

    def reset(self):
        '''Reset the environment.
        
        Returns:
           starting_position: Starting position of the agent.
        '''
        raise Exception("Must be implemented by subclass.")

    def render(self):
        '''Render environment to screen.'''
        raise Exception("Must be implemented by subclass.")

    def step(self, action):
        '''Take action.
        
        Arguments:
           action: action to take.
        
        Returns:
           reward: reward of action taken.
        '''
        raise Exception("Must be implemented by subclass.")

    def possible_actions(self):
        '''Return list of possible actions in current state.
        
        Returns:
          actions: list of possible actions.
        '''
        raise Exception("Must be implemented by subclass.")

    def state(self):
        '''Return current state.

        Returns:
          state: environment-specific representation of current state.
        '''
        raise Exception("Must be implemented by subclass.")

    def state_size(self):
        '''Return the number of elements of the state space.

        Returns:
          state_size: number of elements of the state space.
        '''
        raise Exception("Must be implemented by subclass.")

    def action_size(self):
        '''Return the number of elements of the action space.

        Returns:
          state_size: number of elements of the action space.
        '''
        raise Exception("Must be implemented by subclass.")

    def done(self):
        '''Return whether current episode is finished and environment should be reset.

        Returns:
          done: True if current episode is finished.
        '''
        raise Exception("Must be implemented by subclass.")


class ShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 12  # number of rows in the env.
        self.c = 12  # number of columns
        self.rng = random.Random(seed)  # rng - random number generator (will be used later)
        s = np.zeros((self.r, self.c + 1), dtype=str)  # the state space, filled with "0" initially
        # There are n+1 columns so that we have \n at the end (that's for printing)
        s[:] = 'X'  # We fill the state space with "X"
        s[:, -1] = '\n'  # The last column (it's used only for printing)
        s[self.r // 3:, self.c // 3:2 * self.c // 3] = 'C'  # cliff cells
        s[5 * self.r // 6 - 1, :self.c // 2] = 'X'  # The path we have in between cliffs (look p1 of ass. pdf)
        s[2 * self.r // 3:5 * self.r // 6:, self.c // 2] = 'X'  # Still path
        s[2 * self.r // 3, self.c // 2:2 * self.c // 3] = 'X'  # And more of that path :)
        s[2 * self.r // 3, 2 * self.c // 3] = 'G'  # The Goal state
        self.s = s
        self.reset()

    def reset(self):
        self.x = self.c // 6  # We put the agent on the 1/6-th (rounded below) cell
        rand_number = int(2 * self.rng.random())  # Generate the number in [0, 1] and mult. by 2, which
        # is then "inted" (rounded below). So we have 0 or 1 with equal probabilities
        # That way we represent the statement on the page 1 of the ass. pdf:
        # "In each episode, the agent starts in one of two squares (blue in the figure), each with equal probability."
        if rand_number:
            self.y = 5 * self.r // 6 - 1
        else:
            self.y = self.r // 6
        self.starty = self.y
        self.isdone = False
        return rand_number

    def state(self):
        return self.y * self.c + self.x

    def state_size(self):
        return self.c * self.r

    def action_size(self):
        return 4

    def done(self):
        return self.isdone

    def possible_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')

        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')

        if action == 0:
            if self.y > 0:
                self.y -= 1
        elif action == 1:
            if self.y < self.r - 1:
                self.y += 1
        elif action == 2:
            if self.x > 0:
                self.x -= 1
        elif action == 3:
            if self.x < self.c - 1:
                self.x += 1

        if self.s[self.y, self.x] == 'G':  # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x] == 'C':  # Fall off cliff
            self.x = self.c // 6
            self.y = self.starty
            return -100
        return -1

    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))  # better to look at that in the terminal, but not in some IDE


class WindyShortcutEnvironment(Environment):
    """
    The main difference with the normal ShortcutEnvironment, is that after every action with 50% prob. we can be
    pushed one cell down due to the wind. Take a look at the act method.
    """

    def __init__(self, seed=None):
        self.r = 12
        self.c = 12
        self.rng = random.Random(seed)
        s = np.zeros((self.r, self.c + 1), dtype=str)
        s[:] = 'X'
        s[:, -1] = '\n'
        s[self.r // 3:, self.c // 3:2 * self.c // 3] = 'C'
        s[5 * self.r // 6 - 1, :self.c // 2] = 'X'
        s[2 * self.r // 3:5 * self.r // 6:, self.c // 2] = 'X'
        s[2 * self.r // 3, self.c // 2:2 * self.c // 3] = 'X'
        s[2 * self.r // 3, 2 * self.c // 3] = 'G'
        self.s = s
        self.reset()

    def reset(self):
        self.x = self.c // 6
        rand_number = int(2 * self.rng.random())
        if rand_number:
            self.y = 5 * self.r // 6 - 1
        else:
            self.y = self.r // 6
        self.starty = self.y
        self.isdone = False
        return rand_number

    def state(self):
        return self.y * self.c + self.x

    def state_size(self):
        return self.c * self.r

    def action_size(self):
        return 4

    def done(self):
        return self.isdone

    def possible_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')

        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')

        if action == 0:
            if self.y > 0:
                self.y -= 1
        elif action == 1:
            if self.y < self.r - 1:
                self.y += 1
        elif action == 2:
            if self.x > 0:
                self.x -= 1
        elif action == 3:
            if self.x < self.c - 1:
                self.x += 1

        if self.rng.random() < 0.5:  # <-----  HERE IS THE DIFFERENCE
            # Wind!
            if self.y < self.r - 1:
                self.y += 1

        if self.s[self.y, self.x] == 'G':  # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x] == 'C':  # Fall off cliff
            self.x = self.c // 6
            self.y = self.starty
            return -100
        return -1

    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))

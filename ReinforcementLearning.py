import gym
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle


class Environment(ABC):
    """
    Abstract superclass for all RL environment classes
    """

    def __init__(self, nstates, nactions):
        """
        Creates Environment class with a given number of states and actions
        stored in attributes nstates and nactions
        :param nstates: number of states (positive integer)
        :param nactions: number of actions (positive integer)
        """
        self.nstates = nstates
        self.nactions = nactions

    @abstractmethod
    def state(self):
        """
        Gives current environment state
        :return: current state
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Takes an action in the environment
        :param action: action that must be taken
        :return: depends on the specific environment
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the environment
        :return: the state to which the environment is reset
        """
        pass


class GymEnvironment(Environment):
    """
    Wrapper class for OpenAI Gym environments
    Inherits from abstract class Environment
    """

    def __init__(self, gym_env):
        """
        Creates a wrapper class for an OpenAI Gym environment object
        :param gym_env: the original OpenAI Gym object
        The original environment is stored in self.gym_env
        """
        super().__init__(gym_env.nS, gym_env.nA)
        self.gym_env = gym_env

    def state(self):
        """
        Gives current OpenAI Gym environment state
        :return: current state self.gym_env.s
        """
        return self.gym_env.s

    def step(self, action):
        """
        Takes an action in the OpenAI Gym environment
        Calls self.gym_env.step(action)
        :param action: action that must be taken
        :return: next_state, reward, done, info
        """
        next_state, reward, done, info = self.gym_env.step(action)
        return next_state, reward, done, info

    def reset(self):
        """
        Resets the OpenAI Gym environment
        Calls self.gym_env.reset()
        :return: the state to which the environment is reset
        """
        state = self.gym_env.reset()
        return state

    @staticmethod
    def make(environment_name):
        """
        Static method to create a GymEnvironment object wrapping an OpenAI Gym environment
        :param environment_name: the name of the OpenAI Gym environment (string)
        :return: GymEnvironment object wrapping the OpenAI Gym object
        """
        gym_env = gym.make(environment_name)
        return GymEnvironment(gym_env)


class FrozenLake(GymEnvironment):
    """
    Wrapper class for the OpenAI Gym FrozenLake-v0 enviroment
    Inherits from class GymEnvironment
    """

    def __init__(self, gym_env):
        """
        Creates a wrapper class for the OpenAI Gym FrozenLake-v0 environment
        :param gym_env: the original OpenAI Gym FrozenLake-v0 object
        The original environment is stored in self.gym_env
        """
        super().__init__(gym_env)
        self._grid = np.array([[1, 0, 0, 0],
                               [0, -1, 0, -1],
                               [0, 0, 0, -1],
                               [-1, 0, 0, 2]])

    def render(self):
        """
        Renders the FrozenLake environment
        Calls self.gym_env.render()
        """
        self.gym_env.render()

    def plot(self, values=None, show_state=False, policy=None, update=False, title=None):
        """
        Plots the FrozenLake environment
        :param values: state-value array - optional, if given, grid cells are colored according to these values
                       if not given, the starting cell is colored brown, the ending cell green, and the holes blue
        :param show_state: if True, the current environment state is indicated with a red square, default is False
        :param policy: Policy object - optional, if given, the policy is plotted using red arrows
        :param update: if True, an existing plot of the environment is updated,
                       default is False, in which case a new plot is created
        :param title: plot title (string), optional
        """

        # update existing figure
        if update:
            fig = plt.gcf()
            ax = fig.axes[0]
            ax.clear()
        # create new figure
        else:
            fig = plt.figure()
            ax = plt.axes()

        # no state-values given -> color grid cells start, end and holes
        if values is None:
            cmap = colors.ListedColormap(['lightblue', 'snow', 'peru', 'lightgreen'])
            bounds = np.linspace(-1.5, 2.5, 5)
            norm = colors.BoundaryNorm(bounds, cmap.N)
            ax.matshow(self._grid, cmap=cmap, norm=norm)
        # state-values given -> use colormap Blues to color the grid cells
        else:
            values = np.reshape(values.copy(), (4, 4), order="c")
            values[values < 1e-9] = np.NaN
            values = ax.matshow(values, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
            if not update:
                fig.colorbar(values, ax=ax)

        # policy given -> plot arrow using quiver
        if policy is not None:
            x, y, u, v = [], [], [], []
            uv = [[-0.5, 0.0],  # left = 0
                  [0.0, -0.5],  # down = 1
                  [0.5, 0.0],  # right = 2
                  [0.0, 0.5]]  # up = 3
            uv = np.array(uv)
            for state in range(policy.prob.shape[0]):
                ij = np.unravel_index(state, self._grid.shape, "C")
                if self._grid[ij] == 0 or self._grid[ij] == 1:
                    p = policy.prob[state, :]
                    b = p > 0
                    p = uv[b, :] * p[b, np.newaxis]
                    for i in range(len(p)):
                        x.append(ij[1])
                        y.append(ij[0])
                        u.append(p[i, 0])
                        v.append(p[i, 1])
            ax.quiver(x, y, u, v, scale=1, units="xy", color="red")

        # set axes properties: aspect ratio, ticks ans labels
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(-0.5, 3.5, 5))
        ax.set_xticklabels([])
        ax.set_yticks(np.linspace(-0.5, 3.5, 5))
        ax.set_yticklabels([])
        ax.tick_params(axis=u'both', which=u'both', length=0)

        # plot grid and set axis limits
        plt.grid()
        plt.xlim([-0.5, 3.5])
        plt.ylim([3.5, -0.5])

        # plotting the current state is asked
        if show_state:
            ij = np.unravel_index(self.state(), self._grid.shape, "C")
            ax.add_patch(Rectangle(np.array(ij)[::-1] - 0.5, 1.0, 1.0,
                                   edgecolor='red', fill=False,
                                   linestyle="-", linewidth=2.0))
        # title is given
        if title:
            fig.suptitle(title)

        # show the plot
        if not update:
            fig.show()
        fig.canvas.draw()

    @staticmethod
    def make(is_slippery=True, time_limit=True):
        """
        Static method to create a FrozenLake object wrapping the OpenAI Gym FrozenLake-v0 environment
        :param is_slippery: if False, the FrozenLake environment is deterministic
                            if True (default), the FrozenLake is stochastic
        :param time_limit: If False, the number of actions (steps) are not limited
                           if True (default), the number of actions is limited to 100
        :return: FrozenLake object wrapping the OpenAI Gym FrozenLake-v0 object
        """
        gym_env = gym.make("FrozenLake-v0", is_slippery=is_slippery)
        if not time_limit:
            gym_env = gym_env.env
        return FrozenLake(gym_env)

    @staticmethod
    def num_of_wins(episodes):
        """
        Static method that counts the number of wins in a given list of episodes (plays)
        :param episodes: list of Episode objects
        :return: number of wins (integer)
        """
        wins = [episode.percepts[episode.n-1, 3] == 15 for episode in episodes]
        return np.array(wins).sum()



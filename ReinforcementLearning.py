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

        # create TimeLimit object
        gym_env = gym.make("FrozenLake-v0", is_slippery=is_slippery)

        # get original FrozenLakeEnv object
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


class Agent:
    """
    Class implementing the RL agent
    """

    def __init__(self, env, strategy=None):
        """
        Creates an Agent object
        :param env: Environment object
        :param strategy: LearningStrategy object (optional)
        """

        # set attributes
        self.env = env
        self.strategy = strategy

        # if strategy is given: initialize its MDP
        if strategy is not None:
            self.strategy.set_MDP(self.env.num_of_states(),
                                  self.env.num_of_actions())

    def step(self, action, update_function=None):
        """
        Lets the agent take a step (action) in the environment
        :param action: step (action)
        :param update_function: callback function called after the step is taken
                                must accept a Percept object as input argument
        :return: Percept object holding the state, action, reward, next_state
        """

        # state before step is taken
        state = self.env.state()

        # take step
        next_state, reward, done, info = self.env.step(action)

        # create Percept object
        percept = Percept(state, action, reward, next_state, done)

        # callback function
        if update_function:
            update_function(percept)
        return percept

    def walk(self, policy, update_function=None):
        """
        Lets the agent run one episode following a given policy
        :param policy: Policy object
        :param update_function: callback function called after each step
                                (see method step)
        :return: Episode object
        """

        # initialize episode, state and done flag
        episode = Episode()
        state = self.env.reset()
        done = False

        # continue taking steps as long as episode is not done
        while not done:
            percept = self.step(policy.next_action(state), update_function)
            done = percept.done
            state = percept.next_state
            episode.add(percept)
        return episode

    def run(self, policy, num_of_episodes, update_function=None):
        """
        Lets the agent run many episodes following a given policy
        :param policy: Policy object
        :param num_of_episodes: number of episodes (positive integer)
        :param update_function: callback function called after each step
                                (see method step)
        :return: list of Episode objects
        """
        episodes = []
        for i in range(num_of_episodes):
            episode = self.walk(policy, update_function)
            episodes.append(episode)
        return episodes


class Percept:
    """
    Class implementing the RL percept
    A percept is created when the RL agent takes an action
    It holds the state, the action, the reward, and the next_state
    """

    def __init__(self, state, action, reward, next_state, done=None):
        """
        Creates a Percept object
        :param state: the state in which the agent is before the action is taken
        :param action: the action the agent takes
        :param reward: the reward the agent receives
        :param next_state: the next state the agent arrives at
        :param done: flag indicating if the episode is done or not (optional)
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def to_array(self):
        """
        Converts the Percept object into a numpy array
        :return: numpy array [state, action, reward, next_state]
        """
        return np.array([self.state, self.action,
                         self.reward, self.next_state])

    def get_sa_indices(self):
        """
        Gives the state and action
        :return: state, action
        """
        return self.state, self.action

    def get_sas_indices(self):
        """
        Gives the state, action, and next_state
        :return: state, action, next_state
        """
        return self.state, self.action, self.next_state


class Episode:
    """
    Class implementing the RL episode
    An episode is the path from the begin state to a final state
    It consists of a list of percepts
    """

    def __init__(self):
        """
        Creates an Episode object
        Initializes attribute percepts which is a n x 4 numpy array
        and attribute n which is equal to the number of percepts
        A new Percept object is added using method add()
        """
        self.percepts = np.empty((0, 4))
        self.n = 0

    def add(self, percept):
        """
        Adds a new Percept object to the Episode object
        The Percept object is converted into an array using its to_array() method
        and added as last row to self.percepts
        :param percept: Percept object
        """
        self.percepts = np.vstack((self.percepts, percept.to_array()))
        self.n += 1

    def to_list(self):
        """
        Converts the Episode object to a list of Percept objects
        :return: list of Percept objects
        """
        percept_list = []
        for i in range(self.n):
            percept = Percept(self.percepts[i, 0],  # state
                              self.percepts[i, 1],  # action
                              self.percepts[i, 2],  # reward
                              self.percepts[i, 3],  # next_state
                              i == (self.n-1))  # done
            percept_list.append(percept)
        return percept_list


class Policy:
    """
    Superclass for all RL policy classes
    """

    def __init__(self, prob, env=None):
        """
        Creates a Policy object
        :param prob: nstates x nactions numpy array
                     with nstates the number of states
                     and nactions the number of actions
                     nstates and nactions are stored in attributes with the same name
        :param env: Environment object (optional)
        """
        self.prob = prob
        self.env = env
        self.nstates = prob.shape[0]
        self.nactions = prob.shape[1]

    def next_action(self, state):
        """
        Selects the next action in a given state
        The probabilities of selecting the possible actions are given by self.prob[state, :]
        :param state: state
        :return: action
        """
        p = self.prob[state, :]
        action = np.random.choice(self.prob.shape[1], size=1, p=p)
        return action.item()


class UniformRandomPolicy(Policy):
    """
    Class implementing a uniform random policy
    """

    def __init__(self, *args):
        """
        Creates a UniformRandomPolicy object for
        (1) a given environment or
        (2) a given number of states and actions
        :param args: (1) Environment object or
                     (2) the number of states (nstates) and the number of actions (nactions)
        """

        # UniformRandomPolicy(env)
        if len(args) == 1:
            env = args[0]
            nstates = env.nstates
            nactions = env.nactions

        # UniformRandomPolicy(nstates, nactions)
        else:
            env = None
            nstates = args[0]
            nactions = args[1]

        # probability matrix
        prob = np.ones((nstates, nactions)) / nactions

        # call superclass constructor
        super().__init__(prob, env)



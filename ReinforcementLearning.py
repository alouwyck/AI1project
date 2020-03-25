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

    def seed(self, seed):
        """
        Sets the OpenAI Gym environment random seed
        Calls self.gym_env.seed()
        :param seed: integer
        """
        self.gym_env.seed(seed)

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
            self.strategy.set_MDP(self.env.nstates,
                                  self.env.nactions)

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

    def learn(self, plot_frequency=0):
        """
        Lets the agent run many episodes while applying the given learning strategy.
        The optimal policy and value functions are stored in attribute strategy
        :param plot_frequency: the number of episodes after which the plot must be updated
                               if plot_frequency = 0 (default), no plot is made
        """

        # if plot is asked: initialize variable update_plot
        if plot_frequency:
            update_plot = False

        # loop through episodes
        for i in range(1, self.strategy.num_of_episodes+1):

            # initialize variables state and done
            state = self.env.reset()
            done = False

            # loop through one episode
            while not done:
                percept = self.step(self.strategy.policy.next_action(state))
                self.strategy.learn(percept, i)
                done = percept.done
                state = percept.next_state

            # plot intermediate results if asked
            if plot_frequency and not (i % plot_frequency):
                self.env.plot(values=self.strategy.Vs.copy(),
                              policy=self.strategy.policy,
                              title="episode "+str(i),
                              update=update_plot)
                update_plot = True

        #self.strategy.policy = self.strategy.mdp.policy_improvement(self.strategy.Vs)
        #if plot_frequency:
        #    self.env.plot(self.strategy.policy, self.strategy.Vs.copy(),
        #                  title="Optimal policy", update=update_plot)

    def value_iteration(self, num_of_episodes, gamma=1.0,
                        decay_rate=0.01, eps_min=0.01, eps_max=1.0,
                        precision=1e-3, maxiter=100, seed=None):

        if seed is not None:
            np.random.seed(seed)

        mdp = EmpiricalMDP(self.env.nstates, self.env.nactions, gamma)
        policy = UniformRandomPolicy(self.env)
        states = range(self.env.nstates)
        actions = range(self.env.nactions)
        nsa = self.env.nstates * self.env.nactions
        v = np.zeros(self.env.nstates)
        q = np.zeros((self.env.nstates, self.env.nactions))
        epsilon = eps_max
        if gamma < 1.0:
            epsilon *= (1-gamma)/gamma

        # loop through episodes
        for i in range(1, num_of_episodes + 1):

            # initialize variables state and done
            state = self.env.reset()
            done = False

            # loop through one episode
            while not done:

                # step
                percept = self.step(policy.next_action(state))
                done = percept.done
                state = percept.next_state
                mdp.update(percept)

                # prepare matrices
                prob = np.reshape(policy.prob, (nsa, ), order="c")
                PR = np.reshape(np.sum(mdp.Psas * mdp.Rsas, axis=2),
                                (nsa, ), order="c")
                gPsa = mdp.gamma * np.reshape(mdp.Psas,
                                              (nsa, self.env.nstates),
                                              order="c")

                # evaluate
                rmax = np.max(mdp.Rsas)
                delta = np.Inf
                j = 0
                while (delta > (precision*rmax)) and (j < maxiter):

                    delta = 0.0
                    j += 1

                    u = v.copy()
                    v = np.max(
                               np.reshape(prob * (PR + np.dot(gPsa, v)),
                                          (self.env.nstates, self.env.nactions),
                                          order="c"),
                               axis=1)
                    delta = max(delta, np.max(np.abs(u-v)))

                    # for s in states:
                    #     u = v[s]
                    #     v[s] = np.max(policy.prob[s, :] *
                    #                   (np.sum(mdp.Psas[s, :, :] * mdp.Rsas[s, :, :], axis=1) +
                    #                    np.squeeze(np.dot(mdp.Psas[s, :, :], v)) * gamma))
                    #     delta = max(delta, abs(u-v[s]))

                # improve

                q = np.reshape(PR + np.dot(gPsa, v),
                               (self.env.nstates, self.env.nactions),
                               order="c")
                policy.prob[:, :] = epsilon / self.env.nactions
                is_max = q == np.max(q, axis=1).reshape((self.env.nstates, 1))
                for s in states:
                    a = np.random.choice(np.where(is_max[s, :])[0])
                    policy.prob[s, a] += 1 - epsilon

                # for s in states:
                #     q[s, :] = np.sum(mdp.Psas[s, :, :] * mdp.Rsas[s, :, :], axis=1) + \
                #               np.squeeze(np.dot(mdp.Psas[s, :, :], v)) * gamma
                #     a = np.random.choice(np.where(q[s, :] == np.max(q[s, :]))[0])
                #     policy.prob[s, :] = epsilon / self.env.nactions
                #     policy.prob[s, a] += 1 - epsilon

            # epsilon
            epsilon = eps_min + (eps_max - eps_min) * np.exp(-decay_rate * i)

        return v, q, policy, mdp


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

    def improve(self, Qsa, epsilon):
        """
        Improves the policy using the given action-value function Q
        by applying the epsilon-greedy method
        :param Qsa: action-values Q, nstates x nactions numpy array
        :param epsilon: epsilon probability
        """
        self.prob[:, :] = epsilon / self.nactions
        is_max = Qsa == np.max(Qsa, axis=1).reshape((self.nstates, 1))
        for s in range(self.nstates):
            a = np.random.choice(np.where(is_max[s, :])[0])
            self.prob[s, a] += 1 - epsilon

    @staticmethod
    def seed(seed):
        """
        Sets the policy's random seed
        Calls numpy.random.seed()
        :param seed: integer
        """
        np.random.seed(seed)


class UniformRandomPolicy(Policy):
    """
    Class implementing a uniform random policy
    Inherits from class Policy
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


class MarkovDecisionProcess:
    """
    Class implementing the Markov Decision Process (MDP)
    Provides the dynamic programming algorithms policy evaluation, policy iteration
    and value iteration to solve the MDP
    """

    def __init__(self, states, actions, Psas, Rsas, gamma=1.0, outer=100, inner=0):
        """
        Creates a MarkovDecisionProcess object
        :param states: list or array with the states
                       the number of states nstates = len(states)
        :param actions: list or array with the actions
                        the number of actions nactions = len(actions)
        :param Psas: state transition matrix
                     numpy array with nstates x nactions x nstates probabilities
        :param Rsas: reward matrix
                     numpy array with nstates x nactions x nstates rewards
        :param gamma: discount factor, default is 1.0
        :param outer: default number of outer iterations, optional, default is 100
                      used in policy iteration and value iteration
        :param inner: default number of inner iterations, optional, default is 0
                      used in policy evaluation
                      if inner = 0, the system of equations is solved directly
        All input arguments are assigned to attributes with the same name
        The same is true for nstates and nactions
        """
        self.states = states
        self.nstates = len(self.states)
        self.actions = actions
        self.nactions = len(self.actions)
        self.Psas = Psas
        self.Rsas = Rsas
        self.gamma = gamma
        self.outer = outer
        self.inner = inner

    def policy_evaluation(self, policy, inner=None):
        """
        Evaluates a given policy by calculating the state-value and action-value
        functions of the MDP corresponding to the policy
        :param policy: Policy object
        :param inner: number of iterations, default is given by self.inner
                      if inner = 0, the system of Bellman expectation equations
                      is solved applying the direct solver numpy.linalg.solve
                      if inner > 0, an iterative dynamic programming algorithm
                      is applied
        :return: Vs, Qsa
                 Vs is the state-value function, a numpy array with nstates values
                 Qsa is the action value function an nstates x nactions numpy array
        """

        # if inner is not given, set to self.inner
        if inner is None:
            inner = self.inner

        # get the reduced MRP probability and reward matrices
        Pss, Rs = self.__reduce_matrices(policy)

        # direct solver
        if inner < 1:
            I = np.identity(self.nstates)
            Vs = np.linalg.solve(I - self.gamma * Pss, Rs)

        # iterative solver (dynamic programming)
        else:
            Vs = np.zeros((self.nstates, 1))
            for i in range(inner):
                Vs = Rs + np.dot(self.gamma * Pss, Vs)

        # Qsa
        Qsa = np.zeros((self.nstates, self.nactions))
        for s in range(self.nstates):
            Qsa[s, :] = np.sum(self.Psas[s, :, :] * self.Rsas[s, :, :], axis=1) + \
                        np.squeeze(np.dot(self.Psas[s, :, :], Vs)) * self.gamma

        # output
        return Vs, Qsa

    def __reduce_matrices(self, policy):
        """
        Reduces MDP to MRP by squeezing Psas and Rsas
        :param policy: Policy object
        :return: Pss, Rs
                 Pss is the reduced nstates x nstates probability matrix
                 Rs is the reduced reward vector of length nstates
        """
        Pss = np.zeros((self.nstates, self.nstates))
        Rss = np.zeros((self.nstates, self.nstates))
        for s in range(self.nstates):
            Pss[s, :] = np.dot(policy.prob[s, :], self.Psas[s, :, :])
            Rss[s, :] = np.dot(policy.prob[s, :], self.Psas[s, :, :]*self.Rsas[s, :, :])
        Rs = np.sum(Rss, 1, keepdims=True)
        return Pss, Rs

    def policy_improvement(self, Vs):
        """
        Finds the policy corresponding to the given state-value function
        The method assigns a 100% probability to the action with the largest action-value
        If there is more than one action with maximum value
        the method gives equal probability to all of these actions with maximum value
        instead of selecting one randomly
        :param Vs: state-value function, numpy vector with nstates state-values
        :return: Policy object
        """
        Qsa = np.zeros((self.nstates, self.nactions))
        prob = np.zeros((self.nstates, self.nactions))
        for s in range(self.nstates):
            Qsa[s, :] = np.sum(self.Psas[s, :, :]*self.Rsas[s, :, :], axis=1) + \
                        np.squeeze(np.dot(self.Psas[s, :, :], Vs)) * self.gamma
            b = Qsa[s, :] == np.max(Qsa[s, :])
            prob[s, b] = 1.0 / np.sum(b)
        return Policy(prob)

    def policy_iteration(self, inner=None, outer=None):
        """
        Finds the MDP's optimal policy and value functions applying policy iteration
        which iteratively performs policy evaluation calling method policy_evaluation()
        and policy improvement calling method policy_improvement()
        :param inner: number of inner iterations, default is given by attribute self.inner
                      this parameter determines how the policy evaluation matrix system is solved:
                      if inner = 0 it is solved directly
                      if inner > 0 it is solved iteratively
                      see method policy.evaluation()
        :param outer: number of outer iterations, default is given by attribute self.outer
                      this parameter determines how many successive evaluation and improvement
                      steps are performed
        :return: policy, Vs, Qsa
                 policy is a Policy object with the optimal policy
                 Vs is a numpy vector with the nstates optimal state-values
                 Qsa is a numpy matrix with the nstates x nactions optimal action-values
        """

        # input inner and outer given?
        if inner is None:
            inner = self.inner
        if outer is None:
            outer = self.outer

        # start with uniform random policy
        policy = UniformRandomPolicy(self.nstates, self.nactions)

        # successively perform policy evaluation and improvement
        for i in range(outer):
            Vs, Qsa = self.policy_evaluation(policy, inner=inner)
            policy = self.policy_improvement(Vs)

        # finally do a last evaluation step
        Vs, Qsa = self.policy_evaluation(policy, inner=inner)

        # output
        return policy, Vs, Qsa

    def value_iteration(self, inner=None, outer=None):
        """
        Finds the MDP's optimal policy and value functions applying value iteration
        The algorithm doesn't evaluate and improve the policy explicitely
        but it calls methods policy_improvement() and policy_evaluation() after the iterative process
        :param inner: number of inner iterations, default is given by attribute self.inner
                      this parameter determines how the final policy evaluation step is performed:
                      if inner = 0 matrix system is solved directly
                      if inner > 0 matrix system is solved iteratively
                      see method policy.evaluation()
        :param outer: number of outer iterations, default is given by attribute self.outer
                      this parameter determines the number of iterations of the value iteration algorithm
        :return: policy, Vs, Qsa
                 policy is a Policy object with the optimal policy
                 Vs is a numpy vector with the nstates optimal state-values
                 Qsa is a numpy matrix with the nstates x nactions optimal action-values
        """

        # input inner and outer given?
        if inner is None:
            inner = self.inner
        if outer is None:
            outer = self.outer

        # start with Vs = 0
        Vs = np.zeros((self.nstates, 1))

        # states
        states = range(self.nstates)

        # iterations
        for i in range(outer):
            for s in states:
                Vs[s] = np.max(
                               np.sum(self.Psas[s, :, :] * self.Rsas[s, :, :], axis=1) + \
                               np.squeeze(np.dot(self.Psas[s, :, :], Vs)) * self.gamma
                               )

        # finally do a last improvement and evaluation step
        policy = self.policy_improvement(Vs)
        Vs, Qsa = self.policy_evaluation(policy, inner=inner)

        # output
        return policy, Vs, Qsa


class GymMDP(MarkovDecisionProcess):
    """
    Class implementing the MDP of an OpenAI Gym environment
    Inherits from MarkovDecisionProcess
    """

    def __init__(self, env, gamma=1.0, outer=100, inner=0):
        """
        Creates a GymMDP object
        Derives MDP attributes states, actions, Psas and Rsas
        from the given OpenAI Gym environment
        :param env: GymEnvironment object
        :param gamma: discount factor, optional, default is 1.0
        :param outer: number of outer iterations, optional, default is 100
        :param inner: number of inner iterations, optional, default is 0
        """

        # input: gym environment
        self.env = env

        # states and actions
        nstates = env.nstates
        states = range(nstates)
        nactions = env.nactions
        actions = range(nactions)

        # probabilities and rewards from the gym environment
        envP = env.gym_env.P

        # function to check if state is terminal
        def is_terminal(state):
            Pstate = envP[state]
            is_end = True
            action = 0
            while is_end and action < nactions:
                is_end = len(Pstate[action]) == 1
                if is_end:
                    is_end = (Pstate[action][0][1] == state) & Pstate[action][0][3]
                action += 1
            return is_end

        # allocate Psas and Rsas
        Psas = np.zeros((nstates, nactions, nstates))
        Rsas = np.zeros((nstates, nactions, nstates))

        # loop through all states and actions to define Psas and Rsas
        for state in states:  # number of states
            if not is_terminal(state):  # no terminal state
                for action in actions:  # number of actions
                    for to_state in range(len(envP[state][action])):  # number of next states
                        p = envP[state][action][to_state][0]  # probability
                        to = envP[state][action][to_state][1]  # next state
                        r = envP[state][action][to_state][2]  # reward
                        Psas[state, action, to] += p
                        Rsas[state, action, to] += r

        # call constructor of superclass
        super().__init__(states, actions, Psas, Rsas, gamma, outer, inner)


class EmpiricalMDP(MarkovDecisionProcess):
    """
    Class implementing an empirical Markov Decision Process built up while the agent is learning
    Inherits from class MarkovDecisionProcess
    """

    def __init__(self, nstates, nactions, gamma=1.0, outer=100, inner=0):
        """
        Creates an EmpiricalMDP object
        :param nstates: number of states (positive integer)
        :param nactions: number of actions (positive integer)
        :param gamma: discount factor, optional, default is 1.0
        :param outer: number of outer iterations, optional, default is 100 (see MarkovDecisionProcess)
        :param inner: number of inner iterations, optional, default is 0 (see MarkovDecisionProcess)
        """

        # states and actions
        states = range(nstates)
        actions = range(nactions)

        # allocate probability and reward matrices
        Psas = np.zeros((nstates, nactions, nstates))
        Rsas = np.zeros((nstates, nactions, nstates))

        # call superclass constructor
        super().__init__(states, actions, Psas, Rsas, gamma, outer, inner)

        # allocate frequency matrices
        self.Nsas = np.zeros((nstates, nactions, nstates), dtype=np.int_)
        self.Nsa = np.zeros((nstates, nactions), dtype=np.int_)

    def update(self, percept):
        """
        Updates the MDP using data from the given percept
        :param percept: Percept object
        """
        i = percept.get_sas_indices()
        j = percept.get_sa_indices()
        self.Nsas[i] += 1
        self.Nsa[j] += 1
        self.Rsas[i] += (percept.reward - self.Rsas[i]) / self.Nsas[i]
        self.Psas[j] = self.Nsas[j] / self.Nsa[j]


class NaiveStrategy:
    """
    Class implementing a naive learning strategy in which the agent runs a given
    number of episodes following the given policy and updating the MDP
    after which the MDP can be solved applying value iteration
    """

    def __init__(self, num_of_episodes, policy, gamma=1.0):
        """
        Creates a NaiveStrategy object
        :param num_of_episodes: number of episodes (positive integer)
        :param policy: Policy object
        :param gamma: discount factor, optional, default is 1.0
        """
        self.num_of_episodes = num_of_episodes
        self.policy = policy
        self.gamma = gamma
        self.mdp = None
        self.Qsa = None
        self.Vs = None

    def set_MDP(self, nstates, nactions):
        """
        Initializes the empirical MDP, stored in attribute self.mdp
        Also allocates the value functions stored in attributes self.Vs and self.Qsa
        :param nstates: number of states (positive integer)
        :param nactions: number of actions (positive integer)
        """
        self.mdp = EmpiricalMDP(nstates, nactions, self.gamma)
        self.Qsa = np.zeros((nstates, nactions))
        self.Vs = np.zeros(nstates)

    def learn(self, percept, episode_id):
        """
        Updates the empirical MDP
        This method is called by the agent after each learning step
        When the last episode is finished, the MDP is solved applying value iteration
        (see MarkovDecisionProcess)
        The result is found in attributes self.Qsa and self.Vs
        :param percept: Percept object
        :param episode_id: episode number (positive integer)
        """
        self.mdp.update(percept)
        if episode_id == self.num_of_episodes:
            self.policy, self.Vs, self.Qsa = self.mdp.value_iteration()


class LearningStrategy(NaiveStrategy, ABC):
    """
    Abstract superclass for all classes implementing a reinforcement learning algorithm
    Inherits from NaiveStrategy and ABC
    """

    def __init__(self, num_of_episodes, policy, decay_rate,
                 gamma=1.0, epsilon_min=0.01, epsilon_max=1.0):
        """
        Creates a LearningStrategy object
        :param num_of_episodes: number of episodes (positive integer)
        :param policy: Policy object with initial policy
        :param decay_rate: decay rate
        :param gamma: discount factor, optional, default is 1.0
        :param epsilon_min: minimum epsilon value, optional, default is 0.01
        :param epsilon_max: maximum epsilon value, optional, default is 1.0
        decay rate and minimum and maximum epsilon are used to determine the
        probabilities in the epsilon-greedy policy improvement step
        """
        super().__init__(num_of_episodes, policy, gamma)
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.__epsilon = epsilon_max
        self.__depsilon = epsilon_max - epsilon_min
        self._monitor = None
        self.monitor = None

    def set_monitor(self, Vs=False, Qsa=False, policy=False, check=False):
        self._monitor = dict()
        self.monitor = dict()
        if Vs:
            self.monitor["Vs"] = []
            if callable(Vs):
                self._monitor["Vs"] = lambda strategy: Vs(strategy.Vs)
            else:
                self._monitor["Vs"] = lambda strategy: strategy.Vs
        if Qsa:
            self.monitor["Qsa"] = []
            if callable(Qsa):
                self._monitor["Qsa"] = lambda strategy: Qsa(strategy.Qsa)
            else:
                self._monitor["Qsa"] = lambda strategy: strategy.Qsa
        if policy:
            self.monitor["policy"] = []
            if callable(policy):
                self._monitor["policy"] = lambda strategy: policy(strategy.policy.prob)
            else:
                self._monitor["policy"] = lambda strategy: strategy.policy.prob
        if check:
            self.monitor["check"] = []
            self._monitor["check"] = check

    def learn(self, percept, episode_id):
        """
        Updates the empirical MDP and the value functions by iteratively
        applying an evaluation and an improvement step
        This method is called by the agent after each learning step
        (see MarkovDecisionProcess)
        The result is found in attributes self.Qsa and self.Vs
        :param percept: Percept object
        :param episode_id: episode number (positive integer)
        """
        self.mdp.update(percept)
        self._evaluate(percept)
        self._improve(percept)
        if percept.done:
            self._update_epsilon(episode_id)
            self._update_monitor()

    def _update_epsilon(self, episode_id):
        """
        Updates epsilon after each episode using the decay rate, the given epsilon range,
        and the episode number
        :param episode_id: episode number (positive integer)
        """
        self.__epsilon = self.epsilon_min + self.__depsilon * \
                         np.exp(-self.decay_rate * episode_id)

    def _update_monitor(self):
        if self._monitor:
            for arg, value in self._monitor.items():
                self.monitor[arg].append(value(self))

    @abstractmethod
    def _evaluate(self, percept):
        """
        Evaluation step
        :param percept: Percept object
        """
        pass

    def _improve(self, percept):
        """
        Improvement step
        :param percept: Percept object
        """
        self.policy.improve(self.Qsa, self.__epsilon)


class QLearning(LearningStrategy):
    """
    Class implementing the Q-Learning algorithm
    Inherits from LearningStrategy
    """

    def __init__(self, num_of_episodes, policy, learning_rate, decay_rate,
                 gamma=1.0, epsilon_min=0.01, epsilon_max=1.0):
        """
        Creates a QLearning object
        :param num_of_episodes: number of episodes (positive integer)
        :param policy: Policy object with initial policy
        :param learning_rate: learning rate
        :param decay_rate: decay rate
        :param gamma: discount factor, optional, default is 1.0
        :param epsilon_min: minimum epsilon value, optional, default is 0.01
        :param epsilon_max: maximum epsilon value, optional, default is 1.0
        decay rate and minimum and maximum epsilon are used to determine the
        probabilities in the epsilon-greedy policy improvement step
        """
        super().__init__(num_of_episodes, policy, decay_rate, gamma, epsilon_min, epsilon_max)
        self.learning_rate = learning_rate

    def learn(self, percept, episode_id):
        """
        Updates the empirical MDP and the value functions Q and V by iteratively
        applying an evaluation and an improvement step
        This method is called by the agent after each learning step
        (see MarkovDecisionProcess)
        The result is found in attributes self.Qsa and self.Vs
        :param percept: Percept object
        :param episode_id: episode number (positive integer)
        """
        super().learn(percept, episode_id)
        #self.Vs = np.max(self.Qsa, axis=1)
        self.Vs[percept.state] = np.max(self.Qsa[percept.state, :])

    def _evaluate(self, percept):
        """
        Performs the Q-Learning evaluation step in which the action-value function Q is calculated
        from the new percept
        :param percept: Percept object
        """
        sa = percept.get_sa_indices()
        sas = percept.get_sas_indices()
        self.Qsa[sa] = self.Qsa[sa] + self.learning_rate * \
                       (self.mdp.Rsas[sas] + self.gamma *
                        np.max(self.Qsa[percept.next_state, :]) - self.Qsa[sa])

    def _improve(self, percept):
        """
        Performs the Q-Learning improvement step in which the policy is improved every learning step
        applying a epsilon-greedy method
        :param percept: Percept object
        """
        #for s in range(self.mdp.nstates):
        #    self._improve_policy(s)
        self._improve_policy(percept.state)


class MonteCarlo(QLearning):
    """
    Class implementing the Monte Carlo algorithm
    Inherits from QLearning
    """

    def __init__(self, num_of_episodes, policy, learning_rate, decay_rate,
                 gamma=1.0, epsilon_min=0.01, epsilon_max=1.0):
        """
        Creates a MonteCaro object
        :param num_of_episodes: number of episodes (positive integer)
        :param policy: Policy object with initial policy
        :param learning_rate: learning rate
        :param decay_rate: decay rate
        :param gamma: discount factor, optional, default is 1.0
        :param epsilon_min: minimum epsilon value, optional, default is 0.01
        :param epsilon_max: maximum epsilon value, optional, default is 1.0
        decay rate and minimum and maximum epsilon are used to determine the
        probabilities in the epsilon-greedy policy improvement step
        """
        super().__init__(num_of_episodes, policy, learning_rate, decay_rate,
                         gamma, epsilon_min, epsilon_max)
        self.__percepts = []

    def _evaluate(self, percept):
        """
        Performs the Monte Carlo evaluation step in which the action-value function Q is calculated
        from the new percepts sampled during an episode
        :param percept: Percept object
        """
        self._evaluate_with_condition(percept, percept.done)

    def _evaluate_with_condition(self, percept, condition):
        """
        Performs the evaluation step if condition is satisfied, i.e if condition is True
        :param percept: Percept object
        :param condition: boolean
        """
        self.__percepts.insert(0, percept)
        if condition:
            for percept in self.__percepts:
                super()._evaluate(percept)

    def _improve(self, percept):
        """
        Performs the Monte Carlo improvement step in which the policy is improved every episode
        applying a epsilon-greedy method
        :param percept: Percept object
        """
        self._improve_with_condition(percept, percept.done)

    def _improve_with_condition(self, percept, condition):
        """
        Performs the improvement step if condition is satisfied, i.e if condition is True
        :param percept: Percept object
        :param condition: boolean
        """
        if condition:
            super()._improve(percept)
            self._reset()

    def _reset(self):
        """
        Resets necessary variables when the improvement step is done
        """
        self.__percepts = []


class NStepQLearning(MonteCarlo):
    """
    Class implementing the N-Step Q-Learning algorithm
    Inherits from MonteCarlo
    """

    def __init__(self, num_of_episodes, policy, learning_rate, Nstep, decay_rate,
                 gamma=1.0, epsilon_min=0.01, epsilon_max=1.0):
        """
        Creates a MonteCaro object
        :param num_of_episodes: number of episodes (positive integer)
        :param policy: Policy object with initial policy
        :param learning_rate: learning rate
        :param Nstep: step size N (positive integer)
        :param decay_rate: decay rate
        :param gamma: discount factor, optional, default is 1.0
        :param epsilon_min: minimum epsilon value, optional, default is 0.01
        :param epsilon_max: maximum epsilon value, optional, default is 1.0
        decay rate and minimum and maximum epsilon are used to determine the
        probabilities in the epsilon-greedy policy improvement step
        """
        super().__init__(num_of_episodes, policy, learning_rate, decay_rate,
                         gamma, epsilon_min, epsilon_max)
        self.Nstep = Nstep
        self.__N = 0

    def _evaluate(self, percept):
        """
        Performs the N-Step Q-Learning evaluation step in which the action-value function Q is calculated
        from the new percepts sampled during the last N learning steps
        :param percept: Percept object
        """
        self.__N += 1
        self._evaluate_with_condition(percept, self.__N == self.Nstep)

    def _improve(self, percept):
        """
        Performs the N-Step Q-Learning improvement step in which the policy is improved every N learning steps
        applying a epsilon-greedy method
        :param percept: Percept object
        """
        self._improve_with_condition(percept, self.__N == self.Nstep)
        self.__percepts = []

    def _reset(self):
        """
        Resets necessary variables when the improvement step is done
        """
        super()._reset()
        self.__N = 0


class ValueIteration(LearningStrategy):
    """
    Class implementing the Value Iteration algorithm
    Inherits from LearningStrategy
    """

    def __init__(self, num_of_episodes, policy, decay_rate,
                 precision=1e-3, maxiter=1000, gamma=1.0, epsilon_min=0.01, epsilon_max=1.0):
        """
        Creates a ValueIteration object
        :param num_of_episodes: number of episodes (positive integer)
        :param policy: Policy object with initial policy
        :param decay_rate: decay rate
        :param precision: criterion of convergence for the evaluation step, optional, default is 1e-3
        :param maxiter: maximum number of iterations for the evaluation step (positive integer)
                        optional, default is 1000
        :param gamma: discount factor, optional, default is 1.0
        :param epsilon_min: minimum epsilon value, optional, default is 0.01
        :param epsilon_max: maximum epsilon value, optional, default is 1.0
        decay rate and minimum and maximum epsilon are used to determine the
        probabilities in the epsilon-greedy policy improvement step
        """
        super().__init__(num_of_episodes, policy, decay_rate, gamma, epsilon_min, epsilon_max)
        self.precision = precision
        self.maxiter = maxiter
        self.__precision = precision
        if self.gamma < 1:
            self.__precision *= 1/self.gamma - 1

    def _evaluate(self, percept):
        """
        Performs the Value Iteration evaluation step in which the state-value function V is calculated
        from the new percept
        :param percept: Percept object
        """

        # prepare matrices
        shp = (self.mdp.nstates, self.mdp.nactions)
        nsa = self.mdp.nstates * self.mdp.nactions
        prob = np.reshape(self.policy.prob, (nsa,), order="c")
        PR = np.reshape(np.sum(self.mdp.Psas * self.mdp.Rsas, axis=2),
                        (nsa,), order="c")
        gPsa = self.gamma * np.reshape(self.mdp.Psas,
                                       (nsa, self.mdp.nstates),
                                       order="c")

        # calculate Vs iteratively
        precision = self.__precision * np.max(self.mdp.Rsas)
        delta = np.Inf
        i = 0
        while (delta > precision) and (i < self.maxiter):
            i += 1
            delta = 0.0
            u = self.Vs.copy()
            self.Vs = np.max(
                             np.reshape(prob * (PR + np.dot(gPsa, self.Vs)),
                                        shp, order="c"),
                             axis=1)
            delta = max(delta, np.max(np.abs(u - self.Vs)))

        # update Qsa
        self.Qsa = np.reshape(PR + np.dot(gPsa, self.Vs), shp, order="c")



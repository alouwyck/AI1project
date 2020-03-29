import numpy as np
from abc import ABC, abstractmethod

# https://www.datahubbs.com/multi_armed_bandits_reinforcement_learning_1/

class Bandit:
    """
    implementation of the k-armed bandit
    """

    def __init__(self, k, mu=None):
        """
        construct a k-armed bandit
        :param k: the number of arms (required)
        :param mu: array with the mean score for each arm (length is k)
                   if mu is not given, the mean score
                   is sampled from a standard normal distribution
        object has the following attributes:
        k: the number of arms
        mu: the mean score for each arm (length is k)
        total_pulls: the total number of pulls
        total_score: the total score
        npulls: the number of pulls for each arm (length is k)
        score: the total score for each arm (length is k)
        """

        # number of arms
        self.k = k

        # mean score for each arm
        if mu is None:
            self.mu = np.random.normal(0, 1, k)
        else:
            self.mu = np.array(mu)

        # number of plays
        self.total_pulls = 0

        # total score
        self.total_score = 0

        # number of pulls for each arm
        self.npulls = np.zeros(k)

        # score for each arm
        self.score = np.zeros(k)

    def pull(self, a):
        """
        pull arm a of the k-armed bandit
        and update attributes total_pulls, total_score, npulls and score
        :param a: a is the number of the arm that is pulled
                  arms are numbered from 0 to k-1
        :return: score returned by the arm, which is a number
                 score is sampled from a normal distribution with
                 mean score mu of the arm and variance 1
        """

        # score
        mu_arm = self.mu[a]
        score = np.random.normal(mu_arm, 1, 1)
        score = score.item()

        # update
        self.total_pulls += 1
        self.total_score += score
        self.npulls[a] += 1
        self.score[a] += score

        # return score
        return score

    def reset(self):
        """
        reset the k-armed bandit
        attributes total_pulls, total_score, npulls and score are set to zero
        """
        self.total_pulls = 0
        self.total_score = 0
        self.npulls = np.zeros(self.k)
        self.score = np.zeros(self.k)


class Player(ABC):
    """
    abstract parent class to implement a k-armed bandit player
    """

    def __init__(self, bandit):
        """
        construct a k-armed bandit player
        :param bandit: k-armed bandit object
        """
        self.bandit = bandit

    def play(self, n, echo=False):
        """
        play the k-armed bandit game n times
        :param n: number of times the player pulls the bandit
        :param echo: if True, scores are printed
                     default is False, in which case nothing is printed
        :return: the total score of all games
        """

        # reset bandit
        self.bandit.reset()

        # play the game n times by choosing a random arm
        for i in range(n):
            arm_number = self._select_arm()
            score = self.bandit.pull(arm_number)
            # print score of the game if echo is asked
            if echo:
                print("Pull " + str(self.bandit.total_pulls) +
                      ": arm " + str(arm_number) +
                      " gives score " + str(score))

        # print total score if echo is asked
        if echo:
            print("TOTAL SCORE: " + str(self.bandit.total_score))

        # return total score
        return self.bandit.total_score

    @abstractmethod
    def _select_arm(self):
        """
        protected method to select the arm to pull
        :return: index of arm to pull
        """
        pass


class RandomPlayer(Player):
    """
    class to implement a k-armed bandit player who chooses arms randomly
    """

    def __init__(self, bandit):
        """
        construct a k-armed bandit player pulling randomly
        :param bandit: k-armed bandit object
        """
        super().__init__(bandit)

    def _select_arm(self):
        """
        select arm randomly
        :return: index of arm to pull
        """
        return np.random.randint(0, self.bandit.k)


class GreedyPlayer(Player):
    """
    class to implement a k-armed bandit player who chooses arms greedily
    """

    def __init__(self, bandit):
        """
        construct a greedy k-armed bandit player
        :param bandit: k-armed bandit object
        """
        super().__init__(bandit)

    def _select_arm(self):
        """
        select arm greedily
        :return: index of arm to pull
        """
        # no pulls yet: choose arm randomly
        if self.bandit.total_pulls == 0:
            arm_id = np.random.randint(0, self.bandit.k)
        # choose arm with largest mean score
        else:
            mean_scores = np.zeros(self.bandit.k)
            isnotzero = self.bandit.npulls > 0
            mean_scores[isnotzero] = self.bandit.score[isnotzero] / self.bandit.npulls[isnotzero]
            arm_id = np.argmax(mean_scores)
        return arm_id


class EpsilonGreedyPlayer(Player):
    """
    class to implement a k-armed bandit player who chooses arms epsilon-greedily
    """

    def __init__(self, bandit, eps):
        """
        construct a epsilon-greedy k-armed bandit player
        :param bandit: k-armed bandit object
        :param eps: value of epsilon, must be in interval [0,1]
        """
        super().__init__(bandit)
        self.eps = eps

    def _select_arm(self):
        """
        select arm epsilon-greedily
        :return: index of arm to pull
        """

        # random number between 0 and 1
        p = np.random.rand()
        # p < eps or no pulls yet: choose arm randomly
        if p < self.eps or self.bandit.total_pulls == 0:
            arm_id = np.random.randint(0, self.bandit.k)
        # choose arm with largest mean score
        else:
            mean_scores = np.zeros(self.bandit.k)
            isnotzero = self.bandit.npulls > 0
            mean_scores[isnotzero] = self.bandit.score[isnotzero] / self.bandit.npulls[isnotzero]
            arm_id = np.argmax(mean_scores)
        return arm_id


# test
bandit = Bandit(5, np.arange(1, 6))
player1 = RandomPlayer(bandit)
player2 = GreedyPlayer(bandit)
player3 = EpsilonGreedyPlayer(bandit, 0.1)

npulls = 1000

player1.play(npulls)
print(bandit.total_pulls)
print(bandit.total_score)
print(bandit.npulls)
print(bandit.score)

player2.play(npulls)
print(bandit.total_pulls)
print(bandit.total_score)
print(bandit.npulls)
print(bandit.score)

player3.play(npulls)
print(bandit.total_pulls)
print(bandit.total_score)
print(bandit.npulls)
print(bandit.score)

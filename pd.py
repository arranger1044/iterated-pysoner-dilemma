import numpy


class IPDGame(object):

    """
    Representing a two players Iterated Prisoner's Dilemma game (IPD)
    """

    def __init__(self,
                 players,
                 payoffs,
                 obs_noise=0.0):
        """
        WRITEME
        """
        self._payoffs = payoffs
        self._players = players
        self._n_players = len(players)

        #
        # noise in payoffs as seen by players
        self._noise = obs_noise

        #
        # a game will have a history of actions and rewards when simulated
        self.actions = None
        self.rewards = None

    def _gen_noisy_channel(self):
        return numpy.random.randn(self._payoffs.shape) * self._noise

    def payoffs(self):
        """
        WRITEME
        """
        return self._payoffs  # + self._gen_noisy_channel()

    def get_actions(self, iter_start, iter_end):
        """
        Return the actions for the previous iterates
        """
        # print(iter_start, iter_end)
        return self.actions[:, iter_start:iter_end]

    def _get_payoffs(self, actions):
        """
        Accessing matrix payoffs
        """
        return self._payoffs[[[a] for a in actions]]

    @classmethod
    def build_payoff_matrix(cls, n_players, n_actions):
        """
        For a 2 players game with 2 actions the matrix
        2 x 2 x 2 (n_actions x n_actions x n_players)
        """
        geometry = [n_actions for i in range(n_players)] + [n_players]
        return numpy.zeros(geometry)

    @classmethod
    def repr_action(cls, action):
        """
        0 = 'C' = 'Cooperate'
        1 = 'D' = 'Defect'
        """
        action_str = '*'
        if action == 0:
            action_str = 'C'
        elif action == 1:
            action_str = 'D'

        return action_str

    @classmethod
    def visualize_actions_text(cls, players, actions, rewards):
        """
        WRITEME
        """
        n_iters = actions.shape[1]
        n_players = len(players)

        for p in range(n_players):
            print('\n{0}:'.format(players[p]), end=' ')
            for i in range(n_iters):
                print(IPDGame.repr_action(actions[p, i]), end='')
            print('  [{}]'.format(format(rewards[p].sum())), end='')
        print('\n')

    def simulate(self,
                 iters=None,
                 printing=False):
        """
        WRITEME
        """
        if iters is None:
            iters = self._n_iters

        #
        # allocating players rewards and actions for the game
        self.rewards = numpy.zeros((self._n_players, iters))
        self.actions = numpy.zeros((self._n_players, iters), dtype=int)

        for i in range(iters):
            for p_id, p in enumerate(self._players):
                self.actions[p_id, i] = p.act(self, i)
                # print('\t{0}: {1}'.format(p, IPDGameSimulation.repr_action(actions[p.id, i])))
            self.rewards[:, i] = self._get_payoffs(self.actions[:, i])
            # print(self._game.actions)

        #
        # printing
        if printing:
            IPDGame.visualize_actions_text(self._players,
                                           self.actions,
                                           self.rewards)

        return self.actions, self.rewards


class IPDPairwiseCompetition(object):

    def __init__(self,
                 players,
                 payoffs,
                 n_iters,
                 obs_noise=0.0):
        """
        WRITEME
        """
        self._n_iters = n_iters
        self._players = players
        self._payoffs = payoffs

        self._noise = obs_noise
        # self.curr_iter = 0
        self._eps = 1e-5

    def simulate(self, n_iters=None, printing=True):
        """
        WRITEME
        """

        if n_iters is None:
            n_iters = self._n_iters

        #
        # allocating stats matrix
        n_players = len(self._players)
        scores = numpy.zeros((n_players, n_players))
        #
        # an array for #victories, #draws, #losses
        victories = numpy.zeros((n_players, 3), dtype=int)
        player_instances = []
        # draws = numpy.zeros(n_players, dtype=int)

        for p_1 in range(n_players):

            player_1 = self._players[p_1]()
            player_1.set_ingame_id(0)
            player_instances.append(player_1)

            for p_2 in range(p_1, n_players):

                player_2 = self._players[p_2]()
                player_2.set_ingame_id(1)
                #
                # create a new 2 player ipd game
                game = IPDGame([player_1, player_2],
                               self._payoffs,
                               self._noise)
                #
                # collecting stats
                actions, rewards = game.simulate(n_iters, printing)
                scores[p_1, p_2] = rewards[0, :].sum()
                scores[p_2, p_1] = rewards[1, :].sum()

                n_draws = (numpy.abs(rewards[0, :] - rewards[1, :]) < self._eps).sum()
                p_1_victories = (rewards[0, :] > rewards[1, :]).sum()
                p_2_victories = (rewards[0, :] < rewards[1, :]).sum()
                victories[p_1, 1] += n_draws
                victories[p_2, 1] += n_draws
                victories[p_1, 0] += p_1_victories
                victories[p_2, 0] += p_2_victories
                victories[p_1, 2] += victories[p_2, 0]
                victories[p_2, 2] += victories[p_1, 0]
                # victories[p_1] = (rewards[0, :] > rewards[1, :]).sum()
                # victories[p_2] = (rewards[0, :] < rewards[1, :]).sum()

                # if scores[p_1, p_2] > scores[p_2, p_1]:
                #     victories[p_1] += 1
                # elif scores[p_1, p_2] < scores[p_2, p_1]:
                #     victories[p_2] += 1

        if printing:
            IPDPairwiseCompetition.visualize_stats_text(player_instances,
                                                        scores,
                                                        victories,
                                                        n_iters)

        return scores, victories

    @classmethod
    def visualize_stats_text(cls, players, scores, victories, n_iters):

        n_players = len(players)
        assert scores.shape[0] == n_players
        assert len(victories) == n_players

        print('Score matrix')
        print(scores)
        print('Final scores')
        print('Player:\tType:\tScore:\t#Wins\t#Draws\t#Losses:')
        for i, p in enumerate(players):
            print('{0}\t{1}\t{2:.4f}\t[{3}\t{4}\t{5}]'.format(p.id,
                                                              p._type,
                                                              scores[i, :].sum() /
                                                              (n_players * n_iters),
                                                              victories[i, 0],
                                                              victories[i, 1],
                                                              victories[i, 2]))

from abc import ABCMeta
from abc import abstractmethod


class IPDPlayer(metaclass=ABCMeta):

    """
    WRITEME
    """

    id_counter = 0

    @classmethod
    def reset_id_counter(cls):
        """
        WRITEME
        """
        IPDPlayer.id_counter = 0

    def __init__(self,
                 player_type):
        """
        WRITEME
        """
        #
        # global id
        self.id = IPDPlayer.id_counter
        IPDPlayer.id_counter += 1

        #
        # local (game) id (0 or 1)
        self._ingame_id = None

        self._type = player_type

    def set_ingame_id(self, id):
        """
        WRITEME
        """
        self._ingame_id = id

    @abstractmethod
    def act(self, game, iter):
        """
        WRITEME
        """

    def __repr__(self):
        """
        WRITEME
        """
        return "player #{0} ({1})".format(self.id, self._type)


class AllCPlayer(IPDPlayer):

    """
    A Player that always cooperates
    """

    def __init__(self):
        """
        WRITEME
        """
        IPDPlayer.__init__(self, 'AllC')

    def act(self, game, iter):
        """
        Return 'C'
        """
        return 0


class AllDPlayer(IPDPlayer):

    """
    A Player that always defects
    """

    def __init__(self):
        """
        WRITEME
        """
        IPDPlayer.__init__(self, 'AllD')

    def act(self, game, iter):
        """
        Return 'D'
        """
        return 1


class RANDPlayer(IPDPlayer):

    """
    A Player that randomly defects with probability \theta
    """

    def __init__(self, theta=0.5):
        """
        Just storing theta
        """
        IPDPlayer.__init__(self, 'RAND')
        self._prob = theta

    def act(self, game, iter):
        """
        Return 'D' with probability theta
        """
        if numpy.random.rand() < self._prob:
            return 1
        else:
            return 0


class M1SPlayer(IPDPlayer):

    """

    Memory-One Stochastic Player

    A player that chooses its action based on a
    conditional probability distribution over the previous moves
    (Memory one)
    At the first round it generate a default policy
    """

    def __init__(self, cond_probs, default_policy, player_type=None):
        """
        cond_probs is a matrix (2x2)
        """

        if player_type is None:
            player_type = 'M1S'

        IPDPlayer.__init__(self, player_type)

        self._cond_probs = cond_probs
        self._default_policy = default_policy

    def act(self, game, iter):
        """
        The probability to emit 'C' is given by the past actions
        """
        if iter == 0:
            return self._default_policy
        else:
            #
            # look at previous actions
            prev_actions = game.get_actions(iter - 1, iter)  # .reshape(2)
            #
            # ask the probability to generate one decision
            coop_prob = self._cond_probs[[[a] for a in prev_actions]]
            random_choice = numpy.random.rand()
            # print('p', self._cond_probs, coop_prob, self._ingame_id)
            if numpy.random.rand() < coop_prob:
                return 0
            else:
                return 1


class TFTPlayer(M1SPlayer):

    """
    Tit-for-Tat

    TFT can be seen as a special case of the   Memory-One Stochastic Player
    where the cond prob distribution for cooperating is (1, 0, 1, 0)
    and the default policy is too cooperate

    """

    def __init__(self):
        """
        Just calling M1SPlayer constructor with right parameters
        """
        #
        # this code is ugly, I need to create the index first
        tft_cond_probs = None
        M1SPlayer.__init__(self, tft_cond_probs, 0, 'TFT')

        self._cond_probs_0 = numpy.zeros((2, 2))
        self._cond_probs_1 = numpy.zeros((2, 2))

        self._cond_probs_1[0, :] = 1.
        self._cond_probs_0[:, 0] = 1.

    def set_ingame_id(self, id):

        IPDPlayer.set_ingame_id(self, id)

        if id == 0:
            self._cond_probs = self._cond_probs_0
        elif id == 1:
            self._cond_probs = self._cond_probs_1

        # print(self._cond_probs, self._ingame_id)


class STFTPlayer(TFTPlayer):

    """
    Suspicious Tit-for-Tat

    Like TFT, but default policy is to defect on the first move

    """

    def __init__(self):
        """
        Just calling TFTPlayer constructor with right parameters
        """

        TFTPlayer.__init__(self)
        self._default_policy = 1
        self._type = 'STFT'


class GTFTPlayer(TFTPlayer):

    """
    Generous Tit-for-Tat

    Like TFT, but defects not always but only with a probability theta

    """

    def __init__(self, theta=0.3):
        """
        Just calling M1SPlayer constructor with right parameters
        """
        #
        # this code is ugly, I need to create the index first
        gtft_cond_probs = None
        M1SPlayer.__init__(self, gtft_cond_probs, 0, 'GTFT')

        self.theta = theta

        self._cond_probs_0 = numpy.array([[theta for i in range(2)] for j in range(2)])
        self._cond_probs_1 = numpy.array([[theta for i in range(2)] for j in range(2)])

        self._cond_probs_1[0, :] = 1.
        self._cond_probs_0[:, 0] = 1.


class ATFTPlayer(TFTPlayer):

    """
    Adaptive Tit-for-Tat
    1-memory model smoothed with previous adversarial observation
    """

    def __init__(self, theta=0.6):
        TFTPlayer.__init__(self)
        self._type = 'ATFT'
        self._theta = theta
        self._world = None

    def set_ingame_id(self, id):
        IPDPlayer.set_ingame_id(self, id)
        self._world = None

    def act(self, game, iter):

        if iter == 0:
            self._world = 0
            return self._default_policy

        prev_actions = game.get_actions(iter - 1, iter)
        adv_id = 1 - self._ingame_id
        prev_actions = prev_actions.reshape(prev_actions.shape[0])
        if prev_actions[adv_id] == 0:
            world_update = (1 - self._world)
        else:
            world_update = (0 - self._world)

        self._world = self._world + self._theta * world_update

        if self._world >= 0.5:
            return 0
        else:
            return 1


class WSLSPlayer(M1SPlayer):

    """
    Win-Stay Lose-Switch

    If the player won last iter, continue with that strategy,
    if it lost, switch to the opposite
    """

    def __init__(self):
        """
        Just calling M1SPlayer constructor with right parameters
        """

        wsls_cond_probs = numpy.zeros((2, 2))
        M1SPlayer.__init__(self, wsls_cond_probs, 0, 'WSLS')

        self._cond_probs[0, 0] = 1.
        self._cond_probs[1, 1] = 1.


class GRIMPlayer(IPDPlayer):

    """
    Grim Trigger

    Starts cooperating, when the other player defects, always defects
    """

    def __init__(self):
        IPDPlayer.__init__(self, 'GRIM')
        self._defecting = False

    def act(self, game, iter):
        if self._defecting:
            return 1
        else:
            if iter > 0:
                #
                # look at the previous action
                prev_actions = game.get_actions(iter - 1, iter)
                adv_id = 1 - self._ingame_id
                prev_actions = prev_actions.reshape(prev_actions.shape[0])
                if prev_actions[adv_id] == 1:
                    self._defecting = True
                    return 1
                else:
                    return 0

            return 0

    def set_ingame_id(self, id):
        IPDPlayer.set_ingame_id(self, id)
        #
        # this is ugly btw
        self._defecting = False


class HBPlayer(IPDPlayer):

    """
    History-Based Player
    """

    def __init__(self, default_policy, history_length=None, player_type=None):
        """
        history_length == None means full history
        """

        if player_type is None:
            player_type = 'HBP'

        IPDPlayer.__init__(self, player_type)

        self._default_policy = default_policy
        self._history_length = history_length

    # def act(self, game, iter):
    #     """
    #     WRITEME
    #     """
    #     if iter == 0:
    #         return self._default_policy

    def _get_history(self, game, iter):
        """
        """
        adv_id = 1 - self._ingame_id
        iter_start = 0

        if self._history_length is not None:
            iter_start = iter - self._history_length

        return game.get_actions(iter_start, iter)[adv_id, :]


class SMPlayer(HBPlayer):

    """
    Soft-Majority

    Cooperates on the first move, and cooperates as long as the number
    of times the opponent has cooperated is greater than or equal to
    the number of times it has defected, else it defects
    """

    def __init__(self, history_length=None):
        """
        WRITEME
        """
        HBPlayer.__init__(self, 0, history_length, 'SM')

    def act(self, game, iter):
        #
        # apply default policy
        if iter == 0:
            return self._default_policy

        #
        # get history
        adv_action_history = self._get_history(game, iter)
        # print('history', adv_action_history)
        n_defects = adv_action_history.sum()
        if n_defects * 2 <= len(adv_action_history):
            return 0
        else:
            return 1


class HMPlayer(HBPlayer):

    """
    Hard-Majority

    Defects on the first move, and defects if the number of defections
    of the opponent is greater than or equal to the number of times
    it has cooperated, otherwise it cooperates
    """

    def __init__(self, history_length=None):
        """
        WRITEME
        """
        HBPlayer.__init__(self, 1, history_length, 'HM')

    def act(self, game, iter):
        #
        # apply default policy
        if iter == 0:
            return self._default_policy

        #
        # get history
        adv_action_history = self._get_history(game, iter)
        # print('history', adv_action_history)
        n_defects = adv_action_history.sum()
        if n_defects * 2 >= len(adv_action_history):
            return 1
        else:
            return 0


class PHBPlayer(HBPlayer):

    """
    Predictive History Based Player

    Tries to predict the opponent move based on its history and
    then tries to maximise its payoff. Using exponential smoothing

    Starts by being cooperative
    """

    def __init__(self, history_length=None, alpha=0.6):
        """
        WRITEME
        """
        HBPlayer.__init__(self, 0, history_length, 'PHB')
        self._alpha = alpha

    def _predict_next(self, history):
        #
        # exponential smoothing on history
        s = history[0]
        for i in range(1, len(history)):
            s = self._alpha * history[i] + (1 - self._alpha) * s

        #
        # binarizing
        if s > 0.5:
            return 1
        else:
            return 0

    def act(self, game, iter):
        #
        # apply default policy
        if iter == 0:
            return self._default_policy

        #
        # get history
        adv_action_history = self._get_history(game, iter)
        adv_id = 1 - self._ingame_id
        adv_next = self._predict_next(adv_action_history)

        strategy = None
        if adv_id == 1:
            strategy = game.payoffs()[:, adv_next, self._ingame_id]
        elif adv_id == 0:
            strategy = game.payoffs()[adv_next, :, self._ingame_id]

        # print('strategy', strategy)
        # print('payoffs', game.payoffs(),
        #       adv_action_history, adv_next, self._ingame_id, strategy, numpy.argmax(strategy))
        return numpy.argmax(strategy)


# class SHBPlayer(PHBPlayer):

#     """
#     Superrational History Based Player

#     Tries to predict the opponent move based on its history and
#     then tries to maximise the collective payoff. Using exponential smoothing

#     Starts by being cooperative
#     """

#     def __init__(self, history_length=None, alpha=0.6):
#         """
#         WRITEME
#         """
#         HBPlayer.__init__(self, 0, history_length, 'SHB')
#         self._alpha = alpha

#     def act(self, game, iter):
#         #
#         # apply default policy
#         if iter == 0:
#             return self._default_policy

#         #
#         # get history
#         adv_action_history = self._get_history(game, iter)
#         adv_id = 1 - self._ingame_id
#         adv_next = self._predict_next(adv_action_history)

#         strategy = None
#         if adv_id == 1:
#             strategy = game.payoffs()[:, adv_next, :].sum(axis=1)
#         elif adv_id == 0:
#             strategy = game.payoffs()[adv_next, :, :].sum(axis=1)

#         # print('strategy', strategy)
#         # print('payoffs', game.payoffs(),
#         #       adv_action_history, adv_next, self._ingame_id, strategy, numpy.argmax(strategy
#         #                                                                                 ))
#         return numpy.argmax(strategy)

if __name__ == '__main__':
    #
    # creating two players
    player_1 = WSLSPlayer
    player_2 = RANDPlayer
    player_3 = AllDPlayer
    player_4 = AllCPlayer
    player_5 = TFTPlayer
    player_6 = GRIMPlayer
    player_7 = STFTPlayer
    player_8 = GTFTPlayer
    player_9 = SMPlayer
    player_10 = HMPlayer
    player_11 = ATFTPlayer
    player_12 = PHBPlayer
    # player_13 = SHBPlayer()

    player_list = [player_1,
                   player_2,
                   player_3,
                   player_4,
                   player_5,
                   player_6,
                   player_7,
                   player_8,
                   player_9,
                   player_10,
                   player_11,
                   player_12,
                   # player_13
                   ]

    #
    # creating the game
    R = 3.
    S = 0.
    T = 5.
    P = 1.
    matrix_payoff = numpy.array([[[R, R], [S, T]], [[T, S], [P, P]]])
    print('Payoff Matrix\n', matrix_payoff)

    #
    # simulation
    n_iters = 20
    ipd_game = IPDPairwiseCompetition(player_list, matrix_payoff, n_iters)
    ipd_game.simulate(printing=True)

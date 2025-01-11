'''
Simple pokerbot, written in Python.
'''
from typing import List, Optional

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import math

from skeleton.config import *
import numpy as np

import torch
from agents.model import SimpleNet

NUM_PLAYERS = 2
NUM_RANKS = 13
NUM_SUITS = 4
NUM_ACTIONS = 4
NUM_CARDS = NUM_RANKS * NUM_SUITS
NUM_PAIRS = NUM_CARDS * (NUM_CARDS + 1) // 2
NUM_STREETS = 5

suitNames = ['c', 'd', 'h', 's']
cardNames = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']



def encode_card(card):
    code = [0 for _ in range(NUM_RANKS + NUM_SUITS)]
    code[card.rank] = 1.
    code[card.suit + NUM_RANKS] = 1.
    return code

def encode_rank(rank):
    code = [0 for _ in range(NUM_RANKS + NUM_SUITS)]
    code[rank] = 1.
    return code

def encode_suit(suit):
    code = [0 for _ in range(NUM_RANKS + NUM_SUITS)]
    code[suit + NUM_RANKS] = 1.
    return code

CHIP_LIST = [2.]
while CHIP_LIST[-1] <= STARTING_STACK:
    CHIP_LIST.append(np.ceil(CHIP_LIST[-1] * 1.25))
CHIP_LIST[-1] = STARTING_STACK



def encode_chips(chips, big_blind = 2, max_stack = 400, log_bin_step = 1.5):
    """
    Encodes chip stacks into multiple features:
      1) One-hot vector for log-scale bins
      2) Logarithm of chip count
      3) Ratio to big blind

    :param chips: (int) The number of chips in the stack.
    :param big_blind: (int) Current big blind (for ratio-based feature).
    :param max_stack: (int) Approximate maximum stack you expect in your game.
    :param log_bin_step: (float) Multiplicative step for log-scale bins.

    :return: (np.array) Concatenated feature vector.
    """

    # --------- 1) LOG-SCALE ONE-HOT ENCODING ---------
    # Build log-scale bin boundaries up to max_stack
    # Start from 1 chip to avoid log(0).
    bins = [1]
    while bins[-1] < max_stack:
        bins.append(math.ceil(bins[-1] * log_bin_step))
    # Make sure bins are unique and sorted (in case of rounding collisions).
    bins = sorted(list(set(bins)))

    # bins might look like: [1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 24, ... up to ~max_stack]

    # Figure out where 'chips' would fall among these bins
    # We'll do a simple approach:
    #   bin[i] <= chips < bin[i+1]  => that bin gets a '1', all others get '0'
    # You could also do "soft binning" if you prefer.
    one_hot = [0.0] * len(bins)
    if chips <= 0:
        # If stack is 0 or negative (busted), everything stays 0
        pass
    else:
        # Find the bin index
        # i = largest index where bins[i] <= chips
        # E.g., using binary search with 'bisect'
        from bisect import bisect_right
        idx = bisect_right(bins, chips) - 1
        # Clip at the last bin if chips exceed bins
        idx = min(idx, len(bins) - 1)
        one_hot[idx] = 1.0

    # --------- 2) LOG OF THE STACK ---------
    # log(1 + chips) to avoid log(0)
    log_stack = math.log1p(max(chips, 0))

    # --------- 3) RATIO TO BIG BLIND ---------
    ratio_starting = chips / STARTING_STACK

    # Combine them into one vector
    # You could keep them separate if your network deals with multiple inputs,
    # or you can concatenate them into one long vector.
    feature_vector = np.array(one_hot + [log_stack, ratio_starting], dtype=np.float32)
    return feature_vector



def print_poker_beliefs(belief):
    # Device and tensors initialization
    device = belief.device
    tensor = torch.zeros((13, 13), device=device)
    cnt = torch.zeros((13, 13), device=device)

    # Rank labels for output
    rank_labels = [str(n) for n in range(2, 10)] + list('TJQKA')

    # Create indices for all possible combinations
    suits = torch.arange(4, device=device)
    ranks = torch.arange(13, device=device)

    s0, s1 = torch.meshgrid(suits, suits, indexing='ij')
    r0, r1 = torch.meshgrid(ranks, ranks, indexing='ij')

    # Different suits case (r0 >= r1)
    diff_suits_mask = (s0 != s1).unsqueeze(-1).unsqueeze(-1)
    valid_ranks_diff = (r0.unsqueeze(0).unsqueeze(0) >= r1.unsqueeze(0).unsqueeze(0))

    # Same suits case (r0 < r1)
    same_suits_mask = (s0 == s1).unsqueeze(-1).unsqueeze(-1)
    valid_ranks_same = (r0.unsqueeze(0).unsqueeze(0) < r1.unsqueeze(0).unsqueeze(0))

    # Combine masks
    mask = (diff_suits_mask & valid_ranks_diff) | (same_suits_mask & valid_ranks_same)

    # Sum valid beliefs and counts
    tensor = torch.sum(
        belief.reshape(4, 4, 13, 13) * mask,
        dim=(0, 1)
    )
    cnt = torch.sum(mask, dim=(0, 1))

    # Normalize and convert to numpy
    tensor = (tensor / cnt).detach().cpu().numpy()

    # Print formatted output
    print('      ' + '     '.join(rank_labels))
    for i, row in enumerate(tensor):
        formatted_row = " ".join(f"{value:5.2f}" for value in row)
        print(f"{rank_labels[i]:<3} {formatted_row}")

    return tensor

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''

        self.agent = SimpleNet()
        state_dict = torch.load(f"agents/baseline.pth", weights_only = False)
        self.agent.load_state_dict(state_dict)

        self.num_envs = 1
        self.states: List[Optional[RoundState | TerminalState]] = [None for _ in range(self.num_envs)]
        self.bounties = np.zeros((self.num_envs, NUM_PLAYERS, NUM_RANKS))

        self.belief = torch.zeros((self.num_envs, NUM_PLAYERS, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS))
        self.bounties_masks = np.zeros((self.num_envs, NUM_PLAYERS, NUM_RANKS))
        self.raise_target = [0 for _ in range(self.num_envs)]

        self.safe_mode = False
        self.unsafe_mode = False

    def get_board_obs(self, env_id):
        round_state = self.states[env_id]
        board = round_state.deck[:round_state.street]

        board_tensor = np.zeros((NUM_STREETS, NUM_RANKS + NUM_SUITS))
        for i in range(round_state.street):
            board_tensor[i] = encode_card(board[i])

        return board_tensor

    def get_hand_obs(self, envs_id):
        round_state = self.states[envs_id]
        active = round_state.button % 2

        if round_state.hands[active] == []:
            # opponent's hand
            return -1, -1, -1, -1

        return round_state.hands[active][0].suit, round_state.hands[active][1].suit, round_state.hands[active][0].rank, round_state.hands[active][1].rank

    def get_bounty_obs(self, env_id):
        round_state = self.states[env_id]
        active = round_state.button % 2

        bounty_tensor = np.zeros((NUM_RANKS + NUM_SUITS))
        bounty = self.bounties[env_id, active]
        bounty_tensor[:NUM_RANKS] = bounty / np.sum(np.abs(bounty), axis = -1, keepdims = True)

        bounty_belief_tensor = np.zeros((2, NUM_RANKS + NUM_SUITS))
        bounty_belief = self.bounties_masks[env_id, :, :]
        assert np.all(np.sum(np.abs(bounty_belief), axis = -1, keepdims = True) > 0)
        bounty_belief_tensor[:, :NUM_RANKS] = bounty_belief / np.sum(np.abs(bounty_belief), axis = -1, keepdims = True)

        return np.vstack([bounty_tensor, bounty_belief_tensor])
    # We may want to encode the number of rounds here

    def get_state_info(self, env_id):
        round_state = self.states[env_id]

        active = round_state.button % 2
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1 - active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        my_pip_tensor = np.array(encode_chips(my_pip))
        opp_pip_tensor = np.array(encode_chips(opp_pip))
        my_cb_tensor = np.array(encode_chips(my_contribution))
        opp_cb_tensor = np.array(encode_chips(opp_contribution))
        raise_target_tensor = np.array(encode_chips(self.raise_target[env_id]))

        return np.vstack([my_pip_tensor, opp_pip_tensor, my_cb_tensor, opp_cb_tensor, raise_target_tensor])

    def get_legal_action_mask(self, env_id):
        # actions:
        # fold: 0
        # call: 1
        # check: 2
        # raise 25% more: 3

        # ca: call
        # ra, ca: min_raise
        legal_action_mask = np.zeros(4)
        round_state = self.states[env_id]
        legal_actions = round_state.legal_actions()

        if FoldAction in legal_actions and self.raise_target[env_id] == 0:
            legal_action_mask[0] = 1
        if CallAction in legal_actions and self.raise_target[env_id] == 0:
            legal_action_mask[1] = 1
        if CheckAction in legal_actions:
            legal_action_mask[2] = 1
        if RaiseAction in legal_actions:
            if self.raise_target[env_id] != 0:
                legal_action_mask[2] = 1
            if self.raise_target[env_id] != round_state.raise_bounds()[1]:
                legal_action_mask[3] = 1

        return legal_action_mask

    def get_obs_tensor(self):
        board_obs = np.array([self.get_board_obs(i) for i in range(self.num_envs)])
        state_info = np.array([self.get_state_info(i) for i in range(self.num_envs)])

        hand_obs = np.array([self.get_hand_obs(i) for i in range(self.num_envs)])

        bounty_obs = np.array([self.get_bounty_obs(i) for i in range(self.num_envs)])

        current_players = np.zeros(self.num_envs)
        legal_action_masks = np.zeros((self.num_envs, 4))
        for i in range(self.num_envs):
            round_state = self.states[i]
            active = round_state.button % 2
            current_players[i] = active
            legal_action_masks[i] = self.get_legal_action_mask(i)

        current_players = torch.tensor(current_players, dtype = torch.int)
        board_obs_tensor = torch.tensor(board_obs, dtype = torch.float)
        hand_obs_tensor = torch.tensor(hand_obs, dtype = torch.int)
        bounty_obs_tensor = torch.tensor(bounty_obs, dtype = torch.float)
        state_info_tensor = torch.tensor(state_info, dtype = torch.float)
        legal_action_tensor = torch.tensor(legal_action_masks, dtype = torch.float)
        return (self.belief, state_info_tensor, board_obs_tensor, hand_obs_tensor, bounty_obs_tensor), current_players, legal_action_tensor

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        my_bounty = round_state.bounties[active]  # your current bounty rank

        if my_bankroll > 3.15 * (NUM_ROUNDS - round_num) + 20:
            self.safe_mode = True
        else:
            self.safe_mode = False

        # if my_bankroll < -3 * (NUM_ROUNDS - round_num) - 100:
        #     self.unsafe_mode = True
        # else:
        #     self.unsafe_mode = False

        print(f"# # # # # # # Round {round_num}! # # # # # # # #")

        if round_num % ROUNDS_PER_BOUNTY == 1:
            for i in range(self.num_envs):
                self.bounties_masks[i] = 1.
        else:
            # swap role
            for i in range(self.num_envs):
                self.bounties_masks[i] = self.bounties_masks[i, [1, 0]]

        for i in range(self.num_envs):
            bounty_rank = cardNames.index(my_bounty)
            self.bounties[i][active] = np.array(encode_rank(bounty_rank))[:NUM_RANKS]
            self.bounties[i][1 - active] = self.bounties_masks[i][1 - active]

        print("bounty mask:")
        print(f"My revealed: {self.bounties_masks[0][active]}")
        print(f"Opp revealed: {self.bounties_masks[0][1 - active]}")

        self.belief[:] = 1.

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed

        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        for player_id in range(NUM_PLAYERS):
            if terminal_state.deltas[player_id] >= 0:  # "hit bounty revealed"
                bounty_hits = terminal_state.bounty_hits[player_id]
                if not bounty_hits:
                    # Zero out certain ranks
                    mask1 = np.zeros(NUM_RANKS, dtype = bool)
                    for card in previous_state.deck[:previous_state.street]:
                        mask1[card.rank] = True
                    if FoldAction not in previous_state.legal_actions():
                        # That means we actually showed our cards?
                        mask1[previous_state.hands[player_id][0].rank] = True
                        mask1[previous_state.hands[player_id][1].rank] = True
                    self.bounties_masks[0, player_id, mask1] = 0.
                else:
                    # partial reveal logic
                    if FoldAction not in previous_state.legal_actions():
                        mask1 = np.zeros(NUM_RANKS, dtype = bool)
                        for card in previous_state.deck[:previous_state.street]:
                            mask1[card.rank] = True
                        mask1[previous_state.hands[player_id][0].rank] = True
                        mask1[previous_state.hands[player_id][1].rank] = True
                        # keep only these ranks => zero out others
                        self.bounties_masks[0, player_id, ~mask1] = 0.

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
            print(my_delta >= 0)
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")
            print(my_delta <= 0)


    def traceback(self, round_state, action, traceback_active):
        active = round_state.button % 2
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        if active == traceback_active and action is not None:
            return
        if round_state.previous_state is not None:
            self.traceback(round_state.previous_state, round_state.previous_action, traceback_active)
        if action is not None:
            print(round_state)
            print(f"Action: {str(action)}")

            for i in range(self.num_envs):

                action_sequence = None
                raise_target_sequence = None
                if isinstance(action, FoldAction):
                    action_sequence = [0]
                    raise_target_sequence = [0]
                elif isinstance(action, CallAction):
                    action_sequence = [1]
                    raise_target_sequence = [0]
                elif isinstance(action, CheckAction):
                    action_sequence = [2]
                    raise_target_sequence = [0]
                else:
                    actual_raise = action.amount
                    action_sequence = [3]
                    raise_target_sequence = [0]

                    pre_cb = my_contribution - my_pip
                    rb_min, rb_max = round_state.raise_bounds()  # e.g. [min_raise, max_raise]

                    # For example:
                    self.raise_target[i] = rb_min
                    while action_sequence[-1] != 2:
                        raise_target_sequence.append(self.raise_target[i])

                        raise_25par = min(
                            int(np.ceil(pre_cb * 0.5 + self.raise_target[i] * 1.25)),
                            rb_max
                        )

                        raise_min = self.raise_target[i]
                        raise_max = max(raise_min + 1, raise_25par)  # handle all-in
                        if actual_raise < raise_max:
                            action_sequence.append(2)
                            self.raise_target[i] = 0
                        else:
                            action_sequence.append(3)
                            self.raise_target[i] = raise_max

                self.states[i] = round_state
                for action, raise_target in zip(action_sequence, raise_target_sequence):
                    self.raise_target[i] = raise_target

                    with torch.no_grad():
                        obs, current_players, legal_action_tensor = self.get_obs_tensor()
                        action_tensor = torch.tensor([action])
                        _, _, _, _, action_probs, _ = self.agent.get_logits_value(current_players, obs, legal_action_tensor, action_tensor)
                        self.belief[i, active] *= action_probs[i, :, :, :, :, action]
                        self.belief[i, active] /= self.belief[i, active].max()

                print(f"action {action}, raise target {raise_target}")
                self.raise_target[i] = 0

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        if game_state.round_num < 100:
            print("Tracing back ... ")
            print("".join(["#" for _ in range(25)]))

        if round_state.previous_state is not None:
            self.traceback(round_state.previous_state, round_state.previous_action, active)

        if game_state.round_num < 100:
            print("".join(["#" for _ in range(25)]))
            print("Traceback done!")
            print("")
            print("Time to decide ...")

            print(round_state)
        for i in range(self.num_envs):
            self.states[i] = round_state

        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        agent_action = None
        for i in range(self.num_envs):
            while agent_action is None:
                with torch.no_grad():
                    obs, current_players, legal_action_tensor = self.get_obs_tensor()
                    action, _, _, _, action_probs, _ = self.agent.get_logits_value(current_players, obs, legal_action_tensor)

                    if game_state.round_num < 100:
                        print("My range:")
                        print_poker_beliefs(self.belief[0, active])
                        print("Opponent range:")
                        print_poker_beliefs(self.belief[0, 1 - active])
                        print("Fold:")
                        print_poker_beliefs(action_probs[0, :, :, :, :, 0])
                        print("Check:")
                        print_poker_beliefs(action_probs[0, :, :, :, :, 1])
                        print("Call:")
                        print_poker_beliefs(action_probs[0, :, :, :, :, 2])
                        print("Raise:")
                        print_poker_beliefs(action_probs[0, :, :, :, :, 3])

                    self.belief[i, active] *= action_probs[i, :, :, :, :, action[i]]
                    self.belief[i, active] /= self.belief[i, active].max()


                if action == 0:  # Fold
                    agent_action = FoldAction()
                elif action == 1:  # Call
                    agent_action = CallAction()
                elif self.raise_target[i] == 0:
                    if action == 2:
                        agent_action = CheckAction()
                    else:
                        # We want to set up a raise
                        self.raise_target[i] = round_state.raise_bounds()[0]
                else:
                    # RAISE SEQUENCE
                    pre_cb = my_contribution - my_pip
                    rb_min, rb_max = round_state.raise_bounds()  # e.g. [min_raise, max_raise]

                    # For example:
                    raise_25par = min(
                        int(np.ceil(pre_cb * 0.5 + self.raise_target[i] * 1.25)),
                        rb_max
                    )

                    raise_min = self.raise_target[i]
                    raise_max = max(raise_min + 1, raise_25par)  # handle all-in
                    if action == 2:  # Proceed with raise
                        actual_raise = np.random.randint(raise_min, raise_max)
                        agent_action = RaiseAction(actual_raise)
                        self.raise_target[i] = 0
                    else:
                        # "Stack" the raise
                        self.raise_target[i] = raise_max

        if self.safe_mode:
            agent_action = FoldAction()
        elif self.unsafe_mode:
            if opp_contribution <= 10 and isinstance(agent_action, FoldAction):
                agent_action = RaiseAction(round_state.raise_bounds()[0])

        print(f"final decision {agent_action}")
        return agent_action



if __name__ == '__main__':
    run_bot(Player(), parse_args())

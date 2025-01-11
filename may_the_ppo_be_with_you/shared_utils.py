from typing import List, Optional
import numpy as np
import torch
import math
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import RoundState, TerminalState
from skeleton.config import *

# Common constants
NUM_PLAYERS = 2
NUM_RANKS = 13
NUM_SUITS = 4
NUM_ACTIONS = 4
NUM_CARDS = NUM_RANKS * NUM_SUITS
NUM_PAIRS = NUM_CARDS * (NUM_CARDS + 1) // 2
NUM_STREETS = 5

# Card naming constants
SUIT_NAMES = ['c', 'd', 'h', 's']
CARD_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# Chip scaling for encoding
CHIP_LIST = [2.]
while CHIP_LIST[-1] <= STARTING_STACK:
    CHIP_LIST.append(np.ceil(CHIP_LIST[-1] * 1.25))
CHIP_LIST[-1] = STARTING_STACK

def encode_card(card):
    """Encode a card into a one-hot vector representation."""
    code = [0 for _ in range(NUM_RANKS + NUM_SUITS)]
    code[card.rank] = 1.
    code[card.suit + NUM_RANKS] = 1.
    return code

def encode_rank(rank):
    """Encode a rank into a one-hot vector representation."""
    code = [0 for _ in range(NUM_RANKS + NUM_SUITS)]
    code[rank] = 1.
    return code

def encode_suit(suit):
    """Encode a suit into a one-hot vector representation."""
    code = [0 for _ in range(NUM_RANKS + NUM_SUITS)]
    code[suit + NUM_RANKS] = 1.
    return code

def encode_chips(chips, big_blind=2, max_stack=400, log_bin_step=1.5):
    """
    Encodes chip stacks into multiple features:
      1) One-hot vector for log-scale bins
      2) Logarithm of chip count
      3) Ratio to starting stack
    """
    # Build log-scale bin boundaries
    bins = [1]
    while bins[-1] < max_stack:
        bins.append(math.ceil(bins[-1] * log_bin_step))
    bins = sorted(list(set(bins)))

    # One-hot encoding for bins
    one_hot = [0.0] * len(bins)
    if chips > 0:
        from bisect import bisect_right
        idx = min(bisect_right(bins, chips) - 1, len(bins) - 1)
        one_hot[idx] = 1.0

    # Additional features
    log_stack = math.log1p(max(chips, 0))
    ratio_starting = chips / STARTING_STACK

    return np.array(one_hot + [log_stack, ratio_starting], dtype=np.float32)

class PokerObservationMaker:
    """Helper class for creating poker game observations."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.belief = torch.zeros((num_envs, NUM_PLAYERS, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS))
        self.bounties = np.zeros((num_envs, NUM_PLAYERS, NUM_RANKS))
        self.bounties_masks = np.zeros((num_envs, NUM_PLAYERS, NUM_RANKS))
        self.states: List[Optional[RoundState | TerminalState]] = [None for _ in range(num_envs)]
        self.raise_target = [0 for _ in range(num_envs)]

    def get_board_obs(self, env_id):
        """Get board observation tensor."""
        round_state = self.states[env_id]
        board = round_state.deck[:round_state.street]

        board_tensor = np.zeros((NUM_STREETS, NUM_RANKS + NUM_SUITS))
        for i in range(round_state.street):
            board_tensor[i] = encode_card(board[i])

        return board_tensor

    def get_hand_obs(self, env_id):
        """Get hand observation tensor."""
        round_state = self.states[env_id]
        active = round_state.button % 2

        if hasattr(round_state, 'hands') and round_state.hands[active]:
            return (round_state.hands[active][0].suit,
                    round_state.hands[active][1].suit,
                    round_state.hands[active][0].rank,
                    round_state.hands[active][1].rank)
        return -1, -1, -1, -1

    def get_bounty_obs(self, env_id):
        """Get bounty observation tensor."""
        round_state = self.states[env_id]
        active = round_state.button % 2

        bounty_tensor = np.zeros((NUM_RANKS + NUM_SUITS))
        bounty = self.bounties[env_id, active]
        bounty_tensor[:NUM_RANKS] = bounty / np.sum(np.abs(bounty), axis=-1, keepdims=True)

        bounty_belief_tensor = np.zeros((2, NUM_RANKS + NUM_SUITS))
        bounty_belief = self.bounties_masks[env_id, :, :]
        assert np.all(np.sum(np.abs(bounty_belief), axis=-1, keepdims=True) > 0)
        bounty_belief_tensor[:, :NUM_RANKS] = bounty_belief / np.sum(np.abs(bounty_belief), axis=-1, keepdims=True)

        return np.vstack([bounty_tensor, bounty_belief_tensor])

    def get_state_info(self, env_id):
        """Get state information tensor."""
        round_state = self.states[env_id]
        active = round_state.button % 2

        # Extract state information
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack

        # Encode state information
        tensors = [
            encode_chips(my_pip),
            encode_chips(opp_pip),
            encode_chips(my_contribution),
            encode_chips(opp_contribution),
            encode_chips(self.raise_target[env_id])
        ]

        return np.vstack(tensors)

    def get_legal_action_mask(self, env_id):
        """Get legal action mask."""
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
        """Get complete observation tensor."""
        board_obs = np.array([self.get_board_obs(i) for i in range(self.num_envs)])
        state_info = np.array([self.get_state_info(i) for i in range(self.num_envs)])
        hand_obs = np.array([self.get_hand_obs(i) for i in range(self.num_envs)])
        bounty_obs = np.array([self.get_bounty_obs(i) for i in range(self.num_envs)])

        current_players = np.array([self.states[i].button % 2 for i in range(self.num_envs)])
        legal_action_masks = np.array([self.get_legal_action_mask(i) for i in range(self.num_envs)])

        # Convert to tensors (device handling should be done by caller)
        return (
            self.belief,
            torch.tensor(state_info, dtype=torch.float),
            torch.tensor(board_obs, dtype=torch.float),
            torch.tensor(hand_obs, dtype=torch.int),
            torch.tensor(bounty_obs, dtype=torch.float)
        ), torch.tensor(current_players, dtype=torch.int), torch.tensor(legal_action_masks, dtype=torch.float)

def print_poker_beliefs(belief):
    """Pretty print poker beliefs as a matrix."""
    # Device and tensors initialization
    device = belief.device
    tensor = torch.zeros((13, 13), device=device)
    cnt = torch.zeros((13, 13), device=device)

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

    # Combine masks and calculate values
    mask = (diff_suits_mask & valid_ranks_diff) | (same_suits_mask & valid_ranks_same)
    tensor = torch.sum(belief.reshape(4, 4, 13, 13) * mask, dim=(0, 1))
    cnt = torch.sum(mask, dim=(0, 1))

    # Format and print
    tensor = (tensor / cnt).detach().cpu().numpy()
    print('      ' + '     '.join(CARD_NAMES))
    for i, row in enumerate(tensor):
        formatted_row = " ".join(f"{value:5.2f}" for value in row)
        print(f"{CARD_NAMES[i]:<3} {formatted_row}")

    return tensor
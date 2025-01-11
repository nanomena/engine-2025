from skeleton.states import *
from skeleton.config import *
from typing import List, Optional
import numpy as np

import torch
import eval7
import time

NUM_PLAYERS = 2
NUM_RANKS = 13
NUM_SUITS = 4
NUM_ACTIONS = 4
NUM_CARDS = NUM_RANKS * NUM_SUITS
NUM_PAIRS = NUM_CARDS * (NUM_CARDS + 1) // 2
NUM_STREETS = 5


def encode_card(card: eval7.Card):
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

def encode_pair(card1: eval7.Card, card2: eval7.Card):
    a = encode_card(card1)
    b = encode_card(card2)
    if a > b:
        a, b = b, a
    return b * (b - 1) // 2 + a

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


# print(encode_chips(1))
# print(encode_chips(2))
# print(encode_chips(10))
# print(encode_chips(60))

cardNames = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
class PokerVectorEnv:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

        self.belief = torch.zeros((self.num_envs, NUM_PLAYERS, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS)).to(self.device)
        self.bounties = np.zeros((self.num_envs, NUM_PLAYERS, NUM_RANKS))
        self.bounties_idx = [[-1, -1] for _ in range(self.num_envs)]
        self.bounties_masks = np.zeros((self.num_envs, NUM_PLAYERS, NUM_RANKS))
        self.states: List[Optional[RoundState | TerminalState]] = [None for _ in range(self.num_envs)]
        self.round_num = [0 for _ in range(self.num_envs)]
        self.raise_target = [0 for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            self.assign_new_state(i)

        self.init_cnt = 0

        self.obs_belief_shape = self.belief.shape[1:]
        self.obs_state_info_shape = torch.Size(self.get_state_info(0).shape)
        self.obs_board_shape = torch.Size(self.get_board_obs(0).shape)
        self.obs_bounty_shape = torch.Size(self.get_bounty_obs(0).shape)

        self.obs_encoded_size = math.prod(self.obs_belief_shape) + math.prod(self.obs_state_info_shape) + math.prod(self.obs_board_shape) + math.prod(self.obs_bounty_shape) + 4

        self.num_players = NUM_PLAYERS
        self.num_actions = NUM_ACTIONS

    def assign_new_state(self, env_id: int):
        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        self.round_num[env_id] += 1
        if self.round_num[env_id] % ROUNDS_PER_BOUNTY == 1:
            self.bounties_idx[env_id] = [np.random.randint(NUM_RANKS), np.random.randint(NUM_RANKS)]
            self.bounties_masks[env_id] = 1.
            resets = True
        else:
            # swap role
            self.bounties_idx[env_id] = self.bounties_idx[env_id][::-1]
            self.bounties_masks[env_id] = self.bounties_masks[env_id, [1, 0]]
            resets = False

        self.bounties[env_id] = 0.
        for player_id in range(2):
            self.bounties[env_id, player_id, self.bounties_idx[env_id][player_id]] = 1.
        self.belief[env_id] = 1.
        self.states[env_id] = RoundState(0, 0, pips, stacks, hands, self.bounties_idx[env_id], deck.deal(5), None)

        self.raise_target[env_id] = 0
        return resets

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

        current_players = torch.tensor(current_players, dtype = torch.int, device = self.device)
        board_obs_tensor = torch.tensor(board_obs, dtype = torch.float, device = self.device)
        hand_obs_tensor = torch.tensor(hand_obs, dtype = torch.int, device = self.device)
        bounty_obs_tensor = torch.tensor(bounty_obs, dtype = torch.float, device = self.device)
        state_info_tensor = torch.tensor(state_info, dtype = torch.float, device = self.device)
        legal_action_tensor = torch.tensor(legal_action_masks, dtype = torch.float, device = self.device)
        return (self.belief, state_info_tensor, board_obs_tensor, hand_obs_tensor, bounty_obs_tensor), current_players, legal_action_tensor

    def reset(self):
        self.init_cnt += 1
        if self.init_cnt > 1:
            raise RuntimeError("Environment does not support reinitialization")

        return self.get_obs_tensor()

    # Chatgpt really helps!
    def step(self, action_probs: torch.Tensor, actions_t: torch.Tensor):
        """
        Execute one vectorized step in the environment for all envs at once.

        1) Auto-reset envs that are terminal from the previous step.
        2) Collect per-env data (seat, bounty rank, hole cards, etc.).
        3) Vectorized sampling from action_probs.
        4) Vectorized belief updates.
        5) Loop over envs to apply proceed(...) transitions.
           - If terminal => record reward + bounty logic + immediate reset
        6) Return new observations, reward, done flags, etc.
        """
        if self.init_cnt == 0:
            raise RuntimeError("Environment has not been initialized")

        device = self.device
        num_envs = self.num_envs

        # ----------------------------------------------------------------------
        # 1) AUTO-RESET ANY ENV THAT'S TERMINAL FROM PRIOR STEP
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # 2) GATHER ENV-SPECIFIC DATA
        # ----------------------------------------------------------------------
        # We'll collect all we need for action sampling here, including:
        #  - Active seat (0 or 1)
        #  - Bounty rank
        #  - Hole cards (encoded)
        #  - We COULD also gather pips/stacks, but those are typically used
        #    only inside the environment logic for raise calculations.
        # ----------------------------------------------------------------------
        active_seats = []

        for i in range(num_envs):
            round_state = self.states[i]
            # Which seat is acting?
            active = round_state.button % 2
            active_seats.append(active)

        # Convert lists to tensors
        env_idx = torch.arange(num_envs, device = device)
        active_seats_t = torch.tensor(active_seats, device = device, dtype = torch.long)

        # ----------------------------------------------------------------------
        # 4) VECTORIZED BELIEF UPDATE
        # ----------------------------------------------------------------------
        # For each env i, seat=active_seats[i], chosen action=actions_t[i].
        #   self.belief[i, seat] *= action_probs[i, :, :, :, :, action]
        #   self.belief[i, seat] /= max(...)
        # ----------------------------------------------------------------------
        with torch.no_grad():
            gather_action_probs = action_probs[env_idx, :, :, :, :, actions_t]  # [num_envs, ...]

            belief_2d = self.belief.view(num_envs * NUM_PLAYERS, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS)
            offset = env_idx * NUM_PLAYERS + active_seats_t
            seat_belief = belief_2d[offset]  # shape [num_envs, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS]

            seat_belief *= gather_action_probs
            max_per_env = seat_belief.view(num_envs, -1).max(dim = 1).values
            max_per_env = torch.clamp(max_per_env, min = 1e-12)
            seat_belief /= max_per_env.view(-1, 1, 1, 1, 1)

            belief_2d[offset] = seat_belief
            self.belief = belief_2d.view(num_envs, NUM_PLAYERS, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS)

        # ----------------------------------------------------------------------
        # 5) APPLY ENVIRONMENT TRANSITIONS IN A LOOP
        # ----------------------------------------------------------------------
        # We do a small loop because each env has unique RoundState transitions.
        # We'll store the reward + done in CPU arrays, then convert to torch.
        # ----------------------------------------------------------------------
        rewards_np = np.zeros((num_envs, NUM_PLAYERS), dtype = np.float32)
        dones_np = np.zeros((num_envs,), dtype = np.float32)
        resets_np = np.zeros((num_envs,), dtype = np.float32)

        for i in range(num_envs):
            round_state = self.states[i]
            # Potentially it's already terminal (edge case)
            if isinstance(round_state, TerminalState):
                assert 0

            # Gather your usual environment info:
            active = active_seats[i]
            action = actions_t[i].item()
            # print(action)

            my_pip = round_state.pips[active]
            opp_pip = round_state.pips[1 - active]
            my_stack = round_state.stacks[active]
            opp_stack = round_state.stacks[1 - active]
            continue_cost = opp_pip - my_pip
            my_contribution = STARTING_STACK - my_stack
            opp_contribution = STARTING_STACK - opp_stack

            # Now do the same logic from your original code:
            if action == 0:  # Fold
                new_state = round_state.proceed(FoldAction())
            elif action == 1:  # Call
                new_state = round_state.proceed(CallAction())
            elif self.raise_target[i] == 0:
                if action == 2:
                    # Check
                    new_state = round_state.proceed(CheckAction())
                else:
                    # We want to set up a raise
                    self.raise_target[i] = round_state.raise_bounds()[0]
                    new_state = round_state
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
                    new_state = round_state.proceed(RaiseAction(actual_raise))
                    self.raise_target[i] = 0
                else:
                    # "Stack" the raise
                    self.raise_target[i] = raise_max
                    new_state = round_state

            # Update state
            self.states[i] = new_state

            # Check if environment is now terminal
            if isinstance(new_state, TerminalState):
                dones_np[i] = 1.0
                terminal_state = new_state
                previous_state = terminal_state.previous_state

                # Reward
                rewards_np[i] = np.array(terminal_state.deltas, dtype = np.float32) / STARTING_STACK

                # Bounty logic
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
                            self.bounties_masks[i, player_id, mask1] = 0.
                        else:
                            # partial reveal logic
                            if FoldAction not in previous_state.legal_actions():
                                mask1 = np.zeros(NUM_RANKS, dtype = bool)
                                for card in previous_state.deck[:previous_state.street]:
                                    mask1[card.rank] = True
                                mask1[previous_state.hands[player_id][0].rank] = True
                                mask1[previous_state.hands[player_id][1].rank] = True
                                # keep only these ranks => zero out others
                                self.bounties_masks[i, player_id, ~mask1] = 0.

                # Immediately reset so that next step sees a fresh env
                self.assign_new_state(i)

        # ----------------------------------------------------------------------
        # 6) BUILD AND RETURN OBSERVATIONS, REWARDS, DONES
        # ----------------------------------------------------------------------
        obs, current_players, legal_action_tensor = self.get_obs_tensor()
        rewards = torch.tensor(rewards_np, dtype = torch.float, device = device)
        dones = torch.tensor(dones_np, dtype = torch.float, device = device)
        resets = torch.tensor(resets_np, dtype = torch.float, device = device)

        return obs, rewards, dones, resets, current_players, legal_action_tensor

    def encode_obs(self, obs):
        belief, state_info, board, hand, bounty = obs

        flat_belief = belief.flatten(start_dim = 1)
        flat_state_info = state_info.flatten(start_dim = 1)
        flat_board = board.flatten(start_dim = 1)
        flat_hand = hand.flatten(start_dim = 1)
        flat_bounty = bounty.flatten(start_dim = 1)

        # Concatenate all flattened tensors along a new dimension (dim=0)
        combined_tensor = torch.cat((flat_belief, flat_state_info, flat_board, flat_hand, flat_bounty), dim = 1)

        return combined_tensor

    def decode_obs(self, combined_tensor):
        parts = torch.split(combined_tensor, [math.prod(self.obs_belief_shape), math.prod(self.obs_state_info_shape), math.prod(self.obs_board_shape), 4, math.prod(self.obs_bounty_shape)], dim = 1)

        belief_decoded = parts[0].reshape([-1,] + list(self.obs_belief_shape))
        state_info_decoded = parts[1].reshape([-1,] + list(self.obs_state_info_shape))
        board_decoded = parts[2].reshape([-1,] + list(self.obs_board_shape))
        hand_decoded = parts[3].reshape([-1, 4]).int()
        bounty_decoded = parts[4].reshape([-1,] + list(self.obs_bounty_shape))

        return belief_decoded, state_info_decoded, board_decoded, hand_decoded, bounty_decoded

def print_poker_beliefs(belief):
    tensor = torch.zeros((13, 13), device = belief.device)
    cnt = torch.zeros((13, 13), device = belief.device)
    for s0 in range(4):
        for s1 in range(4):
            for r0 in range(13):
                for r1 in range(13):
                    if s0 != s1:
                        if r0 < r1:
                            continue
                        tensor[r0, r1] += belief[s0, s1, r0, r1]
                        cnt[r0, r1] += 1
                    else:
                        if r0 >= r1:
                            continue
                        tensor[r0, r1] += belief[s0, s1, r0, r1]
                        cnt[r0, r1] += 1

    tensor /= cnt
    tensor = tensor.detach().cpu().numpy()
    for i, row in enumerate(tensor):
        formatted_row = " ".join(f"{value:5.2f}" for value in row)
        print(formatted_row)

if __name__ == '__main__':
    from network import SimpleNet

    envs = PokerVectorEnv(100, "cuda" if torch.cuda.is_available() else "cpu")
    obs, current_players, legal_action_masks = envs.reset()

    print(envs.obs_belief_shape)
    print(envs.obs_state_info_shape)
    print(envs.obs_board_shape)
    print(envs.obs_bounty_shape)
    print(current_players.shape, legal_action_masks.shape)

    agent = SimpleNet()


    state_dict = torch.load(f"models/baseline.pth", weights_only = True)
    agent.load_state_dict(state_dict)

    agent = agent.to(envs.device)

    for i in range(10000):
        start_time = time.time()

        belief, state_info, board, hand, bounty = obs

        encoded_obs = envs.encode_obs(obs)
        _, _, recovered_board, _, _ = envs.decode_obs(encoded_obs)

        # print(obs)
        # print(envs.bounties[0])
        # print(envs.bounties_masks[0])

        assert(torch.all(board == recovered_board))

        # print_poker_beliefs(belief[0, 0])
        # print_poker_beliefs(belief[0, 1])

        actions, log_prob, entropies, values, action_probs, residual = agent.get_logits_value(current_players, obs, legal_action_masks)

        # print(entropies)
        # print(legal_action_masks)
        # print(action_logits)

        #
        # print(card_obs_tensor.shape)
        # print(state_infos_tensor.shape)


        # action_probs = torch.rand(envs.num_envs, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS, NUM_ACTIONS, device = envs.device)
        # action_probs *= legal_action_masks.view(envs.num_envs, 1, 1, 1, 1, NUM_ACTIONS)
        # action_probs /= action_probs.sum(dim = -1, keepdim = True)
        # action_probs = 0.5 * (action_probs + action_probs.permute(0, 2, 1, 4, 3, 5))

        obs, rewards, dones, resets, current_players, legal_action_masks = envs.step(action_probs, actions)
        #
        # print("Fold:")
        # print_poker_beliefs(action_probs[0, :, :, :, :, 0])
        # print("Check:")
        # print_poker_beliefs(action_probs[0, :, :, :, :, 1])
        # print("Call:")
        # print_poker_beliefs(action_probs[0, :, :, :, :, 2])
        # print("Raise:")
        # print_poker_beliefs(action_probs[0, :, :, :, :, 3])
        #
        # print(f"Action ! {actions}")
        # print(rewards)


        end_time = time.time()
        duration = end_time - start_time

        # if rewards[0, 0] != 0:
        #     print(rewards[0, 0])

        print(f"Iteration {i + 1}: {duration:.2f} seconds")
        # print(f"Rewards, max {rewards[:, 0].max()}, deviation {rewards[:, 0].std()}")

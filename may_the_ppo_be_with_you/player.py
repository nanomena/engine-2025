'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random

from config import *
import numpy as np

import torch
from models.network import SimpleNet

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
    code[cardNames.index(card[0])] = 1.
    code[suitNames.index(card[1]) + NUM_RANKS] = 1.
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
        state_dict = torch.load(f"models/model.pth", weights_only = False)
        self.agent.load_state_dict(state_dict)

        self.belief = torch.zeros((1, NUM_PLAYERS, NUM_SUITS, NUM_SUITS, NUM_RANKS, NUM_RANKS))
        self.bounties_masks = torch.zeros((1, NUM_PLAYERS, NUM_RANKS))

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

        if round_num % ROUNDS_PER_BOUNTY == 1:
            self.bounties_masks[0] = 0.
        else:
            # swap role
            self.bounties_masks[0] = self.bounties_masks[0, [1, 0]]

        self.belief[0] = 1.

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

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
            print(my_delta >= 0)
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")
            print(my_delta <= 0)

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

        def traceback(round_state, action):
            if round_state.previous_state != None and (round_state.button % 2 != active or action is None):
                traceback(round_state.previous_state, round_state.previous_action)
                if action is not None:
                    print(round_state)
                    print(f"Action: {str(action)}")

        print("Tracing back ... ")
        print("".join(["#" for _ in range(25)]))
        if round_state.previous_state is not None:
            traceback(round_state.previous_state, round_state.previous_action)
        print("".join(["#" for _ in range(25)]))
        print("Traceback done!")
        print("")
        print("Time to decide ...")
        print(round_state)
        
        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        if RaiseAction in legal_actions:
            if random.random() < 0.5:
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())

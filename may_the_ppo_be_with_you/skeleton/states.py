'''
Encapsulates game and round state information for the player.
'''
from collections import namedtuple
from .actions import FoldAction, CallAction, CheckAction, RaiseAction
from .config import *
import math
import eval7
from dataclasses import dataclass, field

GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])
TerminalState = namedtuple('TerminalState', ['deltas', 'bounty_hits', 'previous_state'])

@dataclass
class _RoundState:
    '''Base class for round state using dataclass for better representation'''
    button: int
    street: int
    pips: list
    stacks: list
    hands: list
    bounties: list
    deck: list
    previous_state: 'RoundState'  # Type hinting for a recursive data structure
    previous_action: any = field(default = None)

    def __str__(self):
        '''Enhanced string representation for debugging'''
        hands_str = ', '.join(f"Player {idx + 1}: {''.join([str(card) for card in hand]) if hand else 'Empty'}"
                              for idx, hand in enumerate(self.hands))
        pips_str = ', '.join(f"Player {idx + 1}: {pip}" for idx, pip in enumerate(self.pips))
        stacks_str = ', '.join(f"Player {idx + 1}: {stack}" for idx, stack in enumerate(self.stacks))

        return (f"RoundState(Button: {self.button}, Street: {self.street},\n"
                f"  Pips: [{pips_str}],\n"
                f"  Stacks: [{stacks_str}],\n"
                f"  Hands: [{hands_str}],\n"
                f"  Bounties: {self.bounties},\n"
                f"  Deck: {[str(card) for card in self.deck] if self.deck else 'Empty'},\n"
                f"  Previous Action: {type(self.previous_action).__name__ if self.previous_action else 'None'}\n)")

class RoundState(_RoundState):
    '''
    Encodes the game tree for one round of poker.
    '''
    def get_bounty_hits(self):
        '''
        Determines if each player hit their bounty card during the round.

        Returns:
            tuple[bool, bool]: A tuple containing two booleans where:
                - First boolean indicates if Player 1's bounty was hit
                - Second boolean indicates if Player 2's bounty was hit
        '''
        community_cards = [] if self.street == 0 else self.deck[:self.street]
        cards0 = self.hands[0] + community_cards
        cards1 = self.hands[1] + community_cards

        # Convert eval7 cards to ranks for bounty comparison
        card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        ranks0 = [card_ranks[card.rank] for card in cards0]
        ranks1 = [card_ranks[card.rank] for card in cards1]

        return (self.bounties[0] in ranks0, self.bounties[1] in ranks1)

    def showdown(self):
        '''
        Compares the players' hands and computes payoffs at showdown.
        '''
        if self.hands[0] == [] or self.hands[1] == []:
            return TerminalState([0, 0], None, self)

        community_cards = self.deck[:5]
        score0 = eval7.evaluate(community_cards + self.hands[0])
        score1 = eval7.evaluate(community_cards + self.hands[1])

        assert self.stacks[0] == self.stacks[1]
        if score0 > score1:
            delta = STARTING_STACK - self.stacks[1]
        elif score0 < score1:
            delta = self.stacks[0] - STARTING_STACK
        else:
            delta = 0  # Split pot

        bounty_hit_0, bounty_hit_1 = self.get_bounty_hits()

        return TerminalState([delta, -delta], (bounty_hit_0, bounty_hit_1), self)

    def legal_actions(self):
        '''
        Returns a set which corresponds to the active player's legal moves.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]

        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}

        # continue_cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns a tuple of the minimum and maximum legal raises.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1-active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting.
        '''
        if self.street == 5:
            return self.showdown()
        new_street = 3 if self.street == 0 else self.street + 1
        return RoundState(1, new_street, [0, 0], self.stacks, self.hands, self.bounties,
                          self.deck, self, None)

    def proceed(self, action):
        '''
        Advances the game tree by one action performed by the active player.
        '''
        active = self.button % 2

        if isinstance(action, FoldAction):
            delta = self.stacks[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.stacks[1]
            return TerminalState([delta, -delta], self.get_bounty_hits(), self)

        if isinstance(action, CallAction):
            if self.button == 0:  # sb calls bb
                return RoundState(1, 0, [BIG_BLIND] * 2,
                                  [STARTING_STACK - BIG_BLIND] * 2,
                                  self.hands, self.bounties, self.deck, self, action)

            # both players acted
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution

            state = RoundState(self.button + 1, self.street, new_pips, new_stacks,
                               self.hands, self.bounties, self.deck, self, action)
            return state.proceed_street()

        if isinstance(action, CheckAction):
            if (self.street == 0 and self.button > 0) or self.button > 1:  # both players acted
                state = RoundState(self.button + 1, self.street, self.pips,
                                   self.stacks, self.hands, self.bounties, self.deck, self, action)
                return state.proceed_street()

            # let opponent act
            return RoundState(self.button + 1, self.street, self.pips, self.stacks,
                              self.hands, self.bounties, self.deck, self, action)

        # isinstance(action, RaiseAction)
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution

        return RoundState(self.button + 1, self.street, new_pips, new_stacks,
                          self.hands, self.bounties, self.deck, self, action)
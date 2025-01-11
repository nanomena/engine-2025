from shared_utils import *
from agents.model import SimpleNet
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        """Initialize the poker bot."""
        self.agent = SimpleNet()
        state_dict = torch.load(f"agents/baseline.pth", weights_only=False)
        self.agent.load_state_dict(state_dict)

        # Initialize observation maker with single environment
        self.num_envs = 1
        self.obs_maker = PokerObservationMaker(self.num_envs)

        # Game state tracking
        self.safe_mode = False
        self.unsafe_mode = False

    def handle_new_round(self, game_state, round_state, active):
        """Handle start of new round."""
        # Update game mode based on bankroll
        remaining_rounds = NUM_ROUNDS - game_state.round_num
        self.safe_mode = game_state.bankroll > 3.15 * remaining_rounds + 20

        print(f"# # # # # # # Round {game_state.round_num}! # # # # # # # #")

        # Update bounty masks
        if game_state.round_num % ROUNDS_PER_BOUNTY == 1:
            self.obs_maker.bounties_masks[0] = 1.
        else:
            self.obs_maker.bounties_masks[0] = self.obs_maker.bounties_masks[0, [1, 0]]

        # Update bounties
        bounty_rank = CARD_NAMES.index(round_state.bounties[active])
        self.obs_maker.bounties[0][active] = np.array(encode_rank(bounty_rank))[:NUM_RANKS]
        self.obs_maker.bounties[0][1 - active] = self.obs_maker.bounties_masks[0][1 - active]

        print("bounty mask:")
        print(f"My revealed: {self.obs_maker.bounties_masks[0][active]}")
        print(f"Opp revealed: {self.obs_maker.bounties_masks[0][1 - active]}")

        self.obs_maker.belief[:] = 1.

    def handle_round_over(self, game_state, terminal_state, active):
        """Handle end of round."""
        previous_state = terminal_state.previous_state

        # Process bounty reveals
        for player_id in range(NUM_PLAYERS):
            if terminal_state.deltas[player_id] >= 0:
                if not terminal_state.bounty_hits[player_id]:
                    # Zero out ranks for non-bounty hits
                    mask1 = np.zeros(NUM_RANKS, dtype=bool)
                    for card in previous_state.deck[:previous_state.street]:
                        mask1[card.rank] = True
                    if FoldAction not in previous_state.legal_actions():
                        mask1[previous_state.hands[player_id][0].rank] = True
                        mask1[previous_state.hands[player_id][1].rank] = True
                    self.obs_maker.bounties_masks[0, player_id, mask1] = 0.
                else:
                    # Handle partial reveal logic
                    if FoldAction not in previous_state.legal_actions():
                        mask1 = np.zeros(NUM_RANKS, dtype=bool)
                        for card in previous_state.deck[:previous_state.street]:
                            mask1[card.rank] = True
                        mask1[previous_state.hands[player_id][0].rank] = True
                        mask1[previous_state.hands[player_id][1].rank] = True
                        self.obs_maker.bounties_masks[0, player_id, ~mask1] = 0.

        # Print bounty hit information
        if terminal_state.bounty_hits[active]:
            print(f"I hit my bounty of {previous_state.bounties[active]}!")
            print(terminal_state.deltas[active] >= 0)
        if terminal_state.bounty_hits[1-active]:
            print(f"Opponent hit their bounty of {previous_state.bounties[1-active]}!")
            print(terminal_state.deltas[active] <= 0)

    def traceback(self, round_state, action, traceback_active):
        """Reconstruct belief state from game history."""
        active = round_state.button % 2

        if active == traceback_active and action is not None:
            return

        if round_state.previous_state is not None:
            self.traceback(round_state.previous_state, round_state.previous_action, traceback_active)

        if action is not None:
            print(round_state)
            print(f"Action: {str(action)}")

            self.process_history_action(round_state, action, active)

    def process_history_action(self, round_state, action, active):
        """Process a historical action to update beliefs."""
        action_sequence = []
        raise_target_sequence = []

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
            action_sequence, raise_target_sequence = self.process_history_raise(round_state, action, active)

        # Update beliefs based on action sequence
        self.obs_maker.states[0] = round_state
        for act, raise_target in zip(action_sequence, raise_target_sequence):
            self.obs_maker.raise_target[0] = raise_target

            with torch.no_grad():
                obs, current_players, legal_action_tensor = self.obs_maker.get_obs_tensor()
                action_tensor = torch.tensor([act])
                _, _, _, _, action_probs, _ = self.agent.get_logits_value(current_players, obs, legal_action_tensor, action_tensor)
                self.obs_maker.belief[0, active] *= action_probs[0, :, :, :, :, act]
                self.obs_maker.belief[0, active] /= self.obs_maker.belief[0, active].max()

        print(f"action {act}, raise target {raise_target}")
        self.obs_maker.raise_target[0] = 0

    def process_history_raise(self, round_state, action, active):
        """Process a historical raise action."""
        action_sequence = [3]
        raise_target_sequence = [0]
        actual_raise = action.amount

        my_stack = round_state.stacks[active]
        my_pip = round_state.pips[active]
        my_contribution = STARTING_STACK - my_stack
        pre_cb = my_contribution - my_pip
        rb_min, rb_max = round_state.raise_bounds()

        self.obs_maker.raise_target[0] = rb_min
        while action_sequence[-1] != 2:
            raise_target_sequence.append(self.obs_maker.raise_target[0])

            raise_25par = min(
                int(np.ceil(pre_cb * 0.5 + self.obs_maker.raise_target[0] * 1.25)),
                rb_max
            )

            raise_min = self.obs_maker.raise_target[0]
            raise_max = max(raise_min + 1, raise_25par)

            if actual_raise < raise_max:
                action_sequence.append(2)
                self.obs_maker.raise_target[0] = 0
            else:
                action_sequence.append(3)
                self.obs_maker.raise_target[0] = raise_max

        return action_sequence, raise_target_sequence

    def get_action(self, game_state, round_state, active):
        """Determine the next action to take."""
        # Print debug information for early rounds
        if game_state.round_num < 100:
            print("Tracing back ... ")
            print("".join(["#" for _ in range(25)]))

        # Reconstruct history
        if round_state.previous_state is not None:
            self.traceback(round_state.previous_state, round_state.previous_action, active)

        if game_state.round_num < 100:
            print("".join(["#" for _ in range(25)]))
            print("Traceback done!")
            print("")
            print("Time to decide ...")
            print(round_state)

        # Update current state
        self.obs_maker.states[0] = round_state

        # Get action from agent
        agent_action = None
        while agent_action is None:
            agent_action = self.get_agent_action(active)

        # Apply safety modes
        if self.safe_mode:
            agent_action = FoldAction()
        elif self.unsafe_mode:
            opp_contribution = STARTING_STACK - round_state.stacks[1-active]
            if opp_contribution <= 10 and isinstance(agent_action, FoldAction):
                agent_action = RaiseAction(round_state.raise_bounds()[0])

        print(f"final decision {agent_action}")
        return agent_action

    def get_agent_action(self, active):
        """Get action from the neural network agent."""
        with torch.no_grad():
            obs, current_players, legal_action_tensor = self.obs_maker.get_obs_tensor()
            action, _, _, _, action_probs, _ = self.agent.get_logits_value(current_players, obs, legal_action_tensor)

            # Debug output beliefs and action probabilities
            if self.obs_maker.states[0].previous_state is None or self.obs_maker.states[0].street == 0:
                print("My range:")
                print_poker_beliefs(self.obs_maker.belief[0, active])
                print("Opponent range:")
                print_poker_beliefs(self.obs_maker.belief[0, 1 - active])
                print("Fold:")
                print_poker_beliefs(action_probs[0, :, :, :, :, 0])
                print("Check:")
                print_poker_beliefs(action_probs[0, :, :, :, :, 1])
                print("Call:")
                print_poker_beliefs(action_probs[0, :, :, :, :, 2])
                print("Raise:")
                print_poker_beliefs(action_probs[0, :, :, :, :, 3])

            # Update beliefs based on chosen action
            self.obs_maker.belief[0, active] *= action_probs[0, :, :, :, :, action[0]]
            self.obs_maker.belief[0, active] /= self.obs_maker.belief[0, active].max()

            return self.convert_action_to_move(action[0])

    def convert_action_to_move(self, action):
        """Convert neural network action to poker move."""
        if action == 0:  # Fold
            return FoldAction()
        elif action == 1:  # Call
            return CallAction()
        elif self.obs_maker.raise_target[0] == 0:
            if action == 2:  # Check
                return CheckAction()
            else:  # Initialize raise
                self.obs_maker.raise_target[0] = self.obs_maker.states[0].raise_bounds()[0]
                return None
        else:
            return self.handle_raise_sequence(action)

    def handle_raise_sequence(self, action):
        """Handle raise action sequence."""
        round_state = self.obs_maker.states[0]
        active = round_state.button % 2
        my_stack = round_state.stacks[active]
        my_pip = round_state.pips[active]

        my_contribution = STARTING_STACK - my_stack
        pre_cb = my_contribution - my_pip
        rb_min, rb_max = round_state.raise_bounds()

        raise_25par = min(
            int(np.ceil(pre_cb * 0.5 + self.obs_maker.raise_target[0] * 1.25)),
            rb_max
        )

        raise_min = self.obs_maker.raise_target[0]
        raise_max = max(raise_min + 1, raise_25par)

        if action == 2:  # Execute raise
            actual_raise = np.random.randint(raise_min, raise_max)
            self.obs_maker.raise_target[0] = 0
            return RaiseAction(actual_raise)
        else:  # Stack raise
            self.obs_maker.raise_target[0] = raise_max
            return None


if __name__ == '__main__':
    run_bot(Player(), parse_args())
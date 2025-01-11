import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

torch.set_num_threads(1)

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    """Simple orthogonal initialization helper for layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# Claude!
class SimpleNet(nn.Module):
    def __init__(
            self,
            num_players: int = 2,
            card_obs_channels: int = 8,
            card_feature_dim: int = 17,  # 13 ranks + 4 suits
            chip_encodings: int = 5,
            embedding_dim: int = 24,
            fuse_hidden_dim: int = 128,
            num_features: int = 10,
            num_suits: int = 4,
            num_ranks: int = 13,
            num_actions: int = 4
    ):
        super().__init__()
        self.num_players = num_players
        self.num_suits = num_suits
        self.num_ranks = num_ranks
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.num_features = num_features

        # Initialize embeddings directly with orthogonal initialization
        suit_embeddings = torch.empty(num_suits, embedding_dim)
        rank_embeddings = torch.empty(num_ranks, embedding_dim)
        nn.init.orthogonal_(suit_embeddings, 1.0)
        nn.init.orthogonal_(rank_embeddings, 1.0)

        self.suit_embeddings = nn.Parameter(suit_embeddings)
        self.rank_embeddings = nn.Parameter(rank_embeddings)

        # Process card observations (board + bounty info)
        # Split the input into rank and suit parts
        self.rank_encoder = layer_init(
            nn.Linear(num_ranks, embedding_dim),
            std = 1.0
        )
        self.suit_encoder = layer_init(
            nn.Linear(num_suits, embedding_dim),
            std = 1.0
        )

        # Process state info tensor (chip-related)
        self.chip_encoder = nn.Sequential(
            layer_init(nn.Linear(chip_encodings * 17, fuse_hidden_dim)),
            nn.ReLU()
        )

        # Process belief features
        self.belief_feature_encoder = nn.Sequential(
            layer_init(nn.Linear(embedding_dim * 2, fuse_hidden_dim)),
            nn.ReLU()
        )

        # Fusion layers
        fuse_input_dim = embedding_dim * card_obs_channels + fuse_hidden_dim * 2
        self.fuse_layers = nn.Sequential(
            layer_init(nn.Linear(fuse_input_dim, fuse_hidden_dim)),
            nn.LayerNorm(fuse_hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(fuse_hidden_dim, fuse_hidden_dim)),
            nn.LayerNorm(fuse_hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(fuse_hidden_dim, num_features * embedding_dim)),
        )

        # Actor heads (one per player) - outputs embedding_dim for each action
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        for _ in range(num_players):
            actor_head = nn.Sequential(
                layer_init(nn.Linear(num_features, num_features)),
                nn.ReLU(),
                layer_init(nn.Linear(num_features, num_actions), std = 0.01)
            )
            self.actor_heads.append(actor_head)

            critic_head = nn.Sequential(
                layer_init(nn.Linear(num_features, num_features)),
                nn.ReLU(),
                layer_init(nn.Linear(num_features, 1), std = 0.01)
            )
            self.critic_heads.append(critic_head)

    def get_card_embedding(self, suit_idx, rank_idx):
        """Get card embedding as sum of suit and rank embeddings."""
        return self.suit_embeddings[suit_idx] + self.rank_embeddings[rank_idx]

    def process_card_obs(self, card_obs):
        """Process a single card observation by splitting into rank and suit parts."""
        B = card_obs.shape[0]

        # Split input into rank and suit parts
        rank_part = card_obs[:, :13]  # First 13 elements are rank one-hot
        suit_part = card_obs[:, 13:]  # Last 4 elements are suit one-hot

        # Get embeddings
        rank_emb = self.rank_encoder(rank_part)
        suit_emb = self.suit_encoder(suit_part)

        # Sum the embeddings
        return rank_emb + suit_emb

    def compute_pair_embeddings(self, device):
        """
        Returns a [num_suits, num_suits, num_ranks, num_ranks, embedding_dim] tensor
        built via broadcasting (rather than nested loops).
        """
        # shape: [num_suits, embedding_dim] + [num_ranks, embedding_dim]
        # We want [num_suits, num_ranks, embedding_dim] for each single card.
        card_embeddings = (
                self.suit_embeddings.unsqueeze(1)          # [4, 1, embedding_dim]
                + self.rank_embeddings.unsqueeze(0)        # [1, 13, embedding_dim]
        )  # Now shape = [4, 13, embedding_dim]

        # Next, we create pair embeddings by broadcasting:
        # emb1 shape = [4,    1,   13,   1,   embedding_dim]
        # emb2 shape = [1,    4,   1,    13,  embedding_dim]
        # pair_embeddings shape = [4, 4, 13, 13, embedding_dim]
        emb1 = card_embeddings.unsqueeze(1).unsqueeze(3)
        emb2 = card_embeddings.unsqueeze(0).unsqueeze(2)
        pair_embeddings = emb1 * emb2

        return pair_embeddings.to(device)


    def process_belief(self, belief: torch.Tensor):
        """
        Process belief tensor using card embeddings.
        belief shape: [B, 2, 4, 4, 13, 13]
        """
        B = belief.shape[0]
        pair_embeddings = self.compute_pair_embeddings(device = belief.device)

        belief_features = []
        for player in range(2):
            # Extract belief for current player
            player_belief = belief[:, player]  # [B, 4, 4, 13, 13]

            # Use belief as weights for pair embeddings
            weights = player_belief.unsqueeze(-1)  # [B, 4, 4, 13, 13, 1]

            # Weighted sum over all dimensions except batch and embedding
            weighted_sum = (weights * pair_embeddings).sum(dim = (1, 2, 3, 4))  # [B, 96]

            belief_features.append(weighted_sum)

        # Combine belief features from both players
        combined_belief = torch.cat(belief_features, dim = 1)  # [B, 192]
        return self.belief_feature_encoder(combined_belief)

    def forward(
            self,
            belief: torch.Tensor,  # [B, 2, 4, 4, 13, 13]
            state_info: torch.Tensor,  # [B, 5, 17]
            board: torch.Tensor, # [B, 5, 17]
            bounty: torch.Tensor, # [B, 3, 17]
    ):
        """Process all inputs and return fused features."""
        B = belief.shape[0]

        # 1. Process card observations
        card_features = []
        for i in range(board.shape[1]):
            card_feat = self.process_card_obs(board[:, i])
            card_features.append(card_feat)
        for i in range(bounty.shape[1]):
            card_feat = self.process_card_obs(bounty[:, i])
            card_features.append(card_feat)
        card_features = torch.cat(card_features, dim = 1)  # [B, 8*embedding_dim]

        # 2. Process chip information
        chip_features = self.chip_encoder(state_info.view(B, -1))

        # 3. Process belief tensor using weighted embeddings
        belief_features = self.process_belief(belief)

        # 4. Fuse all features
        fused = torch.cat([card_features, chip_features, belief_features], dim = 1)
        fused = self.fuse_layers(fused)

        return fused

    def get_logits_value(
            self,
            current_player: torch.Tensor,
            obs: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor),
            legal_action_mask: torch.Tensor,
            actions: torch.Tensor = None
    ):
        """
        Main forward call that produces actor logits and value.
        Returns:
            actor_logits: shape [B, 4] for the 4 possible actions
            value_4d: shape [B, 4, 4, 13, 13] for the hand distribution values
        """
        belief, state_info, board, hand, bounty = obs

        B = belief.shape[0]

        # 1. Get fused features
        fused = self.forward(belief, state_info, board, bounty)

        pair_embeddings = self.compute_pair_embeddings(device = fused.device)
        flat_pair_emb = pair_embeddings.view(-1, self.embedding_dim)

        # Inner product for value
        fused_flat = fused.view(-1, self.embedding_dim)
        inner_flat = torch.matmul(fused_flat, flat_pair_emb.T)  # [B, N] where N = 4 * 4 * 13 * 13
        inner_5d = inner_flat.view(B, self.num_features, self.num_suits, self.num_suits, self.num_ranks, self.num_ranks)
        inner_5d = inner_5d.permute(0, 2, 3, 4, 5, 1) # [B, 4, 4, 13, 13, 16]

        # 3. Actor heads
        actor_logits = torch.zeros(
            B, self.num_suits, self.num_suits, self.num_ranks, self.num_ranks, self.num_actions,
            device = fused.device
        )  # [B, 4, 4, 13, 13, 4]
        actor_probs = torch.zeros(
            B, self.num_suits, self.num_suits, self.num_ranks, self.num_ranks, self.num_actions,
            device = fused.device
        )
        critic_values = torch.zeros(
            B, self.num_suits, self.num_suits, self.num_ranks, self.num_ranks,
            device = fused.device
        )
        residuals = torch.zeros(B, device = fused.device)
        entropies = torch.zeros(B, device = fused.device)

        for player_idx in range(self.num_players):
            mask = (current_player == player_idx)
            if not torch.any(mask):
                continue

            inner_player = inner_5d[mask]
            inner_player_flat = inner_player.reshape(-1, self.num_features)

            logits_flat = self.actor_heads[player_idx](inner_player_flat)
            value_flat = self.critic_heads[player_idx](inner_player_flat)

            logits = logits_flat.view(-1, self.num_suits, self.num_suits, self.num_ranks, self.num_ranks, self.num_actions)
            value = value_flat.view(-1, self.num_suits, self.num_suits, self.num_ranks, self.num_ranks)

            pl_sym = 0.5 * (logits + logits.permute(0, 2, 1, 4, 3, 5))

            # Expand legal_action_mask to match dimensions
            lam_player = legal_action_mask[mask].view(-1, 1, 1, 1, 1, self.num_actions)

            pre_illegal_prob = F.softmax(pl_sym, dim = -1)
            residuals[mask] = torch.mean(torch.sum(pre_illegal_prob * (1 - lam_player), dim = -1), dim = (1, 2, 3, 4))

            pl_sym = pl_sym + (1 - lam_player) * (-1e10)
            probs = F.softmax(pl_sym, dim = -1)
            entropies[mask] = torch.mean(-torch.sum(probs * torch.log(probs + 1e-10), dim = -1), dim = (1, 2, 3, 4))

            actor_logits[mask] = pl_sym
            actor_probs[mask] = probs
            critic_values[mask] = value

        dist_probs = actor_probs[
            torch.arange(belief.shape[0], device = belief.device),
            hand[:, 0], hand[:, 1], hand[:, 2], hand[:, 3]
        ]  # [num_envs, n_actions]
        local_values = critic_values[
            torch.arange(belief.shape[0], device = belief.device),
            hand[:, 0], hand[:, 1], hand[:, 2], hand[:, 3]
        ]

        dist = torch.distributions.Categorical(dist_probs)
        if actions is None:
            actions = dist.sample()  # shape [num_envs]

        return actions, dist.log_prob(actions), entropies, local_values, actor_probs, residuals

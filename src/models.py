import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1, bias=True):
        super(SpatioTemporalBlock, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.temporal_attention = MultiHeadAttention(E_q=model_dim, E_k=model_dim, E_v=model_dim, E_total=model_dim, nheads=num_heads, dropout=dropout, bias=bias)
        self.spatial_attention = MultiHeadAttention(E_q=model_dim, E_k=model_dim, E_v=model_dim, E_total=model_dim, nheads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x shape: (batch_size, time_steps, num_players, features)
        # Temporal Attention
        # We assume all players can follow the same temporal patterns - so they share weights
        batch_size, time_steps, num_players, features = x.size()
        xt = x.permute(0, 2, 1, 3).contiguous().view(batch_size * num_players, time_steps, features)  # (batch_size * players, time_steps, features)
        temp_attn_output = self.temporal_attention(xt, xt, xt)
        xt = self.layer_norm1(xt + temp_attn_output)  # Add + LayerNorm
        xt = xt.view(batch_size, num_players, time_steps, features).permute(0, 2, 1, 3).contiguous()  # (time_steps, batch_size, players, features)

        # Spatial Attention
        # We take spatial attention across players at each time step
        # shape (batch_size * time_steps, players, features)
        xs = x.view(batch_size * time_steps, num_players, features)
        spat_attn_output = self.spatial_attention(xs, xs, xs)
        xs = self.layer_norm2(xs + spat_attn_output)
        xs = xs.view(batch_size, time_steps, num_players, features) # (batch_size, time_steps, players, features)
        x  = xt + xs
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + ff_output)
        return x


class SpatioTemporalModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(SpatioTemporalModel, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.temporal_pe = nn.Parameter(torch.randn(1, 100, 1, model_dim))  # Positional encoding for temporal dimension
        self.spatiotemporal_blocks = nn.ModuleList([
            SpatioTemporalBlock(model_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, time_steps, players, input_dim)
        batch_size, time_steps, players, _ = x.size()
        # Project features to transformer input dimension
        x = self.input_projection(x) # (batch_size, time_steps, players, model_dim)
        # Add positional encoding for temporal dimension
        #x = x + self.temporal_pe(x)
        x = self.dropout(x)
        for block in self.spatiotemporal_blocks:
            x = block(x)
        x = self.output_projection(x)        
        return x

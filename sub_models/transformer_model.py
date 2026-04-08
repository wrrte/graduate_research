import torch
import torch.nn as nn
import torch.nn.functional as F

from sub_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache


def _format_is_first_flags(is_first, batch_size, seq_len, device):
    if is_first is None:
        return None

    if is_first.dim() == 3 and is_first.shape[-1] == 1:
        is_first = is_first.squeeze(-1)
    if is_first.dim() == 1:
        is_first = is_first.unsqueeze(1)

    if is_first.shape[0] != batch_size:
        raise ValueError(f"Expected is_first batch={batch_size}, got {is_first.shape[0]}")

    if is_first.shape[1] == 1 and seq_len > 1:
        is_first = is_first.expand(batch_size, seq_len)
    if is_first.shape[1] != seq_len:
        raise ValueError(f"Expected is_first length={seq_len}, got {is_first.shape[1]}")

    return (is_first.to(device=device) > 0.5)


def _build_episode_causal_mask(is_first_flags):
    _, seq_len = is_first_flags.shape
    segment_ids = torch.cumsum(is_first_flags.long(), dim=1)
    same_segment = segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(1)
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=is_first_flags.device))
    return same_segment & causal.unsqueeze(0)


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask, is_first=None):
        is_first_flags = _format_is_first_flags(is_first, samples.shape[0], samples.shape[1], samples.device)
        if is_first_flags is not None:
            episode_mask = _build_episode_causal_mask(is_first_flags)
            mask = episode_mask if mask is None else (mask.bool() & episode_mask)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)

    def forward(self, samples, action, mask, is_first=None):
        '''
        Normal forward pass
        '''
        is_first_flags = _format_is_first_flags(is_first, samples.shape[0], samples.shape[1], samples.device)
        if is_first_flags is not None:
            episode_mask = _build_episode_causal_mask(is_first_flags)
            mask = episode_mask if mask is None else (mask.bool() & episode_mask)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype):
        '''
        Reset self.kv_cache_list
        '''
        param_example = next(iter(self.stem.parameters()))
        device = param_example.device
        self.kv_cache_list = []
        self.kv_valid_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))

    def forward_with_kv_cache(self, samples, action, is_first=None):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert samples.shape[1] == 1
        if not hasattr(self, "kv_cache_list") or not hasattr(self, "kv_valid_mask"):
            self.reset_kv_cache_list(samples.shape[0], dtype=samples.dtype)

        is_first_flags = _format_is_first_flags(is_first, samples.shape[0], 1, samples.device)
        if is_first_flags is not None:
            reset_mask = is_first_flags[:, 0]
            if reset_mask.any():
                if self.kv_valid_mask.shape[1] > 0:
                    self.kv_valid_mask[reset_mask] = False
                for idx in range(len(self.kv_cache_list)):
                    if self.kv_cache_list[idx].shape[1] > 0:
                        keep = (~reset_mask).to(dtype=self.kv_cache_list[idx].dtype).view(-1, 1, 1)
                        self.kv_cache_list[idx].mul_(keep)

        self.kv_valid_mask = torch.cat(
            [self.kv_valid_mask, torch.ones((samples.shape[0], 1), dtype=torch.bool, device=samples.device)],
            dim=1,
        )
        mask = self.kv_valid_mask.unsqueeze(1)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats

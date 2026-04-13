import torch.nn as nn


class _ProjectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=2, act="SiLU", dtype=None, device=None):
        super().__init__()
        act_layer = getattr(nn, act)

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, out_dim, bias=True, dtype=dtype, device=device))
        else:
            current_dim = input_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim, bias=True, dtype=dtype, device=device))
                layers.append(nn.LayerNorm(hidden_dim, dtype=dtype, device=device))
                layers.append(act_layer())
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, out_dim, bias=True, dtype=dtype, device=device))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActionConditionedContrastiveNetwork(nn.Module):
    """Predictor/representation pair used by AC-CPC."""

    def __init__(self, feat_size, embed_size, hidden_size, out_size, num_layers=2, act="SiLU", dtype=None, device=None):
        super().__init__()
        self.predictor = _ProjectorMLP(
            input_dim=feat_size,
            hidden_dim=hidden_size,
            out_dim=out_size,
            num_layers=num_layers,
            act=act,
            dtype=dtype,
            device=device,
        )
        self.representation = _ProjectorMLP(
            input_dim=embed_size,
            hidden_dim=hidden_size,
            out_dim=out_size,
            num_layers=num_layers,
            act=act,
            dtype=dtype,
            device=device,
        )

    def forward(self, feats, embed):
        return self.predictor(feats), self.representation(embed)

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_out, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        return self.norm(g * h + (1 - g) * self.skip(x))


class TemporalLSTM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class StaticEncoder(nn.Module):
    def __init__(self, d_static, d_model):
        super().__init__()
        self.grn = GRN(d_static, d_model, d_model)

    def forward(self, s):
        return self.grn(s)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars, d_model):
        super().__init__()
        self.var_grns = nn.ModuleList([
            GRN(1, d_model, d_model) for _ in range(num_vars)
        ])
        self.weight_grn = GRN(num_vars, d_model, num_vars)

    def forward(self, x):
        var_embeds = []
        for i, grn in enumerate(self.var_grns):
            var_embeds.append(grn(x[..., i:i + 1]))
        var_embeds = torch.stack(var_embeds, dim=-2)

        weights = self.weight_grn(x).softmax(dim=-1)
        fused = (weights.unsqueeze(-1) * var_embeds).sum(dim=-2)

        return fused, weights


class ContextEnrichment(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.grn = GRN(d_model * 2, d_model, d_model)

    def forward(self, temporal, context):
        context = context.unsqueeze(1).expand_as(temporal)
        return self.grn(torch.cat([temporal, context], dim=-1))


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.grn = GRN(d_model, d_model, d_model)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=mask)
        out = self.grn(attn_out + x)
        return out, attn_weights


class PredictionHead(nn.Module):
    def __init__(self, d_model, n_quantiles=3):
        super().__init__()
        self.fc = nn.Linear(d_model, n_quantiles)

    def forward(self, x):
        return self.fc(x[:, -1])


class MiniTFT(nn.Module):
    def __init__(self, n_obs, n_known, d_static, d_model=32, n_quantiles=3):
        super().__init__()
        self.static_enc = StaticEncoder(d_static, d_model)

        self.obs_vsn = VariableSelectionNetwork(n_obs, d_model)
        self.known_vsn = VariableSelectionNetwork(n_known, d_model)

        self.lstm = TemporalLSTM(d_model)

        self.enrich = ContextEnrichment(d_model)
        self.attn = TemporalAttention(d_model, num_heads=4)

        self.post_attn_grn = GRN(d_model, d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.head = PredictionHead(d_model, n_quantiles)

    def forward(self, obs, known, static):
        s = self.static_enc(static)

        obs_fused, _ = self.obs_vsn(obs)
        known_fused, _ = self.known_vsn(known)

        x = obs_fused + known_fused

        x = self.lstm(x)

        x = self.enrich(x, s)

        attn_out, attn_weights = self.attn(x)
        x = self.layer_norm(attn_out + x)
        x = self.layer_norm(self.post_attn_grn(x) + x)

        return self.head(x), attn_weights

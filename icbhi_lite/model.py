"""Compact shared backbone + event-conditioned full-cycle aggregation."""
from typing import Dict, Tuple

import torch
from torch import nn

from .frontend import feature_channels


def mask_time(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
    return x * valid[:, None, None, :].to(x.dtype)


class AxisBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), expansion=2, axis=True):
        super().__init__()
        hidden = cin * expansion
        self.axis = axis
        self.time_stride = stride[1]
        self.residual = cin == cout and stride == (1, 1)
        self.expand = nn.Sequential(nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU())
        # Frequency x time: broadband transient / narrowband sustained priors.
        k1 = (7, 3) if axis else (3, 3)
        self.dw1 = nn.Sequential(nn.Conv2d(hidden, hidden, k1, stride, (k1[0]//2, k1[1]//2),
                                         groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.SiLU())
        self.dw2 = nn.Sequential(nn.Conv2d(hidden, hidden, (3, 7), stride, (1, 3),
                                         groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.SiLU()) if axis else nn.Identity()
        self.gate = nn.Sequential(nn.Linear(hidden, max(8, hidden//8)), nn.SiLU(),
                                  nn.Linear(max(8, hidden//8), 2 * hidden)) if axis else nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original = x
        h = mask_time(self.expand(x), lengths)
        out_lengths = (lengths + self.time_stride - 1) // self.time_stride
        a = mask_time(self.dw1(h), out_lengths)
        if self.axis:
            b = mask_time(self.dw2(h), out_lengths)
            denom = (out_lengths * a.shape[-2]).clamp_min(1).to(a.dtype)[:, None]
            pooled = (a + b).sum((2, 3)) / denom
            gate = self.gate(pooled).reshape(a.shape[0], 2, a.shape[1]).softmax(1)
            a = a * gate[:, 0, :, None, None] + b * gate[:, 1, :, None, None]
        h = mask_time(self.project(a), out_lengths)
        if self.residual:
            h = h + original
        return h, out_lengths


class CycleLite(nn.Module):
    def __init__(self, channels=4, width=1.0, axis=True, pooling="events", dropout=0.2):
        super().__init__()
        if width <= 0 or pooling not in ("events", "mean"):
            raise ValueError("Invalid network width or pooling")
        widths = [max(8, round(c * width / 8) * 8) for c in [24, 32, 48, 80, 128]]
        self.pooling = pooling
        self.feature_dim = widths[-1]
        self.stem = nn.Sequential(nn.Conv2d(channels, widths[0], 3, (2, 2), 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.SiLU())
        blocks = []
        cin = widths[0]
        for cout, stride in zip(widths[1:], [(2, 1), (2, 2), (2, 2), (1, 1)]):
            blocks += [AxisBlock(cin, cout, stride, axis=axis), AxisBlock(cout, cout, axis=axis)]
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.event_attention = nn.Conv1d(cin, 2, 1)
        self.head = nn.Sequential(nn.Linear(3 * cin, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 4))
        self.event_weight = nn.Parameter(torch.empty(2, cin))
        self.event_bias = nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.event_weight, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: B, W, C, F, T; lengths: B,W; zero lengths mark absent windows.
        batch, windows = x.shape[0], x.shape[1]
        all_lengths = lengths.reshape(-1)
        keep = all_lengths > 0
        flat = x.flatten(0, 1)[keep]
        valid_lengths = all_lengths[keep]
        flat = mask_time(flat, valid_lengths)
        valid_lengths = (valid_lengths + 1) // 2
        h = mask_time(self.stem(flat), valid_lengths)
        for block in self.blocks:
            h, valid_lengths = block(h, valid_lengths)
        h = h.mean(2)  # valid windows, channels, time
        ntime, channels = h.shape[-1], h.shape[1]
        indices = torch.nonzero(keep).flatten()
        padded = h.new_zeros((batch * windows, channels, ntime)).index_copy(0, indices, h)
        lens = all_lengths.new_zeros(batch * windows).index_copy(0, indices, valid_lengths)
        temporal = torch.arange(ntime, device=x.device)[None, :] < lens[:, None]
        valid = temporal.reshape(batch, windows * ntime)
        features = padded.reshape(batch, windows, channels, ntime).permute(0, 2, 1, 3).flatten(2)
        v = valid[:, None, :].to(features.dtype)
        global_pool = (features * v).sum(-1) / v.sum(-1).clamp_min(1)
        if self.pooling == "events":
            attention_logits = self.event_attention(features)
            attention_logits = attention_logits.masked_fill(~valid[:, None, :], -10000.0)
            attention = attention_logits.softmax(-1)
            events = torch.matmul(attention, features.transpose(1, 2))
        else:
            attention = v.expand(-1, 2, -1) / v.sum(-1, keepdim=True).clamp_min(1)
            events = global_pool[:, None, :].expand(-1, 2, -1)
        combined = torch.cat([global_pool, events.flatten(1)], dim=1)
        logits = self.head(combined)
        event_logits = (events * self.event_weight[None]).sum(-1) + self.event_bias
        return {"logits": logits, "event_logits": event_logits, "attention": attention}


def build_model(cfg):
    m = cfg["model"]
    return CycleLite(channels=len(feature_channels(m["feature_mode"])), width=m["width"],
                     axis=m["axis_blocks"], pooling=m["pooling"], dropout=m["dropout"])


class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model(features, lengths)["logits"]

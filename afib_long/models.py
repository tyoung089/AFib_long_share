"""Single-ECG model and longitudinal model."""
import numpy as np
import torch
import torch.nn as nn


def _block(d_in, d_out, dropout):
    return [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(dropout)]


class SingleECGModel(nn.Module):
    def __init__(self, emb_dim=768, hidden=512, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Sequential(*_block(emb_dim, hidden, dropout))
        self.layer2 = nn.Sequential(*_block(hidden, hidden, dropout), nn.Linear(hidden, 1))

    def forward(self, emb):
        return self.layer2(self.layer1(emb)).squeeze(-1)


class TimeEncoding(nn.Module):
    def __init__(self, d_time=64):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(d_time // 2) * 0.01)
        self.phase = nn.Parameter(torch.zeros(d_time // 2))

    def forward(self, days):
        x = days.unsqueeze(-1) * self.freq + self.phase
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class LongitudinalModel(nn.Module):
    def __init__(self, emb_dim=768, d_model=256, d_time=64, n_heads=4, n_layers=2, d_ff=512,
                 dropout=0.3, demo_dim=2, demo_hidden=32):
        super().__init__()
        self.time_enc = TimeEncoding(d_time)
        self.token_proj = nn.Linear(emb_dim + d_time, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.demo_proj = nn.Sequential(nn.Linear(demo_dim, demo_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.index_marker = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.pred_head = nn.Sequential(*_block(d_model + demo_hidden, 256, dropout), nn.Linear(256, 1))

    def forward(self, emb, days, mask, age_sex):
        """emb (B, L, 768), days (B, L), mask (B, L) True for real ECGs, age_sex (B, 2) -> logits (B,)."""
        tokens = self.token_norm(self.token_proj(torch.cat([emb, self.time_enc(days)], dim=-1)))
        is_last = mask.long().cumsum(dim=1) == mask.long().sum(dim=1, keepdim=True)
        tokens = tokens + self.index_marker * is_last.float().unsqueeze(-1)
        b = emb.shape[0]
        tokens = torch.cat([self.cls_token.expand(b, 1, -1), tokens], dim=1)
        keep = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        out = self.transformer(tokens, src_key_padding_mask=~keep)
        return self.pred_head(torch.cat([out[:, 0], self.demo_proj(age_sex)], dim=-1)).squeeze(-1)


def load_longitudinal(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = LongitudinalModel(**ckpt["config"]["architecture"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt["config"]


@torch.no_grad()
def predict(model, dataset, batch_size=512, device="cpu"):
    """Predicted 1-year AF probability for every index ECG of a LongitudinalDataset (eval mode)."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = [torch.sigmoid(model(*(t.to(device) for t in batch[:4]))).cpu().numpy() for batch in loader]
    return np.concatenate(out) if out else np.zeros(0, np.float32)

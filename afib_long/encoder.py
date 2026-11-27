"""ECG-FM encoder (from fairseq-signals) and the 768-d ECG embedding."""
import dataclasses

import torch
import torch.nn as nn

PRETRAINING_ONLY = ("mask_emb", "quantizer", "project_q", "final_proj")


class ECGEncoder(nn.Module):
    def __init__(self, ecgfm_config):
        super().__init__()
        from fairseq_signals.models.ecg_transformer import ECGTransformerConfig, ECGTransformerModel
        names = {f.name for f in dataclasses.fields(ECGTransformerConfig)} - {"_name", "data", "args"}
        self.config = {k: v for k, v in dict(ecgfm_config).items() if k in names}
        self.config["apply_mask"] = False  # no masking: features are always computed on the full input
        self.net = ECGTransformerModel(ECGTransformerConfig(**self.config))

    def forward(self, x):
        """x: (B, 12, 2500) preprocessed 5-s windows -> (B, 768)."""
        feats = self.net.extract_features(x, padding_mask=None)["x"]  # (B, time steps, 768)
        return feats.sum(dim=1) / (feats != 0).float().sum(dim=1).clamp(min=1)

    def embed_ecg(self, windows):
        """windows: (B, 2, 12, 2500) -> (B, 2, 768), one embedding per 5-s window."""
        b = windows.shape[0]
        return self(windows.reshape(b * 2, *windows.shape[2:])).view(b, 2, -1)

    def freeze_lower_layers(self, n_layers=5):
        for p in self.net.feature_extractor.parameters():
            p.requires_grad = False
        for layer in self.net.encoder.layers[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False


def load_pretrained_ecgfm(path):
    """Original ECG-FM checkpoint (mimic_iv_ecg_physionet_pretrained.pt from Hugging Face)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder = ECGEncoder(ckpt["cfg"]["model"])
    state = {k: v for k, v in ckpt["model"].items() if k.split(".")[0] not in PRETRAINING_ONLY}
    encoder.net.load_state_dict(state, strict=True)
    return encoder


def load_encoder(path, device="cpu"):
    """Fine-tuned encoder saved by `save_encoder` (this is the format of the released encoder file)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder = ECGEncoder(ckpt["config"]["ecgfm"])
    encoder.net.load_state_dict(ckpt["state_dict"], strict=True)
    return encoder.to(device), ckpt


def save_encoder(encoder, path, **extra):
    config = {"ecgfm": encoder.config, "sample_rate": 500, "window_samples": 2500, "embedding_dim": 768}
    torch.save({"config": config, "state_dict": encoder.net.state_dict(), **extra}, path)

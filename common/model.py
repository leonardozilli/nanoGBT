import inspect
import math
import os

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

from common.config import GBTConfig


class RoPE(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def build_cache(self, seq_len, device):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[2]:
            return

        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2, device=device).float() / self.d)
        )

        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.einsum("i,j->ij", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)

        self.cos_cached = idx_theta2.cos()[None, None, :, :]
        self.sin_cached = idx_theta2.sin()[None, None, :, :]

    def rotate_half(self, x):
        d_2 = self.d // 2

        x1 = x[..., :d_2]
        x2 = x[..., d_2:]

        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[2]
        device = q.device

        self.build_cache(seq_len, device)

        cos = self.cos_cached[:, :, :seq_len, :]  # (1, 1, T, d)
        sin = self.sin_cached[:, :, :seq_len, :]  # (1, 1, T, d)

        q = (q * cos) + (self.rotate_half(q) * sin)  # (B, nh, T, hs)
        k = (k * cos) + (self.rotate_half(k) * sin)  # (B, nh, T, hs)

        return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.rope = RoPE(config.n_embd // config.n_head)
        if not self.flash:
            # if flash attn is not available, create the mask manually
            print("Flash attention not available.")
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        q, k = self.rope(q, k)

        if self.flash:
            y = scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p, is_causal=True
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            assert isinstance(self.tril, torch.Tensor)
            attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side (B, T, C)
        y = self.c_proj_dropout(self.c_proj(y))

        return y


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = nn.SiLU()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * self.swish(gate)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h_dim = int(8 / 3 * config.n_embd)
        self.exp = nn.Linear(config.n_embd, 2 * self.h_dim, bias=False)
        self.lu = SwiGLU()
        self.c_proj = nn.Linear(self.h_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.exp(x)
        x = self.lu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms_1 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = MultiHeadAttention(config)
        self.rms_2 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.mlp = MLP(config)
        self.scale = 1 / math.sqrt(2 * config.n_layer)  # residual scaling

    def forward(self, x):
        attn_out = self.attn(self.rms_1(x))
        mlp_out = self.mlp(self.rms_2(x))

        x = x + self.scale * attn_out
        x = x + self.scale * mlp_out

        return x


class GBT(nn.Module):
    """Gioachino Belli Transformer"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "rms_f": nn.RMSNorm(config.n_embd, eps=1e-6),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer["wte"].weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        """Initialize linear and embedding layers using a normal distribution, as per GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.transformer["wte"](idx)
        x = self.transformer["drop"](tok_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["rms_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=512,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_id=None,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, loss = self(idx_cond)  # (B, T, C)
            logits = logits[:, -1, :] / temperature  # (B, C)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            if eos_id is not None and (idx_next == eos_id).any():
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
                break

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "GBT":
        if os.path.isdir(checkpoint_path):
            pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(
                    f"No .pt file found in directory: {checkpoint_path}"
                )
            checkpoint_path = os.path.join(checkpoint_path, pt_files[0])

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config_dict = checkpoint["config"]
        config = GBTConfig(**config_dict)
        model = cls(config)

        state_dict = checkpoint["model_state_dict"]
        for k, v in list(state_dict.items()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod.") :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = (
            "cuda" in self.config.device
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=fused_available,
        )
        return optimizer

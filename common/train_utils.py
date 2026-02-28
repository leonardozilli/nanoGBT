import math
import os
from pathlib import Path

import numpy as np
import torch

from common.config import GBTConfig, TrainConfig
from common.model import GBT


class DataLoader:
    def __init__(self, B, T, device, data_path, split="train"):
        self.B = B
        self.T = T
        self.device = device
        self.current_position = 0

        self.tokens = np.fromfile(
            os.path.join(data_path, f"{split}.bin"), dtype=np.uint16
        )
        print(f"loaded {len(self.tokens)} {split} tokens from {data_path}")

        self.reset()

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        if self.current_position + B * T + 1 > len(self.tokens):
            self.reset()

        buf = (
            torch.from_numpy(
                self.tokens[self.current_position : self.current_position + B * T + 1]
            )
            .to(self.device)
            .long()
        )
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        return x, y


def get_lr(step, config: TrainConfig):
    lr_max = config.learning_rate
    lr_min = 0.2 * lr_max  # 20% of max

    if step < config.warmup_steps:
        # linear warmupt 0 -> lr_max
        return lr_max * (step + 1) / config.warmup_steps
    else:
        # cosine decay lr_max -> lr_min
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def sample(
    model: GBT,
    tokenizer,
    config: GBTConfig,
    prompt: str = "",
    max_new_tokens: int = 100,
):
    model.eval()
    try:
        sos_token = tokenizer.special_tokens["SOS"]
    except KeyError:
        sos_token = prompt
    input_ids = tokenizer.encode(sos_token)
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids

    context = torch.tensor([input_ids], dtype=torch.long, device=config.device)
    pred = model.generate(context, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(pred[0].tolist(), skip_special_tokens=False)
    print(generated_text)

    return generated_text


def eval(model: GBT, val_loader: DataLoader, step: int, config: GBTConfig):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_losses = []
        for _ in range(10):
            try:
                x, y = val_loader.next_batch()
            except StopIteration:
                break
            x, y = x.to(config.device), y.to(config.device)
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        print(f"step {step}: val loss {val_loss}")

        return val_loss


def save_model(model, optimizer, val_loss, model_config, step, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "config": model_config.__dict__,
        },
        output_dir / f"checkpoint_{step}_{val_loss:.4f}.pt",
    )

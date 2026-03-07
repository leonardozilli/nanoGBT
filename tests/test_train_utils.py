import math
from typing import cast

import numpy as np
import torch

from common.config import GBTConfig, TrainConfig
from common.model import GBT
from common.train_utils import DataLoader, get_lr, sample, save_model


def test_get_lr_warmup_and_decay_bounds():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100)

    lr_start = get_lr(0, cfg)
    lr_warmup_end = get_lr(9, cfg)
    lr_decay_start = get_lr(10, cfg)
    lr_at_max = get_lr(100, cfg)

    assert math.isclose(lr_start, 1e-4, rel_tol=1e-9)
    assert math.isclose(lr_warmup_end, 1e-3, rel_tol=1e-9)
    assert math.isclose(lr_decay_start, 1e-3, rel_tol=1e-9)
    assert math.isclose(lr_at_max, 1e-4, rel_tol=1e-9)


def test_dataloader_next_batch(tmp_path):
    data_dir = tmp_path / "encoded"
    data_dir.mkdir()

    values = np.arange(1, 21, dtype=np.uint16)
    train_path = data_dir / "train.bin"
    values.tofile(train_path)

    loader = DataLoader(B=2, T=3, device="cpu", data_dir=str(data_dir), split="train")

    x1, y1 = loader.next_batch()
    assert x1.shape == (2, 3)
    assert y1.shape == (2, 3)
    assert x1.dtype == torch.long
    assert y1.dtype == torch.long
    assert torch.equal(y1[:, :-1], x1[:, 1:])
    assert loader.current_position == 6

    x2, y2 = loader.next_batch()
    assert loader.current_position == 12
    assert torch.equal(y2[:, :-1], x2[:, 1:])

    x3, y3 = loader.next_batch()
    assert loader.current_position == 18
    assert torch.equal(y3[:, :-1], x3[:, 1:])

    x4, y4 = loader.next_batch()
    assert loader.current_position == 6
    assert torch.equal(y4[:, :-1], x4[:, 1:])


class _DummyModel:
    def __init__(self):
        self.called_with: tuple[torch.Tensor, int] | None = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def generate(self, context, max_new_tokens):
        self.called_with = (context.clone(), max_new_tokens)
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def state_dict(self):
        return {"weights": torch.tensor([1.0])}


class _DummyTokenizer:
    def __init__(self):
        self.special_tokens = {"BOS": "<SONNET>"}
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text):
        self.encode_calls.append(text)
        return [7, 8]

    def decode(self, batch, skip_special_tokens=False):
        self.decode_calls.append((batch, skip_special_tokens))
        return "decoded-text"


def test_sample_uses_bos_and_returns_decoded_text():
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    cfg = GBTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        device="cpu",
    )

    out = sample(cast(GBT, model), tokenizer, cfg, prompt="ignored", max_new_tokens=5)

    assert out == "decoded-text"
    assert model.eval_called is True
    assert tokenizer.encode_calls == ["<SONNET>"]
    assert model.called_with is not None
    context, max_new_tokens = model.called_with
    assert context.shape == (1, 2)
    assert context.dtype == torch.long
    assert max_new_tokens == 5
    assert tokenizer.decode_calls == [([1, 2, 3], False)]


class _DummyOptimizer:
    def state_dict(self):
        return {"lr": 1e-3}


def test_save_model_writes_checkpoint(tmp_path):
    model = _DummyModel()
    optimizer = _DummyOptimizer()
    model_cfg = GBTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.1,
        device="cpu",
    )

    save_model(model, optimizer, 1.23, model_cfg, step=42, output_dir=str(tmp_path))

    ckpt_path = tmp_path / "checkpoint_42.pt"
    assert ckpt_path.exists()

    payload = torch.load(ckpt_path, map_location="cpu")
    assert set(payload.keys()) == {
        "model_state_dict",
        "optimizer_state_dict",
        "loss",
        "config",
    }
    assert payload["loss"] == 1.23
    assert payload["config"]["vocab_size"] == 32

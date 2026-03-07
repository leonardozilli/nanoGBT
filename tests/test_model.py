from typing import cast

import torch

from common.model import GBT


class _DummyModel:
    def __init__(self, logits_per_step: list[list[float]], block_size: int = 8):
        self.config = type("cfg", (), {"block_size": block_size})()
        self._logits_per_step = logits_per_step
        self._step = 0
        self.seen_context_lengths: list[int] = []

    def __call__(self, idx):
        batch_size, context_len = idx.shape
        self.seen_context_lengths.append(context_len)

        step = min(self._step, len(self._logits_per_step) - 1)
        step_logits = torch.tensor(self._logits_per_step[step], dtype=torch.float)
        vocab_size = step_logits.numel()

        logits = torch.zeros(batch_size, context_len, vocab_size, dtype=torch.float)
        logits[:, -1, :] = step_logits

        self._step += 1
        return logits, None


def _argmax_multinomial(probs, num_samples=1):
    return probs.argmax(dim=-1, keepdim=True)


def test_generate_appends_max_new_tokens_without_eos(monkeypatch):
    monkeypatch.setattr(torch, "multinomial", _argmax_multinomial)

    model = _DummyModel(
        logits_per_step=[[0.1, 0.2, 0.3, 10.0]] * 5,
        block_size=8,
    )
    idx = torch.tensor([[1, 2]], dtype=torch.long)

    out = GBT.generate(cast(GBT, model), idx, max_new_tokens=5)

    assert out.shape == (1, 7)
    assert out[0, -5:].tolist() == [3, 3, 3, 3, 3]


def test_generate_stops_on_eos_id(monkeypatch):
    monkeypatch.setattr(torch, "multinomial", _argmax_multinomial)

    eos_id = 4
    model = _DummyModel(
        logits_per_step=[[0.1, 0.2, 0.3, 0.4, 10.0], [0.0, 0.0, 0.0, 10.0, 0.0]],
        block_size=8,
    )
    idx = torch.tensor([[1, 2]], dtype=torch.long)

    out = GBT.generate(cast(GBT, model), idx, max_new_tokens=10, eos_id=eos_id)

    assert out.shape == (1, 3)
    assert out[0, -1].item() == eos_id


def test_generate_respects_block_size_context_window(monkeypatch):
    monkeypatch.setattr(torch, "multinomial", _argmax_multinomial)

    model = _DummyModel(logits_per_step=[[0.0, 10.0, 0.0]] * 6, block_size=4)
    idx = torch.tensor([[9, 8, 7, 6, 5]], dtype=torch.long)

    _ = GBT.generate(cast(GBT, model), idx, max_new_tokens=6)

    assert model.seen_context_lengths[0] == 4
    assert all(length <= 4 for length in model.seen_context_lengths)


def test_top_k_masks_tail_tokens(monkeypatch):
    captured_probs = {}

    def fake_multinomial(probs, num_samples=1):
        captured_probs["value"] = probs.clone()
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    model = _DummyModel(logits_per_step=[[1.0, 2.0, 3.0, 4.0, 5.0]], block_size=8)
    idx = torch.tensor([[1, 2]], dtype=torch.long)

    _ = GBT.generate(cast(GBT, model), idx, max_new_tokens=1, top_k=2)

    probs = captured_probs["value"][0]
    assert torch.allclose(probs[:3], torch.zeros_like(probs[:3]))
    assert probs[3] > 0 and probs[4] > 0


def test_top_p_nucleus_masks_low_probability_tail(monkeypatch):
    captured_probs = {}

    def fake_multinomial(probs, num_samples=1):
        captured_probs["value"] = probs.clone()
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    model = _DummyModel(logits_per_step=[[10.0, 1.0, 1.0, 1.0]], block_size=8)
    idx = torch.tensor([[1, 2]], dtype=torch.long)

    _ = GBT.generate(cast(GBT, model), idx, max_new_tokens=1, top_p=0.5)

    probs = captured_probs["value"][0]
    assert probs[0] > 0
    assert torch.allclose(probs[1:], torch.zeros_like(probs[1:]))

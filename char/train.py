import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

# ruff: noqa: E402
import hydra
from omegaconf import DictConfig

from common.config import GBTConfig, TrainConfig
from common.tokenizer import CharTokenizer
from common.trainer import train


@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def main(cfg: DictConfig):
    tokenizer = CharTokenizer(cfg.dataset.tokenizer_path)

    model_config = GBTConfig(
        block_size=cfg.model.block_size,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        vocab_size=tokenizer.vocab_size,
        dropout=cfg.model.dropout,
        device=cfg.training.device,
    )

    train_config = TrainConfig(
        checkpoint=cfg.checkpoint,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        max_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.warmup_steps,
        early_stop_patience=cfg.training.early_stop_patience,
        output_dir=cfg.training.output_dir,
        weight_decay=cfg.training.weight_decay,
        checkpointing_steps=cfg.training.checkpointing_steps,
    )

    train(
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        dataset_name=cfg.dataset.dataset_name,
        data_dir=cfg.dataset.data_dir,
        checkpoint=cfg.get("checkpoint"),
        trace=cfg.training.trace,
    )


if __name__ == "__main__":
    main()

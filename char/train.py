import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

# ruff: noqa: E402
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import common.train_utils as train_utils
from common.config import GBTConfig, TrainConfig
from common.tokenizer import CharTokenizer
from common.trainer import train


@hydra.main(config_path="configs", config_name=None, version_base="1.3")
def main(cfg: DictConfig):
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    log_file = os.path.join(hydra_output_dir, "train.log")

    train_utils.setup_logging(log_file)

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
        checkpoint=cfg.get("checkpoint", None),
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        max_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.warmup_steps,
        early_stop_patience=cfg.training.early_stop_patience,
        output_dir=hydra_output_dir,
        weight_decay=cfg.training.weight_decay,
        checkpointing_steps=cfg.training.checkpointing_steps,
        notes=cfg.notes,
    )

    train_loader = train_utils.DataLoader(
        B=train_config.batch_size,
        T=model_config.block_size,
        device=model_config.device,
        data_dir=cfg.dataset.data_dir,
        split="train",
    )
    val_loader = train_utils.DataLoader(
        B=train_config.batch_size,
        T=model_config.block_size,
        device=model_config.device,
        data_dir=cfg.dataset.data_dir,
        split="val",
    )

    train(
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_name=cfg.dataset.name,
        trace=cfg.training.trace,
    )


if __name__ == "__main__":
    main()

import json
import logging
import math
import os
import shutil
import time
from typing import Any, cast

import torch

import wandb
from common import train_utils
from common.config import GBTConfig, TrainConfig
from common.model import GBT
from common.tokenizer import BPETokenizer, CharTokenizer, UnigramTokenizer

logger = logging.getLogger(__name__)


def train(
    model_config: GBTConfig,
    train_config: TrainConfig,
    tokenizer: CharTokenizer | BPETokenizer | UnigramTokenizer,
    train_loader: Any,
    val_loader: Any,
    dataset_name: str,
    sample_prompt: str | None = None,
    trace: bool = False,
):
    os.makedirs(train_config.output_dir, exist_ok=True)

    if train_config.checkpoint:
        if not os.path.isfile(train_config.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {train_config.checkpoint}")
        logger.info(f"Loading checkpoint from {train_config.checkpoint}")
        model = GBT.from_pretrained(train_config.checkpoint)
    else:
        model = GBT(model_config)

    model = torch.compile(model)
    model = cast(GBT, model)
    model = model.to(model_config.device)

    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(0.9, 0.95),
    )

    torch.set_float32_matmul_precision("high")

    run = (
        train_utils.init_wandb(
            train_config, model_config, model, dataset_name, tokenizer
        )
        if trace
        else None
    )

    generations_table = None
    if trace:
        generations_table = wandb.Table(
            columns=["step", "loss", "text"], log_mode="INCREMENTAL"
        )

    best_val_loss = float("inf")
    best_step_val = -1
    steps_since_improvement = 0
    tmp_checkpoint = os.path.join(train_config.output_dir, "checkpoint_best_tmp.pt")

    time_start = time.time()

    for step in range(train_config.max_steps + 1):
        t0 = time.time()

        lr = train_utils.get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % 100 == 0:
            val_loss, perplexity = train_utils.eval(
                model, val_loader, step, model_config
            )
            if trace:
                wandb.log({"loss/val": val_loss, "ppl/val": perplexity}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step_val = step
                steps_since_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                        "config": model_config.__dict__,
                        "step": best_step_val,
                    },
                    tmp_checkpoint,
                )

                if trace:
                    wandb.log(
                        {
                            "loss/val_best": best_val_loss,
                            "ppl/val_best": perplexity,
                            "best_step": best_step_val,
                        },
                        step=step,
                    )
            else:
                steps_since_improvement += 100

            if steps_since_improvement >= train_config.early_stop_patience:
                logger.info(f"Early stopping at step {step}")
                break

        if step % 500 == 0:
            default_prompt = tokenizer.special_tokens.get("BOS", "")
            generated_text = train_utils.sample(
                model,
                tokenizer,
                model_config,
                prompt=sample_prompt if sample_prompt else default_prompt,
            )
            if trace and generations_table is not None:
                generations_table.add_data(step, val_loss, generated_text)
                wandb.log({"generations": generations_table}, step=step)

        if step % train_config.checkpointing_steps == 0:
            os.makedirs(
                os.path.join(train_config.output_dir, "checkpoints"), exist_ok=True
            )
            train_utils.save_model(
                model,
                optimizer,
                val_loss,
                model_config,
                step,
                os.path.join(train_config.output_dir, "checkpoints"),
            )

        model.train()
        optimizer.zero_grad()
        x, y = train_loader.next_batch()
        x, y = x.to(model_config.device), y.to(model_config.device)
        with torch.autocast(device_type=model_config.device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            t1 = time.time()
            dt = (t1 - t0) * 1000
            tps = (train_loader.B * train_loader.T) / (t1 - t0)
            mem_usage = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(
                f"step{step}, loss: {loss.item():.4f}, lr: {lr:.2e}, dt: {dt:.2f}ms, tok/s: {tps:.0f}, grad_norm: {norm:.2f}"
            )
            if trace:
                wandb.log(
                    {
                        "loss/train": loss.item(),
                        "ppl/train": math.exp(loss.item()),
                        "lr/train": lr,
                        "tokens_per_second/train": tps,
                        "grad_norm/train": norm,
                        "system/memory_allocated_mib": mem_usage,
                    },
                    step=step,
                )

    logger.info(f"Training time: {time.time() - time_start:.2f} seconds")
    if torch.cuda.is_available():
        logger.info(
            f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB"
        )

    best_checkpoint_path = ""
    if best_step_val >= 0 and os.path.isfile(tmp_checkpoint):
        best_checkpoint_path = os.path.join(
            train_config.output_dir, "checkpoints", f"checkpoint_{best_step_val}.pt"
        )
        shutil.copy2(tmp_checkpoint, best_checkpoint_path)
        os.remove(tmp_checkpoint)

    metadata = {
        "model": vars(model_config),
        "tokenizer_type": tokenizer.kind,
        "training": vars(train_config),
        "best_val_loss": best_val_loss,
        "best_step": best_step_val,
        "steps_completed": step,
        "M_params": model.get_num_params() / 1e6,
        "training_time_S": time.time() - time_start,
    }
    if train_config.checkpoint:
        metadata["checkpoint"] = train_config.checkpoint

    if trace and run:
        run.finish()
        metadata["wandb_run_id"] = run.id

    with open(os.path.join(train_config.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

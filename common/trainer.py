import json
import os
import time
from datetime import datetime
from typing import cast

import torch
import wandb

from common import train_utils
from common.config import GBTConfig, TrainConfig
from common.model import GBT
from common.tokenizer import CharTokenizer, SubwordTokenizer


def train(
    model_config: GBTConfig,
    train_config: TrainConfig,
    tokenizer: CharTokenizer | SubwordTokenizer,
    dataset_name: str,
    data_path: str,
    trace: bool = False,
):
    os.makedirs(train_config.output_dir, exist_ok=True)

    model = GBT(model_config)
    model = model.to(model_config.device)
    model = torch.compile(model)
    model = cast(GBT, model)

    print(f"Loading data from {data_path} for dataset {dataset_name}")

    train_loader = train_utils.DataLoader(
        B=train_config.batch_size,
        T=model_config.block_size,
        device=model_config.device,
        data_path=data_path,
        split="train",
    )
    val_loader = train_utils.DataLoader(
        B=train_config.batch_size,
        T=model_config.block_size,
        device=model_config.device,
        data_path=data_path,
        split="val",
    )

    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(0.9, 0.95),
    )

    torch.set_float32_matmul_precision("high")

    run = None
    if trace:
        run = wandb.init(
            project="nanoGBT",
            job_type="train",
            config={
                "architecture": "GBT",
                "dataset": dataset_name,
                "learning_rate": train_config.learning_rate,
                "batch_size": train_config.batch_size,
                "block_size": model_config.block_size,
                "n_layer": model_config.n_layer,
                "n_head": model_config.n_head,
                "n_embd": model_config.n_embd,
                "dropout": model_config.dropout,
                "weight_decay": train_config.weight_decay,
                "max_steps": train_config.max_steps,
                "warmup_steps": train_config.warmup_steps,
                "M_params": model.get_num_params() / 1e6,
                "vocab_size": model_config.vocab_size,
            },
        )

        wandb.watch(model, log="all", log_freq=500)

    generations_table = None
    if trace:
        generations_table = wandb.Table(
            columns=["step", "loss", "text"], log_mode="INCREMENTAL"
        )

    best_val_loss = float("inf")
    early_stop_patience = train_config.early_stop_patience
    steps_since_improvement = 0
    time_start = time.time()

    step = 0
    val_loss = float("inf")

    for step in range(train_config.max_steps + 1):
        t0 = time.time()

        lr = train_utils.get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % 100 == 0:
            val_loss = train_utils.eval(model, val_loader, step, model_config)
            if trace:
                wandb.log({"loss/val": val_loss}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                steps_since_improvement = 0
            else:
                steps_since_improvement += 100

            if steps_since_improvement >= early_stop_patience:
                print(f"Early stopping at step {step}")
                train_utils.save_model(
                    model,
                    optimizer,
                    val_loss,
                    model_config,
                    step,
                    train_config.output_dir,
                )
                break

        if step % 500 == 0:
            generated_text = train_utils.sample(model, tokenizer, model_config)
            if trace and generations_table is not None:
                generations_table.add_data(step, val_loss, generated_text)
                wandb.log({"generations": generations_table}, step=step)

        model.train()
        optimizer.zero_grad()
        x, y = train_loader.next_batch()
        x, y = x.to(model_config.device), y.to(model_config.device)
        with torch.autocast(device_type=model_config.device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            t1 = time.time()
            dt = (t1 - t0) * 1000
            tps = (train_loader.B * train_loader.T) / (t1 - t0)
            mem_usage = torch.cuda.max_memory_allocated() / 1024**2
            print(
                f"step{step}, loss: {loss.item():.4f}, lr: {lr:.2e}, dt: {dt:.2f}ms, tok/s: {tps:.0f}, grad_norm: {norm:.2f}"
            )
            if trace:
                wandb.log(
                    {
                        "loss/train": loss.item(),
                        "train/lr": lr,
                        "train/tokens_per_second": tps,
                        "train/grad_norm": norm,
                        "system/memory_allocated_mib": mem_usage,
                    },
                    step=step,
                )

    print(f"Training time: {time.time() - time_start:.2f} seconds")
    if torch.cuda.is_available():
        print(
            f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB"
        )

    if trace and run:
        run.finish()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_config.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    metadata = {
        "model": vars(model_config),
        "training": vars(train_config),
        "best_val_loss": best_val_loss,
        "steps_completed": step,
        "M_params": model.get_num_params() / 1e6,
    }
    train_utils.save_model(model, optimizer, val_loss, model_config, step, run_dir)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

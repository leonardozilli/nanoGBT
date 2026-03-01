import click

from common.config import GBTConfig, TrainConfig
from common.tokenizer import SubwordTokenizer
from common.trainer import train


@click.command()
@click.option("--block-size", type=int, default=256, help="Block size for the model")
@click.option("--n-embd", type=int, default=368, help="Embedding dimension")
@click.option("--n-head", type=int, default=4, help="Number of attention heads")
@click.option("--n-layer", type=int, default=4, help="Number of transformer layers")
@click.option("--dropout", type=float, default=0.35, help="Dropout rate")
@click.option("--batch-size", type=int, default=32, help="Batch size for training")
@click.option("--learning-rate", type=float, default=3e-4, help="Max learning rate")
@click.option("--weight-decay", type=float, default=0.05, help="Weight decay")
@click.option("--max-steps", type=int, default=8000, help="Maximum training steps")
@click.option("--warmup-steps", type=int, default=400, help="Warmup steps")
@click.option(
    "--early-stop-patience",
    type=int,
    default=300,
    help="Early stopping patience (steps)",
)
@click.option(
    "--output-dir",
    type=str,
    default="models/subw/",
    help="Output directory for checkpoints",
)
@click.option(
    "--checkpoint-steps",
    type=int,
    default=1000,
    help="Checkpointing frequency (steps)",
)
@click.option("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
@click.option("--trace", is_flag=True, default=True, help="Enable wandb tracing")
def train_command(
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    batch_size,
    learning_rate,
    weight_decay,
    max_steps,
    warmup_steps,
    early_stop_patience,
    output_dir,
    device,
    trace,
    checkpoint_steps,
):
    tokenizer = SubwordTokenizer("data/encoded/subw/tokenizer.json")

    model_config = GBTConfig(
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        vocab_size=tokenizer.vocab_size,
        dropout=dropout,
    )
    model_config.device = device

    train_config = TrainConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        early_stop_patience=early_stop_patience,
        output_dir=output_dir,
        weight_decay=weight_decay,
        checkpointing_steps=checkpoint_steps,
    )

    train(
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        dataset_name="subword",
        data_path="data/encoded/subw/",
        trace=trace,
    )


if __name__ == "__main__":
    train_command()

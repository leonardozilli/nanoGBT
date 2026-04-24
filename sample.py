from pathlib import Path

import click
import torch

from common.model import GBT
from common.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SyllableTokenizer,
    UnigramTokenizer,
    load_tokenizer,
)


def resolve_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_top_k(top_k: int) -> int | None:
    if top_k < 0:
        raise click.BadParameter("must be >= 0")
    return top_k if top_k > 0 else None


def normalize_top_p(top_p: float) -> float | None:
    if not 0.0 <= top_p <= 1.0:
        raise click.BadParameter("must be in the range [0.0, 1.0]")
    return top_p if top_p > 0 else None


def get_eos_id(
    tokenizer: CharTokenizer | SyllableTokenizer | BPETokenizer | UnigramTokenizer,
) -> int | None:
    eos_token = tokenizer.special_tokens.get("EOS")
    if not eos_token:
        return None

    eos_id = tokenizer.get_token_id(eos_token)

    return eos_id


@click.command()
@click.option(
    "--checkpoint",
    "model_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model checkpoint (.pt) or directory containing a checkpoint",
)
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default="data/encoded/subw/tokenizer.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to tokenizer file",
)
@click.option(
    "--tokenizer-type",
    type=click.Choice(
        ["auto", "char", "syllable", "bpe", "unigram"], case_sensitive=False
    ),
    default="auto",
    show_default=True,
    help="Tokenizer type (auto infers from file name)",
)
@click.option("--cpu", is_flag=True, help="Force CPU inference")
@click.option("--prompt", default="", help="Prompt used to start generation")
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option("--temperature", "-t", type=float, default=1.0, show_default=True)
@click.option("--top-k", type=int, default=0, show_default=True, help="0 disables")
@click.option("--top-p", type=float, default=0.0, show_default=True, help="0 disables")
@click.option("--seed", type=int, default=None, help="Optional random seed")
@click.option(
    "--skip-special-tokens",
    is_flag=True,
    help="Skip special tokens while decoding",
)
def main(
    model_path: Path,
    tokenizer_path: Path,
    tokenizer_type: str,
    cpu: bool,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None,
    skip_special_tokens: bool,
) -> None:
    if max_new_tokens <= 0:
        raise click.BadParameter("--max-new-tokens must be > 0")
    if temperature <= 0:
        raise click.BadParameter("--temperature must be > 0")

    top_k_value = normalize_top_k(top_k)
    top_p_value = normalize_top_p(top_p)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("high")

    device = resolve_device(cpu)

    model = GBT.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_path, tokenizer_type.lower())
    prompt_text = prompt or tokenizer.special_tokens.get("BOS", "")
    if not prompt_text:
        raise click.ClickException(
            "Prompt is empty and tokenizer has no BOS token. Pass --prompt explicitly."
        )
    else:
        prompt_text = prompt.replace("\\n", "\n")

    input_ids = tokenizer.encode(prompt_text)
    context_tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        output_tokens = model.generate(
            context_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_value,
            top_p=top_p_value,
            eos_id=get_eos_id(tokenizer),
        )

    generated_text = tokenizer.decode(
        output_tokens[0].tolist(), skip_special_tokens=skip_special_tokens
    )

    generated_text = generated_text.replace("<NEWLINE>", "\n")
    click.echo(generated_text)


if __name__ == "__main__":
    main()

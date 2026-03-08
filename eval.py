import click
import torch

from common.eval import evaluate_structure
from common.model import GBT
from common.tokenizer import load_tokenizer


@click.command()
@click.option("--checkpoint", required=True, help="Path to the .pt checkpoint file")
@click.option(
    "--tokenizer-path", required=True, help="Path to vocab.json or spm_model.model"
)
@click.option(
    "--tokenizer-type", type=click.Choice(["char", "unigram", "bpe"]), required=True
)
@click.option(
    "--num-samples", default=10, help="Number of sonnets to generate and evaluate"
)
@click.option("--temperature", default=0.8, help="Generation temperature")
@click.option("--top-k", default=40, help="Top-K sampling")
@click.option("--top-p", default=0.9, help="Top-P (nucleus) sampling")
@click.option("--silent", is_flag=True, help="Suppress generated text output")
def main(
    checkpoint,
    tokenizer_path,
    tokenizer_type,
    num_samples,
    temperature,
    top_k,
    top_p,
    silent,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GBT.from_pretrained(checkpoint).to(device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path, tokenizer_type)

    bos_str = tokenizer.special_tokens.get("BOS", "<SONNET>")
    eos_str = tokenizer.special_tokens.get("EOS", "<END>")

    bos_id = tokenizer.get_token_id(bos_str)
    eos_id = tokenizer.get_token_id(eos_str)

    print(f"Evaluating {num_samples} samples...\n")

    results = {
        "14_lines": 0,
        "correct_structure": 0,
        "valid_stanzas": 0,
        "total_stanzas": 0,
        "is_valid_sonnet": 0,
    }

    with torch.no_grad():
        for i in range(num_samples):
            context = torch.tensor([[bos_id]], dtype=torch.long, device=device)

            pred = model.generate(
                context,
                max_new_tokens=1024 if tokenizer_type == "char" else 512,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=eos_id,
            )

            generated_text = tokenizer.decode(
                pred[0].tolist(), skip_special_tokens=True
            )
            generated_text = generated_text.replace("<NEWLINE>", "\n")

            metrics = evaluate_structure(generated_text)

            if metrics["line_count"] == 0:
                continue
            if metrics["is_14_lines"]:
                results["14_lines"] += 1
            if metrics["is_correct_structure"]:
                results["correct_structure"] += 1
            results["valid_stanzas"] += metrics["valid_stanzas"]
            results["total_stanzas"] += metrics["total_stanzas"]
            if metrics["is_valid_sonnet"]:
                results["is_valid_sonnet"] += 1

            if not silent:
                print(f"--- Sample {i + 1} ---")
                print(generated_text)
                print("----------------\n")

    print("=" * 40)
    print(
        f"14 Lines:          {results['14_lines']}/{num_samples} ({(results['14_lines'] / num_samples) * 100:.1f}%)"
    )
    print(
        f"Correct structure: {results['correct_structure']}/{num_samples} ({(results['correct_structure'] / num_samples) * 100:.1f}%)"
    )
    print(
        f"Valid stanzas:     {results['valid_stanzas']}/{results['total_stanzas']} ({(results['valid_stanzas'] / results['total_stanzas']) * 100:.1f}%)"
    )
    print(
        f"Valid Sonnets:     {results['is_valid_sonnet']}/{num_samples} ({(results['is_valid_sonnet'] / num_samples) * 100:.1f}%)"
    )
    print("=" * 40)


if __name__ == "__main__":
    main()

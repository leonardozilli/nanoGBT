# nanoGBT: Gioachino Belli Transformer

This repository hosts experiments in training custom transformer models to generate poetry in the style of Giuseppe Gioachino Belli.

Three training approaches are implemented:
1. [Character-level](./char/): a GPT-style transformer is trained using character-level tokenization. Includes a model trained directly on the sonnets dataset, as well as a 50M parameter model first pre-trained on a large dataset of 19th-century Italian documents before fine-tuning on Belli's sonnets.
2. [Subword-level](./subw/): the same GPT-style architecture, trained from scratch using a custom BPE tokenizer.
3. [LLM Fine-Tuning](./finetune/): Larger pre-trained language models are fine-tuned on the sonnets using LoRA adapters.

---

## Dataset
Giuseppe Gioachino Belli (1791‒1863) wrote 2279 sonnets portraying everyday life and customs of 19th-century Rome, written almost entirely in the local Romanesco dialect. These sonnets mostly follow a consistent structure of 14 hendecasyllabic lines divided into two quatrains and two tercets, typically with an ABBA ABBA CDC DCD rhyming scheme. 
The main data used for training consists of this collection, with two sonnets removed due to being in dialogue format. 

The raw sonnets are prepared for training by cleaning out orthographic noise (e.g. diacritics variants, editorial annotations, OCR errors), and marked to include special structural tokens:

> *\<SONNET>*  
> *Cuattro angioloni co le tromme in bocca*  
> *se metteranno uno pe cantone*  
> *a ssonà: poi co ttanto de voscione*  
> *cominceranno a ddì: ffora a cchi ttocca.*  
>  
> *\<STANZA>*  
>  
> *Allora vierà ssù una filastrocca*  
> *de schertri da la terra a ppecorone,*  
> *pe rripijjà ffigura de perzone,*  
> *come purcini attorno de la bbiocca.*  
>  
> *\<STANZA>*  
>  
> *E sta bbiocca sarà ddio bbenedetto,*  
> *che ne farà du’ parte, bbianca, e nnera:*  
> *una pe annà in cantina, una sur tetto.*  
>  
> *\<STANZA>*  
>  
> *All’urtimo usscirà ’na sonajjera*  
> *d’Angioli, e, ccome si ss’annassi a lletto,*  
> *smorzeranno li lumi, e bbona sera.*  
> *\<END>*

## Model Architecture

The models trained from scratch use a decoder-only transformer [architecture](./common/model.py), inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)'s implementation of GPT-2, but modernized to use SwiGLU activations and RMSNorm layers.

## Sampling CLI

```bash
python sample.py \
  --model <checkpoint-file> \
  --tokenizer-path <tokenizer-vocab-file> \
  --temperature 0.7 \
  --top-k 40 \
  --max-new-tokens 800
```

Arguments:

- `--model` (required): checkpoint file or a directory containing a `.pt` file.
- `--tokenizer-path`: tokenizer vocabulary json file.
- `--tokenizer-type`: `auto` (default), `char`, or `subword`.
  - If omitted (`auto` mode), the type is determined by the name of the tokenizer vocabulary file (`vocab.json` for character-level, `tokenizer.json` for subword-level).
- `--prompt`: initial text for generation. Defaults to the tokenizer's `BOS` token if available, otherwise an empty string.
- `--max-new-tokens`: number of tokens to generate (default `256`).
- `--temperature` / `-t`: sampling temperature (default `1.0`).
- `--top-k`: top-k sampling cutoff (default `0`, meaning disabled).
- `--top-p`: nucleus sampling cutoff (default `0.0`, meaning disabled).
- `--seed`: optional random seed.
- `--cpu`: force CPU inference.
- `--skip-special-tokens`: omit special tokens during decode.

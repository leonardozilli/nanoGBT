# nanoGBT: Gioachino Belli Transformer

This repository hosts experiments in training custom transformer models to generate poetry in the style of Giuseppe Gioachino Belli.

Three approaches are implemented:
1. [Character-level](./char/): a GPT-style transformer trained using character-level tokenization.
2. [Subword-level](./subw/): the same GPT-style architecture, trained from scratch using custom subword (BPE, Unigram) tokenizers.
3. [LLM Fine-Tuning](./finetune/): Larger pre-trained language models fine-tuned on the sonnets using LoRA adapters.

---

## Dataset
Giuseppe Gioachino Belli (1791‒1863) wrote 2279 sonnets portraying everyday life and customs of 19th-century Rome, using almost entirely the local Romanesco dialect. These sonnets generally follow a consistent structure of 14 hendecasyllabic lines divided into two quatrains and two tercets, typically with an ABBA ABBA CDC DCD rhyming scheme.

The main dataset used for training consists of this collection, with the raw sonnets cleaned to remove orthographic noise (e.g. diacritics variants, editorial annotations, OCR errors), and marked to include special structural tokens:

> *\<SONNET>*  
> *<RHYME_A> Cuattro angioloni co le tromme in bocca*  
> *<RHYME_B> se metteranno uno pe cantone*  
> *<RHYME_B> a ssonà: poi co ttanto de voscione*  
> *<RHYME_A> cominceranno a ddì: ffora a cchi ttocca.*  
>  
> *\<STANZA>*  
>  
> *<RHYME_A> Allora vierà ssù una filastrocca*  
> *<RHYME_B> de schertri da la terra a ppecorone,*  
> *<RHYME_B> pe rripijjà ffigura de perzone,*  
> *<RHYME_A> come purcini attorno de la bbiocca.*  
>  
> *\<STANZA>*  
>  
> *<RHYME_C> E sta bbiocca sarà ddio bbenedetto,*  
> *<RHYME_D> che ne farà du’ parte, bbianca, e nnera:*  
> *<RHYME_C> una pe annà in cantina, una sur tetto.*  
>  
> *\<STANZA>*  
>  
> *<RHYME_D> All’urtimo usscirà ’na sonajjera*  
> *<RHYME_C> d’Angioli, e, ccome si ss’annassi a lletto,*  
> *<RHYME_D> smorzeranno li lumi, e bbona sera.*  
> *\<END>*  

## Model Architecture

The models trained from scratch use a decoder-only transformer [architecture](./common/model.py), inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)'s implementation of GPT-2, but modernized to use RMSNorm, RoPE, SwiGLU and residual scaling.

## Sampling CLI

```bash
python sample.py \
  --model <checkpoint-file> \
  --tokenizer-path <tokenizer-vocab-file> \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --max-new-tokens 800
```

Arguments:

- `--model` (required): checkpoint file or a directory containing a `.pt` file.
- `--tokenizer-path`: tokenizer file path (`vocab.json`, `tokenizer.json` or `.model`).
- `--tokenizer-type`: `auto` (default), `char`, `bpe`, `unigram`
  - In `auto` mode, `vocab.json` maps to character-level, `.model` maps to SentencePiece Unigram, otherwise it defaults to BPE-style JSON tokenizers.
- `--prompt`: initial text for generation. Defaults to the tokenizer's `BOS` token if available, otherwise an empty string.
- `--max-new-tokens`: number of tokens to generate (default `256`).
- `--temperature` / `-t`: sampling temperature (default `1.0`).
- `--top-k`: top-k sampling cutoff (default `0`, meaning disabled).
- `--top-p`: nucleus sampling cutoff (default `0.0`, meaning disabled).
- `--seed`: optional random seed.
- `--cpu`: force CPU inference.
- `--skip-special-tokens`: omit special tokens during decode.

## Eval CLI

```bash
python eval.py \
  --checkpoint <checkpoint-file> \
  --tokenizer-path <tokenizer-vocab-file> \
  --tokenizer-type char \
  --num-samples 30 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9
```

Arguments:

- `--checkpoint` (required): path to a `.pt` checkpoint file.
- `--tokenizer-path` (required): tokenizer file path (`vocab.json`, `tokenizer.json` or `.model`).
- `--tokenizer-type` (required): `char`, `bpe`, or `unigram`.
- `--num-samples`: number of generated samples to evaluate (default `10`).
- `--temperature`: generation temperature (default `0.8`).
- `--top-k`: top-k sampling cutoff (default `40`).
- `--top-p`: nucleus sampling cutoff (default `0.9`).
- `--silent`: suppress per-sample generated text and print only metrics.

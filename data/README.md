This directory contains the datasets and scripts for scraping, cleaning, and preprocessing sonnets for use in training.

### 1. Scrape

Use [scrape.py](./scripts/scrape.py) to acquire sonnet texts from the web.

```bash
python data/scripts/scrape.py \
   --index-url <sonnets-index> \
   --output-dir data/raw/sonnets \
```

Arguments:
- `--index-url`: source index page containing sonnet links.
- `--output-dir`: destination folder for `.txt` files.
- `--limit`: optional limit on number of sonnets to scrape.
- `--sleep`: delay between requests to avoid rate limits (default `0.6` seconds).
- `--verbose`: enables debug logging. Useful for troubleshooting parsing issues.

### 2. Postprocess

Use [postprocess.py](./scripts/postprocess.py) to normalize text and add structure and rhyme tokens.

```bash
python data/scripts/postprocess.py \
   --data-dir data/raw/sonnets \
   --out-dir data/processed/sonnets_rhymes \
   --include-title \
   --mark-rhymes \
   --rhyme-length 2
```

Arguments:
- `--data-dir`: source directory containing raw .txt sonnet files.
- `--out-dir`: destination directory for processed sonnets.
- `--include-title`: prepends `<TITLE>...</TITLE>` tags.
- `--mark-rhymes`: appends `<RHYME_X>` tags to verse lines.
- `--rhyme-length`: controls maximum suffix length used for rhyme tagging.

# Character-Level Modeling

## 1. Plain sonnets

<details>
<summary>Tokenize</summary>

```bash
python data/scripts/postprocess.py --data-dir data/raw/sonnets/ --out-dir data/processed/sonnets
```

```bash
python char/tokenize.py --data-dir data/processed/sonnets --out-dir data/encoded/char/sonnets
```

```
number of sonnets: 2,279 
average length: 545.71 chars 
length of dataset in characters: 1,245,958
Vocabulary size: 90 chars
Characters: '\t\n !(),-.01236:;?ABCDEFGHIJLMNOPQRSTUVYZabcdefghijlmnopqrstuvwxz§«¶»ÀÈÉÊàáèéìíïòóôùúăēŌ–—’'

train split: 1,121,462 tokens
val split: 124,494 tokens
```
</details>

<details>
<summary>Train</summary>

```bash
python char/train.py --config-name sonnets
```

```plaintext
block_size: 768
n_embd: 384
n_layer: 8
n_head: 4
dropout: 0.3

batch_size: 64
learning_rate: 1e-3
weight_decay: 0.1
warmup_steps: 100

n_parameters:    14.19M
training time: ~366 seconds

best_step: 800
best_val_loss: 1.331
best_ppl: 3.787
```
</details>

Evaluation:

```bash
python eval.py \
  --checkpoint <checkpoint-file> \
  --tokenizer-path <path-to-vocab.json> \
  --tokenizer-type char \
  --temperature 0.75 \
  --top-k 20 \
  --top-p 0.9 \
  --num-samples 100 \
  --silent
```

```plaintext
========================================
14 Lines:          71/100 (71.0%)  # n of generated samples that have 14 lines
Correct structure: 71/100 (71.0%)  # n of generated samples that have the correct 4-4-3-3 structure  
Valid stanzas:     126/394 (32.0%) # n of generated stanzas with valid rhyme scheme
Valid Sonnets:     0/100 (0.0%)    # n of generated samples that are valid sonnets (14 lines with correct structure and rhyme scheme)
========================================
```

Generation sample:

```plaintext
Sí, ffra cquer che ddisce c’ha ffatto un gran ber governo de bbarberi.
Tu vvienissi a llei, per dio, tra ppezzi,
scorri sò ll’antri e ll’omo, se sarva.

Via, vadino che cquesto cqui a Ppadre,
te viè un cane che vve ne prese a spesa;
ma ar zu’ sette ar zu’ prencipe bboni
che cc’è er campo de la palla ar cappello.

Che ffa ppe ddí, ssor aria de sena,
sò vvienuta a bbattezza co le cappello,
de li cani nostri poverelli e ’r giacchello?

Che vvor dí? Cc’è cquell’antri mattina
a l’inzeggno de sette e dde scale,
che ssan Pietro co st’antra peccatorello.
```


Phonetic and orthographic patterns of the dialect are clearly present. However, it struggles heavily with rhyming and semantic meaning.

Let's try addressing the former by adding explicit rhyme markings to the training data:

## 2. Sonnets with rhyme annotations

<details>
<summary>Tokenize</summary>

```bash
python data/scripts/postprocess.py --data-dir data/raw/sonnets/ --out-dir data/processed/sonnets_rhymes --mark-rhymes
```

Example:

```plaintext
<SONNET>
<RHYME_A> Lustrissimi co’ questo mormoriale
<RHYME_B> v’addimando benigna perdonanza
<RHYME_B> se gni fiasco de vino igni pietanza
<RHYME_A> non fussi stata robba pella quale.
...
```

```bash
python char/tokenize.py --sonnet-dir data/processed/sonnets_rhymes --out-dir data/encoded/char/sonnets_rhymes/
```

```plaintext
number of sonnets: 2,279 
average length: 573.98 chars
length of dataset in characters: 1,310,375
Vocabulary size: 97 chars
Characters: '\t\n !(),-.01236:;?ABCDEFGHIJLMNOPQRSTUVYZabcdefghijlmnopqrstuvwxz§«¶»ÀÈÉÊàáèéìíïòóôùúăēŌ–—’ⒶⒷⒸⒹⒺⒻⒼ'

train split: 1,179,338 tokens
val split: 151,475 tokens
```
</details>

<details>
<summary>Train</summary>

Same hyperparameters as before.

```bash
python char/train.py --config-name sonnets dataset=sonnets_rhymes
```

```plaintext
best_step: 800
best_val_loss: 1.262
best_ppl: 3.535
```
</details>

Evaluation:

```bash
python eval.py \
  --checkpoint <checkpoint-file> \
  --tokenizer-path <path-to-vocab.json> \
  --tokenizer-type char \
  --temperature 0.75 \
  --top-k 20 \
  --top-p 0.9 \
  --num-samples 100 \
  --silent
```

```plaintext
========================================
14 Lines:          91/100 (91.0%)
Correct structure: 91/100 (91.0%)
Valid stanzas:     196/399 (49.1%)
Valid Sonnets:     0/100 (0.0%)
========================================
```

Generated Sample:

```plaintext
¶
Ⓐ Si cche ccaroggna de scertolassi in bello,
Ⓑ che sse sentissi un par de cuer cappello
Ⓑ che nun te pote le scittà ccor pelo
Ⓐ pe ffasse la scarpa de mignottello?

Ⓐ Io nun ho pperdona ppiú de carne e ddonne!
Ⓑ Eh gguarda a ccasa sc’è un fijjo de cane,
Ⓑ de le scarpe che le cose scià aridotte
Ⓐ d’incornaccio de tutte le connejje.

Ⓒ E nun ve stiede cqua sta capasce
Ⓓ pe vvede che mmorze nostri stoccoli
Ⓒ e mme ne faccio un buscio de penna.

Ⓔ Pe vvia de le stronzie de carrozza
Ⓓ de fà cco ttutti li guai li cappelli
Ⓔ pe li sovrani de fijji e mmesi ar cardi.
§
```

Much better adherence to the 4-4-3-3 structure. Still largely struggling with rhymimg though. 

### 2.1 Rhyme suffixes

We can try prepending also each line's rhyme suffix to force rhyme generation:

```bash
python data/scripts/postprocess.py --data-dir data/raw/sonnets/ --out-dir data/processed/sonnets_rhymes_suffix --mark-rhymes --include-rhyme-suffix
```

Example:
```plaintext
<SONNET>
<RHYME_A> ale | Lustrissimi co’ questo mormoriale
<RHYME_B> anza | v’addimando benigna perdonanza
<RHYME_B> anza | se gni fiasco de vino igni pietanza
<RHYME_A> ale | non fussi stata robba pella quale.
...
```

```plaintext
number of sonnets: 2,279 
average length: 663.25 chars
length of dataset in characters: 1,513,816
Vocabulary size: 98 chars
Characters: '\t\n !(),-.01236:;?ABCDEFGHIJLMNOPQRSTUVYZabcdefghijlmnopqrstuvwxz|§«¶»ÀÈÉÊàáèéìíïòóôùúăēŌ–—’ⒶⒷⒸⒹⒺⒻⒼ'

train split: 1,362,339 tokens
val split: 151,475 tokens
```

```bash
python char/train.py --config-name sonnets dataset=sonnets_rhymes_suffix
```

```plaintext
best_step: 1100
best_val_loss: 1.092
best_ppl: 2.982
```

Eval results:

```plaintext
========================================
14 Lines:          94/100 (94.0%)
Correct structure: 94/100 (94.0%)
Valid stanzas:     363/401 (90.5%)
Valid Sonnets:     32/100 (32.0%)
========================================
```

Generation sample:

```plaintext
¶
Ⓐ ato | Nun pare un paínete che in carnovato
Ⓑ ia | d’arissente a l’inferno a la commedia,
Ⓐ ato | disce ch’er Papa j’aricconta er Cristiato,
Ⓑ ia | ch’er Papa a l’improscinni de la sedia.

Ⓐ ato | E llui se sposera ch’er prelato è accato,
Ⓑ ia | e jj’amanca la sciarle d’una fedia:
Ⓐ ato | e lo sapeva chiamà Ddio che mm’ha ffato
Ⓑ ia | de servivà la spedale e la faccia.

Ⓒ ino | Tra ttutti in priggione un palazzino
Ⓓ ale | che pponno avé ssubbito er fritto male,
Ⓒ ino | che jje sposa er prossimo coll’arino?

Ⓓ ale | Perché ll’antri de la Campanale
Ⓒ ino | e ssò dde la padrona e dde cuer cantino
Ⓓ ale | che sse saría mejjo e nun ze passale.
§
```
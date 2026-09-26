# Nawayathi ⇄ English Translator

A neural machine translator for **Nawayathi**, a language spoken by a small community near Bhatkal, Karnataka. Nawayathi has no standard spelling and isn't covered by mainstream translation tools, so this project is built around **collecting a corpus and fine-tuning a pretrained model on it**, in both directions (Nawayathi → English and English → Nawayathi).

## How it works

Training a translation model from scratch needs hundreds of thousands of sentence pairs, far more than exists for Nawayathi. Instead, this project fine-tunes Meta's pretrained multilingual model [NLLB-200 (distilled, 600M)](https://huggingface.co/facebook/nllb-200-distilled-600M) using **LoRA**, which trains only a few percent of the weights. That lets a small corpus go a long way and fits on an 8 GB GPU.

Nawayathi is written here in **Roman letters**. NLLB marks each sentence's language with a tag, and Nawayathi isn't one of its 200 languages, so the project *borrows* the Marathi tag (`mar_Deva`, see `NWY_TAG` in [config.py](config.py)) and fine-tuning re-teaches it to mean "Nawayathi in Roman letters". The base model is never modified; only a small adapter is saved.

## Setup

Needs Python 3.10+ (developed and tested on 3.14) and, ideally, an NVIDIA GPU with 8 GB+ of memory (it runs on the CPU, but slowly). PowerShell on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch with CUDA first. A plain `pip install torch` is CPU-only on Windows.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The first training run downloads the NLLB model (about 2.5 GB) into your Hugging Face cache.

**Will it fit my GPU?** [docs/SYSTEM_SPECS_AND_LIMITS.md](docs/SYSTEM_SPECS_AND_LIMITS.md) records what an 8 GB laptop GPU (RTX 4060) can do: training speed, memory limits and which settings to avoid. `python benchmark.py specs` and `python benchmark.py train` measure your own machine.

## Workflow

### 1. Build the corpus

The corpus is [data/pairs.tsv](data/pairs.tsv): one sentence pair per line, columns separated by a **tab**.

```tsv
english	nawayathi	source
<English sentence>	<Nawayathi sentence in Roman letters>	<optional: who/where>
```

**The file already starts with about 1,350 English sentences** covering around 30 everyday topics (greetings, family, food, the sea and fishing, the mosque and festivals, the market, health, and a small block that covers who/when/negation systematically), with the `nawayathi` column **empty**. Fill in your translations in Roman letters, in any order. Rows you haven't translated yet are simply ignored, so you can run step 2 whenever you like (you need at least 20 translated rows). The most useful sections come first, and you can delete any section that doesn't suit your community. Edit it in VS Code, Notepad or Google Sheets (Download > .tsv) so it stays UTF-8; Excel's "Text" formats don't, and the checker will tell you if that happens.

To add your own sentences, either add rows to the file, or run the app (step 5) and use its **Contribute** tab. The app also lets you correct a wrong translation on the spot, and the correction goes straight into the corpus.

Tips that matter more than any setting:

- Use **full, natural sentences**, not single words. The model learns from context.
- Several spellings of one sentence are fine and useful. If you want two spellings treated as one word, list them in [data/spelling_variants.tsv](data/spelling_variants.tsv).
- Cover everyday topics broadly (greetings, family, food, numbers, questions, negatives, past/future tense) rather than many near-copies of one pattern.

**How much is enough?** Rough rules of thumb, not guarantees: with a few hundred pairs the pipeline runs but the model mostly memorises; a few thousand pairs starts to generalise on common phrases; more is always better. Keep adding pairs and re-training.

### 2. Check and split the data

```powershell
python data_preprocessing.py
```

Reports how many sentences are translated and how many are still waiting, lists every problem with its line number (non-Roman characters, duplicates, stray tabs, a Nawayathi sentence with no English), and writes `data/processed/{train,val,test}.tsv`. The split is grouped by English sentence, so a test sentence is never also in training.

### 3. Train

```powershell
python train_model.py
```

Saves the best adapter (by validation loss) to `models/nawayathi-lora/` and stops early when it stops improving. Useful options: `--epochs`, `--batch-size`, `--lr`, `--lora-r`.

Before training it checks your batch size and longest sentence against the GPU's free memory and prints a warning, with a suggested `--batch-size` and `--accum`, if they are too big. If you run out of GPU memory, lower `--batch-size` and raise `--accum`.

### 4. Evaluate

```powershell
python evaluate.py
```

Scores the held-out test set in both directions with chrF++ (higher is better) and prints sample translations. With a small test set the numbers are noisy, so read the samples too. To compare language tags, change `NWY_TAG` in `config.py`, re-train, and re-evaluate.

### 5. Translate

```powershell
python app.py                                                 # web app (Translate + Contribute tabs)
python inference.py "How are you?" --direction en2nwy         # command line
python inference.py                                           # interactive; /swap changes direction
```

The web app starts fine before any model exists, so you can begin collecting pairs on day one.

## Project layout

| File | Purpose |
| --- | --- |
| `config.py` | Paths, base model, language tags, defaults |
| `text_normalization.py` | Cleaning applied to both sides of the corpus and to your input |
| `data_preprocessing.py` | Corpus validation, splitting, and adding pairs |
| `train_model.py` | LoRA fine-tuning of NLLB, both directions |
| `evaluate.py` | chrF++ / BLEU on the test split |
| `inference.py` | Translation with beam search |
| `app.py` | Gradio web app |
| `benchmark.py` | Measures your GPU's training/translation speed and memory limits |
| `docs/SYSTEM_SPECS_AND_LIMITS.md` | What the author's machine (8 GB RTX 4060 laptop) can and can't do |
| `tests/` | Unit tests; `RUN_SLOW=1 pytest` also runs a GPU end-to-end test on a toy language |

## Tests

```powershell
pytest                                # fast unit tests
$env:RUN_SLOW = "1"; pytest -s        # + end-to-end test on the GPU (downloads the model once)
```

The end-to-end test uses an invented toy language, never your real corpus.

## License

The code is licensed under the **MIT License**, see [LICENSE](LICENSE). The NLLB model weights are released by Meta under **CC-BY-NC 4.0** (non-commercial), and adapters trained from them inherit that restriction. Check that your intended use complies.

## Contribution

Contributions and feature requests are welcome. The most valuable contribution is **more Nawayathi sentence pairs**. Please fork the repository and submit a pull request.

## Contact

- 📩 Email: [connect@syedmohammedsultan.online](mailto:connect@syedmohammedsultan.online)
- 📷 Instagram: [@sm.sultan.maliki](https://instagram.com/sm.sultan.maliki)
- 🌐 Website: [syedmohammedsultan.online](https://syedmohammedsultan.online)
- 💼 LinkedIn: [syedmohammedsultan](https://www.linkedin.com/in/syedmohammedsultan)

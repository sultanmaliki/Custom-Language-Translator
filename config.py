"""Shared settings for the Nawayathi <-> English translator.

Everything the other scripts need to agree on lives here, so changing (say) the
base model or the language tag is a one-line edit.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths -------------------------------------------------------------------
DATA_DIR = ROOT / "data"
PAIRS_FILE = DATA_DIR / "pairs.tsv"                  # the corpus you grow by hand
VARIANTS_FILE = DATA_DIR / "spelling_variants.tsv"   # optional: variant<TAB>canonical
PROCESSED_DIR = DATA_DIR / "processed"               # train/val/test.tsv (generated)
MODEL_DIR = ROOT / "models" / "nawayathi-lora"       # trained adapter (generated)

# --- Model -------------------------------------------------------------------
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB marks the language of every sentence with a special tag token. Nawayathi
# is not one of NLLB's 200 languages, so we *borrow* the tag of a related one.
# Fine-tuning re-teaches that tag to mean "Nawayathi (Roman script)" for this
# adapter only; the base model on disk is never modified. Marathi is a starting
# guess (NLLB has no Konkani tag) -- once you have data, try "hin_Deva",
# "urd_Arab" or "kan_Knda" and keep whichever scores best in evaluate.py.
# The tag must exist in the model's vocabulary; train_model.py checks this.
EN_TAG = "eng_Latn"
NWY_TAG = "mar_Deva"

# direction name -> (source tag, target tag)
DIRECTIONS = {
    "en2nwy": (EN_TAG, NWY_TAG),
    "nwy2en": (NWY_TAG, EN_TAG),
}

# --- Text normalisation ------------------------------------------------------
# Nawayathi has no standard Roman spelling, so we fold cosmetic differences
# (case, stretched letters like "sooo") together before training and at
# inference. Spelling variants of *words* go in VARIANTS_FILE.
LOWERCASE_NWY = True
COLLAPSE_REPEATS_NWY = True   # "aaaa" -> "aa"

# --- Data / training defaults ------------------------------------------------
MAX_LEN = 96                  # max tokens per sentence (longer pairs are dropped)
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
MIN_PAIRS = 20                # below this we refuse to split
SEED = 42

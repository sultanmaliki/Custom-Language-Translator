"""Translate with the trained adapter (beam search, both directions).

    python inference.py                                        # interactive
    python inference.py "How are you?" --direction en2nwy      # one-off
"""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from text_normalization import load_variants, normalize_english, normalize_nawayathi, use_utf8_console

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def require_language_tags(tokenizer, tags, model_name: str) -> None:
    """An unknown tag silently becomes <unk> and would train/translate against garbage, so fail loudly."""
    for tag in tags:
        if tokenizer.convert_tokens_to_ids(tag) == tokenizer.unk_token_id:
            raise ValueError(f"'{tag}' is not a language tag in {model_name}. Check EN_TAG / NWY_TAG in config.py.")


def split_sentences(text: str) -> list[str]:
    """The model translates one sentence at a time, so long text is split first."""
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


class ModelNotTrainedError(RuntimeError):
    """Raised when there is no trained adapter to load yet."""


class Translator:
    def __init__(self, adapter_dir: Path = config.MODEL_DIR):
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / "adapter_config.json").exists() or not (adapter_dir / "meta.json").exists():
            raise ModelNotTrainedError(
                f"No trained model found in {adapter_dir}. Add sentence pairs to data/pairs.tsv, then run "
                "`python data_preprocessing.py` and `python train_model.py`."
            )
        self.meta = json.loads((adapter_dir / "meta.json").read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["base_model"])
        require_language_tags(self.tokenizer, (self.meta["en_tag"], self.meta["nwy_tag"]), self.meta["base_model"])
        base =AutoModelForSeq2SeqLM.from_pretrained(self.meta["base_model"], dtype=dtype)
        # fold the adapter into the base weights: same output, faster generation
        self.model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().to(self.device).eval()
        self.variants = load_variants()
        self.tags = {
            "en2nwy": (self.meta["en_tag"], self.meta["nwy_tag"]),
            "nwy2en": (self.meta["nwy_tag"], self.meta["en_tag"]),
        }
        self.max_len = self.meta.get("max_len", config.MAX_LEN)

    def _prepare(self, text: str, direction: str) -> str:
        if direction == "en2nwy":
            return normalize_english(text)
        return normalize_nawayathi(text, self.variants)

    @torch.no_grad()
    def _generate(self, texts: list[str], direction: str, num_beams: int, n_best: int, max_length: int) -> list[list[str]]:
        if direction not in self.tags:
            raise ValueError(f"direction must be one of {sorted(self.tags)}, got {direction!r}")
        src_tag, tgt_tag = self.tags[direction]
        self.tokenizer.src_lang = src_tag
        enc = self.tokenizer([self._prepare(t, direction) for t in texts], return_tensors="pt",
                             padding=True, truncation=True, max_length=self.max_len).to(self.device)
        out = self.model.generate(
            **enc,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_tag),
            num_beams=max(num_beams, n_best),
            num_return_sequences=n_best,
            max_length=max_length,
            early_stopping=True,
        )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [decoded[i : i + n_best] for i in range(0, len(decoded), n_best)]

    def translate_batch(self, texts: list[str], direction: str = "en2nwy", num_beams: int = 5,
                        batch_size: int = 16, max_length: int = 128) -> list[str]:
        """Translate many single sentences; returns the best candidate for each."""
        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = self._generate(texts[i : i + batch_size], direction, num_beams, 1, max_length)
            results += [candidates[0] for candidates in chunk]
        return results

    def translate(self, text: str, direction: str = "en2nwy", num_beams: int = 5) -> str:
        """Translate free text, keeping line breaks."""
        lines = text.splitlines() or [text]
        pieces = [split_sentences(line) for line in lines]
        flat = [s for sentences in pieces for s in sentences]
        if not flat:
            return ""
        translated = iter(self.translate_batch(flat, direction, num_beams))
        return "\n".join(" ".join(next(translated) for _ in sentences) for sentences in pieces)

    def alternatives(self, sentence: str, direction: str = "en2nwy", n: int = 3, num_beams: int = 6) -> list[str]:
        """The n best candidate translations of one sentence, best first."""
        return self._generate([sentence], direction, num_beams, n, 128)[0]


def main():
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="sentence to translate (omit for interactive mode)")
    parser.add_argument("--direction", choices=sorted(config.DIRECTIONS), default="en2nwy")
    parser.add_argument("--adapter-dir", type=Path, default=config.MODEL_DIR)
    args = parser.parse_args()

    try:
        translator = Translator(args.adapter_dir)
    except ModelNotTrainedError as e:
        raise SystemExit(str(e))

    if args.text:
        print(translator.translate(args.text, args.direction))
        return

    direction = args.direction
    print("Type a sentence to translate. '/swap' switches direction, an empty line quits.")
    while True:
        text = input(f"[{direction}] ").strip()
        if not text:
            break
        if text == "/swap":
            direction = "nwy2en" if direction == "en2nwy" else "en2nwy"
            print(f"Direction is now {direction}")
            continue
        print(translator.translate(text, direction))


if __name__ == "__main__":
    main()

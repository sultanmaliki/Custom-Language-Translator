"""Fine-tune NLLB on the Nawayathi corpus with LoRA, in both directions at once.

    python data_preprocessing.py     # make train/val/test splits
    python train_model.py            # train, keep the best adapter in models/nawayathi-lora

Only small LoRA adapter weights are trained (a few percent of the model), which is
what makes this fit on an 8 GB GPU and keeps a tiny corpus from wrecking the
pretrained model. Every pair is used twice: English -> Nawayathi and back.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

import config
from data_preprocessing import read_split
from inference import require_language_tags

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

# GPU memory guard. Training memory is dominated by the output layer over NLLB's 256k-word vocabulary,
# so it scales with (batch size x longest target). Measured on an 8 GiB RTX 4060 Laptop (see
# docs/SYSTEM_SPECS_AND_LIMITS.md); it ignores the transformer layers, which are small next to it.
BYTES_PER_VOCAB_TOKEN = 18.3   # bf16 logits + their fp32 copy + gradients + loss, per vocab entry per target token
FIXED_OVERHEAD_GIB = 0.1
# Once activations pass ~70% of the free VRAM, Windows silently spills into system RAM (5-20x slower, no error).
SAFE_FRACTION = 0.7

# Batches have a different padded length every step. PyTorch's allocator caches memory blocks by size, and on
# Windows the driver overflows into system RAM instead of raising out-of-memory, so the allocator never learns to
# free that cache: it grows to 2-3x the VRAM and training runs 3-5x slower (measured, see the docs). Two fixes,
# together worth 3x on the default settings: pad to a multiple of PAD_MULTIPLE so batches share a few shapes,
# and cap the allocator just under the VRAM that is really free, so it frees its cache when it needs to.
PAD_MULTIPLE = 8
ALLOCATOR_SHARE_OF_FREE = 0.92


def allocator_fraction(free_gib: float, total_gib: float) -> float:
    """Share of total VRAM to let PyTorch's allocator use."""
    return ALLOCATOR_SHARE_OF_FREE * free_gib / total_gib


def estimate_activation_gib(vocab_size: int, batch_size: int, longest_target: int) -> float:
    """Approximate GPU memory (GiB) a training step needs on top of the model weights."""
    return FIXED_OVERHEAD_GIB + BYTES_PER_VOCAB_TOKEN * vocab_size * batch_size * longest_target / 2**30


def safe_batch_size(vocab_size: int, longest_target: int, free_gib: float) -> int:
    """Largest batch size whose estimated activations stay within the safe share of free memory."""
    per_example = BYTES_PER_VOCAB_TOKEN * vocab_size * longest_target / 2**30
    return max(1, int((SAFE_FRACTION * free_gib - FIXED_OVERHEAD_GIB) / per_example))


def check_gpu_memory(vocab_size: int, batch_size: int, accum: int, longest_target: int) -> None:
    """Print the memory estimate, and warn (not fail) if it is likely to spill into system RAM."""
    free_gib = torch.cuda.mem_get_info()[0] / 2**30   # free after the model is loaded
    needed = estimate_activation_gib(vocab_size, batch_size, longest_target)
    summary = (f"batch size {batch_size} x longest target {longest_target} tokens needs ~{needed:.1f} GiB "
               f"of GPU memory for activations; {free_gib:.1f} GiB is free")
    if needed <= SAFE_FRACTION * free_gib:
        print(f"GPU memory check: {summary} (OK)")
        return
    safe = safe_batch_size(vocab_size, longest_target, free_gib)
    print(f"WARNING: {summary}.\n"
          "  Training would likely stop with an out-of-memory error, or crawl as memory spills into system RAM.\n"
          f"  Try: --batch-size {safe} --accum {math.ceil(batch_size * accum / safe)} "
          "(same effective batch), or lower --max-len. See docs/SYSTEM_SPECS_AND_LIMITS.md.")


def load_lora_model(base_model: str, device: torch.device, use_amp: bool, lora_r: int = 16):
    """The frozen pretrained model (bf16 on GPU) with trainable LoRA adapters on top."""
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, dtype=torch.bfloat16 if use_amp else torch.float32)
    lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r, lora_alpha=2 * lora_r,
                      lora_dropout=0.1, target_modules=LORA_TARGETS)
    return get_peft_model(model, lora).to(device)


def make_examples(pairs, tokenizer, max_len):
    """Turn each pair into two examples (en->nwy and nwy->en). Over-long ones are dropped, not cut."""
    examples, dropped = [], 0
    for p in pairs:
        for direction, (src_tag, tgt_tag) in config.DIRECTIONS.items():
            src, tgt = (p.english, p.nawayathi) if direction == "en2nwy" else (p.nawayathi, p.english)
            tokenizer.src_lang, tokenizer.tgt_lang = src_tag, tgt_tag
            enc = tokenizer(src, text_target=tgt)
            if len(enc["input_ids"]) > max_len or len(enc["labels"]) > max_len:
                dropped += 1
                continue
            examples.append({"input_ids": enc["input_ids"], "labels": enc["labels"]})
    return examples, dropped


def make_collate(pad_id: int, decoder_start_id: int, pad_multiple: int = 1):
    def collate(batch):
        n, src_len, tgt_len = len(batch), max(len(b["input_ids"]) for b in batch), max(len(b["labels"]) for b in batch)
        src_len, tgt_len = (-(-length // pad_multiple) * pad_multiple for length in (src_len, tgt_len))
        input_ids = torch.full((n, src_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, src_len), dtype=torch.long)
        labels = torch.full((n, tgt_len), -100, dtype=torch.long)
        decoder_input_ids = torch.full((n, tgt_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ids, lab = torch.tensor(b["input_ids"]), torch.tensor(b["labels"])
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
            labels[i, : len(lab)] = lab
            # the decoder reads the target shifted right, starting from the decoder-start token
            decoder_input_ids[i, 0] = decoder_start_id
            decoder_input_ids[i, 1 : len(lab)] = lab[:-1]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids, "labels": labels}
    return collate


def batch_loss(model, batch, device, use_amp, label_smoothing=0.0, reduction="mean"):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"], use_cache=False)
    logits = out.logits.float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1),
                           ignore_index=-100, label_smoothing=label_smoothing, reduction=reduction)


@torch.no_grad()
def validation_loss(model, loader, device, use_amp):
    """Mean per-token loss over the whole validation set."""
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        tokens += (batch["labels"] != -100).sum().item()
        total += batch_loss(model, batch, device, use_amp, reduction="sum").item()
    return total / max(tokens, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--base-model", default=config.BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=4, help="stop after this many epochs without improvement")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=config.MAX_LEN)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for name in ("train", "val"):
        if not (args.data_dir / f"{name}.tsv").exists():
            raise SystemExit(f"{args.data_dir / (name + '.tsv')} not found. Run data_preprocessing.py first.")
    train_pairs = read_split(args.data_dir / "train.tsv")
    val_pairs = read_split(args.data_dir / "val.tsv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found, training on the CPU will be very slow.")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}  (bf16: {use_amp})")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(allocator_fraction(free / 2**30, total / 2**30))

    print(f"Loading {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    require_language_tags(tokenizer, (config.EN_TAG, config.NWY_TAG), args.base_model)
    model = load_lora_model(args.base_model, device, use_amp, args.lora_r)
    model.print_trainable_parameters()

    train_ex, dropped_t = make_examples(train_pairs, tokenizer, args.max_len)
    val_ex, dropped_v = make_examples(val_pairs, tokenizer, args.max_len)
    if dropped_t or dropped_v:
        print(f"Dropped {dropped_t + dropped_v} examples longer than {args.max_len} tokens.")
    print(f"Training examples: {len(train_ex)} ({len(train_pairs)} pairs x 2 directions), validation: {len(val_ex)}")
    if not train_ex:
        raise SystemExit(f"No training examples left: every sentence is longer than --max-len {args.max_len}.")
    if device.type == "cuda":
        check_gpu_memory(model.config.vocab_size, args.batch_size, args.accum,
                         max(len(e["labels"]) for e in train_ex))

    collate = make_collate(tokenizer.pad_token_id, model.config.decoder_start_token_id, PAD_MULTIPLE)
    train_loader = DataLoader(train_ex, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ex, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, stale = float("inf"), 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for i, batch in enumerate(train_loader):
            loss = batch_loss(model, batch, device, use_amp, args.label_smoothing)
            (loss / args.accum).backward()
            running, n_batches = running + loss.item(), n_batches + 1
            if (i + 1) % args.accum == 0 or i + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val = validation_loss(model, val_loader, device, use_amp)
        improved = val < best_val - 1e-4
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {running / n_batches:.3f}  "
              f"val loss {val:.3f}{'  *best*' if improved else ''}  ({time.time() - start:.0f}s)")
        if improved:
            best_val, best_epoch, stale = val, epoch, 0
            model.save_pretrained(args.output_dir)
            # written with every checkpoint so an interrupted run still leaves a usable adapter
            meta = {"base_model": args.base_model, "en_tag": config.EN_TAG, "nwy_tag": config.NWY_TAG,
                    "best_epoch": epoch, "best_val_loss": round(val, 4), "train_pairs": len(train_pairs),
                    "lora_r": args.lora_r, "max_len": args.max_len}
            (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping early.")
                break

    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB in use, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved by PyTorch")
    print(f"Done. Best epoch {best_epoch} (val loss {best_val:.3f}), adapter saved to {args.output_dir}")
    print("Next: python evaluate.py   then   python app.py")


if __name__ == "__main__":
    main()

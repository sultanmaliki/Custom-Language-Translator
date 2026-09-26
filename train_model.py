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


def make_collate(pad_id: int, decoder_start_id: int):
    def collate(batch):
        n, src_len, tgt_len = len(batch), max(len(b["input_ids"]) for b in batch), max(len(b["labels"]) for b in batch)
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

    print(f"Loading {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    require_language_tags(tokenizer, (config.EN_TAG, config.NWY_TAG), args.base_model)
    model =AutoModelForSeq2SeqLM.from_pretrained(args.base_model, dtype=torch.bfloat16 if use_amp else torch.float32)
    lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
                      lora_dropout=0.1, target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()

    train_ex, dropped_t = make_examples(train_pairs, tokenizer, args.max_len)
    val_ex, dropped_v = make_examples(val_pairs, tokenizer, args.max_len)
    if dropped_t or dropped_v:
        print(f"Dropped {dropped_t + dropped_v} examples longer than {args.max_len} tokens.")
    print(f"Training examples: {len(train_ex)} ({len(train_pairs)} pairs x 2 directions), validation: {len(val_ex)}")

    collate = make_collate(tokenizer.pad_token_id, model.config.decoder_start_token_id)
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
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"Done. Best epoch {best_epoch} (val loss {best_val:.3f}), adapter saved to {args.output_dir}")
    print("Next: python evaluate.py   then   python app.py")


if __name__ == "__main__":
    main()

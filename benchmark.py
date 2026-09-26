"""Measure what this machine can do for the translator.

    python benchmark.py specs                       # hardware and software summary
    python benchmark.py train                       # batch size x sentence length: speed and GPU memory
    python benchmark.py varlen --max-length 64      # like real training: a different length every batch
                                                    #   (add --memory-fraction 0.8 --pad-multiple 8 to mirror train_model.py)
    python benchmark.py sustained --seconds 150     # long run: checks for thermal / power throttling
    python benchmark.py infer                       # translation speed and memory (add --cpu for CPU only)

Uses synthetic batches through the same code as train_model.py, so it needs no corpus and no trained
model (the base model is downloaded on first use). Run it after a driver, hardware or library change and
compare with docs/SYSTEM_SPECS_AND_LIMITS.md. Plug the laptop in first.
"""
import argparse
import ctypes
import gc
import os
import platform
import random
import shutil
import subprocess
import sys
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
import train_model as tm
from text_normalization import use_utf8_console

GIB = 2**30


def power_source() -> str:
    """'AC power' / 'battery' (Windows only; a laptop on battery is power-limited and will benchmark worse)."""
    if sys.platform != "win32":
        return "unknown"

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("sys_flag", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full_life", ctypes.c_ulong)]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return "unknown"
    return {0: "battery", 1: "AC power"}.get(status.ac, "unknown")


def gpu_state() -> str:
    """Temperature, clock, power draw and utilisation from nvidia-smi ('' if unavailable)."""
    if not shutil.which("nvidia-smi"):
        return ""
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    try:
        temp, clock, power, util = (x.strip() for x in out.split(","))
    except ValueError:
        return ""
    return f"{temp} C, {clock} MHz, {power} W, util {util}%"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU (torch.cuda.is_available() is False).")
    return torch.device("cuda")


def warn_if_on_battery() -> None:
    if power_source() == "battery":
        print("WARNING: running on battery. Plug in for representative numbers.\n")


def specs(_args) -> None:
    print(f"Python {platform.python_version()} | torch {torch.__version__} | CUDA runtime {torch.version.cuda} "
          f"| cuDNN {torch.backends.cudnn.version()}")
    print(f"OS: {platform.platform()} | CPU: {platform.processor()} ({os.cpu_count()} threads) | power: {power_source()}")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"RAM: {vm.total / GIB:.1f} GiB total, {vm.available / GIB:.1f} GiB available now")
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {p.name} | {total / GIB:.1f} GiB VRAM, {free / GIB:.1f} GiB free now | "
              f"compute capability {p.major}.{p.minor} | {p.multi_processor_count} SMs | "
              f"bf16: {torch.cuda.is_bf16_supported()}")
        print(f"GPU now: {gpu_state()}")
    else:
        print("GPU: none visible to PyTorch")


def synthetic_batch(batch_size: int, length: int, pad_id: int = 1, decoder_start_id: int = 2):
    example = {"input_ids": list(range(1000, 1000 + length)), "labels": list(range(2000, 2000 + length))}
    return tm.make_collate(pad_id, decoder_start_id)([example] * batch_size)   # exact length: no extra padding


def train_step(model, params, optimizer, batch, device) -> None:
    loss = tm.batch_loss(model, batch, device, True, 0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def load_for_training(device):
    model = tm.load_lora_model(config.BASE_MODEL, device, use_amp=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    return model, params, torch.optim.AdamW(params, lr=1e-4)


def train_grid(args) -> None:
    device = require_cuda()
    warn_if_on_battery()
    available_gib = torch.cuda.mem_get_info()[0] / GIB   # what the GPU can physically give us
    model, params, optimizer = load_for_training(device)
    weights_gib = torch.cuda.memory_allocated() / GIB
    free_gib = torch.cuda.mem_get_info()[0] / GIB
    vocab = model.config.vocab_size
    print(f"{available_gib:.2f} GiB of VRAM available; model weights + LoRA take {weights_gib:.2f} GiB, "
          f"leaving {free_gib:.2f} GiB for training. Sizes below are in GiB.")
    print("peak = most memory in use, reserved = most the allocator took from the GPU, est = the memory "
          "guard's estimate (should track peak).")
    print("'spilling' = reserved exceeds what the GPU had, so Windows overflowed into system RAM (very slow). "
          "'guard warns' = train_model.py would warn. Each length stops at the first size that spills or fails.\n")
    print(f"{'length':>6} {'batch':>5} | {'ex/s':>7} {'s/step':>7} {'peak':>6} {'reserved':>8} {'est':>6}")
    for length in args.lengths:
        for batch_size in args.batches:
            batch = synthetic_batch(batch_size, length)
            gc.collect()
            torch.cuda.empty_cache()   # so 'reserved' reflects only this cell, not blocks cached by earlier shapes
            torch.cuda.reset_peak_memory_stats()
            estimate = weights_gib + tm.estimate_activation_gib(vocab, batch_size, length)
            try:
                for _ in range(2):   # warm-up
                    train_step(model, params, optimizer, batch, device)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(args.steps):
                    train_step(model, params, optimizer, batch, device)
                torch.cuda.synchronize()
                seconds = (time.time() - start) / args.steps
            except torch.cuda.OutOfMemoryError:
                print(f"{length:>6} {batch_size:>5} | out of memory", flush=True)
                optimizer.zero_grad(set_to_none=True)
                del batch
                gc.collect()
                torch.cuda.empty_cache()
                break
            peak = torch.cuda.max_memory_allocated() / GIB
            reserved = torch.cuda.max_memory_reserved() / GIB
            spilling = reserved > available_gib
            notes = []
            if spilling:
                notes.append("spilling")
            if tm.estimate_activation_gib(vocab, batch_size, length) > tm.SAFE_FRACTION * free_gib:
                notes.append("guard warns")
            print(f"{length:>6} {batch_size:>5} | {batch_size / seconds:7.1f} {seconds:7.3f} {peak:6.2f} {reserved:8.2f} "
                  f"{estimate:6.2f}   {', '.join(notes)}", flush=True)
            if spilling:
                break


def varlen(args) -> None:
    """Like real training: every batch has a different padded length, which fragments the allocator's
    cache and pushes 'reserved' memory above what fixed-size batches need."""
    device = require_cuda()
    warn_if_on_battery()
    available_gib = torch.cuda.mem_get_info()[0] / GIB
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    model, params, optimizer = load_for_training(device)
    weights_gib = torch.cuda.memory_allocated() / GIB
    free_gib = torch.cuda.mem_get_info()[0] / GIB
    rng = random.Random(0)
    lengths = [rng.randint(args.min_length, args.max_length) for _ in range(args.steps + 3)]
    if args.pad_multiple > 1:   # fewer distinct shapes -> the allocator can reuse its cached blocks
        lengths = [-(-length // args.pad_multiple) * args.pad_multiple for length in lengths]
    for length in lengths[:3]:   # warm-up
        train_step(model, params, optimizer, synthetic_batch(args.batch_size, length), device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for length in lengths[3:]:
        train_step(model, params, optimizer, synthetic_batch(args.batch_size, length), device)
    torch.cuda.synchronize()
    seconds = time.time() - start
    peak = torch.cuda.max_memory_allocated() / GIB
    reserved = torch.cuda.max_memory_reserved() / GIB
    worst = weights_gib + tm.estimate_activation_gib(model.config.vocab_size, args.batch_size, args.max_length)
    notes = []
    if reserved > available_gib:
        notes.append("spilling")
    if tm.estimate_activation_gib(model.config.vocab_size, args.batch_size, args.max_length) > tm.SAFE_FRACTION * free_gib:
        notes.append("guard warns")
    print(f"batch {args.batch_size}, lengths {args.min_length}-{args.max_length} ({args.steps} steps): "
          f"{args.batch_size * args.steps / seconds:.1f} ex/s | peak {peak:.2f} GiB, reserved {reserved:.2f} GiB "
          f"(GPU had {available_gib:.2f}), worst-case estimate {worst:.2f} GiB  {', '.join(notes)}")


def sustained(args) -> None:
    device = require_cuda()
    warn_if_on_battery()
    model, params, optimizer = load_for_training(device)
    batch = synthetic_batch(args.batch_size, args.length)
    for _ in range(3):
        train_step(model, params, optimizer, batch, device)
    print(f"Batch {args.batch_size} x length {args.length} for {args.seconds} s. Start: {gpu_state()}")
    print("Steady throughput and clocks mean no throttling; falling throughput or clocks mean it is throttling.")
    end, steps = time.time() + args.seconds, 0
    torch.cuda.synchronize()
    window_start = time.time()
    while time.time() < end:
        train_step(model, params, optimizer, batch, device)
        steps += 1
        if time.time() - window_start >= args.window:
            torch.cuda.synchronize()
            now = time.time()
            print(f"  {args.batch_size * steps / (now - window_start):6.1f} ex/s | {gpu_state()}", flush=True)
            steps, window_start = 0, time.time()


def infer(args) -> None:
    device = torch.device("cpu") if args.cpu else require_cuda()
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    batches = args.batches or ([1, 16] if args.cpu else [1, 16, 64])
    repeats = args.repeats or (1 if args.cpu else 3)
    if not args.cpu:
        warn_if_on_battery()

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL, dtype=dtype).to(device).eval()
    if device.type == "cuda":
        memory = f"{torch.cuda.memory_allocated() / GIB:.2f} GiB VRAM"
    else:
        try:
            import psutil
            memory = f"{psutil.Process().memory_info().rss / GIB:.2f} GiB RAM"
        except ImportError:
            memory = "RAM not measured (pip install psutil)"
    print(f"Base model loaded on {device.type} in {time.time() - start:.1f} s, {memory}.")
    print(f"Generating exactly {args.length} tokens per sentence with beam search 5, so the cost per token "
          "doesn't depend on what the model says.\n")

    tokenizer.src_lang = config.EN_TAG
    forced_bos = tokenizer.convert_tokens_to_ids(config.NWY_TAG)
    decoder_length = args.length + 2   # + decoder-start token + language tag
    print(f"{'batch':>5} | {'seconds':>8} {'sentences/s':>12} {'ms/step':>8}" + (f" {'peak GiB':>9}" if device.type == "cuda" else ""))
    for batch_size in batches:
        enc = tokenizer(["How are you doing today, my friend?"] * batch_size, return_tensors="pt", padding=True).to(device)
        options = dict(forced_bos_token_id=forced_bos, num_beams=5, min_length=decoder_length, max_length=decoder_length)
        with torch.no_grad():
            model.generate(**enc, **options)   # warm-up
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            for _ in range(repeats):
                model.generate(**enc, **options)
            if device.type == "cuda":
                torch.cuda.synchronize()
        seconds = (time.time() - start) / repeats
        line = f"{batch_size:>5} | {seconds:8.2f} {batch_size / seconds:12.1f} {seconds / args.length * 1000:8.0f}"
        if device.type == "cuda":
            line += f" {torch.cuda.max_memory_allocated() / GIB:9.2f}"
        print(line, flush=True)


def main() -> None:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("specs", help="hardware and software summary").set_defaults(run=specs)

    p = sub.add_parser("train", help="batch size x sentence length grid")
    p.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 96])
    p.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    p.add_argument("--steps", type=int, default=5, help="timed steps per cell")
    p.set_defaults(run=train_grid)

    p = sub.add_parser("varlen", help="training with a different length every batch, like the real thing")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-length", type=int, default=8)
    p.add_argument("--max-length", type=int, default=48)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--memory-fraction", type=float, help="cap the allocator at this share of total VRAM")
    p.add_argument("--pad-multiple", type=int, default=1, help="round every length up to a multiple of this")
    p.set_defaults(run=varlen)

    p = sub.add_parser("sustained", help="long training run to check for throttling")
    p.add_argument("--seconds", type=int, default=150)
    p.add_argument("--window", type=int, default=15, help="seconds between readings")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--length", type=int, default=48)
    p.set_defaults(run=sustained)

    p = sub.add_parser("infer", help="translation speed and memory")
    p.add_argument("--cpu", action="store_true", help="force CPU (a fallback for when the GPU is busy)")
    p.add_argument("--batches", type=int, nargs="+")
    p.add_argument("--length", type=int, default=30, help="tokens generated per sentence")
    p.add_argument("--repeats", type=int)
    p.set_defaults(run=infer)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

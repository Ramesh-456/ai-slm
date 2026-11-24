#!/usr/bin/env python3
"""QLoRA / LoRA supervised fine-tuning of a small Qwen model on the LoomNest tool-calling dataset.

Heavy libraries (torch, transformers, peft, trl, bitsandbytes) are imported lazily, so `--help` and `--dry-run`
work on a machine with none of them. Real training needs an NVIDIA GPU (free Colab T4 is enough for the 1.5B model);
see README.md. Loss is computed on assistant turns only (tool calls and final answers), not on the system prompt,
tool schema or tool results.

STATUS: written against the documented transformers/peft/trl APIs but NOT executed in the authoring environment
(no GPU). Run `--dry-run` first, then a short `--max-steps 20` smoke run on the target GPU.
"""
import argparse
import hashlib
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

NO_GPU_HELP = """\
No CUDA GPU detected - QLoRA training needs one. Options:
  * Google Colab (free T4): Runtime > Change runtime type > T4 GPU, then in a cell:
        !git clone <your repo> && cd <repo>/Backend/ai-slm   (or upload the ai-slm folder)
        !pip install -r requirements.txt
        !python generate_dataset.py --n 3000
        !python train_lora.py --output-dir out/loomnest-shop-lora
        # download out/loomnest-shop-lora (the adapter) when it finishes
  * Any Linux box / cloud VM with an NVIDIA GPU with >= 8 GB VRAM (1.5B model, 4-bit).
Use --dry-run to validate the data and configuration without a GPU."""


# ----------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit tested, no ML libraries)
# ----------------------------------------------------------------------------------------------------------------------

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_messages(messages):
    """HF chat templates expect tool_call arguments as objects; accept OpenAI-style JSON strings too."""
    out = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
                call = json.loads(json.dumps(call))
                args = call["function"].get("arguments")
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args)
                calls.append(call)
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def render_chatml_approx(messages, tools):
    """Approximation of the Qwen2.5 chat template (used for --dry-run and tests; training uses the tokenizer's own)."""
    out = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            text = msg["content"]
            if tools:
                text += ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                         "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                         + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
                         + "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
                           "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
                           "\"arguments\": <args-json-object>}\n</tool_call>")
            out.append("%ssystem\n%s%s\n" % (IM_START, text, IM_END))
        elif role == "user":
            out.append("%suser\n%s%s\n" % (IM_START, msg["content"], IM_END))
        elif role == "assistant":
            body = msg.get("content") or ""
            if body:
                body = "\n" + body
            for call in msg.get("tool_calls") or []:
                fn = call["function"]
                body += "\n<tool_call>\n%s\n</tool_call>" % json.dumps({"name": fn["name"], "arguments": fn["arguments"]},
                                                                       ensure_ascii=False)
            out.append("%sassistant%s%s\n" % (IM_START, body, IM_END))
        elif role == "tool":
            out.append("%suser\n<tool_response>\n%s\n</tool_response>%s\n" % (IM_START, msg["content"], IM_END))
    return "".join(out)


def assistant_spans(text):
    """Character spans (start, end) that the loss should cover: assistant content incl. its closing <|im_end|>."""
    spans = []
    header = IM_START + "assistant\n"
    pos = 0
    while True:
        start = text.find(header, pos)
        if start < 0:
            return spans
        begin = start + len(header)
        end = text.find(IM_END, begin)
        if end < 0:
            spans.append((begin, len(text)))
            return spans
        spans.append((begin, end + len(IM_END)))
        pos = end + len(IM_END)


def labels_from_offsets(input_ids, offsets, spans):
    """-100 for tokens outside the spans; the token id itself inside them."""
    labels = []
    for token_id, (s, e) in zip(input_ids, offsets):
        inside = any(s >= a and e <= b and e > s for a, b in spans)
        labels.append(token_id if inside else -100)
    return labels


def dataset_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:16]


def filter_kwargs(fn, kwargs):
    """Drops keyword arguments the installed library version does not accept (APIs drift between releases)."""
    accepted = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF model id; alternative: Qwen/Qwen2.5-3B-Instruct (needs ~12-16 GB VRAM)")
    p.add_argument("--train-file", default=os.path.join(HERE, "data", "train.jsonl"))
    p.add_argument("--val-file", default=os.path.join(HERE, "data", "val.jsonl"))
    p.add_argument("--tools", default=os.path.join(HERE, "data", "tools_schema.json"))
    p.add_argument("--output-dir", default=os.path.join(HERE, "out", "loomnest-shop-lora"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="override epochs for a quick smoke run (e.g. 20)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank r (alpha = 2 * r)")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=4608, help="examples longer than this are dropped; measured with the Qwen2.5 tokenizer: ~3.3k tokens of every example are the constant system prompt + 24 tool schemas, p95 3.7k, max 4.2k")
    p.add_argument("--batch-size", type=int, default=1, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--no-4bit", action="store_true", help="plain LoRA in 16-bit instead of 4-bit QLoRA (needs more VRAM)")
    p.add_argument("--trainer", choices=("auto", "sft", "hf"), default="auto",
                   help="auto = TRL SFTTrainer, falling back to transformers.Trainer if the installed TRL API differs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="validate data/config and print the plan; imports no ML library")
    return p.parse_args(argv)


def dry_run(args):
    for label, path in (("train", args.train_file), ("val", args.val_file), ("tools", args.tools)):
        if not os.path.exists(path):
            print("missing %s file: %s\n  -> run: python generate_dataset.py" % (label, path), file=sys.stderr)
            return 2
    train, val = load_rows(args.train_file), load_rows(args.val_file)
    with open(args.tools, "r", encoding="utf-8") as fh:
        tools = json.load(fh)
    sample = render_chatml_approx(normalize_messages(train[0]["messages"]), tools)
    spans = assistant_spans(sample)
    trained = sum(b - a for a, b in spans)
    lengths = [len(render_chatml_approx(normalize_messages(r["messages"]), tools)) for r in train[:200]]
    est_tokens = sum(lengths) / len(lengths) / 4.0  # ~4 chars/token measured on this rendering (Qwen2.5 tokenizer)
    steps_per_epoch = max(1, len(train) // (args.batch_size * args.grad_accum))
    print("DRY RUN - nothing was loaded or trained")
    print("base model       : %s (%s)" % (args.base_model, "LoRA 16-bit" if args.no_4bit else "QLoRA 4-bit nf4"))
    print("train / val rows : %d / %d | tools in schema: %d" % (len(train), len(val), len(tools)))
    print("dataset digest   : %s" % dataset_digest([args.train_file, args.val_file, args.tools]))
    print("LoRA             : r=%d alpha=%d dropout=%.2f, targets q/k/v/o/gate/up/down" % (args.rank, 2 * args.rank, args.dropout))
    print("optim            : lr=%g, epochs=%s, per-device batch=%d x grad-accum=%d (effective %d)" % (
        args.lr, args.epochs if args.max_steps < 0 else "max_steps=%d" % args.max_steps, args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum))
    print("approx steps     : %d per epoch" % steps_per_epoch)
    print("approx tokens/ex : %d (max-seq-len %d; longer examples are dropped)" % (est_tokens, args.max_seq_len))
    print("loss masking     : %d assistant span(s) in the first example cover %d of %d chars" % (len(spans), trained, len(sample)))
    print("output           : %s" % args.output_dir)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# Training (lazy imports)
# ----------------------------------------------------------------------------------------------------------------------

def build_dataset(tokenizer, rows, tools, max_len, name):
    from datasets import Dataset  # noqa: WPS433 (lazy import)

    features, dropped = [], 0
    for row in rows:
        messages = normalize_messages(row["messages"])
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) > max_len:
            dropped += 1
            continue
        labels = labels_from_offsets(enc["input_ids"], enc["offset_mapping"], assistant_spans(text))
        if not any(l != -100 for l in labels):
            dropped += 1
            continue
        features.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})
    print("%s: %d examples tokenized, %d dropped (too long / no assistant tokens)" % (name, len(features), dropped))
    if not features:
        raise SystemExit("no usable training examples - raise --max-seq-len?")
    return Dataset.from_list(features)


def train(args):
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. pip install -r requirements.txt (see README.md).", file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print(NO_GPU_HELP, file=sys.stderr)
        return 3

    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    for path in (args.train_file, args.val_file, args.tools):
        if not os.path.exists(path):
            print("missing %s - run python generate_dataset.py first" % path, file=sys.stderr)
            return 2
    transformers.set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open(args.tools, "r", encoding="utf-8") as fh:
        tools = json.load(fh)
    train_ds = build_dataset(tokenizer, load_rows(args.train_file), tools, args.max_seq_len, "train")
    val_ds = build_dataset(tokenizer, load_rows(args.val_file), tools, args.max_seq_len, "val")

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dtype)
    load_kwargs = dict(quantization_config=quant, device_map={"": 0})
    try:  # transformers >= 4.56 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, **load_kwargs)
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    base_args = dict(
        output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch", save_total_limit=2, bf16=bf16, fp16=not bf16,
        optim="paged_adamw_8bit" if quant is not None else "adamw_torch", gradient_checkpointing=True,
        report_to="none", remove_unused_columns=False, seed=args.seed,
    )
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    base_args[strategy_key] = "epoch"
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)

    trainer = None
    if args.trainer in ("auto", "sft"):
        try:
            from trl import SFTConfig, SFTTrainer
            cfg = SFTConfig(**filter_kwargs(SFTConfig.__init__, dict(
                base_args, dataset_kwargs={"skip_prepare_dataset": True}, max_length=args.max_seq_len,
                max_seq_length=args.max_seq_len, packing=False)))
            kwargs = dict(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
            params = inspect.signature(SFTTrainer.__init__).parameters
            kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
            trainer = SFTTrainer(**kwargs)
            print("using trl.SFTTrainer")
        except Exception as exc:  # noqa: BLE001 - TRL's API changes often; fall back instead of failing the run
            if args.trainer == "sft":
                raise
            print("SFTTrainer unavailable with this TRL version (%s: %s); falling back to transformers.Trainer"
                  % (type(exc).__name__, exc))
    if trainer is None:
        trainer = Trainer(model=model, args=TrainingArguments(**filter_kwargs(TrainingArguments.__init__, base_args)),
                          train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
        print("using transformers.Trainer")

    trainer.train()
    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"base_model": args.base_model, "rank": args.rank, "lr": args.lr, "epochs": args.epochs,
                   "max_seq_len": args.max_seq_len, "4bit": not args.no_4bit,
                   "dataset_digest": dataset_digest([args.train_file, args.val_file, args.tools])}, fh, indent=2)
    print("\nadapter saved to %s\nnext: python merge_adapter.py --base-model %s --adapter %s --out out/merged"
          % (adapter_dir, args.base_model, adapter_dir))
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":
    sys.exit(main())

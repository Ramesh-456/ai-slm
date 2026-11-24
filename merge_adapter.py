#!/usr/bin/env python3
"""Merges a trained LoRA adapter into the base model so it can be converted to GGUF (lazy ML imports).

Usage: python merge_adapter.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter out/loomnest-shop-lora/adapter --out out/merged
The base model is loaded in 16-bit (not 4-bit) so the merged weights are clean; ~4 GB RAM for the 1.5B model, CPU is fine.
STATUS: not executed in the authoring environment (no model weights / GPU).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "out", "loomnest-shop-lora", "adapter"))
    p.add_argument("--out", default=os.path.join(HERE, "out", "merged"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.dry_run:
        print("would merge adapter %s into %s and save to %s" % (args.adapter, args.base_model, args.out))
        return 0
    if not os.path.isdir(args.adapter):
        print("adapter folder not found: %s (run train_lora.py first)" % args.adapter, file=sys.stderr)
        return 2
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:  # transformers >= 4.56 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.out)
    print("merged model saved to %s\nnext: python <llama.cpp>/convert_hf_to_gguf.py %s --outfile out/loomnest-shop-slm.f16.gguf --outtype f16"
          % (args.out, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

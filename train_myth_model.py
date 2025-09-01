#!/usr/bin/env python3
"""
Train a small causal LM (default: GPT-2) on a recursive text corpus
split across two roots: myth and fate. Handles hundreds of nested
.txt files, tags each sample with a soft [SOURCE: ...] header, and
packs sequences to fixed block_size for efficient training.

Run (inside venv):
  venv/bin/python ~/Siegfried/ai_training/train_myth_model.py \
    --myth_dir ~/Siegfried/myth \
    --fate_dir ~/Siegfried/fate \
    --output_dir ~/Siegfried/ai_training/models/siegfried-myth-fate-gpt2 \
    --model_name gpt2 \
    --epochs 3 \
    --block_size 512 \
    --batch_size 4

Requires:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install transformers datasets accelerate
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)


def gather_txt_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.txt") if p.is_file()]


def read_file_text(path: Path) -> str:
    # Robust read: ignore undecodable bytes, preserve line breaks
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_dataset(myth_dir: Path, fate_dir: Path, tag_sources: bool = True) -> Dataset:
    myth_files = gather_txt_files(myth_dir)
    fate_files = gather_txt_files(fate_dir)

    if not myth_files and not fate_files:
        raise RuntimeError("No .txt files found under myth_dir or fate_dir.")

    samples: List[Dict[str, str]] = []

    for p in myth_files:
        txt = read_file_text(p)
        if tag_sources:
            txt = f"[SOURCE: MYTH]\n{txt}"
        samples.append({"text": txt})

    for p in fate_files:
        txt = read_file_text(p)
        if tag_sources:
            txt = f"[SOURCE: FATE]\n{txt}"
        samples.append({"text": txt})

    if len(samples) == 0:
        raise RuntimeError("No valid samples constructed from provided folders.")

    ds = Dataset.from_list(samples)
    return ds


def tokenize_and_chunk(ds: Dataset, tokenizer, block_size: int, num_proc: int = 1) -> Dict[str, Dataset]:
    # Tokenize all texts
    def tok(examples):
        return tokenizer(examples["text"])  # returns input_ids + attention_mask

    tokenized = ds.map(
        tok,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc,
    )

    # Concatenate then split into fixed-size blocks
    def group_texts(examples):
        # Concatenate across batch
        concatenated = {}
        for k in examples.keys():
            concatenated[k] = sum(examples[k], [])
        total_length = len(concatenated["input_ids"])  # tokens
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        result = {
            "input_ids": [
                concatenated["input_ids"][i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
            "attention_mask": [
                concatenated["attention_mask"][i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
        }
        # LM labels = input_ids shifted internally by model
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
    )

    # small validation split from tail of dataset
    # Using train_test_split could shuffle samples; here we just slice deterministically
    # to avoid expensive shuffles on huge corpora.
    n = len(lm_datasets)
    val_size = max(1, n // 50)  # ~2%
    train_dataset = lm_datasets.select(range(0, n - val_size)) if n > val_size else lm_datasets
    eval_dataset = lm_datasets.select(range(n - val_size, n)) if n > val_size else lm_datasets.select([])

    return {"train": train_dataset, "eval": eval_dataset}


def main():
    parser = argparse.ArgumentParser(description="Train Siegfried myth+fate causal LM")
    parser.add_argument("--myth_dir", type=str, required=True)
    parser.add_argument("--fate_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 (Ampere+)")
    parser.add_argument("--num_proc", type=int, default=1, help="Parallel map workers for tokenization/grouping")
    args = parser.parse_args()

    set_seed(args.seed)

    myth_dir = Path(args.myth_dir).expanduser()
    fate_dir = Path(args.fate_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Gathering .txt from: myth={myth_dir} fate={fate_dir}")
    raw_ds = build_dataset(myth_dir, fate_dir, tag_sources=True)
    print(f"[+] Raw samples: {len(raw_ds)}")

    print(f"[+] Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # GPT-2 has no pad token; map pad->eos
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # Resize embeddings if tokenizer changed
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # CUDA setup hints
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            model = model.cuda()
        except Exception as e:
            print(f"[!] CUDA move failed: {e}")

    print("[+] Tokenizing and packing...")
    lm_splits = tokenize_and_chunk(raw_ds, tokenizer, args.block_size, num_proc=args.num_proc)
    train_dataset = lm_splits["train"]
    eval_dataset = lm_splits["eval"]
    print(f"[+] Packed sequences: train={len(train_dataset)} eval={len(eval_dataset)} block_size={args.block_size}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    fp16_flag = bool(args.fp16)
    bf16_flag = bool(args.bf16) and hasattr(torch.cuda, "is_available") and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        report_to="none",
        fp16=fp16_flag,
        bf16=bf16_flag,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        do_eval=(len(eval_dataset) > 0),
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        tokenizer=tokenizer,
    )

    print("[+] Training start")
    trainer.train()
    print("[+] Training complete")

    print("[+] Saving final model & tokenizer")
    trainer.save_model()
    tokenizer.save_pretrained(str(out_dir))

    # Quick sample generation to sanity-check vibe
    try:
        from transformers import pipeline
        gen = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        prompt = "Siegfried stood before the dragon, sword drawn, and said:"
        out = gen(prompt, max_length=80, do_sample=True, top_p=0.9, temperature=0.9, num_return_sequences=1)[0]["generated_text"]
        print("\n[+] SAMPLE:\n" + out + "\n")
    except Exception as e:
        print(f"[!] Sample generation skipped: {e}")


if __name__ == "__main__":
    main()

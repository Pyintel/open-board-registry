#!/usr/bin/env python3
"""
PyIntel Embedded Hardware Architect (SFT Fine-Tuning Engine)
Fine-tunes a compact LLM (e.g. Qwen2.5-0.5B-Instruct or Gemma-2-2B-it) on
pyintel/embedded-hardware-cot (10,000 physical constraint reasoning pairs).

Uses PEFT (LoRA) for memory-efficient and parameter-efficient fine-tuning,
optimized for multi-core CPUs (48 Xeon threads) and CUDA GPUs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from huggingface_hub import HfApi, create_repo

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HF_REPO_ID = "pyintel/embedded-architect-0.5b"

def load_hardware_dataset(parquet_path: Path, max_samples: Optional[int] = None) -> Dataset:
    """Loads and formats hardware CoT samples for instruction fine-tuning."""
    print(f"📖 Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} training samples across domains: {df['domain'].unique().tolist()}")
    
    # Hugging Face dataset format
    raw_ds = Dataset.from_pandas(df)
    return raw_ds


def prepare_tokenized_dataset(dataset: Dataset, tokenizer, max_seq_length: int = 1024) -> Dataset:
    """Tokenizes instruction and CoT text for causal language modeling."""
    def tokenize_func(batch):
        instructions = batch["instruction"]
        reasonings = batch["reasoning"]
        responses = batch["response"]
        
        texts = []
        for inst, reason, resp in zip(instructions, reasonings, responses):
            # Native chat formatting with reasoning chain
            messages = [
                {"role": "system", "content": "You are PyIntel Embedded Hardware Architect, an expert systems engineer. Analyze physical constraints, verify memory budgets, eliminate incompatible microcontrollers, and provide verifiable hardware recommendations."},
                {"role": "user", "content": inst},
                {"role": "assistant", "content": f"<|channel>thought\n{reason}<channel|>\n{resp}"}
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(formatted_text)
            
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("⚡ Tokenizing dataset...")
    tokenized_ds = dataset.map(
        tokenize_func,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing hardware CoT samples"
    )
    return tokenized_ds


def run_training(
    data_path: Path,
    output_dir: Path,
    base_model_id: str = DEFAULT_MODEL_ID,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_samples: Optional[int] = None,
    push_to_hub: bool = False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training Pipeline initialized on {device} (Available CPUs: {os.cpu_count()})")
    
    # 1. Load Tokenizer & Model
    print(f"📦 Loading base model and tokenizer: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 2. Configure LoRA (PEFT)
    print("🧠 Configuring LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # 3. Prepare Dataset
    raw_ds = load_hardware_dataset(data_path, max_samples=max_samples)
    split_ds = raw_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = prepare_tokenized_dataset(split_ds["train"], tokenizer)
    eval_ds = prepare_tokenized_dataset(split_ds["test"], tokenizer)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=["none"],
        dataloader_num_workers=2 if os.cpu_count() > 4 else 0
    )
    
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator
    )
    
    print("\n⚡ Starting Fine-Tuning Execution...")
    train_result = trainer.train()
    print("✅ Training finished successfully!")
    print(f"Train loss: {train_result.training_loss:.4f}")
    
    # 5. Save Model and Tokenizer
    final_output = output_dir / "final_adapter"
    final_output.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"💾 LoRA weights and tokenizer saved to {final_output}")
    
    # 6. Upload to Hugging Face Hub if requested
    token = os.environ.get("HF_TOKEN")
    if push_to_hub and token:
        print(f"📡 Publishing fine-tuned adapter to Hugging Face ({HF_REPO_ID})...")
        try:
            create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True, token=token)
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(final_output),
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"Release: PyIntel Embedded Architect LoRA weights ({base_model_id})"
            )
            print(f"🎉 Model is live (Private) at: https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"HF upload error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train PyIntel Embedded Hardware Architect")
    parser.add_argument("--data", type=Path, default=Path("embedded_hardware_cot.parquet"), help="Path to dataset Parquet")
    parser.add_argument("--output", type=Path, default=Path("checkpoints_hw_sft"), help="Output directory")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap dataset size for quick experiments")
    parser.add_argument("--push-to-hub", action="store_true", help="Push final adapter to Hugging Face")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        output_dir=args.output,
        base_model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub
    )


if __name__ == "__main__":
    main()

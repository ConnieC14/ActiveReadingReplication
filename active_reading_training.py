#!/usr/bin/env python3
"""
Active Reading Fine-tuning Replication
dataset: https://huggingface.co/datasets/facebook/meta-active-reading
model: https://huggingface.co/facebook/meta-wiki-expert/tree/main

Target: 10-20 Wikipedia articles, 5,000 training steps
Model:  meta-llama/Llama-3.2-1B  (QLoRA 4-bit)
GPU:    RTX 3060 (12 GB) locally  |  A100/H100 in cloud

Usage:
# Local RTX 3060
python train_active_reading.py --device local

# Cloud (A100/H100) — unlocks larger batch, more precision
python train_active_reading.py --device cloud

# Resume from checkpoint
python train_active_reading.py --resume_from_checkpoint ./active-reading-medium/checkpoint-2000


Example (5 steps, 5 examples)
python active_reading_training.py --smoke_test
"""
import argparse
from huggingface_hub import HfFileSystem, login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

import pandas as pd
import torch
import ast
import json
import os
import re
from datetime import datetime
import requests
import torch
from dotenv import load_dotenv
import wandb

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN      = os.getenv("HF_TOKEN")
# login(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    choices=["local", "cloud"],
    default="local",
    help="Hardware profile. 'local'=RTX 3060, 'cloud'=A100/H100",
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Path to a trainer checkpoint directory to resume from",
)
parser.add_argument(
    "--smoke_test",
    action="store_true",
    help="Run 5-step smoke test on 5 examples then exit",
)
parser.add_argument(
    "--base_model",
    type=str,
    default="meta-llama/Llama-3.2-1B",
    help="Base model for fine-tuning (default: meta-llama/Llama-3.2-1B)",
)
parser.add_argument(
    "--num_articles",
    type=int,
    default=3,
    help="Number of Wikipedia articles to train on < 15 (default: 15)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=f"./active-reading-finetune-{datetime.now().strftime('%Y%m%d-%H%M')}",
    help="Directory for checkpoints and final adapter",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="ActiveReading",
    help="W&B project name",
)

args = parser.parse_args()
PRETRAIN_RATIO = 10 # 10% of training data from DCLM
WANDB_PROJECT  = args.wandb_project
smoke_test = args.smoke_test
OUTPUT_DIR = args.output_dir

# Active Reading limit for chunk files, None for all chunks
AR_CHUNK_FILES_LIMIT = args.num_articles  # None for all files in active-reading dataset (~11.3TB)

# SimpleQA CSV — downloaded at runtime to extract target Wikipedia titles
SIMPLEQA_CSV_URL  = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SIMPLEQA_CSV_PATH = "./simple_qa_test_set.csv"   # cached locally after first download

# Hardware-specific hyperparameters
if args.device == "local":
    # RTX 3060: 12 GB VRAM, no bfloat16, limited batch
    PER_DEVICE_BATCH   = 2
    GRAD_ACCUM         = 8
    MAX_SEQ_LEN        = 512
    FP16               = False   
    BF16               = False   
    COMPUTE_DTYPE      = torch.float16
    SAVE_STEPS         = 500
    LOGGING_STEPS      = 25
else:
    # Cloud A100/H100: large VRAM, bfloat16 native
    PER_DEVICE_BATCH   = 8
    GRAD_ACCUM         = 16
    MAX_SEQ_LEN        = 1024     # paper used 4096
    FP16               = False
    BF16               = True
    COMPUTE_DTYPE      = torch.bfloat16
    SAVE_STEPS         = 500
    LOGGING_STEPS      = 25

# Expert Domain Setting Hyperparameters
BASE_MODEL    = args.base_model
LEARNING_RATE = 1e-5
MAX_STEPS     = 5 if args.smoke_test else 5000
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
WARMUP_RATIO  = 0.03
WARMUP_STEPS = max(1, int(MAX_STEPS * WARMUP_RATIO))
GRADIENT_CHECKPOINTING = True

print(f"  Active Reading Training — {'TESTING' if args.smoke_test else 'TESTING AT LARGER SCALE'}")
print(f"  Device profile : {args.device.upper()}")
print(f"  Base model     : {BASE_MODEL}")
print(f"  Max steps      : {MAX_STEPS}")
print(f"  Effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"  Max seq len    : {MAX_SEQ_LEN}")
print(f"  Output dir     : {args.output_dir}")

def download_simpleqa_csv():
    if os.path.exists(SIMPLEQA_CSV_PATH):
        print(f"Using cached SimpleQA CSV: {SIMPLEQA_CSV_PATH}")
        return
    print(f"Downloading SimpleQA CSV from {SIMPLEQA_CSV_URL} ...")
    r = requests.get(SIMPLEQA_CSV_URL, timeout=60)
    r.raise_for_status()
    with open(SIMPLEQA_CSV_PATH, "wb") as f:
        f.write(r.content)
    print(f"  Saved to {SIMPLEQA_CSV_PATH}")

def extract_target_titles():
    """
    Parse SimpleQA CSV and return the set of Wikipedia article titles
    referenced by at least one question (the SimpleWikiQA subset).
    Title is extracted from the en.wikipedia.org/wiki/<Title> URL pattern.
    """
    df = pd.read_csv(SIMPLEQA_CSV_PATH)
    print(f"Total SimpleQA questions: {len(df)}")

    titles = set()
    for _, row in df.iterrows():
        try:
            meta = ast.literal_eval(row["metadata"])
        except Exception:
            continue
        for url in meta.get("urls", []):
            m = re.search(r"en\.wikipedia\.org/wiki/(.+?)(?:\?|#|$)", url)
            if m:
                title = requests.utils.unquote(m.group(1)).replace("_", " ")
                titles.add(title)

    print(f"Unique Wikipedia article titles in SimpleWikiQA: {len(titles)}")
    return titles

def format_ar_example(row):
    """
    Convert one row from facebook/meta-active-reading into a plain-text
    training string.
    """
    messages = row["request"]["messages"]
    response = row["response"]["text"]

    prompt_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        prompt_parts.append(f"### {role}\n{content}")

    prompt = "\n\n".join(prompt_parts)
    full_text = f"{prompt}\n\n### ASSISTANT\n{response}"
    return {"text": full_text}


def load_ar_dataset(target_titles):
    """
    Stream facebook/meta-active-reading chunk files, keep only rows whose
    metadata["title"] is in target_titles, format each row, return a Dataset.
    """
    print("\nScanning facebook/meta-active-reading chunk files ...")
    fs    = HfFileSystem()
    files = fs.glob("datasets/facebook/meta-active-reading/chunk.*.jsonl")
    print(f"  Total chunk files available: {len(files)}")

    if AR_CHUNK_FILES_LIMIT:
        files = files[:AR_CHUNK_FILES_LIMIT]
        print(f"  Limited to first {AR_CHUNK_FILES_LIMIT} chunk files for this run")

    ds = load_dataset(
        "json",
        data_files={"train": [f"hf://{f}" for f in files]},
        split="train",
    )
    print(f"  Rows loaded before filter: {len(ds)}")

    ds_filtered = ds.filter(
        lambda x: x["metadata"]["title"] in target_titles,
        desc="Filtering to SimpleWikiQA titles",
    )
    print(f"Found {len(ds_filtered)} rows across {len(files)} chunks")

    return ds_filtered

def build_pretrain_mix(n_target):
    """
    Stream n_target examples from mlfoundations/dclm-baseline-1.0.

    DCLM is a large web-crawl corpus with no domain structure — we take the
    first n_target rows with no title filtering. mix in 10% of pre-training data from DCLM
    """
    if n_target <= 0:
        return Dataset.from_dict({"text": []})

    print(f"\nLoading {n_target} DCLM pre-training examples ...")
    ds = load_dataset(
        "mlfoundations/dclm-baseline-1.0",
        split="train",
        streaming=True,
    )

    examples = []
    for row in tqdm(ds, total=n_target, desc="Loading DCLM examples"):
        text = row.get("text", "").strip()
        if len(text) > 200:
            examples.append({"text": text[:3000]})
        if len(examples) >= n_target:
            break

    print(f"  Loaded {len(examples)} DCLM pre-training examples")
    return Dataset.from_list(examples)

# Extract target Wikipedia titles from SimpleQA
download_simpleqa_csv()
target_titles = extract_target_titles()

# Load Active Reading synthetic data filtered to those titles
ds_formatted = load_ar_dataset(target_titles)

ar_dataset = ds_formatted.map(
        format_ar_example,
        remove_columns=ds_formatted.column_names,
        desc="Formatting AR examples",
)

print(f"  Formatted training examples: {len(ar_dataset)}")

if smoke_test:
    ar_dataset = ar_dataset.shuffle(seed=42).select(range(min(20, len(ar_dataset))))
    print(f"Smoke test: using {len(ar_dataset)} AR examples")

# DCLM pre-training mix (10%)
N_PRETRAIN = max(1, len(ar_dataset) // PRETRAIN_RATIO)
pretrain_ds = build_pretrain_mix(0 if smoke_test else N_PRETRAIN)

# Combine and shuffle
mixed_ds = concatenate_datasets([ar_dataset, pretrain_ds]).shuffle(seed=42)
print(f"\nTotal training examples: {len(mixed_ds)} "
      f"({len(ar_dataset)} AR + {len(pretrain_ds)} DCLM pretrain)")

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Load model in 4-bit QLoRA configuration for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=COMPUTE_DTYPE
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map='auto',
    token=HF_TOKEN
)
model.config.use_cache = False  # required for gradient checkpointing

model = prepare_model_for_kbit_training(model)
# Cast all non-quantized layers explicitly to float16
for name, param in model.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  VRAM after model load: {vram_used:.1f} / {vram_total:.1f} GB")

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type='CAUSAL_LM'
)

# Training config
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name='active-reading-llama-1b-run',
    max_steps=MAX_STEPS,
    num_train_epochs=1,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    max_length=MAX_SEQ_LEN,
    dataset_text_field='text',
    packing=False,
    fp16=FP16,
    bf16=BF16,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type='cosine',
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    report_to='wandb' if not smoke_test else "none",  
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=mixed_ds,
    peft_config=lora_config,
    processing_class=tokenizer,
)

meta = {
    "base_model":           BASE_MODEL,
    "device_profile":       args.device,
    "smoke_test":           smoke_test,
    "ar_chunk_files_limit": AR_CHUNK_FILES_LIMIT,
    "n_ar_examples":        len(ar_dataset),
    "n_pretrain_examples":  len(pretrain_ds),
    "n_total_examples":     len(mixed_ds),
    "max_steps":            MAX_STEPS,
    "completed":            datetime.now().isoformat(),
}
with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved to {OUTPUT_DIR}/training_meta.json")

# Check VRAM before launching
print(f"VRAM before training: {torch.cuda.memory_allocated() / 1e9:.1f} GB used")

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved adapter to {OUTPUT_DIR}")
import os
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, TaskType

# ─── Configuration ─────────────────────────────────────────────────────────────
# Paths
os.environ["TRANSFORMERS_CACHE"] = ... # where to load the base model from..

model_name   = "EleutherAI/pythia-410m"
output_dir   = "pythia_410m_sft_hh_full_sft_trainer" # where to save the trained model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = load_dataset("Dahoas/static-hh")

# LoRA hyperparameters
lora_r       = 32
lora_alpha   = 32
lora_dropout = 0.05

# Training hyperparameters
learning_rate                = 1e-4
num_train_epochs             = 3
max_length                   = 256
per_device_train_batch_size  = 8
per_device_eval_batch_size   = 8
gradient_accumulation_steps  = 1

# ─── Tokenizer & Model Loading ────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=os.environ["TRANSFORMERS_CACHE"]
)
# Ensure pad token for causal LM
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},  # Map all modules to cuda:0
    # Use float32 to avoid mixed precision issues
    torch_dtype=torch.float32,
    cache_dir=os.environ["TRANSFORMERS_CACHE"]
)  

# ─── Apply LoRA (PEFT) ─────────────────────────────────────────────────────────

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"]
)  
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def format_dataset(examples):
    """Format the dataset for SFTTrainer with prompt-completion structure."""
    return {
        "prompt": examples["prompt"],
        "completion": examples["chosen"]
    }

train_dataset = dataset["train"].map(format_dataset, batched=True, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["test"].map(format_dataset, batched=True, remove_columns=dataset["test"].column_names)

# ─── Training Setup ───────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,  
    max_seq_length=max_length,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,  
    logging_steps=1000,
    # Disable fp16 to avoid mixed precision conflicts
    fp16=False,
    bf16=False,
    push_to_hub=True,
    load_best_model_at_end=True,
    max_steps=-1,
    # Add evaluation and save strategies
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    save_steps=10000,
    save_total_limit=3,  # Keep only the last 3 checkpoints
    report_to=["wandb"],
    hub_token= ... # your huggingface token
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

# ─── Run Training ─────────────────────────────────────────────────────────────

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
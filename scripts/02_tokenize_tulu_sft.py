import os
from typing import Dict, List, Any

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

IGNORE_INDEX = -100

MODEL_PATH = "models/OLMo-2-0425-1B-SFT"
DATASET_NAME = "allenai/tulu-3-sft-olmo-2-mixture-0225"

OUTPUT_DIR = "data/tokenized_tulu_olmo2_sft"

MAX_LENGTH = 4096

# 调试时可以设小一点，例如 1000；正式全量设为 None
MAX_EXAMPLES = None
# MAX_EXAMPLES = 1000


def render_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    使用 tokenizer 自带 chat_template。
    如果你当初 SFT 使用的是 open-instruct 训练代码中的专门模板，
    最好把这里替换成当时完全相同的 formatting 函数。
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_text(tokenizer, text: str) -> List[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]


def build_assistant_only_labels(
    tokenizer,
    messages: List[Dict[str, str]],
    max_length: int,
) -> Dict[str, Any]:
    """
    产出：
      input_ids: 完整 conversation
      attention_mask: 全 1
      labels: 非 assistant answer token 为 -100，assistant answer token 保留原 token id

    注意：
      这个实现是“可解释、好检查”的版本。
      为了最大程度复现 SFT，你应该确认 tokenizer chat_template 与训练时一致。
    """
    full_text = render_chat(tokenizer, messages)

    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    labels = [IGNORE_INDEX] * len(input_ids)

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        assistant_content = msg.get("content", "")
        if not assistant_content:
            continue

        # 用“前缀 + 空 assistant”估计 assistant content 的起点。
        prefix_messages = messages[:msg_idx] + [{"role": "assistant", "content": ""}]
        prefix_text = render_chat(tokenizer, prefix_messages)

        prefix_ids = tokenize_text(tokenizer, prefix_text)
        answer_ids = tokenize_text(tokenizer, assistant_content)

        start = len(prefix_ids)
        end = start + len(answer_ids)

        # 截断保护
        if start >= len(input_ids):
            continue
        end = min(end, len(input_ids))

        # 保留 input_ids 上对应位置，而不是 answer_ids，避免模板空格/特殊符号错位时更坏。
        for pos in range(start, end):
            labels[pos] = input_ids[pos]

    num_loss_tokens = sum(1 for x in labels if x != IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "num_tokens": len(input_ids),
        "num_loss_tokens": num_loss_tokens,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token_id is None:
        print("Tokenizer has no pad token. Setting pad_token = eos_token for later batching.")
        tokenizer.pad_token = tokenizer.eos_token

    print("chat_template exists:", tokenizer.chat_template is not None)
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat_template. You need to provide the exact SFT formatting function."
        )

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")

    if MAX_EXAMPLES is not None:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))
        print(f"Using subset: {len(ds)} examples")
    else:
        print(f"Using full dataset: {len(ds)} examples")

    def map_fn(example):
        out = build_assistant_only_labels(
            tokenizer=tokenizer,
            messages=example["messages"],
            max_length=MAX_LENGTH,
        )
        out["example_id"] = example["id"]
        out["source"] = example["source"]
        return out

    print("Tokenizing and building assistant-only labels...")
    tokenized = ds.map(
        map_fn,
        remove_columns=ds.column_names,
        num_proc=8,
        desc="tokenizing",
    )

    print(tokenized)

    # 过滤掉没有 assistant loss token 的样本
    before = len(tokenized)
    tokenized = tokenized.filter(lambda x: x["num_loss_tokens"] > 0, num_proc=8)
    after = len(tokenized)
    print(f"Filtered empty-loss examples: {before} -> {after}")

    print(f"Saving to: {OUTPUT_DIR}")
    tokenized.save_to_disk(OUTPUT_DIR)

    print("Done.")
    print("Output columns:", tokenized.column_names)
    print("First example:")
    ex = tokenized[0]
    print({
        "example_id": ex["example_id"],
        "source": ex["source"],
        "num_tokens": ex["num_tokens"],
        "num_loss_tokens": ex["num_loss_tokens"],
        "input_ids_head": ex["input_ids"][:10],
        "labels_head": ex["labels"][:10],
    })


if __name__ == "__main__":
    main()

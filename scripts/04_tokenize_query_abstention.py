import os
import json
from typing import Dict, List, Any

from datasets import Dataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100

MODEL_PATH = "models/OLMo-2-0425-1B-SFT"

QUERY_JSON_PATH = "data/query_raw/query_with_diverse_abstention.json"
OUTPUT_DIR = "data/tokenized_query_diverse_abstention"

MAX_LENGTH = 4096


def render_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
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

        prefix_messages = messages[:msg_idx] + [{"role": "assistant", "content": ""}]
        prefix_text = render_chat(tokenizer, prefix_messages)

        prefix_ids = tokenize_text(tokenizer, prefix_text)
        answer_ids = tokenize_text(tokenizer, assistant_content)

        start = len(prefix_ids)
        end = start + len(answer_ids)

        if start >= len(input_ids):
            continue

        end = min(end, len(input_ids))

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

    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no chat_template.")

    print(f"Loading query JSON from: {QUERY_JSON_PATH}")
    with open(QUERY_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} query examples.")

    rows = []

    for ex in raw:
        question = ex["question"]

        # 核心：用 abstention_answer 作为 query 的 supervised target
        target = ex.get("abstention_answer")
        if target is None or target == "":
            continue

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": target},
        ]

        tokenized = build_assistant_only_labels(
            tokenizer=tokenizer,
            messages=messages,
            max_length=MAX_LENGTH,
        )

        if tokenized["num_loss_tokens"] == 0:
            continue

        row = {
            **tokenized,
            "query_id": str(ex["question_id"]),
            "question": question,
            "target_answer": target,
            "answerable": bool(ex.get("answerable", False)),
            "source": ex.get("source", ""),
            "prompt_length": ex.get("prompt_length", None),
        }

        rows.append(row)

    query_ds = Dataset.from_list(rows)

    print(query_ds)
    print("Columns:", query_ds.column_names)

    print(f"Saving query dataset to: {OUTPUT_DIR}")
    query_ds.save_to_disk(OUTPUT_DIR)

    print("Done.")

    first = query_ds[0]
    print("First example summary:")
    print({
        "query_id": first["query_id"],
        "source": first["source"],
        "answerable": first["answerable"],
        "num_tokens": first["num_tokens"],
        "num_loss_tokens": first["num_loss_tokens"],
        "question": first["question"],
        "target_answer": first["target_answer"],
    })


if __name__ == "__main__":
    main()

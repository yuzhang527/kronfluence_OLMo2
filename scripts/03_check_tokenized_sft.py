from datasets import load_from_disk
from transformers import AutoTokenizer

IGNORE_INDEX = -100

MODEL_PATH = "models/OLMo-2-0425-1B-SFT"
DATA_DIR = "data/tokenized_tulu_olmo2_sft"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ds = load_from_disk(DATA_DIR)

    print(ds)
    print("columns:", ds.column_names)
    print("num rows:", len(ds))

    for idx in [0, 1, 2]:
        ex = ds[idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        loss_positions = [i for i, x in enumerate(labels) if x != IGNORE_INDEX]

        print("\n" + "=" * 80)
        print("idx:", idx)
        print("example_id:", ex["example_id"])
        print("source:", ex["source"])
        print("num_tokens:", ex["num_tokens"])
        print("num_loss_tokens:", ex["num_loss_tokens"])
        print("first loss pos:", loss_positions[0] if loss_positions else None)
        print("last loss pos:", loss_positions[-1] if loss_positions else None)

        print("\nDecoded full input, first 1200 chars:")
        print(tokenizer.decode(input_ids, skip_special_tokens=False)[:1200])

        if loss_positions:
            start = max(loss_positions[0] - 20, 0)
            end = min(loss_positions[-1] + 20, len(input_ids))
            visible_label_ids = [
                token_id for token_id, label in zip(input_ids[start:end], labels[start:end])
                if label != IGNORE_INDEX
            ]

            print("\nDecoded supervised assistant span around labels:")
            print(tokenizer.decode(visible_label_ids, skip_special_tokens=False)[:1200])

if __name__ == "__main__":
    main()

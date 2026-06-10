from datasets import load_from_disk
from transformers import AutoTokenizer

IGNORE_INDEX = -100

MODEL_PATH = "models/OLMo-2-0425-1B-SFT"
QUERY_DIR = "data/tokenized_query_diverse_abstention"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ds = load_from_disk(QUERY_DIR)

    print(ds)
    print("columns:", ds.column_names)
    print("num rows:", len(ds))

    for idx in range(min(5, len(ds))):
        ex = ds[idx]

        input_ids = ex["input_ids"]
        labels = ex["labels"]

        loss_token_ids = [
            input_id
            for input_id, label in zip(input_ids, labels)
            if label != IGNORE_INDEX
        ]

        print("\n" + "=" * 80)
        print("idx:", idx)
        print("query_id:", ex["query_id"])
        print("source:", ex["source"])
        print("answerable:", ex["answerable"])
        print("num_tokens:", ex["num_tokens"])
        print("num_loss_tokens:", ex["num_loss_tokens"])

        print("\nQuestion:")
        print(ex["question"])

        print("\nTarget answer from JSON:")
        print(ex["target_answer"])

        print("\nDecoded supervised label span:")
        print(tokenizer.decode(loss_token_ids, skip_special_tokens=False))

        print("\nDecoded full input:")
        print(tokenizer.decode(input_ids, skip_special_tokens=False))

if __name__ == "__main__":
    main()

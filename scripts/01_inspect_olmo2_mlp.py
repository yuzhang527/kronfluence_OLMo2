import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/OLMo-2-0425-1B-SFT"

def main():
    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    print("\nModel class:")
    print(type(model))

    print("\nConfig summary:")
    print("model_type:", getattr(model.config, "model_type", None))
    print("num_hidden_layers:", getattr(model.config, "num_hidden_layers", None))
    print("hidden_size:", getattr(model.config, "hidden_size", None))
    print("intermediate_size:", getattr(model.config, "intermediate_size", None))
    print("max_position_embeddings:", getattr(model.config, "max_position_embeddings", None))
    print("vocab_size:", getattr(model.config, "vocab_size", None))

    print("\nLinear modules containing '.mlp.':")
    mlp_linear_names = []
    for name, module in model.named_modules():
        if ".mlp." in name and isinstance(module, nn.Linear):
            mlp_linear_names.append(name)
            print(name, tuple(module.weight.shape))

    print("\nTotal MLP Linear modules:", len(mlp_linear_names))

    print("\nSuggested Kronfluence tracked modules:")
    for name in mlp_linear_names:
        print(f'    "{name}",')

    print("\nTokenizer special tokens:")
    print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("chat_template exists:", tokenizer.chat_template is not None)

    if tokenizer.chat_template is not None:
        print("\nChat template preview:")
        print(tokenizer.chat_template[:1000])

if __name__ == "__main__":
    main()

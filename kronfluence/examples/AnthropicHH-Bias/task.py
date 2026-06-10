'''
Implements the task (the loss function on the Antrhopic Dataset, and the measurement function on the Bias Dataset)
'''
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from kronfluence.task import Task
from utils import bias_agreement_nll_loss

BATCH_TYPE = Dict[str, torch.Tensor]
NUM_BLOCKS = 1 # number of transformer layers blocks to consider while fetching modules
NUM_TRANSFORMER_BLOCKS = 12 # total number of transformer layers blocks in the pythia 410 model
MODEL_NAME = "EleutherAI/pythia-410m"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


class BiasTask(Task):
    def compute_train_loss(self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False,) -> torch.Tensor: # not factor_args.use_empirical_fisher.
    #use_empirical_fisher is False by default for fit_all_factors in fator_args. This means that sample = True will be passed while fitting the analyzer => FIM will be computed using samples from the model, not true labels
        logits = model(
            input_ids=batch["input_ids"], # prompt + chosen tokenized
            attention_mask=batch["attention_mask"],
        ).logits #  B, T, C
        logits = logits[..., :-1, :].contiguous() # B, T-1, C
        logits = logits.view(-1, logits.size(-1)) # B*(T-1), C

        if not sample: # copmute loss by teacher forcing.
            labels = batch["labels"] # prompt + chosen tokenized, but prompt tokens forced to -100
            labels = labels[..., 1:].contiguous() # B, T-1
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return bias_agreement_nll_loss(model, tokenizer, batch)


    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        
        for i in range(NUM_TRANSFORMER_BLOCKS - NUM_BLOCKS, NUM_TRANSFORMER_BLOCKS):
            print(i, end=" ")
            total_modules.append(f"gpt_neox.layers.{i}.attention.query_key_value")
            total_modules.append(f"gpt_neox.layers.{i}.attention.dense")

        for i in range(NUM_TRANSFORMER_BLOCKS - NUM_BLOCKS, NUM_TRANSFORMER_BLOCKS):
            total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_h_to_4h")
            total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_4h_to_h")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]
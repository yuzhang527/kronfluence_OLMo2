import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

from examples.olmo2_sft_mlp.config import IGNORE_INDEX


class Olmo2SFTMLPTask(Task):
    """
    Kronfluence task for OLMo2 causal-LM SFT.

    We track only MLP linear layers:
      model.layers.{i}.mlp.gate_proj
      model.layers.{i}.mlp.up_proj
      model.layers.{i}.mlp.down_proj
    """

    def __init__(self, num_layers: int = 16):
        super().__init__()
        self.num_layers = num_layers

    def _causal_lm_sum_loss(self, batch, model: nn.Module, sample: bool = False):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs.logits.float()

        # causal LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        if not sample:
            return F.cross_entropy(
                shift_logits,
                shift_labels,
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )

        # For sampled Fisher-like mode.
        with torch.no_grad():
            probs = torch.softmax(shift_logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
            sampled_labels[shift_labels == IGNORE_INDEX] = IGNORE_INDEX

        return F.cross_entropy(
            shift_logits,
            sampled_labels,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )

    def compute_train_loss(self, batch, model: nn.Module, sample: bool = False):
        return self._causal_lm_sum_loss(batch=batch, model=model, sample=sample)

    def compute_measurement(self, batch, model: nn.Module):
        return self._causal_lm_sum_loss(batch=batch, model=model, sample=False)

    def get_influence_tracked_modules(self):
        modules = []
        for i in range(self.num_layers):
            modules.extend(
                [
                    f"model.layers.{i}.mlp.gate_proj",
                    f"model.layers.{i}.mlp.up_proj",
                    f"model.layers.{i}.mlp.down_proj",
                ]
            )
        return modules

    def get_attention_mask(self, batch):
        return batch["attention_mask"]

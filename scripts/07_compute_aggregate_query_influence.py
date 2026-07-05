import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_from_disk
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs


IGNORE_INDEX = -100
_LAYER_MLP_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.")


def setup_distributed() -> Tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    return local_rank, global_rank, world_size


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class TokenizedSFTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_examples: Optional[int] = None,
        start: int = 0,
    ):
        self.ds = load_from_disk(data_dir)

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}.")

        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError(
                    f"max_examples must be None or a positive integer, got {max_examples}."
                )
            end = min(start + max_examples, len(self.ds))
            self.ds = self.ds.select(range(start, end))
        elif start > 0:
            self.ds = self.ds.select(range(start, len(self.ds)))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "example_id": ex.get("example_id", str(idx)),
            "source": ex.get("source", ""),
            "num_tokens": ex.get("num_tokens", None),
            "num_loss_tokens": ex.get("num_loss_tokens", None),
        }


def make_sft_collator(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


class Olmo2AggregateQueryLogPTask(Task):
    """
    Train loss:
        Summed causal-LM NLL for each SFT training example.

    Query measurement:
        Mean, across all query examples, of each example's mean supervised-token
        log probability. With aggregate_query_gradients=True, each query batch
        returns its sum divided by the total number of query examples, so the
        accumulated gradient equals the gradient of the global query mean.
    """

    def __init__(
        self,
        tracked_modules: List[str],
        num_query_examples: Optional[int] = None,
    ):
        self.tracked_modules = tracked_modules
        self.num_query_examples = num_query_examples

    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)

        if not sample:
            return F.cross_entropy(
                flat_logits,
                flat_labels,
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )

        with torch.no_grad():
            probs = torch.softmax(flat_logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
            sampled_labels[flat_labels == IGNORE_INDEX] = IGNORE_INDEX

        return F.cross_entropy(
            flat_logits,
            sampled_labels,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ):
        if self.num_query_examples is None or self.num_query_examples <= 0:
            raise ValueError(
                "num_query_examples must be a positive integer when using "
                "aggregate_query_gradients=True."
            )

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(batch_size, seq_len)

        valid_mask = shift_labels != IGNORE_INDEX
        per_query_nll = (token_nll * valid_mask).sum(dim=1) / valid_mask.sum(
            dim=1
        ).clamp_min(1)
        per_query_logp = -per_query_nll

        return per_query_logp.sum() / float(self.num_query_examples)

    def get_influence_tracked_modules(self):
        return self.tracked_modules

    def get_attention_mask(self, batch):
        return batch["attention_mask"]


def infer_mlp_modules(
    model: nn.Module,
    min_layer: int,
    max_layer: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    """Return MLP Linear modules whose transformer layer index is in range."""
    modules: List[str] = []
    selected_layers = set()

    for name, module in model.named_modules():
        if ".mlp." not in name or not isinstance(module, nn.Linear):
            continue

        match = _LAYER_MLP_PATTERN.search(name)
        if match is None:
            continue

        layer_idx = int(match.group(1))
        if layer_idx < min_layer:
            continue
        if max_layer is not None and layer_idx > max_layer:
            continue

        modules.append(name)
        selected_layers.add(layer_idx)

    return modules, sorted(selected_layers)


def save_metadata(dataset: TokenizedSFTDataset, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            ex = dataset.ds[i]
            row = {
                "row_id": i,
                "example_id": ex.get("example_id", str(i)),
                "source": ex.get("source", ""),
                "num_tokens": ex.get("num_tokens", None),
                "num_loss_tokens": ex.get("num_loss_tokens", None),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_score_tensor(scores_obj):
    if isinstance(scores_obj, torch.Tensor):
        return scores_obj
    if isinstance(scores_obj, dict):
        if "all_modules" in scores_obj:
            return scores_obj["all_modules"]
        keys = list(scores_obj.keys())
        raise KeyError(f"Cannot find 'all_modules' in scores. Available keys: {keys}")
    raise TypeError(f"Unexpected score object type: {type(scores_obj)}")


def set_score_arg_if_possible(score_args, name: str, value: Any) -> None:
    """Give a clear failure if the installed Kronfluence lacks a required field."""
    try:
        setattr(score_args, name, value)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to set score_args.{name}={value}. "
            "Your Kronfluence version may not support this argument."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate-query EKFAC influence for OLMo2. Defaults are configured "
            "for 64k train / 300 query / MLP layer indices >= 14 / partitions=2."
        )
    )

    # Paths and output names.
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/OLMo-2-0425-1B-SFT",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="data/0616_tulu_sft_800k",
    )
    parser.add_argument(
        "--query_data_dir",
        type=str,
        default="data/tokenized_query_diverse_abstention_300",
    )
    parser.add_argument(
        "--analysis_name",
        type=str,
        default="olmo2_1b_q300_train64k_mlp14plus_p2",
    )
    parser.add_argument(
        "--factors_name",
        type=str,
        default="ekfac_mlp14plus_cov20k_lambda20k_p2",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        default="aggregate_query_logp_mlp14plus_train64k_p2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/olmo2_1b_q300_train64k_mlp14plus_p2",
    )

    # Dataset ranges.
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_max_examples", type=int, default=64_000)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--query_max_examples", type=int, default=300)

    # Tracked modules. Layer indices are the model's zero-based indices.
    # With OLMo2-1B's 16 blocks, min_layer=14 selects blocks 14 and 15.
    parser.add_argument("--mlp_min_layer", type=int, default=14)
    parser.add_argument("--mlp_max_layer", type=int, default=None)

    # All three micro-batches default to 1.
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--per_device_query_batch_size", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)

    # EKFAC fitting sample caps.
    parser.add_argument("--covariance_max_examples", type=int, default=20_000)
    parser.add_argument("--lambda_max_examples", type=int, default=20_000)

    # Partition settings: defaults are intentionally 2 for memory protection.
    parser.add_argument("--covariance_data_partitions", type=int, default=2)
    parser.add_argument("--covariance_module_partitions", type=int, default=2)
    parser.add_argument("--lambda_data_partitions", type=int, default=2)
    parser.add_argument("--lambda_module_partitions", type=int, default=2)
    parser.add_argument("--score_data_partitions", type=int, default=2)
    parser.add_argument("--score_module_partitions", type=int, default=2)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument("--skip_fit_factors", action="store_true")
    parser.add_argument("--overwrite_factors", action="store_true")
    parser.add_argument("--overwrite_scores", action="store_true")
    parser.add_argument("--query_gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()

    if args.mlp_min_layer < 0:
        raise ValueError("--mlp_min_layer must be >= 0.")
    if args.mlp_max_layer is not None and args.mlp_max_layer < args.mlp_min_layer:
        raise ValueError("--mlp_max_layer must be >= --mlp_min_layer.")

    partition_values = {
        "covariance_data_partitions": args.covariance_data_partitions,
        "covariance_module_partitions": args.covariance_module_partitions,
        "lambda_data_partitions": args.lambda_data_partitions,
        "lambda_module_partitions": args.lambda_module_partitions,
        "score_data_partitions": args.score_data_partitions,
        "score_module_partitions": args.score_module_partitions,
    }
    invalid_partitions = {k: v for k, v in partition_values.items() if v <= 0}
    if invalid_partitions:
        raise ValueError(f"All partition values must be positive: {invalid_partitions}")

    local_rank, global_rank, world_size = setup_distributed()

    try:
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
        distributed_barrier()

        if args.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif args.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        if is_main_process():
            print("=" * 80)
            print("Distributed")
            print("local_rank:", local_rank)
            print("global_rank:", global_rank)
            print("world_size:", world_size)
            print("output_dir:", args.output_dir)
            print("train_max_examples:", args.train_max_examples)
            print("covariance_max_examples:", args.covariance_max_examples)
            print("lambda_max_examples:", args.lambda_max_examples)
            print(
                "MLP layer range:",
                f"[{args.mlp_min_layer}, "
                f"{args.mlp_max_layer if args.mlp_max_layer is not None else 'last'}]",
            )
            print("partitions:", partition_values)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if is_main_process():
            print("=" * 80)
            print("Loading model:", args.model_path)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        model.eval()
        model.config.use_cache = False

        if torch.cuda.is_available():
            model = model.to(f"cuda:{local_rank}")

        tracked_modules, selected_layers = infer_mlp_modules(
            model=model,
            min_layer=args.mlp_min_layer,
            max_layer=args.mlp_max_layer,
        )
        if not tracked_modules:
            raise RuntimeError(
                "No MLP Linear modules matched the requested layer range. "
                "Inspect model.named_modules() and verify --mlp_min_layer / --mlp_max_layer."
            )

        if is_main_process():
            print("=" * 80)
            print("Selected MLP layer indices:", selected_layers)
            print("Tracked MLP Linear modules:", len(tracked_modules))
            with open(
                os.path.join(args.output_dir, "tracked_modules.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                for name in tracked_modules:
                    print(name)
                    f.write(name + "\n")
        distributed_barrier()

        if is_main_process():
            print("=" * 80)
            print("Loading train/query datasets")

        train_dataset = TokenizedSFTDataset(
            data_dir=args.train_data_dir,
            max_examples=args.train_max_examples,
            start=args.train_start,
        )
        query_dataset = TokenizedSFTDataset(
            data_dir=args.query_data_dir,
            max_examples=args.query_max_examples,
            start=args.query_start,
        )

        num_train = len(train_dataset)
        num_query = len(query_dataset)
        if num_train <= 0:
            raise RuntimeError("train_dataset is empty.")
        if num_query <= 0:
            raise RuntimeError("query_dataset is empty.")

        if is_main_process():
            print("train examples:", num_train)
            print("query examples:", num_query)
            save_metadata(
                train_dataset,
                os.path.join(args.output_dir, "train_metadata.jsonl"),
            )
            save_metadata(
                query_dataset,
                os.path.join(args.output_dir, "query_metadata.jsonl"),
            )
        distributed_barrier()

        task = Olmo2AggregateQueryLogPTask(
            tracked_modules=tracked_modules,
            num_query_examples=num_query,
        )

        if is_main_process():
            print("=" * 80)
            print("Preparing model for Kronfluence")

        model = prepare_model(model=model, task=task)
        analyzer = Analyzer(
            analysis_name=args.analysis_name,
            model=model,
            task=task,
            output_dir=args.output_dir,
        )
        analyzer.set_dataloader_kwargs(
            DataLoaderKwargs(
                num_workers=2,
                pin_memory=True,
                collate_fn=make_sft_collator(tokenizer),
            )
        )

        factor_args = FactorArguments(
            strategy="ekfac",
            use_empirical_fisher=False,
            covariance_max_examples=args.covariance_max_examples,
            lambda_max_examples=args.lambda_max_examples,
            covariance_data_partitions=args.covariance_data_partitions,
            covariance_module_partitions=args.covariance_module_partitions,
            activation_covariance_dtype=torch.float32,
            gradient_covariance_dtype=torch.float32,
            eigendecomposition_dtype=torch.float64,
            lambda_data_partitions=args.lambda_data_partitions,
            lambda_module_partitions=args.lambda_module_partitions,
            per_sample_gradient_dtype=torch_dtype,
            lambda_dtype=torch.float32,
            use_iterative_lambda_aggregation=True,
            offload_activations_to_cpu=True,
        )

        if not args.skip_fit_factors:
            if is_main_process():
                print("=" * 80)
                print("Fitting EKFAC factors")
                print("factors_name:", args.factors_name)

            analyzer.fit_all_factors(
                factors_name=args.factors_name,
                dataset=train_dataset,
                factor_args=factor_args,
                per_device_batch_size=args.per_device_batch_size,
                overwrite_output_dir=args.overwrite_factors,
            )
            distributed_barrier()
        elif is_main_process():
            print("=" * 80)
            print("Skipping factor fitting, using existing factors:", args.factors_name)

        score_args = ScoreArguments(
            score_dtype=torch.float32,
            per_sample_gradient_dtype=torch_dtype,
            precondition_dtype=torch_dtype,
            damping_factor=None,
            offload_activations_to_cpu=True,
            module_partitions=args.score_module_partitions,
            data_partitions=args.score_data_partitions,
            query_gradient_low_rank=None,
        )
        set_score_arg_if_possible(score_args, "aggregate_query_gradients", True)
        set_score_arg_if_possible(score_args, "aggregate_train_gradients", False)
        set_score_arg_if_possible(
            score_args,
            "query_gradient_accumulation_steps",
            args.query_gradient_accumulation_steps,
        )

        if is_main_process():
            print("=" * 80)
            print("Computing train scores for averaged query +log(p)")
            print("scores_name:", args.scores_name)
            print("num_train:", num_train)
            print("num_query:", num_query)
            print(
                "aggregate_query_gradients:",
                getattr(score_args, "aggregate_query_gradients", None),
            )
            print(
                "aggregate_train_gradients:",
                getattr(score_args, "aggregate_train_gradients", None),
            )
            print("per_device_query_batch_size:", args.per_device_query_batch_size)
            print("per_device_train_batch_size:", args.per_device_train_batch_size)

        analyzer.compute_pairwise_scores(
            scores_name=args.scores_name,
            factors_name=args.factors_name,
            query_dataset=query_dataset,
            train_dataset=train_dataset,
            score_args=score_args,
            per_device_query_batch_size=args.per_device_query_batch_size,
            per_device_train_batch_size=args.per_device_train_batch_size,
            overwrite_output_dir=args.overwrite_scores,
        )
        distributed_barrier()

        if is_main_process():
            print("=" * 80)
            print("Loading scores")
            scores_obj = analyzer.load_pairwise_scores(scores_name=args.scores_name)
            scores_tensor = extract_score_tensor(scores_obj).detach().cpu().float()

            if scores_tensor.ndim == 1:
                train_scores_avg_query_logp = scores_tensor
            elif scores_tensor.ndim == 2 and scores_tensor.shape[0] == 1:
                train_scores_avg_query_logp = scores_tensor[0]
            else:
                raise RuntimeError(
                    "aggregate_query_gradients=True should return [num_train] or "
                    f"[1, num_train], but got {tuple(scores_tensor.shape)}."
                )

            if train_scores_avg_query_logp.shape[0] != num_train:
                raise RuntimeError(
                    f"Expected train score vector shape [{num_train}], "
                    f"got {tuple(train_scores_avg_query_logp.shape)}"
                )

            final_path = os.path.join(args.output_dir, "train_scores_avg_query_logp.pt")
            torch.save(train_scores_avg_query_logp, final_path)

            raw_scores_path = os.path.join(args.output_dir, "raw_loaded_scores_tensor.pt")
            torch.save(scores_tensor, raw_scores_path)

            ranked_path = os.path.join(
                args.output_dir,
                "train_scores_avg_query_logp_top_bottom.jsonl",
            )
            topk = min(100, num_train)
            top = torch.topk(train_scores_avg_query_logp, k=topk, largest=True)
            bottom = torch.topk(train_scores_avg_query_logp, k=topk, largest=False)

            train_meta = []
            with open(
                os.path.join(args.output_dir, "train_metadata.jsonl"),
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    train_meta.append(json.loads(line))

            with open(ranked_path, "w", encoding="utf-8") as f:
                for rank, idx in enumerate(top.indices.tolist(), 1):
                    row = {
                        "side": "top_positive",
                        "rank": rank,
                        "train_row": idx,
                        "score": float(train_scores_avg_query_logp[idx]),
                        "metadata": train_meta[idx],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                for rank, idx in enumerate(bottom.indices.tolist(), 1):
                    row = {
                        "side": "top_negative",
                        "rank": rank,
                        "train_row": idx,
                        "score": float(train_scores_avg_query_logp[idx]),
                        "metadata": train_meta[idx],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print("=" * 80)
            print("Done.")
            print("Saved final train score vector:", final_path)
            print("Saved raw loaded scores tensor:", raw_scores_path)
            print("Final score shape:", tuple(train_scores_avg_query_logp.shape))
            print("Raw loaded score shape:", tuple(scores_tensor.shape))
            print("aggregated_queries:", num_query)
            print("Saved ranking preview:", ranked_path)

        distributed_barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

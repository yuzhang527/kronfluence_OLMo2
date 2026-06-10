import os
import json
import argparse
from typing import Dict, List, Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs


IGNORE_INDEX = -100


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    return local_rank, global_rank, world_size


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
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

        if max_examples is not None:
            end = min(start + max_examples, len(self.ds))
            self.ds = self.ds.select(range(start, end))
        elif start > 0:
            self.ds = self.ds.select(range(start, len(self.ds)))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
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
    A 类方法的 Task。

    train loss:
        对 SFT train example 使用 summed causal LM NLL。

    query measurement:
        目标是：

            mean over query examples [
                mean over supervised assistant tokens log p(y_t | x_<t)
            ]

        但为了配合 Kronfluence 的 aggregate_query_gradients=True，
        compute_measurement 对每个 query batch 返回：

            sum_{q in current batch} avg_token_logp(q) / num_total_query_examples

        Kronfluence 会遍历所有 query batch 并累积 gradient，
        所以最终 query gradient 等于：

            grad mean_{all query examples} avg_token_logp(q)

    这就避免了把 100 条 query 一次性塞进显存。
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
                "`num_query_examples` must be a positive integer when using "
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

        # shift_logits: [B, T, V]
        # shift_labels: [B, T]
        batch_size, seq_len, vocab_size = shift_logits.shape

        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(batch_size, seq_len)

        valid_mask = shift_labels != IGNORE_INDEX

        per_query_nll = (
            token_nll * valid_mask
        ).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)

        per_query_logp = -per_query_nll

        # 关键：
        # 当前 batch 内 query 的 logp 求和，然后除以全局 query 数。
        # Kronfluence 会对所有 query batch 累积 gradient。
        return per_query_logp.sum() / float(self.num_query_examples)

    def get_influence_tracked_modules(self):
        return self.tracked_modules

    def get_attention_mask(self, batch):
        return batch["attention_mask"]


def infer_mlp_modules(model: nn.Module) -> List[str]:
    modules = []
    for name, module in model.named_modules():
        if ".mlp." in name and isinstance(module, nn.Linear):
            modules.append(name)
    return modules


def save_metadata(dataset: TokenizedSFTDataset, path: str):
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


def set_score_arg_if_possible(score_args, name: str, value: Any):
    """
    兼容不同 kronfluence 版本。

    新版本 ScoreArguments 通常有 aggregate_query_gradients /
    aggregate_train_gradients / query_gradient_accumulation_steps。

    如果 dataclass 没有显式字段，普通 Python 对象一般仍可 setattr。
    如果安装版本完全不支持该参数，后续 compute_pairwise_scores 可能仍会报错，
    那就说明需要升级 kronfluence。
    """
    try:
        setattr(score_args, name, value)
    except Exception as e:
        raise RuntimeError(
            f"Failed to set score_args.{name}={value}. "
            f"Your kronfluence version may not support this argument."
        ) from e


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="models/OLMo-2-0425-1B-SFT")
    parser.add_argument("--train_data_dir", type=str, default="data/tokenized_tulu_olmo2_sft")
    parser.add_argument("--query_data_dir", type=str, default="data/tokenized_query_abstention")

    parser.add_argument("--analysis_name", type=str, default="olmo2_train_scores_aggregate_query_logp")
    parser.add_argument("--factors_name", type=str, default="ekfac_mlp")
    parser.add_argument("--scores_name", type=str, default="aggregate_query_gradient_scores")

    parser.add_argument("--output_dir", type=str, default="outputs/kronfluence_train_scores_aggregate_query_logp")

    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_max_examples", type=int, default=None)

    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--query_max_examples", type=int, default=100)

    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--per_device_query_batch_size", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)

    parser.add_argument("--covariance_max_examples", type=int, default=10000)
    parser.add_argument("--lambda_max_examples", type=int, default=10000)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--skip_fit_factors", action="store_true")
    parser.add_argument("--overwrite_factors", action="store_true")
    parser.add_argument("--overwrite_scores", action="store_true")

    # 如果 kronfluence 版本支持，可以控制 query gradient 内部累积步数。
    # 一般保持 1 即可；真正的 microbatch 大小由 per_device_query_batch_size 控制。
    parser.add_argument("--query_gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()

    local_rank, global_rank, world_size = setup_distributed()

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

    tracked_modules = infer_mlp_modules(model)

    if len(tracked_modules) == 0:
        raise RuntimeError("No MLP Linear modules found. Check model module names.")

    if is_main_process():
        print("=" * 80)
        print("Tracked MLP modules:", len(tracked_modules))

        with open(os.path.join(args.output_dir, "tracked_modules.txt"), "w", encoding="utf-8") as f:
            for name in tracked_modules:
                print(name)
                f.write(name + "\n")

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

    task = Olmo2AggregateQueryLogPTask(
        tracked_modules=tracked_modules,
        num_query_examples=num_query,
    )

    if is_main_process():
        print("=" * 80)
        print("Preparing model for Kronfluence")

    model = prepare_model(model=model, task=task)

    analyzer_kwargs = dict(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
    )

    # 大多数 kronfluence 版本支持 output_dir。
    # 如果你的版本不支持，可以删掉这一行。
    analyzer_kwargs["output_dir"] = args.output_dir

    analyzer = Analyzer(**analyzer_kwargs)

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

        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        eigendecomposition_dtype=torch.float64,

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
    else:
        if is_main_process():
            print("=" * 80)
            print("Skipping factor fitting, using existing factors:", args.factors_name)

    score_args = ScoreArguments(
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch_dtype,
        precondition_dtype=torch_dtype,

        damping_factor=None,
        offload_activations_to_cpu=True,

        module_partitions=1,
        data_partitions=1,

        query_gradient_low_rank=None,
    )

    # A 类核心：先聚合所有 query gradients，再对 train examples 打分。
    set_score_arg_if_possible(score_args, "aggregate_query_gradients", True)

    # 不聚合 train gradients，因为我们需要每个 train example 一个 score。
    set_score_arg_if_possible(score_args, "aggregate_train_gradients", False)

    # 可选。一般不用调大；microbatch 由 per_device_query_batch_size 控制。
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
        print("aggregate_query_gradients:", getattr(score_args, "aggregate_query_gradients", None))
        print("aggregate_train_gradients:", getattr(score_args, "aggregate_train_gradients", None))
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

    if is_main_process():
        print("=" * 80)
        print("Loading scores")

    scores_obj = analyzer.load_pairwise_scores(scores_name=args.scores_name)
    scores_tensor = extract_score_tensor(scores_obj).detach().cpu().float()

    if scores_tensor.ndim == 1:
        train_scores_avg_query_logp = scores_tensor
    elif scores_tensor.ndim == 2:
        if scores_tensor.shape[0] == 1:
            train_scores_avg_query_logp = scores_tensor[0]
        else:
            raise RuntimeError(
                "aggregate_query_gradients=True should return either [num_train] "
                f"or [1, num_train], but got {tuple(scores_tensor.shape)}."
            )
    else:
        raise RuntimeError(f"Unexpected score tensor shape: {tuple(scores_tensor.shape)}")

    if train_scores_avg_query_logp.shape[0] != num_train:
        raise RuntimeError(
            f"Expected train score vector shape [{num_train}], "
            f"got {tuple(train_scores_avg_query_logp.shape)}"
        )

    if is_main_process():
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
        with open(os.path.join(args.output_dir, "train_metadata.jsonl"), "r", encoding="utf-8") as f:
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

    cleanup_distributed()


if __name__ == "__main__":
    main()
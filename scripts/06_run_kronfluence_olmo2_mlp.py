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


class Olmo2AvgQueryLogPTask(Task):
    """
    目标 measurement：

        average_query_logp = mean over query examples [
            mean over supervised assistant tokens log p(y_t | x_<t)
        ]

    这里 compute_measurement 对单个 query batch 返回 +log(p)。
    因为我们建议 per_device_query_batch_size=1，所以它就是单条 query 的平均 logp。

    compute_train_loss 仍然保持 sum NLL，这是 Kronfluence factor 估计更稳的形式。
    """

    def __init__(self, tracked_modules: List[str]):
        self.tracked_modules = tracked_modules

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

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

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
        """
        返回 +log(p)，不是 -log(p)。

        如果 query batch size = 1：
            返回该 query 的 assistant-token 平均 log-likelihood。

        如果 query batch size > 1：
            返回整个 batch 内所有 supervised assistant tokens 的平均 log-likelihood。
            为了“100 条 query 等权平均”，建议 query batch size 保持 1。
        """
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )

        logits = outputs.logits.float()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        token_nll = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )

        valid_mask = flat_labels != IGNORE_INDEX
        denom = valid_mask.sum().clamp_min(1)

        mean_nll = token_nll[valid_mask].sum() / denom

        # 关键：+log(p) = - NLL
        return -mean_nll

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
    """
    Kronfluence 通常返回 dict，其中 all_modules 是聚合后的 score tensor。
    这里做得保守一点，兼容 tensor / dict 两种返回。
    """
    if isinstance(scores_obj, torch.Tensor):
        return scores_obj

    if isinstance(scores_obj, dict):
        if "all_modules" in scores_obj:
            return scores_obj["all_modules"]
        keys = list(scores_obj.keys())
        raise KeyError(f"Cannot find 'all_modules' in scores. Available keys: {keys}")

    raise TypeError(f"Unexpected score object type: {type(scores_obj)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="models/OLMo-2-0425-1B-SFT")
    parser.add_argument("--train_data_dir", type=str, default="data/tokenized_tulu_olmo2_sft")
    parser.add_argument("--query_data_dir", type=str, default="data/tokenized_query_abstention")

    parser.add_argument("--analysis_name", type=str, default="olmo2_train_scores_avg_query_logp")
    parser.add_argument("--factors_name", type=str, default="ekfac_mlp")
    parser.add_argument("--scores_prefix", type=str, default="avg_query_logp_scores")

    parser.add_argument("--output_dir", type=str, default="outputs/kronfluence_train_scores_avg_query_logp")

    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_max_examples", type=int, default=None)

    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--query_max_examples", type=int, default=100)

    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--per_device_query_batch_size", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)

    parser.add_argument("--covariance_max_examples", type=int, default=10000)
    parser.add_argument("--lambda_max_examples", type=int, default=10000)

    parser.add_argument("--query_chunk_size", type=int, default=10)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--skip_fit_factors", action="store_true")
    parser.add_argument("--save_pairwise_chunks", action="store_true")

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
        raise RuntimeError("No MLP Linear modules found. Check model names.")

    if is_main_process():
        print("=" * 80)
        print("Tracked MLP modules:", len(tracked_modules))
        with open(os.path.join(args.output_dir, "tracked_modules.txt"), "w") as f:
            for name in tracked_modules:
                print(name)
                f.write(name + "\n")

    train_dataset = TokenizedSFTDataset(
        data_dir=args.train_data_dir,
        max_examples=args.train_max_examples,
        start=args.train_start,
    )

    full_query_dataset = TokenizedSFTDataset(
        data_dir=args.query_data_dir,
        max_examples=args.query_max_examples,
        start=args.query_start,
    )

    if is_main_process():
        print("=" * 80)
        print("train examples:", len(train_dataset))
        print("query examples:", len(full_query_dataset))
        save_metadata(train_dataset, os.path.join(args.output_dir, "train_metadata.jsonl"))
        save_metadata(full_query_dataset, os.path.join(args.output_dir, "query_metadata.jsonl"))

    task = Olmo2AvgQueryLogPTask(tracked_modules=tracked_modules)

    model = prepare_model(model=model, task=task)

    analyzer = Analyzer(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
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

        analyzer.fit_all_factors(
            factors_name=args.factors_name,
            dataset=train_dataset,
            factor_args=factor_args,
            per_device_batch_size=args.per_device_batch_size,
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

    num_train = len(train_dataset)
    num_query = len(full_query_dataset)

    train_score_sum = torch.zeros(num_train, dtype=torch.float32)
    processed_queries = 0

    if is_main_process():
        print("=" * 80)
        print("Computing train scores for average query +log(p)")
        print("num_train:", num_train)
        print("num_query:", num_query)
        print("query_chunk_size:", args.query_chunk_size)

    for chunk_start in range(0, num_query, args.query_chunk_size):
        chunk_end = min(chunk_start + args.query_chunk_size, num_query)
        chunk_size = chunk_end - chunk_start

        query_chunk = TokenizedSFTDataset(
            data_dir=args.query_data_dir,
            max_examples=chunk_size,
            start=args.query_start + chunk_start,
        )

        scores_name = f"{args.scores_prefix}_q{chunk_start:06d}_{chunk_end:06d}"

        if is_main_process():
            print("=" * 80)
            print(f"Query chunk {chunk_start}:{chunk_end}")
            print("scores_name:", scores_name)

        analyzer.compute_pairwise_scores(
            scores_name=scores_name,
            factors_name=args.factors_name,
            query_dataset=query_chunk,
            train_dataset=train_dataset,
            score_args=score_args,
            per_device_query_batch_size=args.per_device_query_batch_size,
            per_device_train_batch_size=args.per_device_train_batch_size,
        )

        scores_obj = analyzer.load_pairwise_scores(scores_name=scores_name)
        pairwise = extract_score_tensor(scores_obj).detach().cpu().float()

        if pairwise.shape[1] != num_train:
            raise RuntimeError(
                f"Expected pairwise shape [num_query_chunk, {num_train}], "
                f"got {tuple(pairwise.shape)}"
            )

        # pairwise: [query_chunk, train]
        # 目标：train_scores[j] = mean_q influence(q, train_j)
        chunk_train_sum = pairwise.sum(dim=0)

        train_score_sum += chunk_train_sum
        processed_queries += pairwise.shape[0]

        if is_main_process():
            print("pairwise shape:", tuple(pairwise.shape))
            print("processed_queries:", processed_queries)

            chunk_out = os.path.join(
                args.output_dir,
                f"train_score_sum_q{chunk_start:06d}_{chunk_end:06d}.pt",
            )
            torch.save(chunk_train_sum, chunk_out)
            print("saved chunk train-score sum:", chunk_out)

            if args.save_pairwise_chunks:
                pairwise_out = os.path.join(
                    args.output_dir,
                    f"pairwise_q{chunk_start:06d}_{chunk_end:06d}.pt",
                )
                torch.save(pairwise, pairwise_out)
                print("saved pairwise chunk:", pairwise_out)

        del pairwise
        del scores_obj
        torch.cuda.empty_cache()

    if processed_queries != num_query:
        raise RuntimeError(f"processed_queries={processed_queries}, expected={num_query}")

    train_scores_avg_query_logp = train_score_sum / max(processed_queries, 1)

    if is_main_process():
        final_path = os.path.join(args.output_dir, "train_scores_avg_query_logp.pt")
        torch.save(train_scores_avg_query_logp, final_path)

        ranked_path = os.path.join(args.output_dir, "train_scores_avg_query_logp_top_bottom.jsonl")

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
        print("shape:", tuple(train_scores_avg_query_logp.shape))
        print("processed_queries:", processed_queries)
        print("Saved ranking preview:", ranked_path)

    cleanup_distributed()


if __name__ == "__main__":
    main()

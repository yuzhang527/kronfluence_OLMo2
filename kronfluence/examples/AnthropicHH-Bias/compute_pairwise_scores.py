''' 
Implements computation of the pairwise scores for the pythia 410m model.
'''
import torch
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from fit_all_factors import (
    factor_args,
    USE_HALF_PRECISION,
    USE_COMPILE,
    COMPUTE_PER_TOKEN_SCORES,
    QUERY_GRADIENT_RANK,
    analyzer,
    factors_name
)
from utils import get_anthropic_dataset, get_bias_agreement_dataset
from task import tokenizer

# Compute pairwise scores.

score_args = ScoreArguments()
scores_name = factor_args.strategy
if USE_HALF_PRECISION:
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    scores_name += "_half"
if USE_COMPILE:
    scores_name += "_compile"
if COMPUTE_PER_TOKEN_SCORES:
    score_args.compute_per_token_scores = True
    scores_name += "_per_token"
rank = QUERY_GRADIENT_RANK if QUERY_GRADIENT_RANK != -1 else None
if rank is not None:
    score_args.query_gradient_low_rank = rank
    score_args.query_gradient_accumulation_steps = 10
    scores_name += f"_qlr{rank}"

score_args.aggregate_query_gradients=True # False by default. Highly recommend running with True.

anthropic_dataset = get_anthropic_dataset(tokenizer)

bias_dataset = get_bias_agreement_dataset(tokenizer)


QUERY_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 16
analyzer.compute_pairwise_scores(
    scores_name=scores_name,
    score_args=score_args,
    factors_name=factors_name,
    query_dataset=bias_dataset,
    train_dataset=anthropic_dataset,
    per_device_query_batch_size=QUERY_BATCH_SIZE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    overwrite_output_dir=False
)
''' 
The purpose of this script is to fit the factors for the pythia 410m model.
'''

import os
import logging
import numpy as np
import torch
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from transformers import default_data_collator
from utils import get_anthropic_dataset
from termcolor import colored
from task import (
    BiasTask,
    model,
    tokenizer
)
    
CHECKPOINT_DIR = "./checkpoints"
FACTOR_STRATEGY = "ekfac"
QUERY_GRADIENT_RANK = -1
USE_HALF_PRECISION = True
USE_COMPILE = False
QUERY_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 16
PROFILE = False
COMPUTE_PER_TOKEN_SCORES = False
SIZE_OF_DATASET = 10000 # 10k samples from the dataset to estimate the true fisher information matrix
# You can change how many modules to consider for the computation of ekfac factors in the task.py file varying the number of transformer blocks to consier (NUM_BLOCKS)

print(colored("Loading model and dataset...", "green"))


anthropic_dataset = get_anthropic_dataset(tokenizer)
anthropic_dataset = anthropic_dataset.select(range(SIZE_OF_DATASET))
anthropic_indices = np.arange(len(anthropic_dataset)).astype(int)  # -> Python ints
anthropic_dataset = anthropic_dataset.select(anthropic_indices)    # stays an HF Dataset

if os.path.isfile(os.path.join(CHECKPOINT_DIR, "model.pth")):
    print(colored("Checkpoint found", "green"))
    logging.info(f"Loading checkpoint from {CHECKPOINT_DIR}")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "model.pth")))

# Define task and prepare model.
task = BiasTask()
model = prepare_model(model, task)
if USE_COMPILE:
    model = torch.compile(model)
analyzer = Analyzer(
    analysis_name="bias_pythia_410m",
    model=model,
    task=task,
    profile=PROFILE,
)

# Configure parameters for DataLoader.
dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
analyzer.set_dataloader_kwargs(dataloader_kwargs)
# Compute influence factors.
factors_name = FACTOR_STRATEGY
factor_args = FactorArguments(strategy=FACTOR_STRATEGY, use_empirical_fisher=True) # use empirical fisher is false by default.
if USE_HALF_PRECISION:
    factor_args = all_low_precision_factor_arguments(strategy=FACTOR_STRATEGY, dtype=torch.bfloat16)
    factors_name += "_half"
if USE_COMPILE:
    factors_name += "_compile"

analyzer.fit_all_factors(
    factors_name=factors_name,
    dataset=anthropic_dataset,
    per_device_batch_size=None,
    factor_args=factor_args,
    initial_per_device_batch_size_attempt=32,  
    overwrite_output_dir=False,
)

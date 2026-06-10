# Anthropic HH Bias Example

This folder shows **one minimal, self-contained example** of how to use
[`kronfluence`](https://github.com/pomonam/kronfluence) to:

1. Fit EKFAC influence *factors* on a large-language model (the 410 M parameter
   Pythia model) using a subset of 10 k Anthropic-HH training samples.
2. Compute *pairwise* influence scores between that training set and the
   "Stereotypical Bias" evaluation set.

The goal of the example is to be **copy-paste simple**: after installing the
requirements, a single command will produce both the factors and the influence
scores.

---

## 1&nbsp;·&nbsp;Quick start

```bash
# (1) Create a fresh virtual environment – optional but recommended
python -m venv ekfac-venv        # or conda env create -n ekfac python=3.10
source ekfac-venv/bin/activate

# (2) Install the few extra libraries this example needs
pip install -r requirements.txt

# (3) Download the model weights & fit EKFAC factors (≈ 10 min on a V100)
python fit_all_factors.py

# (4) Compute pairwise influence scores (≈ 5 min)
python compute_pairwise_scores.py

# Outputs ⇒ ./influence_results/
```

That's it – run the two scripts and you will obtain:

```
./influence_results/
 ├─ factors_ekfac_half/           # EKFAC blocks (state_dicts)
 └─ scores_ekfac_half.npy         # N×N influence matrix
```

---

## 2&nbsp;·&nbsp;Folder layout

```
AnthropicHH-Bias/
 ├─ fit_all_factors.py         # Step-1 · compute EKFAC factors
 ├─ compute_pairwise_scores.py # Step-2 · compute pairwise scores
 ├─ task.py                    # Loss + measurement definitions
 ├─ utils.py                   # Helper functions (dataset loading, metrics…)
 ├─ SFT_Trainer_Lora.py        # Optional LoRA fine-tuning script
 ├─ requirements.txt           # Extra runtime deps (torch + transformers …)
 └─ README.md                  # ← this file
```

If you only care about influence scores you can ignore `SFT_Trainer_Lora.py` —
it shows how to do an *optional* LoRA fine-tuning pass before computing EKFAC.

---

## 3&nbsp;·&nbsp;What happens under the hood?

1. **`fit_all_factors.py`**
   • loads Pythia-410 M and selects the last transformer block(s) to track
   • streams 10 k examples from Anthropic-HH and fits EKFAC statistics
   • saves the factors below `./influence_results/factors_ekfac_half/`

2. **`compute_pairwise_scores.py`**
   • reloads the same model and factors
   • computes gradient similarities between every train example and every eval
     example ("stereotypical bias" set)
   • stores the final influence matrix `scores_ekfac_half.npy`

All hyper-parameters (batch sizes, half-precision flag, etc.) live at the top of
`fit_all_factors.py` so you only need to edit them **once**.

---

## 4&nbsp;·&nbsp;Troubleshooting / FAQ

• `ImportError: kronfluence` → make sure the library is installed in the current
  environment (`pip install kronfluence==1.0.1`).

• GPU OOM during factor fitting → lower `initial_per_device_batch_size_attempt`
  in `fit_all_factors.py` (e.g. from *32* to *8*).

---
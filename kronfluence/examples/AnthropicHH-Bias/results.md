# Bias-loss results

Lower is better (negative log-likelihood on the Stereotypical Bias dev set).

| Model / Training subset | Bias NLL |
|-------------------------|---------:|
| Base Pythia-410M | 0.513 |
| SFT on full HH (96 k examples) | 0.4871 |
| SFT on random 45 k examples | 0.4987 |
| SFT on **lowest** 45 k EKFAC-ranked examples | 0.5965 |
| SFT on **highest** 45 k EKFAC-ranked examples | **0.3727** |

---

## LoRA adapter checkpoints on Huggingface

```text
ncgc/pythia_410m_hh_full_sft_trainer
ncgc/pythia_410m_sft_hh_random_45k
ncgc/pythia_410m_sft_hh_45k_lowest.bias
ncgc/pythia_410m_sft_hh_45k_highest.bias
```

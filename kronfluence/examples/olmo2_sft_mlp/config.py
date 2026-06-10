from pathlib import Path

# 项目根目录：
# /workspace/luoyuzhang/kronfluence_olmo2
PROJECT_ROOT = Path("/workspace/luoyuzhang/kronfluence_olmo2")

# Kronfluence 仓库目录：
# /workspace/luoyuzhang/kronfluence_olmo2/kronfluence
REPO_ROOT = PROJECT_ROOT / "kronfluence"

MODEL_PATH = PROJECT_ROOT / "models" / "OLMo-2-0425-1B-SFT"

TRAIN_DATASET_NAME = "allenai/tulu-3-sft-olmo-2-mixture-0225"

TOKENIZED_TRAIN_DIR = PROJECT_ROOT / "data" / "tokenized_tulu_olmo2_sft"

QUERY_JSON_PATH = PROJECT_ROOT / "data" / "query_raw" / "query_with_diverse_abstention.json"

TOKENIZED_QUERY_DIR = PROJECT_ROOT / "data" / "tokenized_query_diverse_abstention"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAX_LENGTH = 4096

IGNORE_INDEX = -100

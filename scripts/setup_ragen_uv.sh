#!/usr/bin/env bash

set -euo pipefail

# Environment setup script for RAGEN using uv.
#
# Replaces the conda-based setup with uv for faster, reproducible installs.
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# Validation:
# - Verified on NVIDIA H100, H200, and B200 (Linux x86_64).
# - macOS: most environments work, but flash-attn / vllm may need
#   platform-specific adjustments (see notes below).
#
# Environment coverage:
# - Supports bandit, sokoban, frozenlake, metamathqa, countdown, deepcoder
#
# Optional environments (install with flags):
#   --with-search    Search (HotpotQA) environment (~87 GB data download)
#   --with-webshop   WebShop environment
#   --with-lean      Lean environment
#
# Examples:
#   bash scripts/setup_ragen_uv.sh                   # base only
#   bash scripts/setup_ragen_uv.sh --with-search     # base + search

ENV_NAME="ragen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Parse optional environment flags
WITH_SEARCH=0
WITH_WEBSHOP=0
WITH_LEAN=0
EXTRA_EXTRAS=()
for arg in "$@"; do
    case "$arg" in
        --with-search)  WITH_SEARCH=1; EXTRA_EXTRAS+=("search") ;;
        --with-webshop) WITH_WEBSHOP=1; EXTRA_EXTRAS+=("webshop") ;;
        --with-lean)    WITH_LEAN=1; EXTRA_EXTRAS+=("lean") ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FATAL]${NC} $*" >&2; exit 1; }

print_step() {
    echo
    echo "[setup_ragen] $1"
}

ensure_uv() {
    if ! command -v uv &>/dev/null; then
        echo "uv is required but was not found in PATH." >&2
        echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi
    print_step "Using uv: $(uv --version)"
}

validate_repo_root() {
    if [[ ! -d "${PROJECT_ROOT}/verl" ]]; then
        echo "Could not find the RAGEN repository root from ${SCRIPT_DIR}." >&2
        exit 1
    fi
}

create_or_reuse_venv() {
    if [[ -d "${VENV_DIR}" ]]; then
        print_step "Using existing virtual environment at ${VENV_DIR}"
    else
        print_step "Creating virtual environment at ${VENV_DIR} (Python 3.12)"
        uv venv "${VENV_DIR}" --python 3.12
    fi

    print_step "Activating virtual environment"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
}

install_ragen() {
    local extras_str=""
    if [[ ${#EXTRA_EXTRAS[@]} -gt 0 ]]; then
        local joined
        joined=$(IFS=,; echo "${EXTRA_EXTRAS[*]}")
        extras_str="[${joined}]"
    fi

    print_step "Installing RAGEN in editable mode${extras_str:+ with extras: }${extras_str}"
    uv pip install -e ".${extras_str}" --no-deps
}

install_verl() {
    print_step "Installing verl dependencies (vllm, sglang, flash-attn, etc.)"

    local platform
    platform=$(uname -s)
    if [[ "${platform}" != "Linux" ]]; then
        echo "WARNING: Non-Linux platform detected (${platform})."
        echo "FlashAttention / vllm pre-built wheels are Linux-only."
        echo "Attempting install anyway — this may fail on GPU-specific packages."
        echo "If it fails, install verl minimal: uv pip install -e verl/ --no-deps"
    fi

    # Detect CUDA version for pre-built wheel selection
    local cuda_ver="124"
    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.')
    fi

    pushd verl >/dev/null

    # ---- Step 1: inference frameworks ----
    print_step "  [verl] Installing vllm..."
    if python -c "import vllm" 2>/dev/null; then
        echo "  vllm already installed ($(python -c 'import vllm; print(vllm.__version__)')), skipping."
    else
        # vllm >= 0.10 requires CUDA 13; pin to <0.10 for CUDA 12
        local vllm_constraint="vllm>=0.8.0"
        if [[ "${cuda_ver:0:2}" == "12" ]]; then
            vllm_constraint="vllm>=0.8.0,<0.10.0"
            echo "  CUDA 12 detected, using ${vllm_constraint}"
        fi
        uv pip install "${vllm_constraint}" --no-cache-dir || {
            warn "vllm install failed — this is critical, but continuing..."
            warn "You may need to install vllm manually."
        }
    fi

    print_step "  [verl] Installing sglang (optional)..."
    uv pip install "sglang[all]>=0.5.0" --no-cache-dir 2>/dev/null || {
        warn "sglang install skipped (not critical for most environments)"
    }

    # ---- Step 2: basic packages ----
    print_step "  [verl] Installing basic packages..."
    uv pip install \
        "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
        "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
        "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
        pytest py-spy pre-commit ruff tensorboard \
        "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

    # ---- Step 3: FlashAttention (pre-built wheel) ----
    print_step "  [verl] Installing FlashAttention (pre-built wheel)..."
    local fa_wheel="flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    local fa_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${fa_wheel}"

    if python -c "import flash_attn" 2>/dev/null; then
        echo "  flash-attn already installed, skipping."
    else
        if [[ ! -f "${fa_wheel}" ]]; then
            wget -nv "${fa_url}" || {
                warn "Failed to download flash-attn wheel — skipping"
            }
        fi
        if [[ -f "${fa_wheel}" ]]; then
            uv pip install --no-cache-dir "./${fa_wheel}"
            rm -f "${fa_wheel}"
        fi
    fi

    # ---- Step 3b: FlashInfer (pre-built wheel, skip source build) ----
    print_step "  [verl] Installing FlashInfer (pre-built wheel)..."
    if python -c "import flashinfer" 2>/dev/null; then
        echo "  flashinfer already installed, skipping."
    else
        # Try pre-built wheel first (avoids source-build timeout)
        local fi_wheel="flashinfer_python-0.3.1+cu${cuda_ver}torch2.8-cp312-cp312-linux_x86_64.whl"
        local fi_url="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.3.1/${fi_wheel}"

        if wget -q --spider "${fi_url}" 2>/dev/null; then
            wget -nv "${fi_url}"
            uv pip install --no-cache-dir "./${fi_wheel}" && rm -f "${fi_wheel}"
        else
            warn "FlashInfer pre-built wheel not available at ${fi_url}"
            warn "Skipping flashinfer (source build would timeout)."
            warn "This is OK — training will work without it, just slightly slower."
        fi
    fi

    # ---- Step 4: opencv ----
    print_step "  [verl] Installing opencv..."
    uv pip install opencv-python opencv-fixer 2>/dev/null || true
    python -c "from opencv_fixer import AutoFix; AutoFix()" 2>/dev/null || true

    # ---- Step 5: cudnn ----
    print_step "  [verl] Installing cudnn..."
    uv pip install nvidia-cudnn-cu12==9.10.2.21 2>/dev/null || true

    # ---- Install verl itself ----
    print_step "  [verl] Installing verl in editable mode..."
    uv pip install --no-deps -e .

    popd >/dev/null
    print_step "verl installation complete"
}

install_base_deps() {
    print_step "Installing release environment dependencies"

    # NOTE: These are mostly covered by pyproject.toml [project.dependencies],
    # but we list them here for explicitness and to pin specific versions.
    uv pip install \
        IPython \
        matplotlib \
        gym \
        gym_sokoban \
        gymnasium \
        "gymnasium[toy-text]" \
        debugpy \
        together \
        anthropic \
        "faiss-cpu==1.11.0" \
        "numpy==1.26.4"

    # Pin setuptools<70 (vllm may upgrade it, breaking pkg_resources for gym_sokoban)
    uv pip install "setuptools<70.0.0"
}

setup_search() {
    print_step "Installing search environment dependencies..."
    uv pip install sentence-transformers flask requests

    local DATA_DIR="./search_data"
    local INDICES_DIR="${DATA_DIR}/prebuilt_indices"
    local WIKI_DIR="${DATA_DIR}/wikipedia"

    print_step "Downloading search index data (wiki corpus + FAISS index shards, ~87 GB)..."
    python scripts/download_search_index.py --data_dir "$DATA_DIR"

    # Merge FAISS index shards
    local INDEX_FILE="${INDICES_DIR}/e5_Flat.index"
    if [ -f "$INDEX_FILE" ]; then
        echo "e5_Flat.index already exists ($(du -h "$INDEX_FILE" | cut -f1))"
    else
        print_step "Merging index shards -> e5_Flat.index..."
        if [ -f "${INDICES_DIR}/part_aa" ] && [ -f "${INDICES_DIR}/part_ab" ]; then
            cat "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab" > "$INDEX_FILE"
            rm -f "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab"
            echo "Created e5_Flat.index ($(du -h "$INDEX_FILE" | cut -f1))"
        else
            echo "ERROR: Index shards not found in ${INDICES_DIR}" >&2
            exit 1
        fi
    fi

    # Convert wiki-18.jsonl -> corpus.json
    local CORPUS_FILE="${INDICES_DIR}/corpus.json"
    local WIKI_JSONL="${WIKI_DIR}/wiki-18.jsonl"

    if [ -f "$CORPUS_FILE" ]; then
        echo "corpus.json already exists ($(du -h "$CORPUS_FILE" | cut -f1))"
    else
        print_step "Converting wiki-18.jsonl -> corpus.json..."
        if [ ! -f "$WIKI_JSONL" ]; then
            echo "ERROR: ${WIKI_JSONL} not found" >&2
            exit 1
        fi
        python3 -c "
import json
from tqdm import tqdm

input_path = '${WIKI_JSONL}'
output_path = '${CORPUS_FILE}'

print(f'Reading {input_path}...')
corpus = []
with open(input_path, 'r') as f:
    for line in tqdm(f, desc='Loading wiki-18.jsonl'):
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        text = doc.get('text', doc.get('contents', doc.get('content', '')))
        title = doc.get('title', '')
        if title and text:
            corpus.append(f'{title} {text}')
        elif text:
            corpus.append(text)

print(f'Writing {len(corpus)} documents to {output_path}...')
with open(output_path, 'w') as f:
    json.dump(corpus, f)
print(f'Done! corpus.json = {len(corpus)} docs')
"
    fi

    # Prepare HotpotQA parquet data
    print_step "Preparing HotpotQA parquet data..."
    python scripts/prepare_search_data.py --output_dir data/search

    print_step "Search environment setup complete"
    echo "To start the retrieval server:"
    echo "  CUDA_VISIBLE_DEVICES='' python scripts/retrieval/server.py --port 8001"
}

main() {
    ensure_uv
    validate_repo_root
    cd "${PROJECT_ROOT}"

    create_or_reuse_venv

    print_step "Initializing git submodules"
    git submodule update --init --recursive

    install_ragen
    install_verl
    install_base_deps

    print_step "Downloading project data"
    python scripts/download_data.py

    # Optional: search environment
    if [ "$WITH_SEARCH" -eq 1 ]; then
        setup_search
    fi

    print_step "Setup complete"
    echo
    echo "The virtual environment is at: ${VENV_DIR}"
    echo "Activate it with: source ${VENV_DIR}/bin/activate"
    echo "Or prefix commands with: uv run <command>"
}

main "$@"

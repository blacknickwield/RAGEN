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
    uv pip install -e ".${extras_str}"
}

install_verl() {
    print_step "Installing verl dependencies (vllm, sglang, flash-attn, etc.)"

    # NOTE: The verl install script downloads Linux x86_64 pre-built wheels
    # (flash-attn, flashinfer). On macOS / non-Linux, this step will fail.
    # For macOS development, you can skip verl-specific GPU packages and
    # install a minimal set:
    #   uv pip install -e verl/ --no-deps
    #   uv pip install vllm sglang  # these may also not work on macOS

    local platform
    platform=$(uname -s)
    if [[ "${platform}" != "Linux" ]]; then
        echo "WARNING: Non-Linux platform detected (${platform})."
        echo "FlashAttention / vllm pre-built wheels are Linux-only."
        echo "Attempting install anyway — this may fail on GPU-specific packages."
        echo "If it fails, install verl minimal: uv pip install -e verl/ --no-deps"
    fi

    pushd verl >/dev/null
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
    uv pip install --no-deps -e .
    popd >/dev/null
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

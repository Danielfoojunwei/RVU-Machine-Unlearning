#!/usr/bin/env bash
# download_models.sh -- Download all required models via huggingface-cli.
# Usage: bash scripts/download_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_DIR="${REPO_ROOT}/artifacts"
MANIFEST="${ARTIFACTS_DIR}/model_manifest.json"
mkdir -p "${ARTIFACTS_DIR}"

# Activate venv if available
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
    exit 1
fi

# ───────────────────────────────────────────────────────────────────────
# Helper: download a model and capture its revision hash
# ───────────────────────────────────────────────────────────────────────
declare -A MODEL_REVISIONS

download_model() {
    local repo="$1"
    local local_dir="$2"
    local key="$3"
    shift 3
    # remaining args are extra flags (e.g. --include patterns)
    local extra_args=("$@")

    echo ""
    echo "==> Downloading ${repo} -> ${local_dir}"
    mkdir -p "${local_dir}"

    huggingface-cli download "${repo}" \
        --local-dir "${local_dir}" \
        "${extra_args[@]}" \
        2>&1 | tee /tmp/_hf_download_log.txt

    # Try to capture the revision from the download metadata
    local revision="unknown"
    if [[ -f "${local_dir}/.huggingface/download_metadata.json" ]]; then
        revision=$(python3 -c "
import json
try:
    with open('${local_dir}/.huggingface/download_metadata.json') as f:
        d = json.load(f)
    print(d.get('commit_hash', d.get('revision', 'unknown')))
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")
    fi

    # Fallback: try huggingface_hub Python API
    if [[ "${revision}" == "unknown" ]]; then
        revision=$(python3 -c "
from huggingface_hub import model_info
try:
    info = model_info('${repo}')
    print(info.sha)
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")
    fi

    MODEL_REVISIONS["${key}"]="${revision}"
    echo "    Revision: ${revision}"
}

# ───────────────────────────────────────────────────────────────────────
# 1. Qwen2.5-1.5B-Instruct GGUF (Q4_K_M)
# ───────────────────────────────────────────────────────────────────────
download_model \
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF" \
    "${REPO_ROOT}/models/llm/qwen2.5-1.5b-instruct" \
    "qwen2.5-1.5b-instruct" \
    --include "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# ───────────────────────────────────────────────────────────────────────
# 2. Llama-3.2-1B-Instruct GGUF (Q4_K_M)
# ───────────────────────────────────────────────────────────────────────
download_model \
    "bartowski/Llama-3.2-1B-Instruct-GGUF" \
    "${REPO_ROOT}/models/llm/llama-3.2-1b-instruct" \
    "llama-3.2-1b-instruct" \
    --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# ───────────────────────────────────────────────────────────────────────
# 3. Prompt Guard 2 86M (full model, all files)
# ───────────────────────────────────────────────────────────────────────
download_model \
    "meta-llama/Llama-Prompt-Guard-2-86M" \
    "${REPO_ROOT}/models/guard/prompt-guard-2-86m" \
    "prompt-guard-2-86m"

# ───────────────────────────────────────────────────────────────────────
# 4. BGE-small-en-v1.5 (embedding model, all files)
# ───────────────────────────────────────────────────────────────────────
download_model \
    "BAAI/bge-small-en-v1.5" \
    "${REPO_ROOT}/models/embed/bge-small-en-v1.5" \
    "bge-small-en-v1.5"

# ───────────────────────────────────────────────────────────────────────
# 5. Write model manifest (with revision hashes passed via env)
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "==> Writing model manifest to ${MANIFEST}"

export QWEN_REV="${MODEL_REVISIONS[qwen2.5-1.5b-instruct]:-unknown}"
export LLAMA_REV="${MODEL_REVISIONS[llama-3.2-1b-instruct]:-unknown}"
export PG_REV="${MODEL_REVISIONS[prompt-guard-2-86m]:-unknown}"
export BGE_REV="${MODEL_REVISIONS[bge-small-en-v1.5]:-unknown}"

python3 - "${MANIFEST}" <<'PYEOF'
import json, sys, os
from datetime import datetime, timezone

manifest_path = sys.argv[1]

models = {
    "qwen2.5-1.5b-instruct": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "local_dir": "models/llm/qwen2.5-1.5b-instruct",
        "gguf_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "revision": os.environ.get("QWEN_REV", "unknown"),
    },
    "llama-3.2-1b-instruct": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "local_dir": "models/llm/llama-3.2-1b-instruct",
        "gguf_file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "revision": os.environ.get("LLAMA_REV", "unknown"),
    },
    "prompt-guard-2-86m": {
        "repo": "meta-llama/Llama-Prompt-Guard-2-86M",
        "local_dir": "models/guard/prompt-guard-2-86m",
        "revision": os.environ.get("PG_REV", "unknown"),
    },
    "bge-small-en-v1.5": {
        "repo": "BAAI/bge-small-en-v1.5",
        "local_dir": "models/embed/bge-small-en-v1.5",
        "revision": os.environ.get("BGE_REV", "unknown"),
    },
}

manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "models": models,
}

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
PYEOF

# ───────────────────────────────────────────────────────────────────────
# 6. Verify downloads exist
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying downloads..."

FAIL=0
verify_file() {
    local path="$1"
    local label="$2"
    if [[ -e "${path}" ]]; then
        local size
        size=$(du -sh "${path}" 2>/dev/null | cut -f1)
        echo "    [OK]   ${label}  (${size})"
    else
        echo "    [FAIL] ${label}  -- NOT FOUND at ${path}"
        FAIL=1
    fi
}

verify_file "${REPO_ROOT}/models/llm/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "Qwen2.5-1.5B-Instruct Q4_K_M"

verify_file "${REPO_ROOT}/models/llm/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "Llama-3.2-1B-Instruct Q4_K_M"

verify_file "${REPO_ROOT}/models/guard/prompt-guard-2-86m/config.json" \
    "Prompt Guard 2 86M"

verify_file "${REPO_ROOT}/models/embed/bge-small-en-v1.5/config.json" \
    "BGE-small-en-v1.5"

# ───────────────────────────────────────────────────────────────────────
# 7. Summary
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Model Download Summary"
echo "================================================================"

for dir in \
    "${REPO_ROOT}/models/llm/qwen2.5-1.5b-instruct" \
    "${REPO_ROOT}/models/llm/llama-3.2-1b-instruct" \
    "${REPO_ROOT}/models/guard/prompt-guard-2-86m" \
    "${REPO_ROOT}/models/embed/bge-small-en-v1.5"; do
    if [[ -d "${dir}" ]]; then
        size=$(du -sh "${dir}" 2>/dev/null | cut -f1)
        echo "  $(basename "${dir}"): ${size}"
    fi
done

echo "  Manifest: ${MANIFEST}"
echo "================================================================"

if [[ "${FAIL}" -ne 0 ]]; then
    echo ""
    echo "WARNING: Some model files were not found. Check errors above."
    exit 1
fi

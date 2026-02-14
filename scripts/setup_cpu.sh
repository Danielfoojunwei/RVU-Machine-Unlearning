#!/usr/bin/env bash
# setup_cpu.sh -- Create venv, install pinned dependencies, clone benchmark repos.
# Usage: bash scripts/setup_cpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
VENDOR_DIR="${REPO_ROOT}/vendor"

# ───────────────────────────────────────────────────────────────────────
# 1. Create Python virtual-environment
# ───────────────────────────────────────────────────────────────────────
echo "==> Creating Python virtual-environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel

# ───────────────────────────────────────────────────────────────────────
# 2. Install pinned dependencies
# ───────────────────────────────────────────────────────────────────────
echo "==> Installing pinned Python packages"

pip install \
    "llama-cpp-python==0.3.4" \
    "transformers==4.46.3" \
    "sentence-transformers==3.3.1" \
    "faiss-cpu==1.9.0.post1" \
    "numpy==1.26.4" \
    "scipy==1.14.1" \
    "pandas==2.2.3" \
    "pytest==8.3.4" \
    "hypothesis==6.115.6" \
    "jsonlines==4.0.0" \
    "pydantic==2.10.3" \
    "networkx==3.4.2" \
    "ruff==0.8.4" \
    "mypy==1.13.0" \
    "pyyaml==6.0.2" \
    "matplotlib==3.9.3"

# PyTorch CPU-only wheel from the dedicated index.
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu

# ───────────────────────────────────────────────────────────────────────
# 3. Install the project itself in editable mode
# ───────────────────────────────────────────────────────────────────────
echo "==> Installing rvu package in editable mode"
pip install -e "${REPO_ROOT}"

# ───────────────────────────────────────────────────────────────────────
# 4. Clone and install benchmark vendor repos
# ───────────────────────────────────────────────────────────────────────
mkdir -p "${VENDOR_DIR}"

clone_vendor() {
    local name="$1"
    local url="$2"
    local commit="${3:-}"          # optional pinned commit
    local dest="${VENDOR_DIR}/${name}"

    if [[ -d "${dest}" ]]; then
        echo "    [skip] ${name} already exists at ${dest}"
    else
        echo "    [clone] ${name} <- ${url}"
        git clone --depth 1 "${url}" "${dest}"
        if [[ -n "${commit}" ]]; then
            (cd "${dest}" && git fetch --depth 1 origin "${commit}" && git checkout "${commit}")
        fi
    fi

    # Install if a setup.py / pyproject.toml exists
    if [[ -f "${dest}/setup.py" ]] || [[ -f "${dest}/pyproject.toml" ]]; then
        echo "    [pip]   installing ${name}"
        pip install -e "${dest}" || echo "    [warn] pip install for ${name} failed (non-fatal)"
    else
        echo "    [info]  ${name} has no setup.py/pyproject.toml -- skipping pip install"
    fi
}

echo "==> Cloning benchmark vendor repositories into ${VENDOR_DIR}"

clone_vendor "agentdojo-inspect" \
    "https://github.com/usnistgov/agentdojo-inspect.git"

clone_vendor "InjecAgent" \
    "https://github.com/uiuc-kang-lab/InjecAgent.git"

clone_vendor "BIPIA" \
    "https://github.com/microsoft/BIPIA.git"

clone_vendor "FATH" \
    "https://github.com/Jayfeather1024/FATH.git"

# ───────────────────────────────────────────────────────────────────────
# 5. Make all project scripts executable
# ───────────────────────────────────────────────────────────────────────
echo "==> Setting executable permissions on scripts/"
chmod +x "${REPO_ROOT}"/scripts/*.sh
if compgen -G "${REPO_ROOT}/scripts/*.py" > /dev/null; then
    chmod +x "${REPO_ROOT}"/scripts/*.py
fi

# ───────────────────────────────────────────────────────────────────────
# 6. Print summary
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  RVU CPU Environment Setup Complete"
echo "================================================================"
echo "  Python  : $(python3 --version)"
echo "  Pip     : $(pip --version)"
echo "  Venv    : ${VENV_DIR}"
echo "  Vendor  : ${VENDOR_DIR}"
echo "----------------------------------------------------------------"
echo "  Installed packages:"
pip list --format=columns
echo "================================================================"
echo ""
echo "Activate with:  source ${VENV_DIR}/bin/activate"

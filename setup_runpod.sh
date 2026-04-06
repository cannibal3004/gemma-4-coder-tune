#!/bin/bash
# Run once after pod starts: source setup_runpod.sh
# Using 'source' (not 'bash') so the exports persist in your current shell

# --- Workspace dirs ---
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export OUTPUTS_DIR=/workspace/outputs

mkdir -p /workspace/hf_cache
mkdir -p /workspace/outputs

# Point config.yaml output dir at /workspace
sed -i 's|output_dir: ".*"|output_dir: "/workspace/outputs"|' config.yaml

# --- Miniforge (local container disk — fast for small files) ---
# NOTE: this reinstalls on pod restart, but takes ~2min on local NVMe vs hours on /workspace
CONDA_DIR=/root/miniforge

if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing miniforge to $CONDA_DIR..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    rm /tmp/miniforge.sh
    echo "Miniforge installed."
fi

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# --- Conda env (creates once, persists in /workspace) ---
if ! conda env list | grep -q "^finetune "; then
    echo "Creating finetune env (Python 3.11)..."
    conda create -y -n finetune python=3.11 -q
fi

conda activate finetune

# --- Python deps (skip if already installed) ---
if ! python -c "import unsloth" 2>/dev/null; then
    echo "Installing Python deps..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --cache-dir /tmp/pip-cache
    pip install -q -r requirements.txt --cache-dir /tmp/pip-cache
    echo "Deps installed."
fi

echo ""
echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "Outputs: /workspace/outputs"
echo "Monitor disk: watch -n10 'df -h /workspace && du -sh /workspace/*/'"
echo ""
echo "Ready. Run: python train.py"

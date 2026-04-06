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

# --- Miniconda (installs once, persists in /workspace) ---
CONDA_DIR=/workspace/miniconda

if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing miniconda to $CONDA_DIR..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "Miniconda installed."
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
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -q -r requirements.txt
    echo "Deps installed."
fi

echo ""
echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "Outputs: /workspace/outputs"
echo "Monitor disk: watch -n10 'df -h /workspace && du -sh /workspace/*/'"
echo ""
echo "Ready. Run: python train.py"

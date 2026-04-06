#!/bin/bash
# Run once after pod starts: source setup_runpod.sh
# Using 'source' (not 'bash') so the exports persist in your current shell

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export OUTPUTS_DIR=/workspace/outputs

mkdir -p /workspace/hf_cache
mkdir -p /workspace/outputs

# Point config.yaml output dir at /workspace
sed -i 's|output_dir: ".*"|output_dir: "/workspace/outputs"|' config.yaml

echo "HF_HOME=$HF_HOME"
echo "Outputs -> /workspace/outputs"
echo "Ready."
echo ""
echo "Monitor disk: watch -n10 'df -h /workspace && du -sh /workspace/*/'"

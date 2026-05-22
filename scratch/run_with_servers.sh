#!/bin/bash
set -euo pipefail

echo "========================================================="
echo "🚀 STARTING MULTIMODAL PROMPT OPTIMIZATION WITH ACTUAL VLM & DIFFUSER"
echo "========================================================="

# Create necessary outputs directories
mkdir -p /scratch/Mock_repo_mnlp/outputs/generated_images

# ── env vars consumed by multimodal_optimization.py and metrics.py ──────────
# Rewriter / synthetic-data calls go to port 8002 (Qwen2.5-14B-Instruct-AWQ).
# Judge calls go to port 8000 (Qwen2-VL-7B-Instruct) — set via LOCAL_VLM_URL.
export OPTIMIZER_MODEL="openai/Qwen/Qwen2.5-14B-Instruct-AWQ"
export OPTIMIZER_API_BASE="http://localhost:8002/v1"
export OPTIMIZER_API_KEY="local"
export OPTIMIZER_PROVIDER="local"
export LOCAL_VLM_URL="http://localhost:8000/v1/chat/completions"
export LOCAL_DIFFUSION_URL="http://localhost:8001/v1/images/generations"

# 1. Start vLLM Judge Server in background (Qwen2-VL-7B handles the VLM judge).
# Allocation set to 0.50 — the original 0.55 was generous; 0.40 starved the KV
# cache and crashed init with "No available memory for the cache blocks".
# 0.50 × 40 GB = 20 GB: ~15 GB weights + ~3-4 GB KV cache at max-model-len 8192.
echo "⏳ Launching local vLLM VLM Server (Qwen2-VL-7B-Instruct) — judge..."
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 8192 \
  --trust-remote-code > /scratch/vllm_server.log 2>&1 &
VLM_PID=$!

# 2. Start Diffusion Server in background
echo "⏳ Launching local Stable Diffusion Server (SDXL base 1.0)..."
python3 scratch/local_diffusion_server.py > /scratch/diffusion_server.log 2>&1 &
DIFF_PID=$!

# 3. Start vLLM Rewriter Server in background (Qwen2.5-14B-Instruct-AWQ, 4-bit).
# Text-only instruction-tuned model with stronger rule following than VL-7B —
# previous runs showed VL-7B locked into repetition loops on tag-style output.
# Allocation: 0.25 × 40 GB = 10 GB target. AWQ weights ~7 GB, leaving ~2.5 GB
# for KV cache + overhead. (0.20 = 8 GB failed: KV cache had no room.)
# max-model-len dropped 4096 → 2048: meta-prompt + 120-token output never
# exceeds ~700 tokens, so 2048 is generous and halves KV cache reservation.
echo "⏳ Launching local vLLM Rewriter Server (Qwen2.5-14B-Instruct-AWQ)..."
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8002 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 2048 \
  --quantization awq \
  --trust-remote-code > /scratch/vllm_rewriter.log 2>&1 &
REWRITER_PID=$!

# Function to clean up background servers on exit
cleanup() {
  echo "🧹 Cleaning up background servers (PIDs: $VLM_PID, $DIFF_PID, $REWRITER_PID)..."
  kill $VLM_PID $DIFF_PID $REWRITER_PID 2>/dev/null || true
  wait $VLM_PID $DIFF_PID $REWRITER_PID 2>/dev/null || true
  echo "✅ Background servers terminated successfully!"
}
trap cleanup EXIT

# 3. Wait for ports 8000 and 8001 to be active
echo "⏳ Waiting for servers to be online..."
python3 -c '
import socket
import time
import sys

def wait_for_port(port, name, timeout=300):
    start = time.time()
    print(f"Waiting for {name} on port {port}...", flush=True)
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                print(f"✅ {name} is online!", flush=True)
                return True
        except OSError:
            time.sleep(5)
    print(f"❌ Timeout waiting for {name} on port {port}!", flush=True)
    return False

# vLLM might take a minute or two to download/load weights
if not wait_for_port(8000, "vLLM Judge Server", timeout=420):
    sys.exit(1)
if not wait_for_port(8001, "Stable Diffusion Server", timeout=120):
    sys.exit(1)
if not wait_for_port(8002, "vLLM Rewriter Server", timeout=600):
    sys.exit(1)
'

echo "========================================================="
echo "🎉 ALL SERVERS ONLINE! EXECUTING OPTIMIZER PIPELINE"
echo "========================================================="

# 4. Run the optimizer once per concept — each concept is its own prompt being optimized.
CONCEPTS=(
  "Futuristic cyberpunk skyscraper inside a green rainforest dome, synthwave theme"
  "A vintage Victorian library floating in outer space"
  "Minimalist geometric sculpture on an empty white beach, soft morning light"
)

for i in "${!CONCEPTS[@]}"; do
  CONCEPT="${CONCEPTS[$i]}"
  IDX=$((i + 1))
  TOTAL=${#CONCEPTS[@]}

  # Slugify the concept for use as a per-concept folder name under
  # outputs/generated_images/. The metric reads CONCEPT_LABEL to route saves.
  CONCEPT_LABEL=$(python3 -c '
import re, sys
s = re.sub(r"[^A-Za-z0-9]+", "_", sys.argv[1]).strip("_").lower()
print((s[:60] or "unlabeled"))
' "$CONCEPT")
  export CONCEPT_LABEL

  # Pin the judge target to the user's original concept text. Without this
  # the VLM judge evaluates each image against the drifted synthetic
  # variation (which the synth generator often pads with extra attributes
  # not in the user's concept) and penalizes faithful images.
  export TARGET_CONCEPT="$CONCEPT"

  echo ""
  echo "========================================================="
  echo "🎯 Optimizing concept ${IDX}/${TOTAL}  [folder: ${CONCEPT_LABEL}]"
  echo "    ${CONCEPT}"
  echo "========================================================="

  # Build sample_data JSON safely (handles quotes/specials in the concept text)
  SAMPLE_JSON=$(python3 -c 'import json,sys; c=sys.argv[1]; print(json.dumps([{"concept": c, "output_prompt": c}]))' "$CONCEPT")

  # Rewriter/synthetic-data routed to the 14B-AWQ on port 8002.
  # The judge still hits Qwen2-VL-7B on port 8000 via LOCAL_VLM_URL in metrics.py.
  python3 -m promptomatix.main \
    --raw_input "$CONCEPT" \
    --task "$CONCEPT" \
    --task_type "image_generation" \
    --input_fields concept \
    --output_fields output_prompt \
    --model_name "openai/Qwen/Qwen2.5-14B-Instruct-AWQ" \
    --model_api_base "http://127.0.0.1:8002/v1" \
    --model_api_key "mock" \
    --model_provider "openai" \
    --config_model_name "openai/Qwen/Qwen2.5-14B-Instruct-AWQ" \
    --config_model_api_base "http://127.0.0.1:8002/v1" \
    --config_model_api_key "mock" \
    --config_model_provider "openai" \
    --synthetic_data_size 5 \
    --max_tokens 2048 \
    --config_max_tokens 2048 \
    --config_temperature 0.7 \
    --backend "simple_meta_prompt" \
    --sample_data "$SAMPLE_JSON"
done


echo "========================================================="
echo "🎉 PIPELINE COMPLETE! (${#CONCEPTS[@]} concepts optimized)"
echo "========================================================="

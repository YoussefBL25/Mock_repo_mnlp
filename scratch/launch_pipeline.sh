#!/bin/bash
# launch_pipeline.sh
#
# Fully local pipeline on a single A100-40G:
#
#   Port 8000 — vLLM serving Qwen2-VL-7B-Instruct (~26 GB)
#               Acts as BOTH the VLM judge (text+image) and the optimizer LLM (text-only).
#               Qwen2-VL supports text-only inputs, so no separate optimizer server is needed.
#
#   Port 8001 — Diffusion server (SDXL fp16, ~7 GB)
#
#   Total VRAM: ~33 GB out of 40 GB.
#
# Optional env var overrides:
#   VLM_MODEL       — HF model ID for vLLM (judge + optimizer).
#                     Default: Qwen/Qwen2-VL-7B-Instruct
#                     Override to a path in /shared-ro/models if already downloaded.
#   DIFFUSION_MODEL — HF model ID for SDXL.
#                     Default: stabilityai/stable-diffusion-xl-base-1.0
#   N_FEEDBACK_ITERATIONS / SYNTHETIC_DATA_SIZE / OUTPUT_DIR — forwarded to Python.

set -euo pipefail

REPO=/scratch/Mock_repo_mnlp
LOGS=/scratch/logs/services
mkdir -p "$LOGS"

VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-stabilityai/stable-diffusion-xl-base-1.0}"

echo "============================================================"
echo "  LAUNCH PIPELINE  (fully local, single A100-40G)"
echo "============================================================"
echo "  VLM (judge + optimizer) : $VLM_MODEL  → port 8000"
echo "  Diffusion               : $DIFFUSION_MODEL  → port 8001"
echo "  VRAM budget             : ~26 GB (vLLM) + ~7 GB (SDXL) = ~33 GB / 40 GB"
echo "  Logs                    : $LOGS"
echo "============================================================"
echo ""

# ── 1. Start diffusion server on port 8001 ──────────────────────────────────
echo "[1/2] Starting diffusion server (port 8001)..."
DIFFUSION_MODEL="$DIFFUSION_MODEL" \
    python3 "$REPO/scratch/local_diffusion_server.py" \
    > "$LOGS/diffusion.log" 2>&1 &
DIFFUSION_PID=$!
echo "      PID $DIFFUSION_PID  —  tail -f $LOGS/diffusion.log"

# ── 2. Start vLLM (VLM judge + optimizer) on port 8000 ──────────────────────
echo ""
echo "[2/2] Starting vLLM ($VLM_MODEL) on port 8000..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.65 \
    --max-model-len 32768 \
    --served-model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOGS/vllm.log" 2>&1 &
VLLM_PID=$!
echo "      PID $VLLM_PID  —  tail -f $LOGS/vllm.log"

# ── Wait for both services to be ready ──────────────────────────────────────
echo ""
echo "Waiting for services..."

wait_for_url() {
    local url="$1" label="$2" max_wait="${3:-720}" elapsed=0
    printf "  %-40s" "$label"
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 10; elapsed=$((elapsed + 10))
        printf "."
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo " TIMEOUT"
            echo "ERROR: $label did not become ready within ${max_wait}s."
            echo "       Check $LOGS for details."
            kill "$DIFFUSION_PID" "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi
    done
    printf " ready (%ds)\n" "$elapsed"
}

# Diffusion: /health triggers eager model load (added in local_diffusion_server.py)
wait_for_url "http://localhost:8001/health"    "Diffusion server (port 8001)" 600

# vLLM: /v1/models returns 200 only once the model weights are fully loaded
wait_for_url "http://localhost:8000/v1/models" "vLLM (port 8000)"             720

# ── Export env vars consumed by metrics.py and multimodal_optimization.py ───
export LOCAL_DIFFUSION_URL="http://localhost:8001/v1/images/generations"
export LOCAL_VLM_URL="http://localhost:8000/v1/chat/completions"

# Optimizer: same vLLM instance, text-only inputs, no API key required.
# Use provider="local" so Config skips the OPENAI_API_KEY env var lookup.
export OPTIMIZER_MODEL="openai/Qwen/Qwen2-VL-7B-Instruct"
export OPTIMIZER_API_BASE="http://localhost:8000/v1"
export OPTIMIZER_API_KEY="local"
export OPTIMIZER_PROVIDER="local"

echo ""
echo "============================================================"
echo "  All services ready — starting optimization pipeline"
echo "============================================================"
echo ""

# ── Run the real optimization pipeline ──────────────────────────────────────
python3 "$REPO/promptomatix/examples/scripts/multimodal_optimization.py"
EXIT_CODE=$?

# ── Cleanup ──────────────────────────────────────────────────────────────────
echo ""
echo "Pipeline finished (exit $EXIT_CODE). Stopping services..."
kill "$DIFFUSION_PID" "$VLLM_PID" 2>/dev/null || true

exit $EXIT_CODE

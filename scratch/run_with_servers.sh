#!/bin/bash
set -euo pipefail

echo "========================================================="
echo "🚀 STARTING MULTIMODAL PROMPT OPTIMIZATION WITH ACTUAL VLM & DIFFUSER"
echo "========================================================="

# Create necessary outputs directories
mkdir -p /scratch/Mock_repo_mnlp/outputs/generated_images

# 1. Start vLLM Server in background
echo "⏳ Launching local vLLM VLM Server (Qwen2-VL-7B-Instruct)..."
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 4096 \
  --trust-remote-code > /scratch/vllm_server.log 2>&1 &
VLM_PID=$!

# 2. Start Diffusion Server in background
echo "⏳ Launching local Stable Diffusion Server (SDXL base 1.0)..."
python3 scratch/local_diffusion_server.py > /scratch/diffusion_server.log 2>&1 &
DIFF_PID=$!

# Function to clean up background servers on exit
cleanup() {
  echo "🧹 Cleaning up background servers (PIDs: $VLM_PID, $DIFF_PID)..."
  kill $VLM_PID $DIFF_PID 2>/dev/null || true
  wait $VLM_PID $DIFF_PID 2>/dev/null || true
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
if not wait_for_port(8000, "vLLM Server", timeout=420):
    sys.exit(1)
if not wait_for_port(8001, "Stable Diffusion Server", timeout=120):
    sys.exit(1)
'

echo "========================================================="
echo "🎉 BOTH SERVERS ONLINE! EXECUTING OPTIMIZER PIPELINE"
echo "========================================================="

# 4. Run the actual optimizer!
python3 promptomatix/examples/scripts/multimodal_optimization.py

echo "========================================================="
echo "🎉 PIPELINE COMPLETE!"
echo "========================================================="

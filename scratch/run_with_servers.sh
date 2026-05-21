#!/bin/bash
set -euo pipefail

echo "========================================================="
echo "🚀 STARTING MULTIMODAL PROMPT OPTIMIZATION WITH ACTUAL VLM & DIFFUSER"
echo "========================================================="

# Create necessary outputs directories
mkdir -p /scratch/Mock_repo_mnlp/outputs/generated_images

# Clean up any lingering processes on ports 8000 and 8001
echo "🧹 Checking and cleaning up lingering processes on ports 8000 and 8001..."
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8000/tcp || true
  fuser -k 8001/tcp || true
elif command -v lsof >/dev/null 2>&1; then
  kill -9 $(lsof -t -i:8000) 2>/dev/null || true
  kill -9 $(lsof -t -i:8001) 2>/dev/null || true
else
  python3 -c '
import os
import signal
try:
    import psutil
    for conn in psutil.net_connections():
        if conn.laddr.port in [8000, 8001] and conn.pid:
            print(f"Killing process {conn.pid} on port {conn.laddr.port}")
            os.kill(conn.pid, signal.SIGKILL)
except Exception as e:
    print(f"Python cleanup check bypassed: {e}")
' || true
fi

# 1. Start vLLM Server in background
echo "⏳ Launching local vLLM VLM Server (Qwen2-VL-7B-Instruct)..."
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 4096 \
  --served-model-name Qwen/Qwen2-VL-7B-Instruct \
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
python3 -m promptomatix.main \
  --raw_input "A photo of a {concept}." \
  --task "A photo of a {concept}." \
  --task_type "image_generation" \
  --input_fields concept \
  --output_fields output_prompt \
  --model_name "openai/Qwen/Qwen2-VL-7B-Instruct" \
  --model_api_base "http://127.0.0.1:8000/v1" \
  --model_api_key "mock" \
  --model_provider "openai" \
  --config_model_name "openai/Qwen/Qwen2-VL-7B-Instruct" \
  --config_model_api_base "http://127.0.0.1:8000/v1" \
  --config_model_api_key "mock" \
  --config_model_provider "openai" \
  --synthetic_data_size 10 \
  --max_tokens 2048 \
  --config_max_tokens 2048 \
  --config_temperature 0.7 \
  --backend "simple_meta_prompt" \
  --sample_data '[{"concept": "Futuristic cyberpunk skyscraper inside a green rainforest dome, synthwave theme", "output_prompt": "Futuristic synthwave cyberpunk skyscraper rising inside a massive bioluminescent green rainforest glass dome, dramatic lighting, highly detailed digital art"}, {"concept": "A vintage Victorian library floating in outer space, warm cozy fireplace", "output_prompt": "A warm cozy vintage Victorian library floating in outer space, stars visible through giant glass windows, glowing fireplace, oil painting style"}, {"concept": "Minimalist geometric sculpture on an empty white beach, soft morning light", "output_prompt": "A minimalist geometric sculpture standing on a vast empty white sand beach, illuminated by soft golden morning sunlight, photorealistic"}]'


echo "========================================================="
echo "🎉 PIPELINE COMPLETE!"
echo "========================================================="

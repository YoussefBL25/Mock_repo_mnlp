"""
local_diffusion_server.py

Lightweight local API server wrapping Hugging Face diffusers.
Exposes an OpenAI-compatible `/v1/images/generations` endpoint on Port 8001.
Designed to share single-GPU VRAM with vLLM safely.
"""

import os

# Reduce CUDA allocator fragmentation when sharing the GPU with vLLM —
# vLLM's allocations can leave the free pool fragmented, causing SDXL's
# 512 MB VAE allocation to fail even when total free memory is sufficient.
# Must be set BEFORE torch initializes CUDA, so it lives above the torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
import base64
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize pipeline globally
pipe = None

def load_pipeline():
    global pipe
    if pipe is not None:
        return
        
    print("⏳ Loading Diffusion Pipeline onto single GPU...")
    from diffusers import DiffusionPipeline, AutoencoderKL

    # We use FLUX.1-schnell or SDXL for fast cluster visual optimization
    model_id = os.environ.get("DIFFUSION_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

    try:
        # SDXL's stock VAE has known numerical overflow in fp16 — it produces
        # NaN outputs that save as fully black images (every pixel clamped to
        # zero). The community 'madebyollin/sdxl-vae-fp16-fix' VAE is a
        # drop-in replacement that's numerically stable in fp16, same size,
        # ~167 MB download on first use.
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.to("cuda")

        # NB: do NOT force-cast the VAE (or any submodule) to fp16 here.
        # SDXL's VAE has known numerical instability in pure fp16 and
        # `from_pretrained(torch_dtype=torch.float16)` already keeps certain
        # internal ops at higher precision. A hard cast collapses VAE output
        # quality (images come out partially decoded / washed-out / NaN).
        # The autocast wrapper around inference (below) catches the original
        # dtype-mismatch issue without modifying weights.

        # VRAM optimization configurations
        pipe.enable_attention_slicing()
        # VAE tiling + slicing: the VAE decode step is the peak-memory moment
        # of SDXL inference (1024×1024 latent → image needs ~800 MB transient).
        # Tiling processes the latent in chunks, dropping peak to <100 MB.
        # Required when sharing the GPU with vLLM judge + rewriter on a 40 GB
        # card — without this we OOM at the final decode step.
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        # if hasattr(pipe, "enable_model_cpu_offload"):
        #     pipe.enable_model_cpu_offload()

        print(f"✅ Diffusion Pipeline ({model_id}) Loaded Successfully on GPU!")
    except Exception as e:
        print(f"❌ Error loading diffusion pipeline: {str(e)}")
        sys.exit(1)

@app.route("/health", methods=["GET"])
def health():
    load_pipeline()
    return {"status": "ok", "model": os.environ.get("DIFFUSION_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")}

@app.route("/v1/images/generations", methods=["POST"])
def generate_image():
    global pipe
    load_pipeline()
    
    data = request.json or {}
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    print(f"🖼️ Generating image for prompt: '{prompt}'")
    
    try:
        # Run inference in FP16 with minimal steps for speed.
        # autocast belt-and-suspenders against the float16/float32 mismatch
        # that fires intermittently when vLLM modifies global dtype state.
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            image = pipe(prompt=prompt, num_inference_steps=20).images[0]
            
        # Encode output image directly to base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({
            "created": 1234567,
            "data": [
                {
                    "b64_json": img_str,
                    "url": ""
                }
            ]
        })
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"🚀 Starting Local Diffusion Server on Port {port}...")
    load_pipeline()  # load model before accepting requests to avoid first-request timeout
    app.run(host="0.0.0.0", port=port, debug=False)

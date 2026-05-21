"""
local_diffusion_server.py

Lightweight local API server wrapping Hugging Face diffusers.
Exposes an OpenAI-compatible `/v1/images/generations` endpoint on Port 8001.
Designed to share single-GPU VRAM with vLLM safely.
"""

import os
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
    from diffusers import DiffusionPipeline
    
    # We use FLUX.1-schnell or SDXL for fast cluster visual optimization
    model_id = os.environ.get("DIFFUSION_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    
    try:
        # Load in half precision (float16) to conserve VRAM for vLLM
        pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        )
        pipe.to("cuda")
        
        # VRAM optimization configurations
        pipe.enable_attention_slicing()
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
        # Run inference in FP16 with minimal steps for speed
        with torch.inference_mode():
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

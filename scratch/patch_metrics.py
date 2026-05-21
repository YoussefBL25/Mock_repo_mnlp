import os

path = "/scratch/Mock_repo_mnlp/promptomatix/src/promptomatix/metrics/metrics.py"
if not os.path.exists(path):
    path = "promptomatix/src/promptomatix/metrics/metrics.py"

print(f"Targeting metrics.py at: {path}")

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'diffusion_url = os.environ.get("LOCAL_DIFFUSION_URL"' in line:
        start_idx = i
    if '# 3. ROBUST DEVELOPER FALLBACK: If APIs are not online' in line:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    print(f"Found target block: lines {start_idx + 1} to {end_idx + 1}")
    
    new_block = """            diffusion_url = os.environ.get("LOCAL_DIFFUSION_URL", "http://127.0.0.1:8001/v1/images/generations")
            vlm_url = os.environ.get("LOCAL_VLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
            
            try:
                # 1. Contact local diffusion pipeline to generate image
                payload = {
                    "prompt": candidate_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "response_format": "b64_json"
                }
                headers = {"Content-Type": "application/json"}
                response = requests.post(diffusion_url, json=payload, headers=headers, timeout=45)
                
                if response.status_code == 200:
                    b64_data = response.json()["data"][0]["b64_json"]
                    
                    # Proactively decode and save generated image to disk for user inspection
                    try:
                        import time
                        clean_prompt = "".join([c if c.isalnum() else "_" for c in candidate_prompt[:30].strip()])
                        filename = f"gen_{clean_prompt}_{int(time.time())}.png"
                        
                        output_dir = os.environ.get("GENERATED_IMAGES_DIR", "/scratch/Mock_repo_mnlp/outputs/generated_images")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                            
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, "wb") as fh:
                            fh.write(base64.b64decode(b64_data))
                        print(f"📸 [SYSTEM SUCCESS] Generated image saved to disk: {filepath}")
                    except Exception as save_err:
                        print(f"⚠️ [SYSTEM WARNING] Failed to save generated image to disk: {str(save_err)}")
                else:
                    raise ConnectionError("Diffusion server returned non-200 status code.")
                    
                # 2. Contact local VLM Judge to evaluate prompt adherence and aesthetics
                vlm_payload = {
                    "model": "Qwen/Qwen2-VL-7B-Instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"You are an expert visual evaluation judge. Analyze the provided image against this prompt: '{concept_text}'. Rate these three criteria from 0.0 (poor) to 1.0 (excellent):\\n1) adherence_score: How closely does the image content match the prompt's core semantic details?\\n2) aesthetic_score: Rate the visual quality, details, contrast, composition, and aesthetics.\\n3) artifact_score: Rate the absence of weird artifacts, bad anatomy, blur, or rendering defects (1.0 means no defects, 0.0 means completely distorted).\\n\\nYou MUST respond strictly in valid JSON format inside a ```json``` codeblock like this:\\n```json\\n{{\\n  \\"adherence_score\\": 0.85,\\n  \\"aesthetic_score\\": 0.90,\\n  \\"artifact_score\\": 0.95\\n}}\\n```\\nDo not write any introductory or concluding text. Output only the JSON block."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150
                }
                
                vlm_resp = requests.post(vlm_url, json=vlm_payload, headers=headers, timeout=30)
                if vlm_resp.status_code == 200:
                    raw_content = vlm_resp.json()["choices"][0]["message"]["content"].strip()
                    # Clean up common markdown block issues
                    clean_content = raw_content.strip()
                    if "```json" in clean_content:
                        clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_content:
                        clean_content = clean_content.split("```")[1].split("```")[0].strip()
                    
                    eval_data = json.loads(clean_content)
                    
                    print(f"🤖 [VLM JUDGE FEEDBACK COMPLETED]")
                    print(f"   Raw Judge Content:\\n{raw_content}")
                    print(f"   Breakdown: Adherence (50%): {eval_data.get('adherence_score', 0.0):.2f} | Aesthetics (30%): {eval_data.get('aesthetic_score', 0.0):.2f} | Artifacts (20%): {eval_data.get('artifact_score', 0.0):.2f}")
                    
                    w_adherence = float(eval_data.get("adherence_score", 0.0)) * 0.50
                    w_aesthetic = float(eval_data.get("aesthetic_score", 0.0)) * 0.30
                    w_artifacts = float(eval_data.get("artifact_score", 0.0)) * 0.20
                    score = w_adherence + w_aesthetic + w_artifacts
                else:
                    raise ConnectionError("VLM Judge returned non-200 status code.")
\n"""
    
    lines[start_idx:end_idx] = [new_block]
    
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("✅ metrics.py patched successfully!")
else:
    print("❌ Target block not found!")

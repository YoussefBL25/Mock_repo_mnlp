#!/usr/bin/env python3
"""
multimodal_optimization.py

Real multimodal prompt optimization using Promptomatix.

- Step 1: process_input()  — LLM generates synthetic data, runs initial optimization
- Step 2+: generate_feedback() + optimize_with_feedback()  — N real feedback iterations
- Metrics: VLM-judge + diffusion server when LOCAL_DIFFUSION_URL / LOCAL_VLM_URL are reachable,
           word-overlap fallback otherwise (set by metrics.py automatically)

Required env vars:
  OPTIMIZER_API_KEY   — OpenAI API key (or any string for local vLLM)
  OPTIMIZER_MODEL     — model name, e.g. "openai/gpt-4o-mini" or
                        "openai/Qwen/Qwen2.5-7B-Instruct" for local vLLM
  OPTIMIZER_API_BASE  — (optional) vLLM base URL, e.g. http://localhost:8002/v1
  OPTIMIZER_PROVIDER  — (optional) "openai" (default) | "anthropic" | "local"

Optional:
  N_FEEDBACK_ITERATIONS  — number of feedback loops (default: 2)
  SYNTHETIC_DATA_SIZE    — synthetic samples to generate (default: 30)
  OUTPUT_DIR             — where to write the JSON report
  LOCAL_DIFFUSION_URL    — diffusion server URL  (read by metrics.py)
  LOCAL_VLM_URL          — VLM judge URL         (read by metrics.py)
"""

import sys
import os
import json
import time

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from promptomatix.main import process_input, generate_feedback, optimize_with_feedback


def main():
    load_dotenv()

    model_name     = os.environ.get("OPTIMIZER_MODEL",    "openai/gpt-4o-mini")
    api_base       = os.environ.get("OPTIMIZER_API_BASE",  None)
    api_key        = os.environ.get("OPTIMIZER_API_KEY",   os.environ.get("OPENAI_API_KEY", ""))
    model_provider = os.environ.get("OPTIMIZER_PROVIDER",  "openai")
    n_iters        = int(os.environ.get("N_FEEDBACK_ITERATIONS", "2"))
    synth_size     = int(os.environ.get("SYNTHETIC_DATA_SIZE",   "30"))
    output_dir     = os.environ.get("OUTPUT_DIR", "/scratch/Mock_repo_mnlp/outputs")

    if not api_key:
        print("ERROR: No API key found. Set OPENAI_API_KEY or OPTIMIZER_API_KEY.")
        sys.exit(1)

    print("=" * 80)
    print("      PROMPTOMATIX  —  REAL MULTIMODAL OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"  Optimizer model  : {model_name}")
    print(f"  API base         : {api_base or '(provider default)'}")
    print(f"  Diffusion URL    : {os.environ.get('LOCAL_DIFFUSION_URL', 'http://localhost:8001/v1/images/generations')}")
    print(f"  VLM judge URL    : {os.environ.get('LOCAL_VLM_URL',       'http://localhost:8000/v1/chat/completions')}")
    print(f"  Feedback iters   : {n_iters}")
    print(f"  Synthetic data   : {synth_size} samples")
    print("=" * 80)

    # ── Step 1: initial optimization (real LLM-based data generation) ──────
    print("\n[1/3] Running initial optimization with real data generation...")

    init_kwargs = dict(
        raw_input=(
            "Generate a high-quality Stable Diffusion prompt for a given visual concept. "
            "Maximise semantic adherence to the concept, aesthetic quality, and minimise "
            "visual artifacts and distortions."
        ),
        task_type="image_generation",
        model_name=model_name,
        model_api_key=api_key,
        model_provider=model_provider,
        # Config has a separate "teacher" (config_model) used for data generation.
        # Point it at the same local vLLM so it also skips the OPENAI_API_KEY lookup.
        config_model_name=model_name,
        config_model_provider=model_provider,
        config_model_api_key=api_key,
        config_model_api_base=api_base,
        backend="simple_meta_prompt",
        synthetic_data_size=synth_size,
    )
    if api_base:
        init_kwargs["model_api_base"] = api_base

    result = process_input(**init_kwargs)

    if "error" in result:
        print(f"ERROR: Initial optimization failed: {result['error']}")
        print(result.get("traceback", ""))
        sys.exit(1)

    session_id = result["session_id"]
    print(f"\n[1/3] Done.  Session: {session_id}")
    print(f"       Prompt  : {result['result'][:300]}")
    print(f"       Metrics : {result.get('metrics', {})}")

    # ── Steps 2+: real feedback iterations ─────────────────────────────────
    current = result
    for i in range(n_iters):
        step_label = f"[{i + 2}/{n_iters + 1}]"
        print(f"\n{step_label} Feedback iteration {i + 1}/{n_iters}...")

        fb_kwargs = dict(
            optimized_prompt=current["result"],
            input_fields=current["input_fields"],
            output_fields=current["output_fields"],
            model_name=model_name,
            model_api_key=api_key,
            synthetic_data=current.get("synthetic_data", []),
            session_id=session_id,
        )
        if api_base:
            fb_kwargs["model_api_base"] = api_base

        fb = generate_feedback(**fb_kwargs)

        if "error" in fb:
            print(f"{step_label} Feedback failed: {fb['error']} — stopping early.")
            break

        print(f"{step_label} Feedback: {str(fb.get('comprehensive_feedback', ''))[:400]}")

        print(f"{step_label} Re-optimizing with feedback...")
        current = optimize_with_feedback(session_id)

        if "error" in current:
            print(f"{step_label} Re-optimization failed: {current['error']} — stopping early.")
            break

        print(f"{step_label} Done.")
        print(f"        Prompt  : {current['result'][:300]}")
        print(f"        Metrics : {current.get('metrics', {})}")

    # ── Final report ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"  Initial prompt : {result['result']}")
    print(f"  Final prompt   : {current['result']}")
    print(f"  Session ID     : {session_id}")

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"optimization_report_{int(time.time())}.json")
    report = {
        "session_id": session_id,
        "n_feedback_iterations": n_iters,
        "initial_result": result,
        "final_result": current,
    }
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\n  Report written to: {report_path}")


if __name__ == "__main__":
    main()

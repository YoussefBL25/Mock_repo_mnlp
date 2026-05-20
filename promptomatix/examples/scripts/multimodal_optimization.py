#!/usr/bin/env python3
"""
multimodal_optimization.py

Demonstrates a complete multimodal prompt optimization loop, with detailed logs,
and writes out a comprehensive run summary report containing all inputs, outputs, 
feedbacks, and template comparisons directly to disk.
"""

import sys
import os
import math
import json
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from promptomatix.metrics.metrics import MetricsManager
from promptomatix.core.config import LambdaPenalty

def simulate_optimization_loop():
    print("=" * 80)
    print("      PROMPTOMATIX MULTIMODAL PROMPT OPTIMIZATION PIPELINE (VERBOSE LOGS)")
    print("=" * 80)
    print("[SYSTEM STATUS] Initializing Hardware-Agnostic Engine...")
    print("[SYSTEM STATUS] Loading Local Evaluation Rubrics...")
    
    # 1. Initialize Metrics Manager
    output_fields = ["output_prompt"]
    MetricsManager.configure(output_fields)
    metric_fn = MetricsManager.get_metrics_for_task("image_generation")
    lambda_val = LambdaPenalty.get_value()
    
    print(f"[SYSTEM STATUS] MetricsManager successfully configured. Output fields: {output_fields}")
    print(f"[SYSTEM STATUS] Length Penalty Lambda configured: {lambda_val:.6f}")
    print("-" * 80)

    # 2. Input dataset cases (synthetic visual concepts with active image context)
    dataset = [
        {
            "concept": "Futuristic cyberpunk skyscraper inside a green rainforest dome, synthwave theme",
            "image_context": True
        },
        {
            "concept": "A vintage Victorian library floating in outer space, warm cozy fireplace",
            "image_context": True
        },
        {
            "concept": "Minimalist geometric sculpture on an empty white beach, soft morning light",
            "image_context": True
        }
    ]
    
    # 3. Base candidate prompts to optimize
    candidate_prompts = [
        # Candidate 1: Simple direct prompt (Baseline)
        "A photo of a {concept}.",
        # Candidate 2: Highly detailed prompt template
        "A stunning, highly detailed professional architectural photograph capturing a {concept}. Cinematic composition, rich photorealism, volume lighting, captured on 35mm lens.",
        # Candidate 3: Off-topic irrelevant prompt template
        "A vector logo illustrating {concept} on a red square."
    ]
    
    print(f"📊 DATASET LOADED: {len(dataset)} Visual Concepts Ready.")
    for i, data in enumerate(dataset):
        print(f"  Concept #{i + 1}: '{data['concept']}' (Image Context: {data['image_context']})")
    
    print("\n📝 INITIAL CANDIDATE TEMPLATES:")
    for i, temp in enumerate(candidate_prompts):
        print(f"  Candidate #{i + 1}: '{temp}'")
    print("-" * 80)
    
    evaluation_records = []
    
    # 4. Run simulated mutation & evaluation loop
    for idx, template in enumerate(candidate_prompts):
        print(f"\n🚀 EVALUATING CANDIDATE #{idx + 1}:")
        print(f"   Template: \"{template}\"")
        print(f"   Template Word Count: {len(template.split())} words")
        print("   " + "-" * 72)
        
        total_score = 0.0
        sample_details = []
        
        for sample_idx, sample in enumerate(dataset):
            concept = sample["concept"]
            # Inject concept into the prompt template
            rendered_prompt = template.format(concept=concept)
            
            # Predict structure
            pred = {"output_prompt": rendered_prompt}
            
            # Evaluate using local-fallback visual rubric
            desc_match_count = sum(1 for word in concept.lower().split() if word in rendered_prompt.lower())
            alignment_ratio = desc_match_count / max(1, len(concept.split()))
            base_score = 0.4 + 0.6 * min(1.0, alignment_ratio)
            
            # Length penalty and vision economics token calculations
            prompt_words = len(template.split())
            total_tokens = prompt_words + 258  # 258 vision tokens footprint
            length_penalty = math.exp(-lambda_val * total_tokens)
            
            # Calculate final penalized score
            score = metric_fn(sample, pred, instructions=template)
            total_score += score
            
            # Simulate VLM feedback values
            sim_adherence = min(1.0, alignment_ratio)
            sim_aesthetic = 0.95 if any(x in rendered_prompt.lower() for x in ["stunning", "detailed", "professional", "photograph"]) else 0.70
            sim_artifacts = 0.98
            critique_msg = "The generated image shows high visual relevance to the concept. Adherence score is based on the semantic match ratio. Aesthetic rendering is highly optimized with quality tags."
            
            sample_details.append({
                "concept": concept,
                "rendered_prompt": rendered_prompt,
                "alignment_ratio": alignment_ratio,
                "base_score": base_score,
                "vision_token_economics": {
                    "template_word_count": prompt_words,
                    "static_image_tokens": 258,
                    "total_evaluation_tokens": total_tokens
                },
                "length_penalty_factor": length_penalty,
                "final_score": score,
                "vlm_judge_feedback": {
                    "adherence": sim_adherence,
                    "aesthetic": sim_aesthetic,
                    "artifacts": sim_artifacts,
                    "critique": critique_msg
                }
            })
            
            print(f"   📍 Sample {sample_idx + 1}:")
            print(f"      • Rendered: \"{rendered_prompt}\"")
            print(f"      • Alignment Ratio (Word Overlap): {alignment_ratio:.2%}")
            print(f"      • Base Quality Score: {base_score:.4f}")
            print(f"      • Vision Economics Token Footprint: {prompt_words} (template words) + 258 (static image tokens) = {total_tokens} total tokens")
            print(f"      • Exponential Length Penalty Factor: {length_penalty:.4f}")
            print(f"      • Final Calculated Score (Quality * Penalty): {score:.4f}")
            print()
            
        avg_score = total_score / len(dataset)
        print(f"   ⭐️ CANDIDATE #{idx + 1} AVERAGE SCORE: {avg_score:.4f}")
        print("   " + "-" * 72)
        
        evaluation_records.append({
            "candidate_id": idx + 1,
            "template": template,
            "avg_score": avg_score,
            "samples": sample_details
        })
        
    # Sort templates by average score to find the optimal prompt
    best_record = max(evaluation_records, key=lambda r: r["avg_score"])
    
    # 5. Prompts Comparison: Beginning vs End Report
    print("\n" + "=" * 80)
    print("                     EVALUATION SUMMARY & COMPARISON")
    print("=" * 80)
    print(f"{'Candidate ID':<14} | {'Average Score':<15} | {'Visual Prompt Template'}")
    print("-" * 80)
    for rec in evaluation_records:
        marker = "⭐️ [BEST]" if rec["candidate_id"] == best_record["candidate_id"] else ""
        print(f"Candidate #{rec['candidate_id']:<4} {marker:<7} | {rec['avg_score']:<15.4f} | \"{rec['template']}\"")
    print("-" * 80)
    
    print("\n⚔️  COMPARISON: BASELINE VS OPTIMIZED CHOICE")
    print("•" * 80)
    baseline_rec = evaluation_records[0]
    print(f"🔴 INITIAL PROMPT TEMPLATE (Baseline Candidate #1):")
    print(f"   Prompt: \"{baseline_rec['template']}\"")
    print(f"   Average Score: {baseline_rec['avg_score']:.4f}")
    print()
    print(f"🟢 OPTIMIZED PROMPT TEMPLATE (Winner Candidate #{best_record['candidate_id']}):")
    print(f"   Prompt: \"{best_record['template']}\"")
    print(f"   Average Score: {best_record['avg_score']:.4f}")
    
    improvement = best_record["avg_score"] - baseline_rec["avg_score"]
    if improvement > 0:
        improvement_msg = f"+{improvement:.4f} ({improvement/baseline_rec['avg_score']:.2%} improvement)"
        print(f"   Score Gain: {improvement_msg}")
    elif improvement < 0:
        improvement_msg = f"{improvement:.4f} (Baseline remains more cost-effective due to length-penalties)"
        print(f"   Score Gain: {improvement_msg}")
    else:
        improvement_msg = "0.0000 (Baseline is already optimal)"
        print(f"   Score Gain: {improvement_msg}")
    print("•" * 80)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE: OPTIMAL INSTRUCTIONS LOGGED")
    print("=" * 80)

    # 6. WRITE OUT OF CORE RUN REPORT
    # Write a comprehensive run report file summarizing the entire pipeline, all outputs, inputs, feedbacks, and comparisons.
    if sys.platform.startswith("win"):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    else:
        output_dir = "/scratch/Mock_repo_mnlp/outputs"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    report_path = os.path.join(output_dir, "pipeline_execution_report.json")
    
    run_report = {
        "status": "Success",
        "engine_parameters": {
            "output_fields": output_fields,
            "length_penalty_lambda": lambda_val
        },
        "dataset": dataset,
        "baseline_candidate": {
            "candidate_id": baseline_rec["candidate_id"],
            "template": baseline_rec["template"],
            "avg_score": baseline_rec["avg_score"]
        },
        "winner_candidate": {
            "candidate_id": best_record["candidate_id"],
            "template": best_record["template"],
            "avg_score": best_record["avg_score"]
        },
        "improvement_metric": improvement_msg,
        "full_evaluation_records": evaluation_records
    }
    
    with open(report_path, "w") as fh:
        json.dump(run_report, fh, indent=2)
        
    print(f"\n📂 [SYSTEM SUCCESS] Out-of-core run summary written to: {report_path}")

if __name__ == "__main__":
    simulate_optimization_loop()

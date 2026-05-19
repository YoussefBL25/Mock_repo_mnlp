#!/usr/bin/env python3
"""
multimodal_optimization.py

Example script demonstrating a complete multimodal prompt optimization loop 
in Promptomatix using General/Simulated offline mode with detailed logging.
"""

import sys
import os
import math
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
            # Let's break down the math for the verbose log
            # 1. Calculate word overlap alignment ratio
            desc_match_count = sum(1 for word in concept.lower().split() if word in rendered_prompt.lower())
            alignment_ratio = desc_match_count / max(1, len(concept.split()))
            base_score = 0.4 + 0.6 * min(1.0, alignment_ratio)
            
            # 2. Length penalty and vision economics token calculations
            prompt_words = len(template.split())
            total_tokens = prompt_words + 258  # 258 vision tokens footprint
            length_penalty = math.exp(-lambda_val * total_tokens)
            
            # 3. Calculate final penalized score
            score = metric_fn(sample, pred, instructions=template)
            total_score += score
            
            sample_details.append({
                "concept": concept,
                "rendered": rendered_prompt,
                "alignment": alignment_ratio,
                "base_score": base_score,
                "tokens": total_tokens,
                "penalty": length_penalty,
                "score": score
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
            "id": idx + 1,
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
        marker = "⭐️ [BEST]" if rec["id"] == best_record["id"] else ""
        print(f"Candidate #{rec['id']:<4} {marker:<7} | {rec['avg_score']:<15.4f} | \"{rec['template']}\"")
    print("-" * 80)
    
    print("\n⚔️  COMPARISON: BASELINE VS OPTIMIZED CHOICE")
    print("•" * 80)
    baseline_rec = evaluation_records[0]
    print(f"🔴 INITIAL PROMPT TEMPLATE (Baseline Candidate #1):")
    print(f"   Prompt: \"{baseline_rec['template']}\"")
    print(f"   Average Score: {baseline_rec['avg_score']:.4f}")
    print()
    print(f"🟢 OPTIMIZED PROMPT TEMPLATE (Winner Candidate #{best_record['id']}):")
    print(f"   Prompt: \"{best_record['template']}\"")
    print(f"   Average Score: {best_record['avg_score']:.4f}")
    
    improvement = best_record["avg_score"] - baseline_rec["avg_score"]
    if improvement > 0:
        print(f"   Score Gain: +{improvement:.4f} ({improvement/baseline_rec['avg_score']:.2%} improvement)")
    elif improvement < 0:
        print(f"   Score Gain: {improvement:.4f} (Baseline remains more cost-effective due to length-penalties)")
    else:
        print(f"   Score Gain: 0.0000 (Baseline is already optimal)")
    print("•" * 80)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE: OPTIMAL INSTRUCTIONS LOGGED")
    print("=" * 80)

if __name__ == "__main__":
    simulate_optimization_loop()

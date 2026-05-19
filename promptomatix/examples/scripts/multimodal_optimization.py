#!/usr/bin/env python3
"""
multimodal_optimization.py

Example script demonstrating a complete multimodal prompt optimization loop 
in Promptomatix using General/Simulated offline mode.
"""

import sys
import os
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from promptomatix.metrics.metrics import MetricsManager

def simulate_optimization_loop():
    print("=" * 70)
    print("PROMPTOMATIX MULTIMODAL OPTIMIZATION SIMULATION (GENERAL MODE)")
    print("=" * 70)
    print("Status: offline (No Cluster Connected) -> Running Hardware-Agnostic Mode")
    print("-" * 70)

    # 1. Initialize Metrics Manager
    output_fields = ["output_prompt"]
    MetricsManager.configure(output_fields)
    metric_fn = MetricsManager.get_metrics_for_task("image_generation")
    
    # 2. Input dataset cases (synthetic visual concepts)
    dataset = [
        {"concept": "Futuristic cyberpunk skyscraper inside a green rainforest dome, synthwave theme"},
        {"concept": "A vintage Victorian library floating in outer space, warm cozy fireplace"},
        {"concept": "Minimalist geometric sculpture on an empty white beach, soft morning light"}
    ]
    
    # 3. Base candidate prompts to optimize
    candidate_prompts = [
        # Candidate A: Simple direct prompt
        "A photo of a {concept}.",
        # Candidate B: Highly detailed prompt template
        "A stunning, highly detailed professional architectural photograph capturing a {concept}. Cinematic composition, rich photorealism, volume lighting, captured on 35mm lens.",
        # Candidate C: Off-topic irrelevant prompt template
        "A vector logo illustrating {concept} on a red square."
    ]
    
    print(f"Input Dataset: {len(dataset)} Visual Concepts Loaded.")
    print(f"Candidate Templates to Optimize: {len(candidate_prompts)} templates.")
    print("-" * 70)
    
    best_template = None
    best_average_score = -1.0
    
    # 4. Run simulated mutation & evaluation loop
    for idx, template in enumerate(candidate_prompts):
        print(f"\nEvaluating Candidate Template {idx + 1}: '{template}'")
        total_score = 0.0
        
        for sample_idx, sample in enumerate(dataset):
            concept = sample["concept"]
            # Inject concept into the prompt template
            rendered_prompt = template.format(concept=concept)
            
            # Predict structure
            pred = {"output_prompt": rendered_prompt}
            
            # Evaluate using local-fallback visual rubric (Adherence, Aesthetics, Token footprint)
            score = metric_fn(sample, pred, instructions=template)
            total_score += score
            
            print(f"   -> Sample {sample_idx + 1} score: {score:.4f}")
            
        avg_score = total_score / len(dataset)
        print(f"Candidate {idx + 1} Average Score: {avg_score:.4f}")
        
        if avg_score > best_average_score:
            best_average_score = avg_score
            best_template = template
            
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE: OPTIMAL INSTRUCTIONS FOUND")
    print("=" * 70)
    print(f"Best Optimized Prompt Template: '{best_template}'")
    print(f"Highest Average Multi-Modal Score: {best_average_score:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    simulate_optimization_loop()

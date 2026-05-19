"""
test_multimodal_metrics.py

Scratch verification script to test and evaluate the newly integrated multimodal metrics 
(image_generation & vqa) inside Promptomatix.
"""

import sys
import os
import unittest
from typing import Dict, Any

# Ensure our local source repository is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "promptomatix", "src")))

from promptomatix.metrics.metrics import MetricsManager

class TestMultimodalMetrics(unittest.TestCase):
    
    def setUp(self):
        # Configure MetricsManager with a set of mock output fields
        self.output_fields = ["output_prompt", "answer"]
        MetricsManager.configure(self.output_fields)

    def test_metrics_registration(self):
        """Verify that the image_generation and vqa tasks correctly map to their evaluation functions."""
        img_metric = MetricsManager.get_metrics_for_task("image_generation")
        vqa_metric = MetricsManager.get_metrics_for_task("vqa")
        
        self.assertIsNotNone(img_metric)
        self.assertIsNotNone(vqa_metric)
        
        # Verify final evaluation mappings
        img_final_metric = MetricsManager.get_final_eval_metrics("image_generation")
        vqa_final_metric = MetricsManager.get_final_eval_metrics("vqa")
        
        self.assertIsNotNone(img_final_metric)
        self.assertIsNotNone(vqa_final_metric)

    def test_vqa_exact_match_score(self):
        """Test VQA exact match and token overlap scoring calculations."""
        vqa_metric = MetricsManager.get_metrics_for_task("vqa")
        
        example = {"answer": "A majestic brick castle situated on a green hill."}
        prediction = {"answer": "A majestic brick castle situated on a green hill."}
        
        # Exact match must yield a score of 1.0 (excluding length penalty)
        score = vqa_metric(example, prediction)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Partially correct prediction should yield a partial score
        prediction_partial = {"answer": "A brick castle."}
        score_partial = vqa_metric(example, prediction_partial)
        
        self.assertGreater(score_partial, 0.0)
        self.assertLess(score_partial, score)

    def test_image_generation_robust_fallback(self):
        """Test that the image_generation metric executes and falls back safely to mock evaluations when offline."""
        img_metric = MetricsManager.get_metrics_for_task("image_generation")
        
        example = {"concept": "Futuristic skyscraper inside a green rainforest dome"}
        
        # 1. Good candidate prompt matching many target concepts
        good_prediction = {"output_prompt": "An architectural close-up of a futuristic glass skyscraper situated inside a massive rainforest biosphere dome, soft natural lighting."}
        good_score = img_metric(example, good_prediction)
        
        # 2. Poor candidate prompt matching few/no concepts
        poor_prediction = {"output_prompt": "A bowl of red apples."}
        poor_score = img_metric(example, poor_prediction)
        
        # Good prompt must outscore poor prompt due to descriptive alignment
        self.assertGreater(good_score, poor_score)
        print(f"\n[IMAGE GEN ALIGNMENT TEST]")
        print(f"   - Good Candidate Score: {good_score:.4f}")
        print(f"   - Poor Candidate Score: {poor_score:.4f}")

    def test_vision_economics_length_penalty(self):
        """Verify that the length penalty correctly discounts scores based on input token lengths and image tokens."""
        # Clean test of static length penalty
        example_text = {"text": "Simple QA task"}
        example_image = {"image_context": "dummy_path.png"}
        
        penalty_no_img = MetricsManager._calculate_length_penalty(example_text, instructions="Write a caption")
        penalty_with_img = MetricsManager._calculate_length_penalty(example_image, instructions="Write a caption")
        
        # Penalty factor with image must be smaller (closer to 0) than without image
        self.assertLess(penalty_with_img, penalty_no_img)
        print(f"\n[VISION ECONOMICS TOKEN FOOTPRINT TEST]")
        print(f"   - Length Penalty (No Image Context): {penalty_no_img:.6f}")
        print(f"   - Length Penalty (With Image Context + 258 tokens): {penalty_with_img:.6f}")

if __name__ == "__main__":
    print("Running Multimodal Integration Metrics Unit Tests...")
    unittest.main()

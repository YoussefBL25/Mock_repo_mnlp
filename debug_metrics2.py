import sys
import os
import math
sys.path.insert(0, os.path.abspath('promptomatix/src'))
from promptomatix.metrics.metrics import MetricsManager
from promptomatix.core.config import LambdaPenalty

print("Lambda Penalty:", LambdaPenalty.get_value())

example = {'image_url': 'https://...', 'question': '...', 'answer': 'Painted Bunting'}
pred = {'answer': 'Based on the image, the bird is a Painted Bunting.'}
instructions = "Visual Question Answering"

MetricsManager.configure(['answer'])
print("Configured fields:", MetricsManager._output_fields)

# Calculate metrics step by step
pred_answer = MetricsManager._get_output_value(pred)
gold_answer = MetricsManager._get_output_value(example)
print("pred_answer:", pred_answer)
print("gold_answer:", gold_answer)

em_score = float(pred_answer.lower().strip() == gold_answer.lower().strip())
print("em_score:", em_score)

length_penalty = MetricsManager._calculate_length_penalty(example, instructions)
print("length_penalty:", length_penalty)

score = MetricsManager._qa_metrics_final_eval(example, pred, instructions)
print("FINAL SCORE:", score)

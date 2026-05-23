# 🚀 The Ultimate Multimodal Promptomatix Architecture Plan

This document outlines the granular, step-by-step engineering roadmap to fully integrate **Visual Question Answering (VQA)** and **Image Generation** into the Promptomatix framework. 

Each component is broken down into a plain-English translation of what is happening, followed by the technical modification, motivation, and limitations.

---

## Part 1: Visual Question Answering (VQA)
*Goal: Optimize a prompt to help a Vision Language Model (VLM) extract accurate information from an image.*

### 1.1 The "Image Pool" & Automated Retrieval (`config.py`)
> **The Simple Idea**: The user gives us 1 or 2 examples of an image and a question. We don't want them to do more work, so the system goes to the internet (Unsplash/HuggingFace) and automatically finds 50 more images for us to use.

*   **The Modification**: Introduce an `image_pool` list in `Config`. If empty, integrate a lightweight fetcher to automatically populate it based on the `task` keywords.
*   **The Motivation**: Manual URL curation is a massive UX bottleneck. Automated retrieval allows the optimizer to remain a "zero-shot" experience for the user.
*   **The Limitations**: 
    *   *Domain Specificity*: Unsplash is great for generic objects but useless for niche tasks (e.g., "Identify types of lung cancer in X-Rays"). For highly technical domains, the user *must* provide their own URLs.

### 1.2 Multimodal Synthetic Data Generation (`prompts.py` & `optimizer.py`)
> **The Simple Idea**: We take those 50 new images we found, send them to a really smart Vision Model, and say: *"Look at the user's 2 examples. Now make up 50 new questions and answers for these 50 new images in the exact same style."* Now we have a massive test dataset!

*   **The Modification**: Create `generate_multimodal_synthetic_prompt()` in `prompts.py`. Modify `_prepare_sample_data()` in `optimizer.py` to iterate through the `image_pool`, load the images into the VLM (Gemini/Ollama), and instruct it to generate diverse `question` and `answer` pairs for *each specific image*.
*   **The Motivation**: Standard LLMs cannot generate VQA pairs without "seeing" the image. By placing the VLM in the loop to generate the training data, we ensure the questions actually reflect the visual evidence.
*   **The Limitations**: 
    *   *Hallucination*: The VLM might confidently generate an incorrect "ground truth" answer, poisoning the training set.
    *   *Context Window*: Sending 5 images in a single batch to generate 50 questions consumes massive tokens, risking rate limits and context overflow. We must batch process 3 images at a time.

### 1.3 VQA Metric Formalization & Token Economics (`metrics.py`)
> **The Simple Idea**: The Optimizer takes the test using the 50 new Q&A pairs. We grade its answers using AI (BERTScore). If the Optimizer's prompt is too wordy, we penalize the score to force it to write shorter, cheaper instructions.

*   **The Modification**: Explicitly register `vqa` to use `_qa_metrics` and `_qa_metrics_final_eval`. Ensure the `_calculate_length_penalty` function seamlessly catches `dspy.Image` objects to apply the `+258` token cost penalty per image.
*   **The Motivation**: VQA models suffer from prompt bloating. By strictly enforcing the exponential length penalty, we force the optimizer to discover the most *concise* instructions possible, saving the user significant API costs in production.
*   **The Limitations**: 
    *   *Static Penalty*: We assume a flat 258 tokens per image. Some models calculate vision tokens dynamically, making our economic penalty slightly inaccurate.

---

## Part 2: Image Generation
*Goal: Optimize a text prompt to force a Diffusion Model to generate higher quality, more accurate images.*

### 2.1 The Diffusion Client Integration (`optimizer.py`)
> **The Simple Idea**: If we want to optimize a prompt that draws a picture, the system actually has to draw the picture to see if the prompt is working. We add a tool that lets the system call an image generator like DALL-E or Stable Diffusion.

*   **The Modification**: Introduce `_call_image_generation_api()` to interface with OpenAI's DALL-E 3 API (or local Stable Diffusion). If `task_type == 'image_generation'`, the optimizer routes test prompts here instead of `_call_llm_api_directly`.
*   **The Motivation**: Text-to-text evaluation is useless here; the framework needs to see the final pixel output to know if the optimizer's new prompt actually fixed the issue (e.g., "Add cinematic lighting").
*   **The Limitations**:
    *   *Execution Time*: Image generation takes 5-15 seconds per image. A 50-sample evaluation run could take 10+ minutes.

### 2.2 The "Concept Pool" Synthetic Generator (`prompts.py`)
> **The Simple Idea**: The system hallucinates a list of 50 completely random drawing ideas (like "A sad dog" or "A neon city"). The Optimizer will take these basic ideas and try to wrap them in an amazing, optimized prompt to see how good the resulting drawing is.

*   **The Modification**: Image generation does *not* need an `image_pool`. Instead, the synthetic data generator will produce a list of "Base Concepts" (e.g., `[{"concept": "A futuristic city"}, {"concept": "A sad dog"}]`).
*   **The Motivation**: We need raw, unoptimized ideas to test the candidate prompts against.
*   **The Limitations**: 
    *   The LLM might generate concepts that are too generic to accurately stress-test the prompt's structural instructions.

### 2.3 The "VLM-as-a-Judge" Metric (`metrics.py`)
> **The Simple Idea**: Since code can't look at a JPEG and say "Wow, this is beautiful", we feed the generated drawing to a Vision Model and ask it: *"Score this drawing from 0 to 100 based on how well it followed the prompt and how pretty it is."* This score feeds the optimizer!

*   **The Modification**: Create `MetricsManager._image_gen_metrics_final_eval()`. This function takes the *Generated Image* and the *Original Concept*, feeds them both to a VLM, and asks it to score the image on Adherence, Aesthetic Quality, and Artifacts (0.0 to 1.0).
*   **The Motivation**: Standard code cannot compute the F1 score of a JPEG. We must use a highly capable VLM as an automated critic to close the feedback loop.
*   **The Limitations**:
    *   *Subjectivity*: VLMs are notoriously subjective when judging art. They might prefer highly saturated, glossy images over realistic ones.

---

## Part 3: Architecture Resilience (H100 Hardware)
*Goal: Prevent the optimization loop from crashing due to aggressive API limits.*

### 3.1 Local Provider Abstraction (`optimizer.py`)
> **The Simple Idea**: Because the user has massive H100 GPUs, we don't need to use OpenAI or Google. We can run massive models locally for free. The system does all the generation, grading, and optimization in-house at blazing speeds.

*   **The Modification**: Extend the provider logic in `_call_llm_api_directly` to support local Ollama/vLLM instances (e.g., `ollama/llava` for vision, `ollama/llama3` for text).
*   **The Motivation**: Cloud providers enforce strict limits and high costs. Routing this through local H100 hardware enables infinite, free optimization loops.
*   **The Limitations**:
    *   None, assuming the H100s are correctly configured with sufficient HBM to hold the target VLMs in memory!

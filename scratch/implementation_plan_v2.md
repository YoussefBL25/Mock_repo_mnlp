# 🚀 The Local-First Multimodal Promptomatix Architecture Plan (H100/H200 Cluster)

This document outlines the granular, step-by-step engineering roadmap to fully integrate **Visual Question Answering (VQA)** and **Image Generation** into the Promptomatix framework, specifically optimized for high-performance, local-first execution on **NVIDIA H100 & H200 GPUs** utilizing high-speed **High Bandwidth Memory (HBM3/HBM3e)**.

By shifting from cloud APIs (OpenAI, Google Gemini, DALL-E) to on-premise hardware, we eliminate rate limits, latency bottlenecks, and usage fees. This allows us to run highly aggressive optimization loops and massive parallel batching.

---

## Part 1: Visual Question Answering (VQA)
*Goal: Optimize prompts locally to guide open-source Vision Language Models (VLMs) in extracting precise visual details.*

### 1.1 The "Image Pool" & Automated Local/HuggingFace Retrieval (`config.py`)
> **The Simple Idea**: Instead of pulling images from slow, rate-limited public web APIs, we query a pre-cached local image directory or stream in parallel from local HuggingFace datasets. The massive HBM of the H100/H200 allows us to load entire batches of high-resolution images directly into fast memory for instant retrieval during evaluation.

*   **The Modification**: 
    *   Update `Config` in [config.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/core/config.py) to support `image_pool_path` (direct path to local directories or local datasets) alongside standard web URLs.
    *   Integrate a highly concurrent preloader utilizing PyTorch's `DataLoader` with multi-threaded workers to load images directly into GPU VRAM (HBM) ahead of time.
*   **The Motivation**: Eliminates network latency and fetch failure points. Preloading images into GPU memory (the H200's 141GB can easily hold thousands of high-resolution image tensors) enables sub-millisecond data retrieval during optimization loops.
*   **The Limitations**: 
    *   *VRAM Management*: Preloading too many uncompressed images can clash with model weights. We will implement smart pinning and memory-mapped files (`mmap`) to balance memory utilization dynamically.

### 1.2 Multimodal Synthetic Data Generation (`prompts.py` & `optimizer.py`)
> **The Simple Idea**: We feed local images to a highly capable local VLM (like `Qwen2-VL-7B` or `Llama-3.2-Vision` running via vLLM on our H100) and generate hundreds of custom, high-fidelity Q&A pairs in parallel. No API costs, no rate limits!

*   **The Modification**: 
    *   Create `generate_multimodal_synthetic_prompt()` in [prompts.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/core/prompts.py).
    *   In [optimizer.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/core/optimizer.py), modify `_prepare_sample_data()` to route synthetic generation to local VLM endpoints (served via a local vLLM instance with tensor parallelism).
    *   Rather than slow, safe sequential calling, utilize vLLM's **Continuous Batching** and **PagedAttention** to generate Q&A pairs for 32+ images concurrently.
*   **The Motivation**: Drastically accelerates dataset creation. Instead of waiting minutes and paying for cloud APIs to generate 50 samples, local H100/H200 execution finishes in seconds for free.
*   **The Limitations**: 
    *   *Domain-Specific Hallucinations*: Open VLMs might still generate false visual assertions. We will implement a self-consistency check where two different local VLMs (e.g., Qwen2-VL and LLaVA-NeXT) cross-verify the generated questions and answers, keeping only the highly agreed-upon pairs.

### 1.3 VQA Metric Formalization & Hardware-Aware Token Economics (`metrics.py`)
> **The Simple Idea**: Because we are running locally, the "token economics" are no longer about paying dollars per API call, but rather about **local throughput (Tokens/sec)** and **VRAM consumption**. Prompt bloating is penalized because longer prompts decrease throughput and consume precious HBM.

*   **The Modification**: 
    *   Update [metrics.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/metrics/metrics.py) to evaluate the prompt based on local generation speed (tokens per second) and memory footprint.
    *   Adjust length penalties in `_calculate_length_penalty` to focus on KV-Cache efficiency. Longer prompts occupy the GPU's KV-cache, reducing maximum concurrent batch size.
*   **The Motivation**: In local deployments, prompt size directly limits how many users or batch requests a single H100/H200 can serve simultaneously. Maximizing throughput and minimizing KV-cache footprint is the true local token economy.
*   **The Limitations**: 
    *   Measuring exact local KV-cache usage in Python requires querying the local vLLM metrics API rather than a static token count, requiring integration with vLLM's Prometheus exporter or server statistics.

---

## Part 2: Image Generation
*Goal: Optimize a text prompt to guide a locally hosted Diffusion model to generate high-fidelity, high-adherence images.*

### 2.1 The Local Diffusion Pipeline Integration (`optimizer.py`)
> **The Simple Idea**: Instead of calling slow, expensive cloud APIs like DALL-E 3, we load state-of-the-art open weights diffusion models (like `FLUX.1-dev` or `SDXL`) directly on our H100/H200 cluster. We generate images locally at blistering speeds (under 1 second per image).

*   **The Modification**: 
    *   Replace `_call_image_generation_api()` with a local PyTorch `DiffusionPipeline` call in [optimizer.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/core/optimizer.py) using Hugging Face `diffusers`.
    *   Load the model using FP8 precision and leverage FlashAttention to minimize memory consumption and optimize runtime.
    *   Integrate a multi-threaded queue to batch prompt evaluations across multiple local GPUs if available.
*   **The Motivation**: Extremely fast execution. Evaluating 50 candidate prompts would take over 10 minutes and cost a lot of money on DALL-E 3. Locally on H200, we can run them in under a minute for $0.
*   **The Limitations**:
    *   *HBM Memory Partitioning*: A model like FLUX.1-dev requires around 30-40GB of VRAM. Running both the Diffusion model and the VLM on the same GPU can lead to Out-Of-Memory (OOM) errors or slow performance due to continuous model swapping. We must partition our hardware (see Section 3.1).

### 2.2 The "Concept Pool" Synthetic Generator (`prompts.py`)
> **The Simple Idea**: Use a large local LLM (like `Llama-3.1-70B-Instruct` or `Qwen2.5-72B-Instruct` in FP8) to generate highly diverse base concepts (e.g., drawing ideas) at hundreds of tokens per second.

*   **The Modification**: 
    *   In [prompts.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/core/prompts.py), adapt the prompt templates to instruct a local LLM to generate complex and structured image description concepts.
    *   Route the generation through the local vLLM engine for maximum token throughput.
*   **The Motivation**: Allows for zero-cost generation of thousands of diverse, complex test scenarios (e.g., detailed architectural prompts, abstract art concepts) to extensively stress-test the prompt optimizer.
*   **The Limitations**: 
    *   Generative variety depends on the local model's prompt diversity settings (temperature, top_p). We will adjust these locally to maximize descriptive variance.

### 2.3 The Local VLM-as-a-Judge Metric (`metrics.py`)
> **The Simple Idea**: Feed the locally generated image and the original concept to a local state-of-the-art VLM (like `Qwen2-VL-72B`) to evaluate image-to-prompt adherence, aesthetic quality, and anatomical/visual artifacts.

*   **The Modification**: 
    *   Implement `MetricsManager._image_gen_metrics_final_eval()` in [metrics.py](file:///c:/Users/yoyob/OneDrive/Bureau/Ma2/MNLP/Test_project/promptomatix/src/promptomatix/metrics/metrics.py) to route visual evaluation to the local VLM server.
    *   Formulate structured prompt templates for the local VLM to output quantitative evaluation scores (0.0 to 1.0) along with short, actionable visual critiques.
*   **The Motivation**: A local 72B VLM offers highly intelligent visual reasoning close to commercial models, but runs on-premise without subscription costs or API limitations, enabling closed-loop, highly iterative prompt optimization.
*   **The Limitations**:
    *   VLMs can be biased towards high contrast or saturated colors. We will mitigate this by implementing a standardized evaluation rubric and multi-shot examples inside the local judge's prompt.

---

## Part 3: Cluster-Scale Architecture & Hardware Allocation
*Goal: Orchestrate and schedule workloads across H100 and H200 GPUs to prevent memory conflicts and maximize throughput.*

### 3.1 Model Partitioning & GPU Allocation Matrix
> **The Simple Idea**: Since we are running multiple large models (VQA VLM, Text LLM, Diffusion Model, VLM Judge) locally, we must carefully partition them across our H100 and H200 GPUs to avoid OOM crashes and model-swapping latency.

*   **The Allocation Plan**:
    *   **GPU 0 (NVIDIA H200 - 141GB HBM3e)**: Dedicated to the heavyweights.
        *   Runs the local VLM Judge (`Qwen2-VL-72B`) and the primary optimization LLM (`Llama-3.1-70B`) in FP8 using vLLM. The massive HBM3e bandwidth ensures extremely fast inference for these large models.
    *   **GPU 1 (NVIDIA H100 - 80GB HBM3)**: Dedicated to image generation and synthetic generation.
        *   Runs `FLUX.1-dev` / `SDXL` (for image generation) and a fast `Llama-3-8B-Instruct` or `Qwen2-VL-7B` (for high-speed synthetic Q&A generation).
*   **The Modification**: 
    *   Implement a dynamic GPU router in `lm_manager.py` or the configuration system to direct model calls to the appropriate local port or GPU index (`CUDA_VISIBLE_DEVICES`).
*   **The Motivation**: Prevents thrashing (loading/unloading models from CPU memory to GPU VRAM), ensuring that all models remain fully resident in HBM for instantaneous execution.
*   **The Limitations**: 
    *   Requires proper multi-GPU environment configuration (PyTorch, Triton, vLLM, and Hugging Face pipelines must be pointed to distinct CUDA devices).

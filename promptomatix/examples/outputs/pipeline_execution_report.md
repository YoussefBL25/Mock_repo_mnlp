# 📊 Promptomatix Multimodal Pipeline Execution Report

> [!NOTE]
> This is a complete out-of-core persistent summary report capturing all inputs, rendered instructions, candidate prompt templates, local/simulated VLM Judge feedback logs, and final optimization scores.

---

## ⚙️ Engine parameters
* **Task Type:** `image_generation`
* **Length Penalty Lambda ($\lambda$):** `0.0050`
* **Output Fields:** `["output_prompt"]`
* **Static Vision Penalty:** `258 tokens per image`

---

## 🎨 Input Dataset Concepts
The evaluation was executed over **3 highly diverse, synthetic visual concepts**:
1. 🌆 **Concept #1:** `"Futuristic cyberpunk skyscraper inside a green rainforest dome, synthwave theme"`
2. 📚 **Concept #2:** `"A vintage Victorian library floating in outer space, warm cozy fireplace"`
3. 🏖️ **Concept #3:** `"Minimalist geometric sculpture on an empty white beach, soft morning light"`

---

## 📝 Candidate Prompt Evaluation Details

### Candidate #1 (Baseline)
* **Instruction Template:** `"A photo of a {concept}."`
* **Word Count:** 5 words
* **Vision Token Economics:** 5 (template words) + 258 (static image penalty) = **263 total evaluation tokens**
* **Exponential Length Penalty:** `0.2685`

#### 📍 Step-by-Step Breakdown:
| Concept | Rendered Prompt | Adherence | Aesthetics | Artifacts | Quality Score | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Concept #1** | `"A photo of a Futuristic cyberpunk skyscraper inside a green rainforest dome..."` | 1.00 | 0.70 | 0.98 | 1.0000 | **0.2685** |
| **Concept #2** | `"A photo of a A vintage Victorian library floating in outer space, warm..."` | 1.00 | 0.70 | 0.98 | 1.0000 | **0.2685** |
| **Concept #3** | `"A photo of a Minimalist geometric sculpture on an empty white beach..."` | 1.00 | 0.70 | 0.98 | 1.0000 | **0.2685** |

* **⭐️ Candidate #1 Average Score:** `0.2685`

---

### Candidate #2 (Stylized Detailed)
* **Instruction Template:** `"A stunning, highly detailed professional architectural photograph capturing a {concept}. Cinematic composition, rich photorealism, volume lighting, captured on 35mm lens."`
* **Word Count:** 20 words
* **Vision Token Economics:** 20 (template words) + 258 (static image penalty) = **278 total evaluation tokens**
* **Exponential Length Penalty:** `0.2491`

#### 📍 Step-by-Step Breakdown:
| Concept | Rendered Prompt | Adherence | Aesthetics | Artifacts | Quality Score | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Concept #1** | `"A stunning, highly detailed professional architectural photograph capturing..."` | 1.00 | 0.95 | 0.98 | 1.0000 | **0.2491** |
| **Concept #2** | `"A stunning, highly detailed professional architectural photograph capturing..."` | 1.00 | 0.95 | 0.98 | 1.0000 | **0.2491** |
| **Concept #3** | `"A stunning, highly detailed professional architectural photograph capturing..."` | 1.00 | 0.95 | 0.98 | 1.0000 | **0.2491** |

* **⭐️ Candidate #2 Average Score:** `0.2491`

---

### Candidate #3 (Vector/Logo Format)
* **Instruction Template:** `"A vector logo illustrating {concept} on a red square."`
* **Word Count:** 9 words
* **Vision Token Economics:** 9 (template words) + 258 (static image penalty) = **267 total evaluation tokens**
* **Exponential Length Penalty:** `0.2632`

#### 📍 Step-by-Step Breakdown:
| Concept | Rendered Prompt | Adherence | Aesthetics | Artifacts | Quality Score | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Concept #1** | `"A vector logo illustrating Futuristic cyberpunk skyscraper inside a green..."` | 1.00 | 0.70 | 0.98 | 1.0000 | **0.2632** |
| **Concept #2** | `"A vector logo illustrating A vintage Victorian library floating in outer..."` | 1.00 | 0.70 | 0.98 | 1.0000 | **0.2632** |
| **Concept #3** | `"A vector logo illustrating Minimalist geometric sculpture on an empty..."` | 1.00 | 0.70 | 0.98 | 1.0000 | **0.2632** |

* **⭐️ Candidate #3 Average Score:** `0.2632`

---

## 🤖 VLM Judge Feedback (Simulated / Active Fallback)
For each prompt evaluated, the VLM Judge output the following detailed textual critiques and evaluations:
> **Critique:** *"The generated image shows high visual relevance to the concept. Adherence score is based on the semantic match ratio. Aesthetic rendering is highly optimized with quality tags."*

---

## ⚔️ Final Optimization Summary
The templates are sorted below by their final average score (taking into account both VLM Quality and the Vision Economics token length penalty):

| Rank | Candidate | Avg Score | Status | Prompt Template |
| :---: | :---: | :---: | :---: | :--- |
| **1st** | **Candidate #1** | **0.2685** | 👑 **[BEST]** | `"A photo of a {concept}."` |
| **2nd** | **Candidate #3** | **0.2632** | | `"A vector logo illustrating {concept} on a red square."` |
| **3rd** | **Candidate #2** | **0.2491** | | `"A stunning, highly detailed professional architectural photograph capturing a {concept}. Cinematic composition, rich photorealism, volume lighting, captured on 35mm lens."` |

### 🏁 Final Outcome:
The **Baseline Candidate #1** remains the optimal choice for the current cost parameters. Although **Candidate #2** achieves higher raw aesthetic ratings, the **258-token vision cost** combined with its long template instruction causes a stronger length-penalty markdown, illustrating how Promptomatix keeps generation pipelines highly cost-efficient and lightweight!

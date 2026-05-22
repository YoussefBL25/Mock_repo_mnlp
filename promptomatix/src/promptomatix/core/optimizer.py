"""
Core module for prompt optimization functionality.
"""

import dspy
import ast
import json
from typing import Dict, List, Type, Optional, Union, Tuple
from datetime import datetime
from dspy.evaluate import Evaluate
import nltk
import os
import logging
from pathlib import Path
import requests
from io import BytesIO
import PIL.Image

from ..utils.parsing import parse_dict_strings
from ..utils.paths import OPTIMIZER_LOGS_DIR
from ..core.config import Config
from ..core.session import OptimizationSession
from ..metrics.metrics import MetricsManager
from .prompts import (
    generate_synthetic_data_prompt,
    generate_synthetic_data_validation_prompt,
    generate_meta_prompt,
    generate_meta_prompt_7,
    generate_meta_prompt_image_gen,
    validate_synthetic_data,
    generate_meta_prompt_2
)

# Setup module logger
logger = logging.getLogger(__name__)

class PromptOptimizer:
    """
    Handles the optimization of prompts using either DSPy or meta-prompt backend.
    
    Attributes:
        config (Config): Configuration for optimization
        lm: Language model instance
        optimized_prompt (str): Latest optimized prompt
        data_template (Dict): Template for data structure
        backend (str): Optimization backend ('dspy' or 'simple_meta_prompt')
    """
    
    def __init__(self, config: Config):
        """
        Initialize the optimizer with configuration.
        
        Args:
            config (Config): Configuration object containing optimization parameters
        """
        self.config = config
        self.lm = None
        self.llm_cost = 0
        # Initialize logger
        setup_optimizer_logger()
        self.logger = logger
        self.logger.info("PromptOptimizer initialized")
        
        # Set backend (default to DSPy for backward compatibility)
        self.backend = getattr(config, 'backend', 'simple_meta_prompt')
        if self.backend not in ['dspy', 'simple_meta_prompt']:
            raise ValueError(f"Unsupported backend: {self.backend}. Must be 'dspy' or 'simple_meta_prompt'")
        
        self.logger.info(f"Using backend: {self.backend}")

    def create_signature(self, name: str, input_fields: List[str], 
                        output_fields: List[str]) -> Type[dspy.Signature]:
        """
        Create a DSPy signature for the optimization task.
        
        Args:
            name (str): Name of the signature
            input_fields (List[str]): List of input field names
            output_fields (List[str]): List of output field names
            
        Returns:
            Type[dspy.Signature]: DSPy signature class
        """
        cleaned_name = name.strip('_').strip()
        
        # Parse fields if they're strings
        input_fields = self._parse_fields(input_fields)
        output_fields = self._parse_fields(output_fields)
        
        # Create signature class
        class_body = {
            '__annotations__': {},
            '__doc__': self.config.task
        }
        
        # Add input and output fields
        has_images = any(key in str(input_fields) or key in str(self.config.sample_data) 
                        for key in ['image', 'image_url', 'image_path']) or self.config.task_type == 'vqa'
                        
        if has_images:
            if 'image_context' not in input_fields:
                input_fields.insert(0, 'image_context')
            class_body['__annotations__']['image_context'] = dspy.Image
            class_body['image_context'] = dspy.InputField(desc="The visual context for the task")
            
        for field in input_fields:
            if field == 'image_context': continue
            field = field.strip('"\'')
            class_body['__annotations__'][field] = str
            class_body[field] = dspy.InputField()
            
        for field in output_fields:
            field = field.strip('"\'')
            class_body['__annotations__'][field] = str
            class_body[field] = dspy.OutputField()
        
        return type(cleaned_name, (dspy.Signature,), class_body)
    
    def _parse_fields(self, fields: Union[List[str], str]) -> List[str]:
        """Parse field definitions from string or list."""
        if isinstance(fields, str):
            fields = fields.strip()
            if not (fields.startswith('[') and fields.endswith(']')):
                fields = f"[{fields}]"
            return ast.literal_eval(fields)
        return fields

    def generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data based on sample data in batches."""
        try:
            sample_data, sample_data_group = self._prepare_sample_data()
            template = {key: '...' for key in sample_data.keys()}
            
            # Ensure all input and output fields are included in the template
            input_fields = self._parse_fields(self.config.input_fields)
            output_fields = self._parse_fields(self.config.output_fields)
            for field in input_fields:
                if field not in template:
                    template[field] = '...'
            for field in output_fields:
                if field not in template:
                    template[field] = '...'

            # On average, 4 characters make up a token
            no_of_toks_in_sample_data = len(str(sample_data))/4
            
            # Calculate batch size based on token limits (assuming 16k token limit)
            max_samples_per_batch = min(50, max(1, int(8000 / no_of_toks_in_sample_data)))
            
            all_synthetic_data = []
            remaining_samples = self.config.synthetic_data_size
            max_retries = 3  # Maximum number of retries per batch
            validation_feedback = []  # Store feedback for failed attempts
            
            print(f"📊 Generating {self.config.synthetic_data_size} synthetic samples...")
            
            # Initialize LLM once for all batches
            with dspy.settings.context():
                try:
                    tmp_lm = dspy.LM(
                        self.config.config_model_name,
                        api_key=self.config.config_model_api_key,
                        api_base=self.config.config_model_api_base,
                        max_tokens=self.config.config_max_tokens,
                        cache=False
                    )
                    
                    batch_num = 1
                    while remaining_samples > 0:
                        batch_size = min(max_samples_per_batch, remaining_samples)
                        
                        valid_batch_data = []
                        retry_count = 0
                        
                        while len(valid_batch_data) < batch_size and retry_count < max_retries:
                            try:
                                # Include previous validation feedback in the prompt
                                feedback_section = ""
                                if validation_feedback:
                                    feedback_section = "\n### Previous Validation Feedback:\n" + "\n".join(
                                        f"- Attempt {i+1}: {feedback}" 
                                        for i, feedback in enumerate(validation_feedback[-3:])  # Show last 3 feedbacks
                                    )
                                
                                prompt = self._create_synthetic_data_prompt(
                                    sample_data_group, 
                                    template, 
                                    batch_size - len(valid_batch_data),
                                    feedback_section
                                )
                                
                                response = tmp_lm(prompt)[0]
                                response = self._clean_llm_response(response)
                                
                                try:
                                    batch_data = json.loads(response)
                                    # Validate each sample in the batch
                                    for sample in batch_data:
                                        is_valid, feedback = self._validate_synthetic_data(sample, self.config.task)
                                        if is_valid:
                                            valid_batch_data.append(sample)
                                            if len(valid_batch_data) >= batch_size:
                                                break
                                        else:
                                            validation_feedback.append(feedback)
                                except json.JSONDecodeError as e:
                                    validation_feedback.append(f"Failed to parse JSON response: {str(e)}")
                                
                                retry_count += 1
                            except Exception as e:
                                self.logger.error(f"Error in batch generation: {str(e)}")
                                retry_count += 1
                                continue
                        
                        if valid_batch_data:
                            all_synthetic_data.extend(valid_batch_data)
                            remaining_samples -= len(valid_batch_data)
                            print(f"  ✓ Batch {batch_num}: {len(valid_batch_data)} samples ({len(all_synthetic_data)}/{self.config.synthetic_data_size})")
                        else:
                            print(f"  ⚠️  Batch {batch_num} failed after {max_retries} retries")
                            break
                        
                        batch_num += 1
                    
                    self.llm_cost += sum([x['cost'] for x in getattr(tmp_lm, 'history', []) if x.get('cost') is not None])
                finally:
                    if 'tmp_lm' in locals():
                        del tmp_lm
                
            print(f"✅ Generated {len(all_synthetic_data)} synthetic samples")
            return all_synthetic_data

        except Exception as e:
            print(f"❌ Failed to generate synthetic data: {str(e)}")
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def _prepare_sample_data(self) -> Dict:
        """Prepare sample data for synthetic data generation."""
        if isinstance(self.config.sample_data, str):
            try:
                # First try to parse as JSON
                try:
                    data = json.loads(self.config.sample_data)
                    if isinstance(data, list):
                        return data[0], data
                    else:
                        return data, [data]
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    data = ast.literal_eval(self.config.sample_data)
                    if isinstance(data, list):
                        return data[0], data
                    else:
                        return data, [data]
            except (SyntaxError, ValueError, json.JSONDecodeError) as e:
                self.logger.error(f"Error parsing sample data: {str(e)}")
                raise ValueError(f"Invalid sample data format: {str(e)}")
        elif isinstance(self.config.sample_data, list):
            return self.config.sample_data[0], self.config.sample_data
        elif isinstance(self.config.sample_data, dict):
            return self.config.sample_data, [self.config.sample_data]
        else:
            raise ValueError(f"Unexpected sample_data type: {type(self.config.sample_data)}")

    def _create_synthetic_data_prompt(self, sample_data: Dict, template: Dict, batch_size: int, feedback_section: str = "") -> str:
        """Generate a high-quality prompt for synthetic data creation with specified batch size."""
        
        return generate_synthetic_data_prompt(
            task=self.config.task,
            batch_size=batch_size,
            example_data=json.dumps(sample_data, indent=2),
            template=json.dumps([template], indent=2),
            feedback_section=feedback_section
        )

    def _clean_llm_response(self, response: str) -> str:
        """Clean and format LLM response, extracting content from fenced blocks if present."""
        response = response.strip()
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
                # Strip any language specifiers from the first line
                lines = content.split('\n')
                if lines:
                    first_line = lines[0].strip().lower()
                    if first_line in ['json', 'xml', 'text', 'html', 'python', 'yaml', 'yml', 'bash', 'sh']:
                        content = '\n'.join(lines[1:]).strip()
                    elif any(first_line.startswith(lang) for lang in ['json', 'xml', 'text', 'html', 'python', 'yaml', 'yml']):
                        if len(first_line) < 10:  # small length safeguard
                            content = '\n'.join(lines[1:]).strip()
                return content
        return response


    def run(self, initial_flag: bool = True) -> Dict:
        """
        Run the optimization process using the configured backend.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        if self.backend == 'dspy':
            return self._run_dspy_backend(initial_flag)
        elif self.backend == 'simple_meta_prompt':
            return self._run_meta_prompt_backend(initial_flag)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _run_dspy_backend(self, initial_flag: bool = True) -> Dict:
        """
        Run optimization using DSPy backend.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        try:
            print("Starting DSPy optimization...")
            
            # Create signature
            print("Creating DSPy signature...")
            signature = self.create_signature(
                name=f"{self.config.task_type.upper()}Signature",
                input_fields=self.config.input_fields,
                output_fields=self.config.output_fields
            )
            print("✓ Signature created")

            # Generate synthetic data if needed
            if not self.config.train_data:
                print("Generating synthetic training data...")
                synthetic_data = self.generate_synthetic_data()
                self.config.train_data = synthetic_data[:self.config.train_data_size]
                self.config.valid_data = synthetic_data[self.config.train_data_size:]
                print(f"✓ Generated {len(synthetic_data)} synthetic samples")

            # Prepare datasets
            print("Preparing datasets...")
            trainset, validset = self._prepare_datasets()
            validset_full = (self._prepare_full_validation_dataset() 
                           if self.config.valid_data_full else validset)
            print("✓ Datasets prepared")

            # Initialize trainer
            print("Initializing DSPy trainer...")
            trainer = self._initialize_trainer()
            print("✓ Trainer initialized")

            # Compile program
            print("Creating DSPy program...")
            if self.config.dspy_module == dspy.ReAct:
                program = self.config.dspy_module(signature, tools=self.config.tools)
            else:
                program = self.config.dspy_module(signature)
            print("✓ Program created")
            
            # Get evaluation metrics
            print("Setting up evaluation metrics...")
            eval_metrics = self.get_final_eval_metrics()
            print("✓ Evaluation metrics ready")
            
            # Evaluate initial prompt
            print("Evaluating initial prompt...")
            evaluator = Evaluate(devset=validset_full, metric=eval_metrics)
            initial_score, initial_results = evaluator(program=program, return_outputs=True)
            print(f"✓ Initial score: {initial_score:.4f}")
            
            # Compile optimized program
            print("Compiling optimized program (this may take a while)...")
            compiled_program = self._compile_program(trainer, program, trainset, validset)
            print("✓ Program compilation complete")
            
            # Evaluate optimized prompt
            print("Evaluating optimized prompt...")
            optimized_score, optimized_results = evaluator(
                program=compiled_program, return_outputs=True
            )
            print(f"✓ Optimized score: {optimized_score:.4f}")
            
            try:
                opt_instructions = compiled_program.signature.instructions
            except:
                opt_instructions = compiled_program.predict.signature.instructions
            
            # Prepare and return results
            print("Preparing final results...")
            result = self._prepare_results(
                self.config.task,
                opt_instructions,
                initial_score,
                optimized_score
            )
            
            # Calculate LLM cost from optimizer's LM if available
            if hasattr(self, 'lm') and self.lm is not None:
                self.llm_cost += sum([x['cost'] for x in getattr(self.lm, 'history', []) if x.get('cost') is not None])
            
            print("✓ DSPy optimization complete!")
            return result

        except Exception as e:
            print(f"❌ Error in DSPy optimization: {str(e)}")
            self.logger.error(f"Error in DSPy optimization run: {str(e)}")
            return {'error': str(e), 'session_id': self.config.session_id}

    def _run_meta_prompt_backend(self, initial_flag: bool = True) -> Dict:
        """
        Run optimization using meta-prompt backend with direct API calls.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        try:
            # Generate synthetic data if needed (same as DSPy backend)
            if not self.config.train_data:
                synthetic_data = self.generate_synthetic_data()
                self.config.train_data = synthetic_data[:self.config.train_data_size]
                self.config.valid_data = synthetic_data[self.config.train_data_size:]
            
            # Evaluate initial prompt first.
            # Set OPT_PHASE so the image-generation metric routes saved images into
            # outputs/generated_images/<CONCEPT_LABEL>/synthetic_data/ for this round.
            print("🔧 Evaluating initial prompt...")
            os.environ["OPT_PHASE"] = "synthetic_data"
            initial_score = self._evaluate_prompt_meta_backend(self.config.task)
            print(f"  Initial score: {initial_score:.4f}")

            # Generate meta-prompt. For image_generation tasks, use the image-gen
            # specific rewriter that expands the concept into a rich diffusion prompt
            # rather than the generic schema-preserving rewriter (which short-circuits
            # on prompts < 280 chars and barely changes them).
            if (self.config.task_type or "").lower() == "image_generation":
                meta_prompt = generate_meta_prompt_image_gen(self.config.raw_input)
            else:
                meta_prompt = generate_meta_prompt_7(self.config.raw_input)

            # Get optimized prompt from LLM using direct API calls.
            # Sampling temperature deliberately bumped above 0 so the rewriter
            # can explore phrasing and detail. Evaluation, validation, and
            # synthetic-data calls remain deterministic (temperature defaults
            # to 0.0 in _call_openai_api when no override is passed).
            optimized_prompt_raw = self._call_llm_api_directly(meta_prompt, temperature=0.7)

            # Strip XML wrapper tags that the meta-prompt LLM may produce
            import re as _re
            optimized_prompt = _re.sub(
                r'</?optimized_prompt>', '', optimized_prompt_raw
            ).strip()
            print(f"  Cleaned optimized prompt: {optimized_prompt}")

            # Evaluate optimized prompt — route saved images into final_optimized/
            print("📊 Evaluating optimized prompt...")
            os.environ["OPT_PHASE"] = "final_optimized"
            optimized_score = self._evaluate_prompt_meta_backend(optimized_prompt)
            print(f"  Optimized score: {optimized_score:.4f}")
            os.environ.pop("OPT_PHASE", None)
            
            # Prepare and return results
            result = self._prepare_results(
                self.config.raw_input,
                optimized_prompt,
                initial_score,
                optimized_score
            )
            
            print("✅ Prompt optimization complete!")
            return result
                    
        except Exception as e:
            print(f"❌ Prompt optimization failed: {str(e)}")
            self.logger.error(f"Error in meta-prompt optimization run: {str(e)}")
            return {'error': str(e), 'session_id': self.config.session_id}

    def _evaluate_prompt_meta_backend(self, prompt: str) -> float:
        """
        Evaluate a prompt using the meta-prompt backend by testing it against synthetic data.
        
        Args:
            prompt (str): The prompt to evaluate
            
        Returns:
            float: Average score across all synthetic data samples
        """
        try:
            # Get the appropriate evaluation metric for the task type
            eval_metric = self.get_final_eval_metrics()
            
            # Use all available synthetic data for evaluation
            all_data = self.config.train_data + (self.config.valid_data or [])
            
            if not all_data:
                return 0.0
            
            total_score = 0.0
            valid_evaluations = 0

            # For image generation, the prompt under evaluation IS the diffusion
            # prompt — we want it to reach the diffuser unchanged. Skipping the
            # per-sample LM rewrite avoids the LM fusing the prompt with the
            # synthetic-data variation and emitting a drifted candidate (e.g.
            # rewriting "A vintage Victorian library floating in outer space"
            # into "A library of virtual reality books floating in outer space"
            # because that's the variation the synth generator produced).
            is_image_gen = (self.config.task_type or "").lower() == "image_generation"

            for i, sample in enumerate(all_data):
                try:
                    # Extract image keys if present (consumed by the LM path only)
                    images = []
                    for key in ['image', 'images', 'image_url', 'image_path']:
                        if key in sample and sample[key]:
                            val = sample[key]
                            if isinstance(val, list):
                                images.extend(val)
                            else:
                                images.append(val)

                    if is_image_gen:
                        # Send the prompt straight through. Synthetic samples
                        # still drive iteration count -> diffuser stochasticity
                        # gives us per-call image variance for averaging.
                        prediction = self._create_prediction_object(prompt, sample)
                    else:
                        test_input = self._create_test_input_from_sample(sample)
                        full_test_prompt = f"{prompt}\n\n{test_input}"
                        prediction_text = self._call_llm_api_directly(full_test_prompt, images=images if images else None)
                        prediction = self._create_prediction_object(prediction_text, sample)

                    # Evaluate using the appropriate metric
                    score = eval_metric(sample, prediction, prompt)
                    total_score += score
                    valid_evaluations += 1
                    
                except Exception as e:
                    print(f"  ⚠ Sample {i} evaluation failed: {e}")
                    continue
            
            if valid_evaluations == 0:
                return 0.0
            
            average_score = total_score / valid_evaluations
            return average_score
            
        except Exception as e:
            self.logger.error(f"Error in prompt evaluation: {str(e)}")
            return 0.0

    def _create_test_input_from_sample(self, sample: Dict) -> str:
        """
        Create a test input string from a sample data dictionary.
        
        Args:
            sample (Dict): Sample data containing input fields
            
        Returns:
            str: Formatted test input string
        """
        try:
            # Parse input fields
            input_fields = self._parse_input_fields()
            
            if isinstance(input_fields, (list, tuple)):
                # Multiple input fields
                input_parts = []
                for field in input_fields:
                    if field in sample:
                        input_parts.append(f"{field}: {sample[field]}")
                    else:
                        input_parts.append(f"{field}: [MISSING]")
                return "\n".join(input_parts)
            else:
                # Single input field
                if input_fields in sample:
                    return f"{input_fields}: {sample[input_fields]}"
                else:
                    return f"{input_fields}: [MISSING]"
                
        except Exception as e:
            self.logger.error(f"Error creating test input: {str(e)}")
            # Fallback: return the sample as a simple string
            return str(sample)

    def _create_prediction_object(self, prediction_text: str, sample: Dict) -> Dict:
        """
        Create a prediction object with the expected structure for evaluation.
        
        Args:
            prediction_text (str): Raw prediction text from LLM
            sample (Dict): Original sample for reference
            
        Returns:
            Dict: Prediction object with output fields populated
        """
        try:
            # Parse output fields
            if isinstance(self.config.output_fields, str):
                output_fields = ast.literal_eval(self.config.output_fields)
            else:
                output_fields = self.config.output_fields
            
            # Create prediction object
            prediction = {}
            
            if isinstance(output_fields, (list, tuple)):
                # Multiple output fields - try to extract them from the prediction text
                if len(output_fields) == 1:
                    # Single output field, use the entire prediction text
                    prediction[output_fields[0]] = prediction_text.strip()
                else:
                    # Multiple output fields - this is more complex
                    # For now, assign the prediction text to the first field
                    # In a more sophisticated implementation, you might parse the text
                    prediction[output_fields[0]] = prediction_text.strip()
                    for field in output_fields[1:]:
                        prediction[field] = ""  # Empty for additional fields
            else:
                # Single output field
                prediction[output_fields] = prediction_text.strip()
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error creating prediction object: {str(e)}")
            # Fallback: return a simple object with the prediction text
            return {"output": prediction_text.strip()}

    def _call_llm_api_directly(self, prompt: str, model: str = "", images: list = None,
                                temperature: float = None) -> str:
        """
        Call LLM API directly based on the configured provider.

        Args:
            prompt (str): The prompt to send to the LLM
            model (str): Optional model name
            images (list): Optional list of image paths or URLs
            temperature (float): Optional sampling temperature. When None, each
                provider falls back to its existing default (0.0 for OpenAI-style,
                config_temperature for Anthropic, provider default for Gemini),
                keeping evaluation/validation calls deterministic.

        Returns:
            str: The LLM response
        """
        try:
            # Determine the provider from config
            provider = getattr(self.config, 'config_model_provider', 'openai')
            if hasattr(provider, 'value'):
                provider = provider.value

            if provider.lower() in ('openai', 'local', 'databricks', 'togetherai'):
                return self._call_openai_api(prompt, model, temperature=temperature)
            elif provider.lower() == 'anthropic':
                return self._call_anthropic_api(prompt, temperature=temperature)
            elif provider.lower() == 'gemini':
                return self._call_gemini_api(prompt, model, images, temperature=temperature)
            else:
                raise ValueError(f"Unsupported provider for direct API calls: {provider}")
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API directly: {str(e)}")
            raise

    def _call_gemini_api(self, prompt: str, model: str = "", images: list = None,
                          temperature: float = None) -> str:
        """
        Call Gemini API directly.
        
        Args:
            prompt (str): The prompt to send
            model (str): Model name
            images (list): Optional list of image paths or URLs
            
        Returns:
            str: The API response
        """
        import google.generativeai as genai
        
        if model == "":
            model = self.config.config_model_name
            
        genai.configure(api_key=self.config.config_model_api_key)
        
        contents = [prompt]
        if images:
            for img_path in images:
                if img_path.startswith("http://") or img_path.startswith("https://"):
                    response = requests.get(img_path)
                    response.raise_for_status()
                    img = PIL.Image.open(BytesIO(response.content))
                    contents.append(img)
                else:
                    contents.append(PIL.Image.open(img_path))
                    
        try:
            # Strip the 'gemini/' prefix if it was added for litellm compatibility
            clean_model = model.replace('gemini/', '') if model.startswith('gemini/') else model
            
            llm = genai.GenerativeModel(model_name=clean_model)
            gen_kwargs = {}
            if temperature is not None:
                gen_kwargs["generation_config"] = genai.GenerationConfig(temperature=temperature)
            response = llm.generate_content(contents=contents, **gen_kwargs)
            response_text = response.text
            
            # Calculate cost (approximate)
            input_tokens = len(prompt.split()) * 1.3 
            output_tokens = len(response_text.split()) * 1.3
            self.llm_cost += (input_tokens * 0.00001) + (output_tokens * 0.00003)
            
            return self._clean_llm_response(response_text)
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {str(e)}")
            raise

    def _call_openai_api(self, prompt: str, model: str = "", temperature: float = None) -> str:
        """
        Call OpenAI API directly.

        Args:
            prompt (str): The prompt to send
            model (str): Optional model name
            temperature (float): Optional sampling temperature. Defaults to 0.0
                (deterministic) when not provided — appropriate for eval and
                validation calls. Callers that want sampling (e.g. the rewriter)
                pass an explicit value.

        Returns:
            str: The API response
        """
        if model == "":
            model = self.config.config_model_name

        # Strip provider prefix if present (e.g. "openai/") to match local vLLM served model name
        if model.startswith("openai/"):
            model = model[len("openai/"):]

        from openai import OpenAI

        for prefix in ("openai/", "local/", "azure/", "togetherai/", "databricks/"):
            if model.startswith(prefix):
                model = model[len(prefix):]
                break

        client = OpenAI(
            api_key=self.config.config_model_api_key,
            base_url=self.config.config_model_api_base if self.config.config_model_api_base else None
        )

        effective_temperature = temperature if temperature is not None else 0.0

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=effective_temperature
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Calculate cost (approximate)
            # OpenAI pricing varies by model, this is a rough estimate
            input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
            output_tokens = len(response_text.split()) * 1.3
            
            # Estimate cost (this would need to be updated with actual pricing)
            estimated_cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)  # Rough estimate
            self.llm_cost += estimated_cost

            if model != "o3" and model != "gpt-4.1":
                return self._clean_llm_response(response_text)
            else:
                return response_text
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def _call_anthropic_api(self, prompt: str, temperature: float = None) -> str:
        """
        Call Anthropic API directly.

        Args:
            prompt (str): The prompt to send
            temperature (float): Optional sampling temperature override.
                Falls back to self.config.config_temperature when not provided.

        Returns:
            str: The API response
        """
        import anthropic

        # Configure Anthropic client
        client = anthropic.Anthropic(
            api_key=self.config.config_model_api_key,
            base_url=self.config.config_model_api_base if self.config.config_model_api_base else None
        )

        effective_temperature = temperature if temperature is not None else self.config.config_temperature

        try:
            response = client.messages.create(
                model=self.config.config_model_name,
                max_tokens=self.config.config_max_tokens,
                temperature=effective_temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response content
            response_text = response.content[0].text
            
            # Calculate cost (approximate)
            # Anthropic pricing varies by model, this is a rough estimate
            input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
            output_tokens = len(response_text.split()) * 1.3
            
            # Estimate cost (this would need to be updated with actual pricing)
            estimated_cost = (input_tokens * 0.000015) + (output_tokens * 0.000075)  # Rough estimate
            self.llm_cost += estimated_cost
            
            return self._clean_llm_response(response_text)
            
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            raise

    def _parse_input_fields(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Parse input fields from config."""
        return (ast.literal_eval(self.config.input_fields) 
                if isinstance(self.config.input_fields, str) 
                else self.config.input_fields)

    def _get_base64_image(self, img_src: str) -> str:
        """Convert image URL or path to base64 string for VLM payload formatting."""
        if str(img_src).startswith('data:image'):
            return img_src
            
        try:
            import base64
            import requests
            if str(img_src).startswith('http'):
                response = requests.get(img_src)
                response.raise_for_status()
                img_data = response.content
            else:
                with open(img_src, 'rb') as f:
                    img_data = f.read()
                    
            b64_data = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
        except Exception as e:
            self.logger.warning(f"Failed to convert image {img_src} to base64: {e}")
            return img_src

    def _prepare_dataset(self, data: List[Dict]) -> List[dspy.Example]:
        """Prepare a dataset from input data, injecting base64 image_context if present."""
        input_fields = self._parse_input_fields()
        if not isinstance(input_fields, list):
            input_fields = list(input_fields)
            
        has_image = False
        prepared_data = []
        
        for ex in data:
            ex_copy = ex.copy()
            images = []
            for key in ['image', 'image_url', 'image_path']:
                if key in ex_copy:
                    img_val = ex_copy[key]
                    if isinstance(img_val, list) and img_val:
                        images.extend(img_val)
                    elif img_val:
                        images.append(img_val)
            
            if images:
                has_image = True
                # Inject the first image as base64 DSPy Image context
                b64_url = self._get_base64_image(images[0])
                ex_copy['image_context'] = dspy.Image(url=b64_url)
                
            prepared_data.append(ex_copy)
            
        if has_image and 'image_context' not in input_fields:
            input_fields.insert(0, 'image_context')
        
        if isinstance(input_fields, (list, tuple)):
            return [dspy.Example(**ex).with_inputs(*input_fields) for ex in prepared_data]
        return [dspy.Example(**ex).with_inputs(input_fields) for ex in prepared_data]

    def _prepare_datasets(self):
        """Prepare training and validation datasets."""
        return (
            self._prepare_dataset(self.config.train_data),
            self._prepare_dataset(self.config.valid_data)
        )

    def _prepare_full_validation_dataset(self):
        """Prepare full validation dataset if available."""
        return self._prepare_dataset(self.config.valid_data_full)

    def _initialize_trainer(self):
        """Initialize the DSPy trainer."""
        return dspy.MIPROv2(
            metric=self.get_eval_metrics(),
            init_temperature=0.7,
            auto=self.config.miprov2_init_auto,
            num_candidates=self.config.miprov2_init_num_candidates
        )

    def _compile_program(self, trainer, program, trainset, validset):
        """Compile the program using the trainer."""
        return trainer.compile(
            program,    
            trainset=trainset,
            valset=validset,
            requires_permission_to_run=False,
            max_bootstrapped_demos=self.config.miprov2_compile_max_bootstrapped_demos,
            max_labeled_demos=self.config.miprov2_compile_max_labeled_demos,
            num_trials=self.config.miprov2_compile_num_trials,
            minibatch_size=self.config.miprov2_compile_minibatch_size
        )

    def _prepare_results(self, initial_prompt: str, optimized_prompt: str, 
                        initial_score: float, optimized_score: float) -> Dict:
        """Prepare the final results dictionary."""    
            
        return {    
            'result': optimized_prompt,
            'initial_prompt': initial_prompt,
            'session_id': self.config.session_id,
            'backend': self.backend,
            'metrics': {
                'initial_prompt_score': initial_score,
                'optimized_prompt_score': optimized_score
            },
            'synthetic_data': self.config.train_data + (self.config.valid_data or []),  # Include all synthetic data
            'llm_cost': self.llm_cost
        }

    def get_eval_metrics(self):
        """Get evaluation metrics for the task type."""
        if isinstance(self.config.output_fields, str):
            output_fields = ast.literal_eval(self.config.output_fields)
        else:
            output_fields = self.config.output_fields
        
        MetricsManager.configure(output_fields)
        return MetricsManager.get_metrics_for_task(self.config.task_type)
    
    def get_final_eval_metrics(self):
        """Get final evaluation metrics for the task type."""
        if isinstance(self.config.output_fields, str):
            output_fields = ast.literal_eval(self.config.output_fields)
        else:
            output_fields = self.config.output_fields
        
        MetricsManager.configure(output_fields)
        return MetricsManager.get_final_eval_metrics(self.config.task_type)


    
    def _validate_synthetic_data(self, data: Dict, task: str) -> Tuple[bool, str]:
        """
        Validate the generated synthetic data for quality and consistency.
        
        Args:
            data (Dict): The generated data sample to validate
            task (str): The task to validate the data against
            
        Returns:
            Tuple[bool, str]: (True if valid, feedback message)
        """        
        try:
            prompt = validate_synthetic_data(task, data, self.config.input_fields, self.config.output_fields)

            response = self._call_llm_api_directly(prompt)

            # read and clean the response which is expected to be in json format
            response = self._clean_llm_response(response)
            response = json.loads(response)

            try:
                is_valid = response['is_valid']
                feedback = response['feedback']
            except Exception as e:
                self.logger.error(f"Error parsing validation response: {str(e)}")
                return False, "Invalid validation response"

            if not is_valid:
                return False, feedback
            else:
                return True, "Sample passed all validation checks"
            
            
        except Exception as e:
            error_msg = f"Error in data validation: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

def setup_optimizer_logger():
    """Set up dedicated logger for optimization steps and results."""
    logger.setLevel(logging.DEBUG)

    # Ensure the optimizer logs directory exists
    OPTIMIZER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OPTIMIZER_LOGS_DIR / f"optimizer_{timestamp}.jsonl"

    # File Handler for JSON Lines format
    class JSONLinesHandler(logging.FileHandler):
        def emit(self, record):
            try:
                # Only try to parse JSON if it's an optimization step
                if hasattr(record, 'optimization_step'):
                    msg = record.optimization_step
                else:
                    # For regular log messages, create a simple JSON structure
                    msg = {
                        'timestamp': datetime.now().isoformat(),
                        'level': record.levelname,
                        'message': self.format(record)
                    }
                
                with open(self.baseFilename, 'a') as f:
                    json.dump(msg, f)
                    f.write('\n')
            except Exception as e:
                # Log error to stderr but don't raise to avoid logging loops
                import sys
                print(f"Error in JSONLinesHandler: {str(e)}", file=sys.stderr)
                # Ensure the file is closed even if an error occurs
                self.close()

    # Custom formatter for optimization steps
    class OptimizationStepFormatter(logging.Formatter):
        def format(self, record):
            if hasattr(record, 'optimization_step'):
                return json.dumps(record.optimization_step)
            return super().format(record)

    # Set up file handler with custom formatter
    file_handler = JSONLinesHandler(log_file)
    file_handler.setFormatter(OptimizationStepFormatter())
    logger.addHandler(file_handler)

    logger.info(f"Optimizer logger initialized. Log file: {log_file}") 
import os
import ast
import json
import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any, Dict, Type
import dspy
import requests
from datasets import load_dataset, Dataset

from ..utils.paths import CONFIG_LOGS_DIR
from ..utils.google_lens import (
    fetch_google_image_search_urls,
    fetch_similar_image_urls_from_google_lens,
)
from .prompts import (
    generate_dspy_module_from_task_description_and_sample_data,
    extract_task_description_from_raw_input,
    extract_task_type_from_raw_input,
    extract_tools_from_raw_input,
    generate_sample_data_from_task_description_and_raw_input,
    simplify_human_feedback,
    simplify_human_feedback_2,
    extract_fields_from_sample_data,
    improvise_raw_input,
    convert_few_shot_examples_to_json,
    complete_the_main_example,
    generate_sample_data_from_sample_data,
    get_expected_answer_from_sample_data,
    generate_task_description_from_sample_data,
    improvise_raw_input_tools,
    improvise_raw_input_task
)

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    DATABRICKS = 'databricks'
    LOCAL = 'local'
    TOGETHERAI = 'togetherai'
    GEMINI = 'gemini'

DEFAULT_SYNTHETIC_DATA_SIZE = 30
DEFAULT_TRAIN_RATIO = 0.2

class LambdaPenalty:
    """Class to manage the lambda penalty value for metrics calculations."""
    _value = 0.005  # Default value

    @classmethod
    def get_value(cls) -> float:
        """Get the current lambda penalty value."""
        return cls._value

    @classmethod
    def set_value(cls, value: float) -> None:
        """Set the lambda penalty value."""
        cls._value = value

AGENTIC_TASK_TYPES = {
    'code_generation', 'code_explanation', 'code_completion', 'code_debugging',
    'planning', 'tool_use', 'decision_making', 'process_automation', 
    'reasoning', 'agentic', 'agentic-reasoning'
}

class DSPyModules(str, Enum):
    """Available DSPy modules."""
    PREDICT = 'dspy.Predict'
    CHAIN_OF_THOUGHT = 'dspy.ChainOfThought'
    PROGRAM_OF_THOUGHT = 'dspy.ProgramOfThought'
    REACT = 'dspy.ReAct'

DSPY_MODULE_MAP = {
    DSPyModules.PREDICT: dspy.Predict,
    DSPyModules.CHAIN_OF_THOUGHT: dspy.ChainOfThought,
    DSPyModules.PROGRAM_OF_THOUGHT: dspy.ProgramOfThought,
    DSPyModules.REACT: dspy.ReAct
}

class DatasetConfig:
    """Configuration for a few common huggingface datasets."""
    XSUM = {
        'input_fields': ['document'],
        'output_fields': ['summary'],
        'task_type': 'summarization'
    }
    COMMON_GEN = {
        'input_fields': ['concepts'],
        'output_fields': ['target'],
        'task_type': 'generation'
    }
    AG_NEWS = {
        'input_fields': ['text'],
        'output_fields': ['label'],
        'task_type': 'classification',
        'label_map': {
            0: "World", 
            1: "Sports", 
            2: "Business", 
            3: "Science and Technology"
        }
    }
    SQUAD_V2 = {
        'input_fields': ['question', 'context'],
        'output_fields': ['answers'],
        'task_type': 'qa'
    }

    GSM8K = {
        'input_fields': ['question'],
        'output_fields': ['answer'],
        'task_type': 'qa'
    }

def setup_config_logger():
    """Set up dedicated logger for LLM interactions in configuration.
    
    Creates a file handler that logs all LLM prompts and responses to a dedicated file
    in the logs directory.
    """
    logger.setLevel(logging.DEBUG)

    # Ensure the config logs directory exists
    CONFIG_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = CONFIG_LOGS_DIR / f"llm_interactions_{timestamp}.jsonl"

    # File Handler for JSON Lines format
    class JSONLinesHandler(logging.FileHandler):
        def emit(self, record):
            try:
                # Only try to parse JSON if it's an LLM interaction
                if hasattr(record, 'llm_interaction'):
                    msg = record.llm_interaction
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

    # Custom formatter for LLM interactions
    class LLMInteractionFormatter(logging.Formatter):
        def format(self, record):
            if hasattr(record, 'llm_interaction'):
                return json.dumps(record.llm_interaction)
            return super().format(record)

    # Set up file handler with custom formatter
    file_handler = JSONLinesHandler(log_file)
    file_handler.setFormatter(LLMInteractionFormatter())
    logger.addHandler(file_handler)

    logger.info(f"Config LLM interaction logger initialized. Log file: {log_file}")

def log_llm_interaction(prompt: str, response: str, context: str = None):
    """Log an LLM interaction with structured data.
    
    Args:
        prompt (str): The prompt sent to the LLM
        response (str): The response received from the LLM
        context (str, optional): The context or purpose of this LLM call
    """
    interaction = {
        'timestamp': datetime.now().isoformat(),
        'context': context,
        'prompt': prompt,
        'response': response
    }
    
    # Create a custom log record with the interaction data
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='',
        args=(),
        exc_info=None
    )
    record.llm_interaction = interaction
    logger.handle(record)


class Config:
    """Configuration class for prompt optimization.
    
    This class manages configuration parameters for natural language processing tasks,
    automatically inferring missing parameters and validating the configuration. It supports
    both direct human input and HuggingFace datasets as data sources.
    
    Attributes:
        # Input Source Parameters
        raw_input (str): Raw input text for task initialization
        huggingface_dataset_name (Optional[str]): Name of HuggingFace dataset (e.g., 'squad_v2')
        original_raw_input (Optional[str]): Preserved copy of initial human input
        session_id (Optional[str]): Unique identifier for the optimization session

        # Task Configuration
        task_description (Optional[str]): Detailed description of the NLP task to be performed
        sample_data (Optional[str]): Example input-output pairs for the task
        output_format (Optional[str]): Expected format of the output (e.g., 'json', 'text', 'list')
        style_guide (Optional[str]): Guidelines for output formatting, tone, and style
        constraints (Optional[str]): Task-specific constraints and limitations (e.g., max length)
        context (Optional[str]): Additional context or background information for task execution
        task_type (Optional[str]): Category of NLP task (e.g., 'classification', 'qa', 'generation')
        tools (Optional[List[str]]): External tools or resources needed for task execution
        decouple_task_description_and_raw_input (Optional[bool]): Flag to decouple task description and human input

        # Model Configuration
        model_name (Optional[str]): Name of the language model (e.g., 'gpt-4', 'claude-2')
        model_provider (Optional[str]): Provider of the model (e.g., 'openai', 'anthropic')
        model_api_key (Optional[str]): API key for model access
        model_api_base (Optional[str]): Base URL for API requests
        temperature (float): Sampling temperature for generation (default: 0.7)
        max_tokens (int): Maximum tokens for model output (default: 4000)

        # Training Configuration
        synthetic_data_size (Optional[int]): Number of synthetic examples to generate
        train_ratio (Optional[float]): Fraction of data to use for training (0.0 to 1.0)
        dspy_module (Optional[Any]): DSPy module configuration for task execution
        input_fields (List[str]): Required input field names for structured data
        output_fields (List[str]): Expected output field names for structured data
        metrics (List[str]): Evaluation metrics to use (e.g., ['accuracy', 'f1'])
        trainer (Optional[str]): Training algorithm to use (e.g., 'MIPROv2')
        search_type (Optional[str]): Optimization strategy ('quick_search', 'moderate_search', 'heavy_search')
        backend (str): Optimization backend ('dspy' or 'simple_meta_prompt', default: 'simple_meta_prompt')

        # Data Configuration
        train_data (Optional[Any]): Training dataset
        valid_data (Optional[Any]): Validation dataset
        valid_data_full (Optional[Any]): Complete validation dataset
        train_data_size (Optional[int]): Size of training dataset
        valid_data_size (Optional[int]): Size of validation dataset
        load_data_local (bool): Flag to load data from local files (default: False)
        local_train_data_path (Optional[str]): Path to local training data file
        local_test_data_path (Optional[str]): Path to local test data file

    Raises:
        ValueError: If neither raw_input nor huggingface_dataset_name is provided
        ValueError: If model_provider is specified but not supported
        ValueError: If required fields for specified task_type are missing
        ValueError: If data configuration is invalid or inconsistent

    Examples:
        >>> config = Config(raw_input="Classify text into positive/negative sentiment")
        >>> config = Config(
        ...     huggingface_dataset_name="squad_v2",
        ...     model_name="gpt-4",
        ...     model_provider="openai"
        ... )
    """
    def __init__(self, **kwargs):
        """Initialize configuration parameters for DSPy task.
        
        Args:
            **kwargs: Configuration parameters as described in class attributes.
                     See class documentation for detailed parameter descriptions.
            
        Raises:
            ValueError: If configuration parameters are invalid or inconsistent.
                       See class documentation for specific error conditions.
        """
        setup_config_logger()

        # Input source parameters (required)
        self.raw_input = kwargs.get('raw_input')
        self.raw_input_improvised = kwargs.get('raw_input_improvised', False)
        self.huggingface_dataset_name = kwargs.get('huggingface_dataset_name')
        self.original_raw_input = kwargs.get('original_raw_input')
        self.session_id = None

        # Task configuration
        self.task_description = kwargs.get('task_description')
        self.sample_data = kwargs.get('sample_data')
        self.output_format = kwargs.get('output_format')
        self.style_guide = kwargs.get('style_guide')
        self.constraints = kwargs.get('constraints')
        self.context = kwargs.get('context')
        self.task_type = kwargs.get('task_type')
        self.tools = kwargs.get('tools')
        self.decouple_task_description_and_raw_input = kwargs.get('decouple_task_description_and_raw_input', False)
        self.image_pool = kwargs.get('image_pool') or []
        self.image_pool_target_size = kwargs.get('image_pool_target_size', 50)
        self.image_pool_sources = kwargs.get('image_pool_sources', ['serpapi'])
        self.image_pool_site_filters = kwargs.get('image_pool_site_filters') or []
        self.image_pool_use_lm_query = kwargs.get('image_pool_use_lm_query', True)
        self.image_pool_query = kwargs.get('image_pool_query')
        self.auto_fetch_image_pool = kwargs.get('auto_fetch_image_pool', True)
        self.google_custom_search_api_key = (
            kwargs.get('google_custom_search_api_key')
            or os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
            or os.getenv('GOOGLE_API_KEY')
        )
        self.google_custom_search_cx = (
            kwargs.get('google_custom_search_cx')
            or kwargs.get('google_search_engine_id')
            or os.getenv('GOOGLE_CUSTOM_SEARCH_CX')
            or os.getenv('GOOGLE_SEARCH_ENGINE_ID')
            or os.getenv('GOOGLE_CSE_ID')
        )
        self.serpapi_api_key = (
            kwargs.get('serpapi_api_key')
            or os.getenv('SERPAPI_API_KEY')
        )
        self.serpapi_hl = kwargs.get('serpapi_hl') or os.getenv('SERPAPI_HL') or 'en'
        self.serpapi_gl = kwargs.get('serpapi_gl') or os.getenv('SERPAPI_GL') or 'us'

        # Model configuration
        self.model_name = kwargs.get('model_name')
        self.model_api_key = kwargs.get('model_api_key')
        self.model_api_base = kwargs.get('model_api_base')
        self.model_provider = kwargs.get('model_provider')
        self.temperature = kwargs.get('temperature', 0.7)  # Default temperature
        self.max_tokens = kwargs.get('max_tokens', 2048)  # Default max tokens
        self.config_max_tokens = kwargs.get('config_max_tokens', 2048)  # Default max tokens
        self.config_temperature = kwargs.get('config_temperature', 0.7)
        self.config_model_name = kwargs.get('config_model_name')
        self.config_model_provider = kwargs.get('config_model_provider')
        self.config_model_api_key = kwargs.get('config_model_api_key')
        self.config_model_api_base = kwargs.get('config_model_api_base')

        # Training configuration
        self.synthetic_data_size = kwargs.get('synthetic_data_size')
        self.train_ratio = kwargs.get('train_ratio')
        self.dspy_module = kwargs.get('dspy_module')
        self.input_fields = kwargs.get('input_fields', [])
        self.output_fields = kwargs.get('output_fields', [])
        self.metrics = kwargs.get('metrics', [])
        self.trainer = kwargs.get('trainer')
        self.search_type = kwargs.get('search_type', 'quick_search')
        self.backend = kwargs.get('backend', 'simple_meta_prompt')  # Default to DSPy backend

        # Set lambda penalty value if provided, otherwise use default
        lambda_penalty = kwargs.get('lambda_penalty')
        if lambda_penalty is not None:
            LambdaPenalty.set_value(lambda_penalty)
        self.lambda_penalty = LambdaPenalty.get_value()

        # Data configuration
        self.train_data = kwargs.get('train_data')
        self.valid_data = kwargs.get('valid_data')
        self.valid_data_full = kwargs.get('valid_data_full')
        self.train_data_size = kwargs.get('train_data_size')
        self.valid_data_size = kwargs.get('valid_data_size')
        self.load_data_local = kwargs.get('load_data_local', False)
        self.local_train_data_path = kwargs.get('local_train_data_path')
        self.local_test_data_path = kwargs.get('local_test_data_path')
        self._image_pool_query_cache = self.image_pool_query

        # Validate backend
        if self.backend not in ['dspy', 'simple_meta_prompt']:
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'dspy' or 'simple_meta_prompt'")

        # Validate and populate missing configurations
        if self.huggingface_dataset_name is None and self.raw_input is not None:
            self._populate_config()
        elif self.huggingface_dataset_name is not None:
            self._populate_config_from_huggingface()
        else:
            raise ValueError("Either huggingface_dataset_name or raw_input must be provided")

        # Final validation of required fields
        # self._validate()

    def _populate_config_from_huggingface(self):
        """Populate configuration from HuggingFace dataset."""

        self._set_search_type_config()

        # Checking if the dataset has train split at all. If yes, use it. Else, generate synthetic data.
        train_dataset, test_dataset = self._load_and_process_dataset()
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

        # Usage in _populate_config_from_huggingface
        self._prepare_datasets(train_dataset, test_dataset)
        logger.info(f"Prepared datasets - Train: {len(self.train_data)}, Valid: {len(self.valid_data)}, Valid Full: {len(self.valid_data_full)}")
        
        # Store original human input
        if self.original_raw_input is None:
            self.original_raw_input = self.raw_input

        # setup the LM for config
        tmp_lm = self._setup_model_config()
        logger.info(f"Using {self.model_provider} model: {self.model_name}")

        # Create task description (raw_input) from dataset
        self.raw_input = self._create_task_description(tmp_lm)
        logger.info(f"Task description created: {self.raw_input[:100]}...")

        # Process and extract configurations in order
        self.raw_input = self._process_human_feedback(tmp_lm)
        logger.info("Human input processed")

        # if not self.raw_input_improvised:
        #    self.raw_input_improvised = self._improvise_raw_input(tmp_lm)
        #    logger.info("Human input improvised")

        # Develop prompt template components form raw input
        self._develop_prompt_template_components(tmp_lm)

        if not self.task_description:
            self.task_description = self._extract_task_description(tmp_lm)
            logger.info(f"Task description extracted: {self.task_description[:100]}...")

        if not self.input_fields or not self.output_fields:
            self.input_fields, self.output_fields = self._extract_fields(tmp_lm)
            logger.info(
                f"Fields extracted - Input: {self.input_fields}, "
                f"Output: {self.output_fields}"
            )

        if not self.task_type:
            self.task_type = self._extract_task_type(tmp_lm)
            logger.info(f"Task type determined: {self.task_type}")

        # Extract tools only for relevant task types
        self.tools = self._extract_tools(tmp_lm)
        if self.tools:
            logger.info(f"Tools extracted for {self.task_type} task")

        # Set training configuration
        self.trainer = self._set_trainer()
        logger.info(f"Using trainer: {self.trainer}")

        self.dspy_module = self._set_dspy_module(tmp_lm)
        logger.info(f"Selected DSPy module: {self.dspy_module.__name__}")

        self._resolve_image_pool(tmp_lm=tmp_lm)
        if self.image_pool:
            logger.info(f"Image pool prepared with {len(self.image_pool)} items")

        # Set data configuration with defaults if needed
        self.synthetic_data_size = self.synthetic_data_size or DEFAULT_SYNTHETIC_DATA_SIZE
        logger.info(f"Synthetic data size: {self.synthetic_data_size}")

        self.train_ratio = self.train_ratio or DEFAULT_TRAIN_RATIO
        logger.info(f"Train ratio: {self.train_ratio}")

        # At the end, after all LLM calls:
        self.llm_cost = sum([x['cost'] for x in getattr(tmp_lm, 'history', []) if x.get('cost') is not None])

        # Cleanup
        del tmp_lm
        logger.info("Configuration population completed successfully")

        
    def _populate_config(self):
        """Populate missing configuration parameters in a specific order.
        
        This method handles the core configuration flow:
        1. Sets search type configuration
        2. Initializes language model
        3. Processes human input and feedback
        4. Extracts task-specific configurations
        5. Sets training parameters
        
        The order of operations is important as later configurations may 
        depend on earlier ones (e.g., tools extraction depends on task type).
        
        Note:
            - All LLM interactions are logged
            - Configuration values are validated as they're set
            - Default values are used when appropriate
        """
        # Initialize search configuration
        self._set_search_type_config()

        # Store original human input
        if self.original_raw_input is None:
            self.original_raw_input = self.raw_input

        # Setup language model for configuration
        tmp_lm = self._setup_model_config()
        logger.info(f"Using {self.model_provider} model: {self.model_name}")

        # Process and extract configurations in order
        self.raw_input = self._process_human_feedback(tmp_lm)
        logger.info("Human input processed")

        # Develop prompt template components form raw input
        self._develop_prompt_template_components(tmp_lm)

        # if not self.raw_input_improvised:
        #    self.raw_input_improvised = self._improvise_raw_input(tmp_lm)
        #    logger.info("Human input improvised")

        if not self.task_description:
            self.task_description = self._extract_task_description(tmp_lm)
            logger.info(f"Task description extracted: {self.task_description[:100]}...")

        if not self.sample_data:
            self.sample_data = self._extract_sample_data(tmp_lm)
            logger.info(f"Sample data extracted: {self.sample_data[:100]}...")

        if not self.task_type:
            self.task_type = self._extract_task_type(tmp_lm)
            logger.info(f"Task type determined: {self.task_type}")

        if not self.input_fields or not self.output_fields:
            self.input_fields, self.output_fields = self._extract_fields(tmp_lm)
            logger.info(
                f"Fields extracted - Input: {self.input_fields}, "
                f"Output: {self.output_fields}"
            )

        # Extract tools only for relevant task types
        self.tools = self._extract_tools(tmp_lm)
        if self.tools:
            logger.info(f"Tools extracted for {self.task_type} task")

        # Set training configuration
        self.trainer = self._set_trainer()
        logger.info(f"Using trainer: {self.trainer}")

        self.dspy_module = self._set_dspy_module(tmp_lm)
        logger.info(f"Selected DSPy module: {self.dspy_module.__name__}")

        # Load user-provided data if available
        self._load_user_provided_data()
        self._resolve_image_pool(tmp_lm=tmp_lm)
        if self.image_pool:
            logger.info(f"Image pool prepared with {len(self.image_pool)} items")

        # Set data configuration with defaults if needed (only if data not provided)
        if not self.train_data and not self.valid_data:
            self.synthetic_data_size = self.synthetic_data_size or DEFAULT_SYNTHETIC_DATA_SIZE
            logger.info(f"Synthetic data size: {self.synthetic_data_size}")

            self.train_ratio = self.train_ratio or DEFAULT_TRAIN_RATIO
            logger.info(f"Train ratio: {self.train_ratio}")
            
            # Splitting the datasets
            self._calculate_dataset_sizes()
        else:
            logger.info("Using user-provided data, skipping synthetic data generation")

        # At the end, after all LLM calls:
        self.llm_cost = sum([x['cost'] for x in getattr(tmp_lm, 'history', []) if x.get('cost') is not None])

        # Cleanup
        del tmp_lm
        logger.info("Configuration population completed successfully")

    def _create_task_description(self, tmp_lm):
        """Create task description from dataset."""
        prompt = generate_task_description_from_sample_data(self.sample_data)
        response = tmp_lm(prompt)[0]

        if 'task description' in response.lower():
            return response.lower().split('task description')[1].strip()
        else:
            return response

    def _resolve_image_pool(self, tmp_lm=None) -> None:
        """Prepare the image pool for multimodal/VQA tasks.

        If the user provided an image_pool, keep it and only normalize/deduplicate it.
        If image_pool is empty and the task is VQA, seed it from existing examples and
        then auto-populate additional public image URLs based on task keywords.
        """
        self.image_pool = self._normalize_image_pool(self.image_pool)

        if str(self.task_type).lower() != 'vqa':
            return

        if self.image_pool:
            return

        seed_images = self._collect_image_references()
        if seed_images:
            self.image_pool.extend(seed_images)

        if not self.auto_fetch_image_pool:
            self.image_pool = self._deduplicate_list(self.image_pool)
            return

        remaining = max(0, int(self.image_pool_target_size) - len(self.image_pool))
        if remaining <= 0:
            self.image_pool = self._deduplicate_list(self.image_pool)
            return

        query = self._build_image_pool_query(tmp_lm=tmp_lm)
        if not query:
            self.image_pool = self._deduplicate_list(self.image_pool)
            return

        fetched_images = self._fetch_public_image_candidates(query=query, limit=remaining)
        self.image_pool.extend(fetched_images)
        self.image_pool = self._deduplicate_list(self.image_pool)

    def _normalize_image_pool(self, image_pool: Any) -> List[str]:
        """Normalize image_pool into a flat list of image references."""
        if image_pool is None:
            return []

        if isinstance(image_pool, (str, Path)):
            return [str(image_pool)]

        normalized = []
        if isinstance(image_pool, list):
            for item in image_pool:
                if isinstance(item, dict):
                    for key in ['image', 'image_url', 'image_path', 'url', 'path']:
                        value = item.get(key)
                        if value:
                            normalized.append(str(value))
                            break
                elif item:
                    normalized.append(str(item))

        return self._deduplicate_list(normalized)

    def _collect_image_references(self) -> List[str]:
        """Collect image references from sample, train, and validation data."""
        image_refs = []
        for source in [self.sample_data, self.train_data, self.valid_data, self.valid_data_full]:
            image_refs.extend(self._extract_image_refs_from_source(source))
        return self._deduplicate_list(image_refs)

    def _extract_image_refs_from_source(self, source: Any) -> List[str]:
        """Extract image references from one data source."""
        refs = []
        if source is None:
            return refs

        items = source
        if isinstance(source, str):
            try:
                items = json.loads(source)
            except Exception:
                try:
                    items = ast.literal_eval(source)
                except Exception:
                    return refs

        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return refs

        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ['image', 'image_url', 'image_path']:
                value = item.get(key)
                if isinstance(value, list):
                    refs.extend(str(v) for v in value if v)
                elif value:
                    refs.append(str(value))
        return refs

    def _build_image_pool_query(self, tmp_lm=None) -> str:
        """Build a concise image-search query, optionally using an LM."""
        if self._image_pool_query_cache:
            return self._image_pool_query_cache

        if self.image_pool_query:
            self._image_pool_query_cache = self.image_pool_query.strip()
            return self._image_pool_query_cache

        hint_phrases = []
        for source in [self.sample_data, self.train_data]:
            hint_phrases.extend(self._collect_image_search_hints_from_source(source))

        concise_phrases = []
        for phrase in hint_phrases:
            normalized = self._normalize_image_search_phrase(phrase)
            if normalized:
                concise_phrases.append(normalized)

        deterministic_hint_query = " ".join(self._deduplicate_list(concise_phrases)[:3]).strip()

        lm_query = ""
        if self.image_pool_use_lm_query:
            lm_query = self._generate_image_pool_query_with_lm(
                tmp_lm=tmp_lm,
                hint_phrases=hint_phrases,
            )
            lm_query = self._normalize_image_search_phrase(lm_query)

        query_parts = []
        if lm_query:
            query_parts.append(lm_query)
        elif deterministic_hint_query:
            query_parts.append(deterministic_hint_query)

        fallback_text = " ".join(
            part for part in [
                self.raw_input,
                getattr(self, 'question', None),
                self.context,
            ] if part
        )
        fallback_tokens = self._extract_keywords(fallback_text)[:4]
        query_parts.extend(token for token in fallback_tokens if token not in query_parts)

        if self.image_pool_site_filters and 'serpapi' in self.image_pool_sources:
            query_parts.extend([f"site:{domain}" for domain in self._deduplicate_list(self.image_pool_site_filters)])

        self._image_pool_query_cache = " ".join(self._deduplicate_list(query_parts)[:8])
        return self._image_pool_query_cache

    def _generate_image_pool_query_with_lm(self, tmp_lm=None, hint_phrases: Optional[List[str]] = None) -> str:
        """Use the configured LM to compress task context into a short image-search query."""
        hints = self._deduplicate_list((hint_phrases or []))[:6]
        question_hints = []
        for source in [self.sample_data, self.train_data]:
            question_hints.extend(self._extract_text_fields_from_source(source, ['question', 'prompt', 'query']))

        prompt = f"""Generate one concise web image search query.

Task:
{self.raw_input or ''}

Task description:
{self.task_description or ''}

Sample questions:
{json.dumps(question_hints[:3], ensure_ascii=True)}

High-signal labels or answers:
{json.dumps(hints, ensure_ascii=True)}

Rules:
- Return only the query text
- Use 2 to 6 words
- Prefer the most specific visual entity or class
- No punctuation
- No explanations
- No full sentences
"""

        lm = tmp_lm
        created_tmp_lm = False
        try:
            if lm is None:
                lm = self._setup_model_config()
                created_tmp_lm = True
            response = lm(prompt)[0].strip()
            log_llm_interaction(prompt, response, "image_pool_query_generation")
            return response
        except Exception as exc:
            logger.warning(f"LM image query generation failed, falling back to deterministic query: {exc}")
            return ""
        finally:
            if created_tmp_lm:
                try:
                    del lm
                except Exception:
                    pass

    def _normalize_image_search_phrase(self, text: str) -> str:
        """Trim a free-form field down to a short phrase suitable for image search."""
        if not isinstance(text, str):
            return ""

        cleaned = re.sub(r"\s+", " ", text).strip(" ,.;:!?")
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        boilerplate_fragments = [
            'you are', 'your task', 'tasked with', 'objective', 'analyze',
            'analysis system', 'assistant', 'question', 'answer', 'identify',
            'depicted', 'shown', 'image', 'picture', 'photo'
        ]
        for fragment in boilerplate_fragments:
            lowered = lowered.replace(fragment, " ")

        lowered = re.sub(r"\s+", " ", lowered).strip(" ,.;:!?")
        if not lowered:
            return ""

        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", lowered)
        if not tokens:
            return ""

        stopwords = {
            'a', 'an', 'and', 'are', 'be', 'depicted', 'for', 'from', 'image',
            'images', 'in', 'is', 'kind', 'object', 'of', 'on', 'photo',
            'picture', 'shown', 'species', 'that', 'the', 'this', 'to', 'what'
        }
        filtered = [token for token in tokens if token not in stopwords]
        if not filtered:
            filtered = tokens

        return " ".join(filtered[:4]).strip()

    def _collect_image_search_hints_from_source(self, source: Any) -> List[str]:
        """Extract label-like fields that improve image search specificity."""
        hint_fields = [
            'answer', 'answers', 'label', 'labels', 'class_name', 'category',
            'title', 'object', 'species', 'caption', 'description'
        ]
        values = []
        if source is None:
            return values

        items = source
        if isinstance(source, str):
            try:
                items = json.loads(source)
            except Exception:
                try:
                    items = ast.literal_eval(source)
                except Exception:
                    return values

        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return values

        for item in items:
            if not isinstance(item, dict):
                continue
            for field_name in hint_fields:
                value = item.get(field_name)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
                elif isinstance(value, dict):
                    for nested_key in ['text', 'label', 'answer', 'value']:
                        nested_value = value.get(nested_key)
                        if isinstance(nested_value, str) and nested_value.strip():
                            values.append(nested_value.strip())
                elif isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, str) and entry.strip():
                            values.append(entry.strip())
                        elif isinstance(entry, dict):
                            for nested_key in ['text', 'label', 'answer', 'value']:
                                nested_value = entry.get(nested_key)
                                if isinstance(nested_value, str) and nested_value.strip():
                                    values.append(nested_value.strip())
        return values

    def _extract_text_fields_from_source(self, source: Any, field_names: List[str]) -> List[str]:
        """Extract text fields such as question/prompt from a dataset-like source."""
        values = []
        if source is None:
            return values

        items = source
        if isinstance(source, str):
            try:
                items = json.loads(source)
            except Exception:
                try:
                    items = ast.literal_eval(source)
                except Exception:
                    return values

        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return values

        for item in items:
            if not isinstance(item, dict):
                continue
            for field_name in field_names:
                value = item.get(field_name)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
        return values

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract a small set of useful task keywords for public image retrieval."""
        stopwords = {
            'a', 'an', 'and', 'answer', 'answers', 'are', 'bird', 'by', 'for', 'from',
            'given', 'identify', 'image', 'images', 'in', 'is', 'kind', 'of', 'on',
            'picture', 'question', 'species', 'task', 'tasked', 'objective', 'primary',
            'developing', 'analysis', 'system', 'designed', 'depicted', 'your', 'the',
            'this', 'to', 'visual', 'what', 'with'
        }
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text.lower())
        keywords = [token for token in tokens if token not in stopwords]
        return self._deduplicate_list(keywords)

    def _fetch_public_image_candidates(self, query: str, limit: int) -> List[str]:
        """Fetch public image URLs from configured providers and validate them."""
        candidates = []

        for source_name in self.image_pool_sources:
            if len(candidates) >= limit:
                break

            if source_name == 'serpapi':
                candidates.extend(
                    self._fetch_serpapi_image_candidates(
                        query=query,
                        limit=limit - len(candidates),
                    )
                )
            elif source_name == 'google_cse':
                candidates.extend(
                    self._fetch_google_cse_image_candidates(
                        query=query,
                        limit=limit - len(candidates),
                    )
                )

        return self._resolve_existing_image_urls(candidates, limit=limit)

    def _fetch_serpapi_image_candidates(self, query: str, limit: int) -> List[str]:
        """Fetch image candidates from SerpApi using the seed image first, then text search."""
        if limit <= 0:
            return []

        if not self.serpapi_api_key:
            logger.warning("SerpApi image retrieval requested but SERPAPI_API_KEY is not configured.")
            return []

        candidates = []
        seed_images = self._collect_image_references()

        if seed_images:
            try:
                candidates.extend(
                    fetch_similar_image_urls_from_google_lens(
                        image_ref=seed_images[0],
                        api_key=self.serpapi_api_key,
                        limit=max(limit * 2, limit),
                        hl=self.serpapi_hl,
                        gl=self.serpapi_gl,
                    )
                )
            except Exception as exc:
                logger.warning(f"SerpApi Google Lens image retrieval failed: {exc}")

        if len(candidates) < limit:
            try:
                candidates.extend(
                    fetch_google_image_search_urls(
                        query=query,
                        api_key=self.serpapi_api_key,
                        limit=max((limit - len(candidates)) * 2, limit - len(candidates)),
                        hl=self.serpapi_hl,
                        gl=self.serpapi_gl,
                    )
                )
            except Exception as exc:
                logger.warning(f"SerpApi Google Images retrieval failed: {exc}")

        return self._deduplicate_list(candidates)[:max(limit * 3, limit)]

    def _fetch_google_cse_image_candidates(self, query: str, limit: int) -> List[str]:
        """Fetch image result URLs from Google's Custom Search JSON API."""
        if limit <= 0:
            return []

        if not self.google_custom_search_api_key or not self.google_custom_search_cx:
            logger.warning(
                "Google image retrieval requested but GOOGLE_CUSTOM_SEARCH_API_KEY/GOOGLE_API_KEY "
                "or GOOGLE_CUSTOM_SEARCH_CX/GOOGLE_CSE_ID is not configured."
            )
            return []

        endpoint = "https://www.googleapis.com/customsearch/v1"
        start = 1
        collected = []

        while len(collected) < limit and start <= 91:
            num = min(10, limit - len(collected))
            params = {
                "key": self.google_custom_search_api_key,
                "cx": self.google_custom_search_cx,
                "q": query,
                "searchType": "image",
                "num": num,
                "start": start,
                "safe": "active",
                "imgType": "photo",
            }

            try:
                response = requests.get(endpoint, params=params, timeout=15)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.warning(f"Google Custom Search image retrieval failed: {exc}")
                break

            items = payload.get("items", []) or []
            if not items:
                break

            for item in items:
                link = item.get("link")
                mime = (item.get("mime") or "").lower()
                if link and (not mime or mime.startswith("image/")):
                    collected.append(link)
                image_meta = item.get("image") or {}
                thumbnail_link = image_meta.get("thumbnailLink")
                if thumbnail_link:
                    collected.append(thumbnail_link)
                if len(collected) >= max(limit * 3, limit):
                    break

            next_page = payload.get("queries", {}).get("nextPage", [])
            if not next_page:
                break
            start = next_page[0].get("startIndex", start + num)

        return self._deduplicate_list(collected)[:max(limit * 3, limit)]

    def _resolve_existing_image_urls(self, candidates: List[str], limit: int) -> List[str]:
        """Keep only candidate URLs that resolve to readable image resources."""
        resolved = []
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; promptomatix/0.1)",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }

        for candidate in self._deduplicate_list(candidates):
            if len(resolved) >= limit:
                break

            response = None
            try:
                response = requests.get(
                    candidate,
                    allow_redirects=True,
                    stream=True,
                    timeout=10,
                    headers=headers,
                )
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                final_url = response.url or candidate
                final_suffix = Path(final_url.split("?", 1)[0]).suffix.lower()
                if (
                    content_type.startswith("image/")
                    or final_suffix in {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
                ) and final_url not in resolved:
                    resolved.append(final_url)
            except Exception:
                continue
            finally:
                try:
                    response.close()
                except Exception:
                    pass

        return resolved

    def _deduplicate_list(self, values: List[str]) -> List[str]:
        """Deduplicate a list while preserving order."""
        seen = set()
        deduped = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    def _improvise_raw_input(self, tmp_lm):
        """Improvise the human input."""
        prompt = improvise_raw_input(self.raw_input)
        response = tmp_lm(prompt)[0]

        log_llm_interaction(prompt, response, "improvise_raw_input")
        return response

    def _develop_prompt_template_components(self, tmp_lm):
        """
        Develop prompt template components from raw input.
        
        Flexibly extracts components from raw_input regardless of their order.
        Components are marked with [COMPONENT_NAME] tags.
        
        Components (all optional):
        - [TASK]: Main task description
        - [FEW_SHOT_EXAMPLES]: Example input-output pairs
        - [CONTEXT]: Additional context or background
        - [QUESTION]: The actual question or prompt
        - [INSTRUCTIONS]: Instructions for the LLM
        - [RULES]: Rules for the LLM
        - [OUTPUT_FORMAT]: Output format for the LLM
        - [TOOLS]: Tools for the LLM
        
        If no components are marked with tags, the entire input is treated as the task.
        """
        # If raw_input is None or empty, initialize empty components
        if not self.raw_input:
            self.task = None
            self.few_shot_examples = None
            self.task_context = None
            self.question = None
            return
        

        # Define all possible components
        components = ['TASK', 'FEW_SHOT_EXAMPLES', 'CONTEXT', 'QUESTION', 'INSTRUCTIONS', 'RULES', 'OUTPUT_FORMAT', 'TOOLS']
        
        # Initialize component dictionary
        extracted = {comp: None for comp in components}
        
        # Check if any component markers exist in the input
        has_markers = any(f'[{comp}]' in self.raw_input for comp in components)
        
        if not has_markers:
            # If no markers found, treat entire input as task
            prompt = improvise_raw_input(self.raw_input)
            self.task = tmp_lm(prompt)[0]
            self.few_shot_examples = None
            self.task_context = None
            self.question = None
            return
        
        # Find all component positions
        positions = {}
        for comp in components:
            start = self.raw_input.find(f'[{comp}]')
            if start != -1:
                positions[comp] = start
        
        # If no valid positions found, return with None values
        if not positions:
            self.task = None
            self.few_shot_examples = None
            self.task_context = None
            self.question = None
            return
        
        # Sort components by their position in the text
        ordered_components = sorted(positions.items(), key=lambda x: x[1])
        
        # Extract content between components
        for i, (comp, pos) in enumerate(ordered_components):
            start = pos + len(comp) + 2  # +2 for the brackets
            
            # If this is the last component, extract until the end
            if i == len(ordered_components) - 1:
                content = self.raw_input[start:].strip()
            else:
                # Extract until the next component
                next_comp = ordered_components[i + 1][0]
                end = self.raw_input.find(f'[{next_comp}]')
                content = self.raw_input[start:end].strip()
            
            extracted[comp] = content if content else None
        
        # Store components as instance attributes
        self.task = extracted['TASK']
        self.few_shot_examples = extracted['FEW_SHOT_EXAMPLES']
        self.task_context = extracted['CONTEXT']
        self.question = extracted['QUESTION']   
        self.instructions = extracted['INSTRUCTIONS']
        self.rules = extracted['RULES']
        self.output_format = extracted['OUTPUT_FORMAT']
        self.tools = extracted['TOOLS']

        # Lets rephrase and improvise some aspets of the input  
        if self.task:
            prompt = improvise_raw_input_task(self.task)
            self.task = tmp_lm(prompt)[0]

        # self.few_shot_examples - keep as is
        # self.task_context - keep as is

        if self.question:
            prompt = improvise_raw_input(self.question)
            self.question = tmp_lm(prompt)[0]

        if self.instructions:
            prompt = improvise_raw_input(self.instructions)
            self.instructions = tmp_lm(prompt)[0]

        if self.rules:
            prompt = improvise_raw_input(self.rules)
            self.rules = tmp_lm(prompt)[0]

        # self.output_format - keep as is

        if self.tools:
            prompt = improvise_raw_input_tools(self.tools)
            self.tools = tmp_lm(prompt)[0]

        # Combine the above components to form the final task description

        if not self.task:
            self.task = "You are a helpful assistant."
        
        if self.task_context:
            self.task += f"\n\nTask context: {self.task_context}"

        if self.question:
            self.task += f"\n\nQuestion: {self.question}"

        if self.instructions:
            self.task += f"\n\nInstructions: {self.instructions}"

        if self.rules:
            self.task += f"\n\nRules: {self.rules}"

        if self.tools:
            self.task += f"\n\nAvailable functions: {self.tools}"

        if self.few_shot_examples:
            self.task += f"\n\nFew shot examples: {self.few_shot_examples}"

        if self.output_format:
            self.task += f"\n\nOutput format: {self.output_format}"

        # TODO: Add log here



    def _set_search_type_config(self):
        """Configure parameters based on search type.
        
        Sets data and optimizer related parameters according to the search strategy
        (quick, moderate, or heavy search).
        
        Raises:
            ValueError: If search_type is invalid.
        """
        search_configs = {
            'quick_search': {
                # DATA related
                'synthetic_data_size': 30,
                'train_ratio': 0.2,
                # OPTIMIZER related
                'miprov2_init_auto': None,
                'miprov2_init_num_candidates': 5,
                'miprov2_compile_max_bootstrapped_demos': 4,
                'miprov2_compile_max_labeled_demos': 2,
                'miprov2_compile_num_trials': 10,
                'miprov2_compile_minibatch_size': 1
            },
            'moderate_search': {
                # DATA related
                'synthetic_data_size': 100,
                'train_ratio': 0.2,
                # OPTIMIZER related
                'miprov2_init_auto': None,
                'miprov2_init_num_candidates': 10,
                'miprov2_compile_max_bootstrapped_demos': 0,
                'miprov2_compile_max_labeled_demos': 0,
                'miprov2_compile_num_trials': 10,
                'miprov2_compile_minibatch_size': 10
            },
            'heavy_search': {
                # DATA related
                'synthetic_data_size': 300,
                'train_ratio': 0.2,
                # OPTIMIZER related
                'miprov2_init_auto': None,
                'miprov2_init_num_candidates': 20,
                'miprov2_compile_max_bootstrapped_demos': 0,
                'miprov2_compile_max_labeled_demos': 0,
                'miprov2_compile_num_trials': 30,
                'miprov2_compile_minibatch_size': 10
            }
        }

        if self.search_type not in search_configs:
            raise ValueError(f"Invalid search type: {self.search_type}")
        
        # Update instance attributes with the configuration
        for key, value in search_configs[self.search_type].items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def _validate(self) -> None:
        """Validate all configuration parameters after population.
        
        This validation ensures that all required fields are present and correctly typed
        after running either _populate_config() or _populate_config_from_huggingface().
        
        Raises:
            ValueError: If any required parameters are missing or invalid
            TypeError: If parameters are of incorrect type
        """
        # Required fields with their expected types
        required_fields = {
            'task_description': str,
            'task_type': str,
            'model_name': str,
        }
        
        # Validate required fields
        for field, expected_type in required_fields.items():
            value = getattr(self, field)
            if value is None:
                raise ValueError(f"{field} must be provided")
            if not isinstance(value, expected_type):
                raise TypeError(f"{field} must be of type {expected_type.__name__}, got {type(value).__name__}")

        # Validate model_provider separately - allow both string and ModelProvider enum
        if self.model_provider is None:
            raise ValueError("model_provider must be provided")
        if not isinstance(self.model_provider, (str, ModelProvider)):
            raise TypeError("model_provider must be either a string or ModelProvider enum")
        
        # Convert string to ModelProvider enum if needed
        if isinstance(self.model_provider, str):
            try:
                self.model_provider = ModelProvider(self.model_provider.lower())
            except ValueError:
                valid_providers = [p.value for p in ModelProvider]
                raise ValueError(f"Invalid model provider. Must be one of: {valid_providers}")

        # Validate training parameters
        if self.train_ratio is not None:
            if not isinstance(self.train_ratio, (int, float)):
                raise TypeError("train_ratio must be a number")
            if not 0 < self.train_ratio < 1:
                raise ValueError("train_ratio must be between 0 and 1")
        
        if self.synthetic_data_size is not None:
            if not isinstance(self.synthetic_data_size, int):
                raise TypeError("synthetic_data_size must be an integer")
            if self.synthetic_data_size < 1:
                raise ValueError("synthetic_data_size must be positive")

        # Validate model parameters
        if not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        
        if not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        # Validate dataset configuration
        if self.load_data_local:
            if not self.local_train_data_path or not self.local_test_data_path:
                raise ValueError("Local data paths must be provided when load_data_local is True")
            if not Path(self.local_train_data_path).exists():
                raise FileNotFoundError(f"Training data file not found: {self.local_train_data_path}")
            if not Path(self.local_test_data_path).exists():
                raise FileNotFoundError(f"Test data file not found: {self.local_test_data_path}")
        else:
            if not self.huggingface_dataset_name:
                raise ValueError("HuggingFace dataset name must be provided when load_data_local is False")

        # Validate data splits
        if self.train_data is None and self.valid_data is None:
            raise ValueError("Either train_data or valid_data must be provided")
        
        # Validate task-specific configuration
        if self.task_type in AGENTIC_TASK_TYPES and self.tools is None:
            logger.warning(f"No tools specified for agentic task type: {self.task_type}")

        # Validate DSPy configuration
        if self.dspy_module not in DSPY_MODULE_MAP.values():
            raise ValueError(f"Invalid DSPy module: {self.dspy_module}")

        logger.info("Configuration validation completed successfully")

    def to_dict(self) -> Dict:
        """Convert config to a serializable dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value  # Convert Enum to string
            elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                config_dict[key] = value  # Keep basic types as is
            else:
                # For other types, convert to string representation
                config_dict[key] = str(value)
        return config_dict

    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from a JSON or YAML file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Config: Loaded configuration object
        """
        import json
        import yaml
        
        ext = filepath.split('.')[-1].lower()
        with open(filepath, 'r') as f:
            if ext == 'json':
                data = json.load(f)
            elif ext in ('yml', 'yaml'):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        return cls(**data)
    
    def save(self, filepath: str) -> None:
        """Save configuration to a file.
        
        Args:
            filepath: Path where to save the configuration
        """
        import json
        import yaml
        
        ext = filepath.split('.')[-1].lower()
        with open(filepath, 'w') as f:
            if ext == 'json':
                json.dump(self.to_dict(), f, indent=2)
            elif ext in ('yml', 'yaml'):
                yaml.dump(self.to_dict(), f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

    def _setup_model_config(self):
        """Set up model configuration based on provider.
        
        Returns:
            dspy.LM: Configured language model instance
        
        Raises:
            ValueError: If model provider is invalid
            EnvironmentError: If required API key is missing
        """
        PROVIDER_CONFIGS = {
            ModelProvider.OPENAI: {
                'api_base': 'https://api.openai.com/v1',
                'env_key': 'OPENAI_API_KEY',
                'default_model': 'gpt-4o'
            },
            ModelProvider.ANTHROPIC: {
                'api_base': 'https://api.anthropic.com/v1',
                'env_key': 'ANTHROPIC_API_KEY',
                'default_model': 'claude-2'
            },
            ModelProvider.DATABRICKS: {
                'api_base': 'https://api.databricks.com/v1',
                'env_key': 'DATABRICKS_API_KEY',
                'default_model': None
            },
            ModelProvider.LOCAL: {
                'api_base': 'http://127.0.0.1:8000/v1',
                'env_key': None,
                'default_model': None
            },
            ModelProvider.TOGETHERAI: {
                'api_base': 'https://api.together.xyz/',
                'env_key': 'TOGETHERAI_API_KEY',
                'default_model': 'together_ai/mistralai/Mistral-Small-24B-Instruct-2501'
            },
            ModelProvider.GEMINI: {
                'api_base': None,
                'env_key': 'GOOGLE_API_KEY',
                'default_model': 'gemini-1.5-flash'
            }
        }

        # Teacher
        PROVIDER_CONFIGS_FOR_CONFIG_MODEL = {
            ModelProvider.OPENAI: {
                'api_base': 'https://api.openai.com/v1',
                'env_key': 'OPENAI_API_KEY',
                'default_model': 'gpt-4o'
            },
            ModelProvider.ANTHROPIC: {
                'api_base': 'https://api.anthropic.com/v1',
                'env_key': 'ANTHROPIC_API_KEY',
                'default_model': 'claude-3-sonnet'
            },
            ModelProvider.DATABRICKS: {
                'api_base': 'https://api.databricks.com/v1',
                'env_key': 'DATABRICKS_API_KEY',
                'default_model': None
            },
            ModelProvider.LOCAL: {
                'api_base': 'http://127.0.0.1:8000/v1',
                'env_key': None,
                'default_model': None
            },
            ModelProvider.TOGETHERAI: {
                'api_base': 'https://api.together.xyz/',
                'env_key': 'TOGETHERAI_API_KEY',
                'default_model': 'together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo'
            },
            ModelProvider.GEMINI: {
                'api_base': None,
                'env_key': 'GOOGLE_API_KEY',
                'default_model': 'gemini-1.5-flash'
            }
        }

        # If no model name provided, set defaults
        if not self.model_name:
            if self.model_provider is None:
                self.model_provider = ModelProvider.OPENAI
                # self.model_provider = ModelProvider.TOGETHERAI
            self.model_name = PROVIDER_CONFIGS[ModelProvider.OPENAI]['default_model']
            # self.model_name = PROVIDER_CONFIGS[ModelProvider.TOGETHERAI]['default_model']

        if not self.config_model_name:
            if self.config_model_provider is None:
                self.config_model_provider = ModelProvider.OPENAI
                # self.config_model_provider = ModelProvider.TOGETHERAI
            self.config_model_name = PROVIDER_CONFIGS_FOR_CONFIG_MODEL[ModelProvider.OPENAI]['default_model']
            # self.config_model_name = PROVIDER_CONFIGS_FOR_CONFIG_MODEL[ModelProvider.TOGETHERAI]['default_model']
        
        # Convert string to enum if needed
        if isinstance(self.model_provider, str):
            # Handle modelprovider prefix
            if self.model_provider.startswith('modelprovider.'):
                self.model_provider = self.model_provider.split('.')[-1]
            try:
                self.model_provider = ModelProvider(self.model_provider.lower())
            except ValueError:
                valid_providers = [p.value for p in ModelProvider]
                raise ValueError(f"Invalid model provider. Must be one of: {valid_providers}")
            if not self.model_name:
                self.model_name = PROVIDER_CONFIGS[ModelProvider.OPENAI]['default_model']
                # self.model_name = PROVIDER_CONFIGS[ModelProvider.TOGETHERAI]['default_model']

        if isinstance(self.config_model_provider, str):
            # Handle modelprovider prefix
            if self.config_model_provider.startswith('modelprovider.'):
                self.config_model_provider = self.config_model_provider.split('.')[-1]
            try:
                self.config_model_provider = ModelProvider(self.config_model_provider.lower())
            except ValueError:
                valid_providers = [p.value for p in ModelProvider]
                raise ValueError(f"Invalid model provider. Must be one of: {valid_providers}")
            if not self.config_model_provider:
                self.config_model_provider = PROVIDER_CONFIGS_FOR_CONFIG_MODEL[ModelProvider.OPENAI]['default_model']
                # self.config_model_provider = PROVIDER_CONFIGS_FOR_CONFIG_MODEL[ModelProvider.TOGETHERAI]['default_model']

        # Get provider configuration
        try:
            provider_config = PROVIDER_CONFIGS[self.model_provider]
        except KeyError:
            valid_providers = [p.value for p in ModelProvider]
            raise ValueError(f"Unsupported model provider. Must be one of: {valid_providers}")
        
        try:
            config_provider_config = PROVIDER_CONFIGS_FOR_CONFIG_MODEL[self.config_model_provider]
        except KeyError:
            valid_providers = [p.value for p in ModelProvider]
            raise ValueError(f"Unsupported model provider. Must be one of: {valid_providers}")

        # Set API base
        if self.model_api_base is None:
            self.model_api_base = provider_config['api_base']
        if self.config_model_api_base is None:
            self.config_model_api_base = config_provider_config['api_base']

        # Set API key if required
        if provider_config['env_key']:
            if self.model_api_key is None:
                try:
                    self.model_api_key = os.environ[provider_config['env_key']]
                except KeyError:
                    raise EnvironmentError(
                        f"Missing {provider_config['env_key']} environment variable "
                        f"required for {self.model_provider.value}"
                    )
        else:
            if self.model_api_key is None:
                self.model_api_key = None

        if config_provider_config['env_key']:
            if self.config_model_api_key is None:
                try:
                    self.config_model_api_key = os.environ[config_provider_config['env_key']]
                except KeyError:
                    raise EnvironmentError(
                        f"Missing {config_provider_config['env_key']} environment variable "
                        f"required for {self.config_model_provider.value}"
                    )
        else:
            if self.config_model_api_key is None:
                self.config_model_api_key = None

        # Initialize language model
        try:
            tmp_lm = dspy.LM(
                self.config_model_name,
                api_key=self.config_model_api_key,
                api_base=self.config_model_api_base,
                max_tokens=self.config_max_tokens,
                temperature=self.config_temperature,
                cache=True
            )
            logger.info(f"Successfully initialized {self.config_model_provider.value} model: {self.config_model_name}")
            
            # Log model configuration (excluding sensitive data)
            log_llm_interaction(
                context="Model initialization",
                prompt="N/A",
                response=f"Initialized {self.config_model_provider.value} model: {self.config_model_name}"
            )
            
            return tmp_lm
            
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise RuntimeError(f"Failed to initialize language model: {str(e)}")

    def _extract_task_description(self, tmp_lm) -> str:
        """Extract or set task description from human input.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            str: Task description
        """
        # Use human input directly if decoupling is enabled
        if self.decouple_task_description_and_raw_input:
            return self.raw_input_improvised
        
        # Extract task description if not provided
        if self.task_description is None:
            prompt = extract_task_description_from_raw_input(self.task)
            response = tmp_lm(prompt)[0]
            
            # Log LLM interaction
            log_llm_interaction(prompt=prompt, response=response, context="Task description extraction")
            
            # Clean up response if needed
            if "Task description:" in response:
                response = response.split("Task description:")[1].strip()
            elif "Task Description:" in response:
                response = response.split("Task Description:")[1].strip()
            
            if 'none' not in response.lower():
                return response
            
        return self.task_description

    def _extract_sample_data(self, tmp_lm) -> str:
        """Extract sample data from task description and human input.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            str: Extracted sample data
        """
        if self.sample_data is not None:
            return self.sample_data
        

        
        
        if self.few_shot_examples is not None:
            # if self.few_shot_examples exists and can be loaded as JSON directly return it
            try:
                _ = json.loads(self.few_shot_examples)
                return self.few_shot_examples
            except Exception as e:
                logger.warning(f"Provided few shot examples are not valid JSON: {str(e)}")

            # If the few shot examples are not valid JSON, convert them to JSON
            prompt = convert_few_shot_examples_to_json(self.few_shot_examples)
            response = tmp_lm(prompt)[0]
            if "```json" in response:
                response = response.split("```json")[1].strip()
            if "```" in response:
                response = response.split("```")[0].strip()
            return response
        
        
        cxt_output = ""
        if self.task_context is not None or self.question is not None:
            # Use the context as the sample data so only generate the expected output 
            prompt = complete_the_main_example(
                
                self.task,
                self.question,
                self.task_context
            )
            cxt_output = tmp_lm(prompt)[0]
            if "```json" in cxt_output:
                cxt_output = cxt_output.split("```json")[1].strip()
            if "```" in cxt_output:
                cxt_output = cxt_output.split("```")[0].strip()

            complete_sample = f"INPUT:\n{self.task_context}\n\nOUTPUT:{cxt_output}\n\n"

            prompt = generate_sample_data_from_sample_data(
                self.task,
                complete_sample
            )

            response = tmp_lm(prompt)[0]
        
            # Log LLM interaction
            log_llm_interaction(
                prompt=prompt, 
                response=response, 
                context="Sample data extraction"
            )
            
        
        else:
            # if either self.question or self.context is not None, use it to generate sample data
            prompt = generate_sample_data_from_task_description_and_raw_input(
                self.task_description, 
                self.task
            )
            
            response = tmp_lm(prompt)[0]

            if "```json" in response:
                response = response.split("```json")[1].strip()
            if "```" in response:
                response = response.split("```")[0].strip()
            
            # Log LLM interaction
            log_llm_interaction(
                prompt=prompt, 
                response=response, 
                context="Sample data extraction"
            )

            prompt = get_expected_answer_from_sample_data(
                self.task,
                response
            )

            response = tmp_lm(prompt)[0]
            
            

        # Clean response by removing markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].strip()
        if "```" in response:
            response = response.split("```")[0].strip()
        
        return response

    def _extract_fields(self, tmp_lm) -> tuple[list, list]:
        """Extract input and output fields from task description and sample data.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            tuple[list, list]: Tuple of (input_fields, output_fields)
        """
        # Skip if both input and output fields are already defined
        if self.input_fields and self.output_fields:
            return self.input_fields, self.output_fields
        
        # Handle different sample_data formats
        try:
            if isinstance(self.sample_data, str):
                # If it's a string, try to parse it as JSON first, then as literal_eval
                try:
                    parsed_data = json.loads(self.sample_data)
                except json.JSONDecodeError:
                    parsed_data = ast.literal_eval(self.sample_data)
                
                # Handle both list and dict formats
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    allowed_fields = list(parsed_data[0].keys())
                elif isinstance(parsed_data, dict):
                    allowed_fields = list(parsed_data.keys())
                else:
                    allowed_fields = []
            elif isinstance(self.sample_data, list) and len(self.sample_data) > 0:
                # If it's already a list of dictionaries
                allowed_fields = list(self.sample_data[0].keys())
            elif isinstance(self.sample_data, dict):
                # If it's already a dictionary
                allowed_fields = list(self.sample_data.keys())
            else:
                allowed_fields = []
        except Exception as e:
            logger.warning(f"Failed to parse sample data: {str(e)}")
            allowed_fields = []

        # Add missing fields based on task type - use if statements instead of elif
        if self.task_type == 'summarization' and 'summary' not in allowed_fields:
            allowed_fields.append('summary')
        if self.task_type == 'classification' and 'label' not in allowed_fields:
            allowed_fields.append('label')
        if self.task_type == 'generation' and 'target' not in allowed_fields:
            allowed_fields.append('target')
        if (self.task_type == 'question-answering' or self.task_type == 'qa') and 'answer' not in allowed_fields:
            allowed_fields.append('answer')
        if self.task_type == 'translation' and 'translation' not in allowed_fields:
            allowed_fields.append('translation')
        if self.task_type == 'text-classification' and 'label' not in allowed_fields:
            allowed_fields.append('label')
        if self.task_type == 'text-generation' and 'text' not in allowed_fields:
            allowed_fields.append('text')

        prompt = extract_fields_from_sample_data(
            self.task_description, 
            self.sample_data,
            allowed_fields
        )
        
        response = tmp_lm(prompt)[0]
        
        # Log LLM interaction
        log_llm_interaction(
            prompt=prompt, 
            response=response, 
            context="Field extraction"
        )
        
        # Clean response
        if "```json" in response:
            response = response.split("```json")[1].strip()
        if "```" in response:
            response = response.split("```")[0].strip()
        
        if 'none' not in response.lower():
            try:
                fields = json.loads(response)
                return fields.get('input_fields', []), fields.get('output_fields', [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse fields JSON response")
                return [], []
            
        return [], []

    def _extract_task_type(self, tmp_lm) -> Optional[str]:
        """Extract task type from task description and sample data.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            Optional[str]: Extracted task type or None if not determined
        """
        if self.task_type is not None:
            return self.task_type

        prompt = extract_task_type_from_raw_input(
            self.task_description, 
            self.task, 
            self.sample_data
        )
        
        response = tmp_lm(prompt)[0]
        
        # Log LLM interaction
        log_llm_interaction(
            prompt=prompt, 
            response=response, 
            context="Task type extraction"
        )
        
        # Clean response
        if "```json" in response:
            response = response.split("```json")[1].strip()
        if "```" in response:
            response = response.split("```")[0].strip()
        
        if 'none' not in response.lower():
            try:
                # Validate against known task types
                task_type = response.lower().strip()
                if 'task type' in task_type and 'reasoning' in task_type:
                    return task_type.split('task type:')[1].split('reasoning:')[0].strip()
                elif 'task type' in task_type:
                    return task_type.split('task type:')[1].strip()
                elif 'reasoning' in task_type:
                    return task_type.split('reasoning:')[1].strip()

                logger.warning(f"Unknown task type: {task_type}")
            except Exception as e:
                logger.warning(f"Failed to process task type: {str(e)}")
            
        return None

    def _extract_tools(self, tmp_lm) -> Optional[str]:
        """Extract tools for agentic/programming tasks.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            Optional[str]: Extracted tools configuration or None
        """
        # Skip if not an agentic task type
        if self.task_type not in AGENTIC_TASK_TYPES:
            return None
        
        # Return existing tools if already defined
        if self.tools is not None:
            return self.tools

        prompt = extract_tools_from_raw_input(self.raw_input)
        response = tmp_lm(prompt)[0]
        
        # Log LLM interaction
        log_llm_interaction(
            prompt=prompt, 
            response=response, 
            context="Tools extraction"
        )
        
        # Clean response
        if "```json" in response:
            response = response.split("```json")[1].strip()
        if "```" in response:
            response = response.split("```")[0].strip()
        
        if 'none' not in response.lower():
            return response
        
        return None

    def _process_human_feedback(self, tmp_lm) -> str:
        """Process and simplify human feedback if present in input.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            str: Processed human input
        """
        if not self.raw_input or 'feedback' not in self.raw_input.lower():
            return self.raw_input

        prompt = simplify_human_feedback_2(self.raw_input)
        response = tmp_lm(prompt)[0]
        
        # Log LLM interaction
        log_llm_interaction(
            prompt=prompt,
            response=response,
            context="Processing human feedback"
        )
        
        logger.info("Human feedback processed and simplified")
        return response

    def _set_dspy_module(self, tmp_lm) -> Type[dspy.Module]:
        """Set appropriate DSPy module based on task description and sample data.
        
        Args:
            tmp_lm: Language model instance
        Returns:
            Type[dspy.Module]: DSPy module class
        """
        # testing `dspy.Predict`
        # TODO: remove this
        return dspy.Predict
    
        if self.dspy_module:
            return self.dspy_module

        prompt = generate_dspy_module_from_task_description_and_sample_data(
            self.task_description, 
            self.sample_data
        )
        
        response = tmp_lm(prompt)[0]
        
        # Log LLM interaction
        log_llm_interaction(
            prompt=prompt,
            response=response,
            context="DSPy module selection"
        )

        # If the response is 'react' and tools are not defined, set the module to Predict
        if 'react' in response.lower().strip() and not self.tools:
            response = 'dspy.Predict'
        
        try:
            module_type = DSPyModules(response.strip())
            return DSPY_MODULE_MAP[module_type]
        except (KeyError, ValueError):
            logger.warning(f"Invalid DSPy module '{response}', defaulting to Predict")
            return dspy.Predict

    def _set_trainer(self):
        """Set trainer based on task type."""
        return 'MIPROv2'

    def _load_and_process_dataset(self) -> tuple[Dataset, Dataset]:
        """Load and preprocess dataset from local files or HuggingFace.
        
        Returns:
            tuple[Dataset, Dataset]: Processed (train_dataset, test_dataset)
        """
        train_dataset = None
        test_dataset = None

        if self.load_data_local:
            train_dataset, test_dataset = self._load_local_datasets()
        else:
            train_dataset, test_dataset = self._load_huggingface_datasets()

        # Apply dataset-specific configurations and preprocessing
        train_dataset, test_dataset = self._apply_dataset_config(train_dataset, test_dataset)
        
        # Apply common preprocessing steps
        train_dataset, test_dataset = self._apply_common_preprocessing(train_dataset, test_dataset)
        
        return train_dataset, test_dataset

    def _load_local_datasets(self) -> tuple[Dataset, Dataset]:
        """Load datasets from local CSV files."""
        import pandas as pd
        
        train_df = pd.read_csv(self.local_train_data_path)
        test_df = pd.read_csv(self.local_test_data_path)
        
        return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

    def _load_huggingface_datasets(self) -> tuple[Dataset, Dataset]:
        """Load datasets from HuggingFace."""
        options = [("main",), (), ("generation",)]
        dataset = None
        
        for option in options:
            try:
                dataset = load_dataset(self.huggingface_dataset_name, *option)
                logger.info(f"Successfully loaded dataset: {self.huggingface_dataset_name}")
                break
            except Exception as e:
                logger.debug(f"Failed to load with option {option}: {str(e)}")
                continue
        
        if dataset is None:
            raise ValueError(f"Failed to load dataset: {self.huggingface_dataset_name}")
        
        train_dataset = dataset.get('train')
        test_dataset = dataset.get('test') or dataset.get('validation')
        
        return train_dataset, test_dataset

    def _apply_dataset_config(self, train_dataset: Dataset, test_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Apply dataset-specific configurations and preprocessing."""
        dataset_name = self.huggingface_dataset_name.lower()
        
        if 'xsum' in dataset_name:
            config = DatasetConfig.XSUM
        elif 'common_gen' in dataset_name:
            config = DatasetConfig.COMMON_GEN
        elif 'ag_news' in dataset_name:
            config = DatasetConfig.AG_NEWS
            # Apply label mapping for AG News
            if train_dataset is not None:
                train_dataset = train_dataset.map(
                    lambda x: {'label': config['label_map'][x['label']]}
                )
            if test_dataset is not None:
                test_dataset = test_dataset.map(
                    lambda x: {'label': config['label_map'][x['label']]}
                )
        elif 'squad_v2' in dataset_name:
            config = DatasetConfig.SQUAD_V2
            train_dataset, test_dataset = self._process_squad_dataset(train_dataset, test_dataset)

        elif 'gsm8k' in dataset_name:
            config = DatasetConfig.GSM8K
            train_dataset, test_dataset = self._process_gsm8k_dataset(train_dataset, test_dataset)
        
        # Apply configuration
        self.input_fields = config['input_fields']
        self.output_fields = config['output_fields']
        self.task_type = config['task_type']
        
        return train_dataset, test_dataset
    
    def _process_gsm8k_dataset(self, train_dataset: Dataset, test_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Special processing for GSM8K dataset to handle math reasoning steps.
        
        Args:
            train_dataset: HuggingFace Dataset for training
            test_dataset: HuggingFace Dataset for testing/validation
            
        Returns:
            tuple[Dataset, Dataset]: Processed (train_dataset, test_dataset)
        """
        # Set input and output fields
        self.input_fields = ['question']
        self.output_fields = ['answer']
        self.task_type = 'qa'  # Set task type to question-answering
        
        def process_answer(example):
            """Keep the full answer including reasoning steps."""
            try:
                # GSM8K answers are formatted as:
                # reasoning steps
                # ####
                # final_answer
                if '####' in example['answer']:
                    # Keep both reasoning and answer
                    return example
                return {
                    'question': example['question'],
                    'answer': example['answer']
                }
            except Exception:
                return {
                    'question': example['question'],
                    'answer': ''
                }
        
        # Apply processing to both datasets if they exist
        if train_dataset is not None:
            train_dataset = train_dataset.map(
                process_answer,
                desc="Processing GSM8K training dataset"
            )
            train_dataset = train_dataset.filter(lambda x: x['answer'] != '')
        
        if test_dataset is not None:
            test_dataset = test_dataset.map(
                process_answer,
                desc="Processing GSM8K test dataset"
            )
            test_dataset = test_dataset.filter(lambda x: x['answer'] != '')
        
        return train_dataset, test_dataset
    
    def _process_squad_dataset(self, train_dataset: Dataset, test_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Special processing for SQuAD dataset."""
        def transform_example(example):
            try:
                answer = (example['answers']['text'][0] if isinstance(example['answers'], dict) 
                         else example['answers'][0]) if example['answers'] else ""
                return {
                    'question': example['question'],
                    'context': example['context'],
                    'answers': answer
                }
            except (KeyError, IndexError):
                return {'question': "", 'context': "", 'answers': ""}

        def is_valid_example(example):
            return (example['question'] and 
                    example['context'] and 
                    isinstance(example['question'], str) and 
                    isinstance(example['context'], str))

        for dataset in (train_dataset, test_dataset):
            if dataset is not None:
                dataset = dataset.map(
                    transform_example,
                    remove_columns=dataset.column_names,
                    desc="Transforming dataset"
                )
                dataset = dataset.filter(is_valid_example)
        
        return train_dataset, test_dataset

    def _apply_common_preprocessing(self, train_dataset: Dataset, test_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Apply common preprocessing steps to both datasets."""
        for dataset in (train_dataset, test_dataset):
            if dataset is not None:
                # Select relevant columns
                if self.input_fields and self.output_fields:
                    dataset = dataset.select_columns(self.input_fields + self.output_fields)
                # Shuffle dataset
                dataset = dataset.shuffle(seed=42)
        
        return train_dataset, test_dataset

    def _prepare_datasets(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        """Prepare sample, training, and validation datasets.
        
        This method handles:
        1. Creating sample data from available datasets
        2. Calculating dataset sizes
        3. Preparing training and validation splits
        4. Converting datasets to list of dictionaries format
        
        Args:
            train_dataset: HuggingFace Dataset for training
            test_dataset: HuggingFace Dataset for testing/validation
        """
        # Create sample data
        self._create_sample_data(train_dataset, test_dataset)
        
        # Calculate dataset sizes
        self._calculate_dataset_sizes()
        
        # Prepare training and validation datasets
        self._prepare_train_valid_splits(train_dataset, test_dataset)

    def _create_sample_data(self, train_dataset: Optional[Dataset], test_dataset: Optional[Dataset]) -> None:
        """Create sample data from available datasets.
        
        Args:
            train_dataset: HuggingFace Dataset for training
            test_dataset: HuggingFace Dataset for testing/validation
        """
        SAMPLE_SIZE = 3
        
        def dataset_to_dict_list(dataset, size: int) -> List[Dict]:
            """Convert dataset slice to list of dictionaries."""
            if dataset is None:
                return None
            sample = dataset[:size]
            return [dict(zip(sample.keys(), values)) for values in zip(*sample.values())]
        
        # Try to get samples from train dataset first, then test dataset
        self.sample_data = (
            dataset_to_dict_list(train_dataset, SAMPLE_SIZE) or 
            dataset_to_dict_list(test_dataset, SAMPLE_SIZE)
        )
        
        logger.debug(f"Created sample data with {len(self.sample_data) if self.sample_data else 0} examples")

    def _calculate_dataset_sizes(self) -> None:
        """Calculate sizes for training and validation datasets."""
        self.train_data_size = int(self.synthetic_data_size * self.train_ratio)
        self.valid_data_size = self.synthetic_data_size - self.train_data_size
        
        logger.debug(f"Calculated dataset sizes - Train: {self.train_data_size}, Valid: {self.valid_data_size}")

    def _prepare_train_valid_splits(self, train_dataset: Optional[Dataset], test_dataset: Optional[Dataset]) -> None:
        """Prepare training and validation dataset splits.
        
        Args:
            train_dataset: HuggingFace Dataset for training
            test_dataset: HuggingFace Dataset for testing/validation
        """
        def dataset_slice_to_dict_list(dataset, start: int, size: int) -> List[Dict]:
            """Convert dataset slice to list of dictionaries."""
            if dataset is None:
                return None
            if start+size == 0:
                data_slice = dataset[start:]
            else:
                data_slice = dataset[start:start + size]
            return [dict(zip(data_slice.keys(), values)) for values in zip(*data_slice.values())]
        
        # Prepare training data
        self.train_data = dataset_slice_to_dict_list(train_dataset, 0, self.train_data_size)
        
        # Prepare validation data
        if test_dataset is not None:
            # Regular validation set
            self.valid_data = dataset_slice_to_dict_list(
                test_dataset, 
                -self.valid_data_size, 
                self.valid_data_size
            )
            
            # Full validation set (first 100 examples)
            FULL_VALID_SIZE = 100
            self.valid_data_full = dataset_slice_to_dict_list(
                test_dataset,
                0,
                FULL_VALID_SIZE
            )
        
        logger.debug(
            f"Prepared datasets - "
            f"Train: {len(self.train_data) if self.train_data else 0}, "
            f"Valid: {len(self.valid_data) if self.valid_data else 0}, "
            f"Valid Full: {len(self.valid_data_full) if self.valid_data_full else 0}"
        )

    def _load_user_provided_data(self):
        """Load user-provided training and validation data from direct input or files."""
        # If data is already provided directly, keep it as is
        if self.train_data or self.valid_data:
            logger.info("Using directly provided train_data/valid_data")
            return
        
        # If local data paths are provided, load from files
        if self.local_train_data_path:
            logger.info(f"Loading training data from: {self.local_train_data_path}")
            
            import pandas as pd
            
            # Only calculate dataset sizes if not explicitly provided
            if not self.train_data_size or not self.valid_data_size:
                self._calculate_dataset_sizes()
            
            # Load training data with size restriction
            train_df = pd.read_csv(self.local_train_data_path)
            if self.train_data_size and len(train_df) > self.train_data_size:
                train_df = train_df.head(self.train_data_size)
                logger.info(f"Restricted training data to {self.train_data_size} samples")
            
            self.train_data = train_df.to_dict('records')
            logger.info(f"Loaded {len(self.train_data)} training samples")
            
            # Load test/validation data if path provided
            if self.local_test_data_path:
                logger.info(f"Loading test data from: {self.local_test_data_path}")
                test_df = pd.read_csv(self.local_test_data_path)
                if self.valid_data_size and len(test_df) > self.valid_data_size:
                    test_df = test_df.head(self.valid_data_size)
                    logger.info(f"Restricted validation data to {self.valid_data_size} samples")
                
                self.valid_data = test_df.to_dict('records') 
                logger.info(f"Loaded {len(self.valid_data)} validation samples")

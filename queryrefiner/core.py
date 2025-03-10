# queryrefiner/core.py

"""Core module containing the QueryRefiner class for query refinement and code generation."""

import json
import re
import torch
from transformers import LogitsProcessor, LogitsProcessorList
import logging
from .utils import validate_syntax
from .models import load_models

class QueryRefiner:
    """A class to refine natural language queries and generate code using transformer models.

    This class integrates CodeBERT for constraint detection, CodeT5p for query restructuring,
    and a code generation model (e.g., CodeLlama) with logits bias to restrict dangerous APIs.

    Args:
        code_model (str): The model name for code generation (default: "codellama/CodeLlama-7b-hf").
        device (int): The device ID for computation (-1 for CPU, 0+ for GPU, default: 0).
        dangerous_tokens (list): List of dangerous tokens to restrict (default: ["eval", "exec", ...]).
        log_level (int): Logging level (default: logging.INFO).
    """

    def __init__(self, code_model="codellama/CodeLlama-7b-hf", device=0, dangerous_tokens=None, log_level=logging.INFO):
        """Initialize the QueryRefiner with models and configurations."""
        # Configure logging
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)

        # Load models
        self.logger.info("Loading models...")
        self.tokenizer_bert, self.model_bert, self.tokenizer_t5, self.model_t5, self.generator = load_models(code_model, device)

        # Initialize dangerous token constraints
        self.dangerous_tokens = dangerous_tokens or ["eval", "exec", "os.system", "subprocess.call", "subprocess.Popen"]
        self.logits_processor = LogitsProcessorList([self._get_dangerous_token_bias()])

    def _get_dangerous_token_bias(self):
        """Create a LogitsProcessor to bias against dangerous tokens."""
        return DangerousTokenBias(self.generator.tokenizer, self.dangerous_tokens)

    def analyze_query(self, query):
        """Analyze the input query and extract task, condition, and property.

        Args:
            query (str): The natural language query to analyze.

        Returns:
            dict: A dictionary containing 'task', 'condition', and 'property'.
        """
        self.logger.debug(f"Analyzing query: {query}")
        condition_pattern = r'([><]=?\s*\d+)|(\(\d+,\s*\d+\))|(between\s+\d+\s+and\s+\d+)'
        condition_match = re.search(condition_pattern, query)
        condition = condition_match.group(0).strip() if condition_match else ""

        task_pattern = r'(generate|create)\s+a\s+function'
        task_match = re.search(task_pattern, query)
        task = task_match.group(0) if task_match else "generate function"

        property_part = query.replace(task, "").replace(condition, "").strip()
        property_value = re.search(r"with\s+(\w+\s+digits)", property_part)
        property_value = property_value.group(1) if property_value else "unknown property"

        return {"task": task, "condition": condition, "property": property_value}

    def restructure_query(self, query):
        """Restructure the query into a structured JSON format using CodeT5p.

        Args:
            query (str): The natural language query to restructure.

        Returns:
            dict: A JSON-compatible dictionary with 'task', 'condition', and 'property'.
        """
        initial_data = self.analyze_query(query)
        prompt = (
            "Convert the query into a JSON object with keys 'task', 'condition', and 'property'.\n"
            "Examples:\n"
            "{\"task\": \"generate function\", \"condition\": \"> 10\", \"property\": \"even digits\"}\n"
            "{\"task\": \"create function\", \"condition\": \"(2, 4)\", \"property\": \"unknown property\"}\n"
            f"Query: {query}\n"
            "Output only the JSON object:\n"
        )

        inputs = self.tokenizer_t5(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        self.logger.debug(f"Inputs: {inputs}")

        outputs = self.model_t5.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=50,
            num_beams=5,
            do_sample=False,
            temperature=0.3,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer_t5.pad_token_id
        )

        generated_text = self.tokenizer_t5.decode(outputs[0], skip_special_tokens=True).strip()

        try:
            structured_data = json.loads(generated_text)
            if not all(key in structured_data for key in ["task", "condition", "property"]):
                raise json.JSONDecodeError("Missing required keys", generated_text, 0)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}, falling back to initial data. Raw output: {generated_text}")
            structured_data = initial_data

        return structured_data

    def detect_constraints(self, query):
        """Detect constraints in the query using CodeBERT.

        Args:
            query (str): The natural language query to check.

        Returns:
            dict: A dictionary containing prediction class, logits, and detected issues.
        """
        inputs = self.tokenizer_bert(query, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model_bert(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        issues = []
        if predicted_class == 0:
            if ">" not in query and "<" not in query and "(" not in query:
                issues.append("Missing condition (e.g., '> 10' or '(2, 4)')")
            if "with" not in query:
                issues.append("Missing property (e.g., 'with even digits')")
        return {
            "predicted_class": predicted_class,
            "logits": logits.tolist(),
            "issues": issues
        }

    def generate_code(self, prompt, structured_query=None, max_attempts=3):
        """Generate code with constraints to avoid dangerous APIs.

        Args:
            prompt (str): The input prompt for code generation.
            structured_query (dict, optional): Precomputed structured query.
            max_attempts (int): Maximum retry attempts for syntax errors.

        Returns:
            str: The generated code.

        Raises:
            ValueError: If no syntactically correct code is generated after max attempts.
        """
        if structured_query is None:
            structured_query = self.restructure_query(prompt)
        code_prompt = f"{prompt}\n# Condition: {structured_query['condition']}\n# Property: {structured_query['property']}"
        for attempt in range(max_attempts):
            try:
                code = self.generator(
                    code_prompt,
                    max_length=300,
                    num_return_sequences=1,
                    temperature=0.7,
                    logits_processor=self.logits_processor
                )[0]["generated_text"]
                if validate_syntax(code):
                    return code
                self.logger.warning(f"Attempt {attempt + 1}: Generated code has syntax error, retrying...")
            except Exception as e:
                self.logger.error(f"Generation error: {str(e)}")
                raise
        raise ValueError("Failed to generate syntactically correct code after max attempts")

    def refine_and_generate(self, prompt):
        """Refine the query and generate code in a single workflow.

        Args:
            prompt (str): The input prompt to process.

        Returns:
            str: The generated code.
        """
        constraints = self.detect_constraints(prompt)
        self.logger.info(f"Constraints Detection: {json.dumps(constraints, indent=4)}")

        structured_query = self.restructure_query(prompt)
        self.logger.info(f"Structured Query: {json.dumps(structured_query, indent=4)}")

        code = self.generate_code(prompt, structured_query)
        return code

class DangerousTokenBias(LogitsProcessor):
    """LogitsProcessor to bias against dangerous tokens during code generation.

    Args:
        tokenizer: The tokenizer associated with the generation model.
        dangerous_tokens (list): List of tokens to penalize.
    """

    def __init__(self, tokenizer, dangerous_tokens):
        self.dangerous_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in dangerous_tokens]

    def __call__(self, input_ids, scores):
        """Apply bias to logits to penalize dangerous tokens.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            scores (torch.Tensor): Logit scores to modify.

        Returns:
            torch.Tensor: Modified logit scores.
        """
        scores[:, self.dangerous_token_ids] = scores[:, self.dangerous_token_ids] - 100.0
        return scores
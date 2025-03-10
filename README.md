# QueryRefiner

A Python library for query refinement and code generation using CodeBERT, CodeT5p, and CodeLlama.

## Installation

```bash
pip install queryrefiner
```

## Usage

```python
from queryrefiner import QueryRefiner
```

# Initialize QueryRefiner
refiner = QueryRefiner(code_model="codellama/CodeLlama-7b-hf", device=0)

# Refine query and generate code
prompt = "generate a function to find numbers > 10 with even digits"
code = refiner.refine_and_generate(prompt)
print("Generated Code:", code)

# Features

Query analysis and restructuring using CodeT5p.
Constraint detection using CodeBERT.
Code generation with logits bias to restrict dangerous APIs (e.g., eval, exec).
Syntax validation and retry mechanism.

# Requirements
Python >= 3.6
transformers >= 4.30.0
torch >= 1.9.0
datasets >= 2.0.0

## Development

### Clone the Repository

```bash
git clone https://github.com/laqeey/queryrefiner.git
cd queryrefiner
```

### Install Dependencies

```bash
pip install -e .
```

### Run Tests
```bash
python -m unittest discover -s tests
```

### Run Examples
```bash
python examples/example_usage.py
```

# License
MIT License.

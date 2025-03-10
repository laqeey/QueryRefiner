# examples/example_usage.py

"""Example usage of the QueryRefiner library."""

from queryrefiner import QueryRefiner
import logging

def main():
    """Main function to demonstrate QueryRefiner usage."""
    # Initialize QueryRefiner with debug logging
    refiner = QueryRefiner(code_model="codellama/CodeLlama-7b-hf", device=0, log_level=logging.DEBUG)

    # Example query
    prompt = "generate a function to find numbers > 10 with even digits"

    # Refine query and generate code
    code = refiner.refine_and_generate(prompt)
    print("Generated Code:\n", code)

if __name__ == "__main__":
    """Execute the main function."""
    main()
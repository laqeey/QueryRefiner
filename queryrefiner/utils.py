# queryrefiner/utils.py

"""Utility functions for the QueryRefiner library."""

import ast

def validate_syntax(code):
    """Validate the syntax of the generated code.

    Args:
        code (str): The code string to validate.

    Returns:
        bool: True if the syntax is valid, False otherwise.

    Notes:
        Prints the syntax error message if validation fails.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return False
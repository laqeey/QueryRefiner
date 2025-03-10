# tests/test_queryrefiner.py

"""Unit tests for the QueryRefiner library."""

import unittest
from queryrefiner import QueryRefiner

class TestQueryRefiner(unittest.TestCase):
    """Test cases for the QueryRefiner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.refiner = QueryRefiner(device=-1)  # Use CPU for testing

    def test_analyze_query(self):
        """Test the analyze_query method."""
        query = "generate a function to find numbers > 10 with even digits"
        result = self.refiner.analyze_query(query)
        self.assertEqual(result["task"], "generate a function")
        self.assertEqual(result["condition"], "> 10")
        self.assertEqual(result["property"], "even digits")

    def test_restructure_query(self):
        """Test the restructure_query method."""
        query = "generate a function to find numbers > 10 with even digits"
        result = self.refiner.restructure_query(query)
        self.assertIn("task", result)
        self.assertIn("condition", result)
        self.assertIn("property", result)

if __name__ == "__main__":
    """Run the unit tests."""
    unittest.main()
"""
Script to run all tests for fks_analyze.
"""

import unittest
import os
import sys

# Adjust sys.path to include the tests directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def discover_and_run_tests():
    """Discover and run all test files in the tests directory."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = discover_and_run_tests()
    sys.exit(0 if success else 1)

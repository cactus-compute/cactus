#!/usr/bin/env python3
"""Run all cactus python tests.

Usage:
    python python/test.py            # run all tests
    python python/test.py -v         # verbose
    python python/test.py -k graph   # only graph tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2 if "-v" in sys.argv else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

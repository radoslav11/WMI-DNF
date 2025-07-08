#!/usr/bin/env python3
"""
Unit tests for WMI-DNF examples.

This module contains comprehensive unit tests for all examples in the examples/ directory.
Each test imports the corresponding example and runs it with standard parameters,
verifying that the results are within acceptable tolerance of expected values.
"""

import unittest
import sys
import os
import time

# Add the parent directory to the path so we can import examples
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all examples
from examples.simple_constant import run_example as run_simple_constant
from examples.linear_weight import run_example as run_linear_weight
from examples.two_variables import run_example as run_two_variables
from examples.simple_boolean import run_example as run_simple_boolean
from examples.boolean_real_simple import run_example as run_boolean_real_simple
from examples.complex_boolean import run_example as run_complex_boolean
from examples.boolean_example import run_example as run_boolean_example
from examples.advanced_latte import run_example as run_advanced_latte
from examples.two_variables_simple_square import (
    run_example as run_two_variables_simple_square,
)
from examples.four_variables_mixed import (
    run_example as run_four_variables_mixed,
)


class TestExamples(unittest.TestCase):
    """Test cases for WMI-DNF examples."""

    def setUp(self):
        """Set up test environment."""
        # Create temp directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")

    def test_simple_constant(self):
        """Test simple constant weight example."""
        result = run_simple_constant(eps=0.25, delta=0.15, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 2.0)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            1.0,
            "Result should be within 1.0 of expected value",
        )

    def test_linear_weight(self):
        """Test linear weight function example."""
        result = run_linear_weight(eps=0.25, delta=0.15, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 2.0)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            1.0,
            "Result should be within 1.0 of expected value",
        )

    def test_two_variables(self):
        """Test two variables polynomial weight example."""
        result = run_two_variables(eps=0.25, delta=0.15, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 2.0)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            1.0,
            "Result should be within 1.0 of expected value",
        )

    def test_simple_boolean(self):
        """Test simple boolean variables example."""
        result = run_simple_boolean(eps=0.25, delta=0.15, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 0.92)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            0.5,
            "Result should be within 0.5 of expected value",
        )

    def test_boolean_real_simple(self):
        """Test boolean + real variables example."""
        result = run_boolean_real_simple(eps=0.25, delta=0.15, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 1.0)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            0.5,
            "Result should be within 0.5 of expected value",
        )

    def test_complex_boolean(self):
        """Test complex boolean variables example."""
        result = run_complex_boolean(eps=0.05, delta=0.05, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 0.5)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            0.3,
            "Result should be within 0.3 of expected value",
        )

    def test_boolean_example(self):
        """Test boolean and real variables example."""
        result = run_boolean_example(eps=0.25, delta=0.15, verbose=False)

        self.assertIsNotNone(result["result"])
        self.assertEqual(result["expected"], 2.0)
        self.assertIsInstance(result["execution_time"], float)
        self.assertGreater(result["execution_time"], 0)
        self.assertIsInstance(result["error"], float)

        # Check that result is within reasonable tolerance
        self.assertLess(
            result["error"],
            1.0,
            "Result should be within 1.0 of expected value",
        )

    def test_advanced_latte(self):
        """Test advanced LattE integration example."""
        result = run_advanced_latte(eps=0.2, delta=0.1, verbose=False)

        # This example might fail if LattE is not installed
        if result["success"]:
            self.assertIsNotNone(result["result"])
            self.assertIsInstance(result["execution_time"], float)
            self.assertGreater(result["execution_time"], 0)
            self.assertIsNone(result["expected"])  # No simple expected value
            self.assertIsNone(result["error"])  # No error calculation
        else:
            # LattE not available, check that we handled it gracefully
            self.assertIsNone(result["result"])
            self.assertIsNone(result["expected"])
            self.assertIsNone(result["error"])
            self.assertFalse(result["success"])

    def test_two_variables_simple_square(self):
        """Test simple square integration example."""
        result = run_two_variables_simple_square(
            eps=0.2, delta=0.1, verbose=False
        )

        # This example might fail if LattE is not installed
        if result["success"]:
            self.assertIsNotNone(result["result"])
            self.assertEqual(result["expected"], 400.0)
            self.assertIsInstance(result["execution_time"], float)
            self.assertGreater(result["execution_time"], 0)
            self.assertIsInstance(result["error"], float)

            # Check that result is within reasonable tolerance
            self.assertLess(
                result["error"],
                200.0,
                "Result should be within 200.0 of expected value",
            )
        else:
            # LattE not available, check that we handled it gracefully
            self.assertIsNone(result["result"])
            self.assertEqual(result["expected"], 400.0)
            self.assertIsNone(result["error"])
            self.assertFalse(result["success"])

    def test_four_variables_mixed(self):
        """Test four variables mixed active/free example."""
        result = run_four_variables_mixed(eps=0.2, delta=0.1, verbose=False)

        # This example might fail if LattE is not installed
        if result["success"]:
            self.assertIsNotNone(result["result"])
            self.assertEqual(result["expected"], 2400.0)
            self.assertIsInstance(result["execution_time"], float)
            self.assertGreater(result["execution_time"], 0)
            self.assertIsInstance(result["error"], float)

            # Check that result is within reasonable tolerance
            self.assertLess(
                result["error"],
                1000.0,
                "Result should be within 1000.0 of expected value",
            )

            # Check active/free variable separation
            self.assertIsInstance(result["active_vars"], list)
            self.assertIsInstance(result["free_vars"], list)
            self.assertEqual(
                result["active_vars"], [0, 1]
            )  # x, y should be active
            self.assertEqual(
                result["free_vars"], [2, 3]
            )  # z, w should be free

            # Check interior point
            self.assertIsInstance(result["interior_point"], list)
            self.assertEqual(len(result["interior_point"]), 4)
        else:
            # LattE not available, check that we handled it gracefully
            self.assertIsNone(result["result"])
            self.assertEqual(result["expected"], 2400.0)
            self.assertIsNone(result["error"])
            self.assertFalse(result["success"])

            # Active/free variables should still be computed
            self.assertIsInstance(result["active_vars"], list)
            self.assertIsInstance(result["free_vars"], list)


class TestExamplePerformance(unittest.TestCase):
    """Performance tests for examples."""

    def test_all_examples_complete_in_reasonable_time(self):
        """Test that all examples complete in reasonable time."""
        examples = [
            ("simple_constant", run_simple_constant),
            ("linear_weight", run_linear_weight),
            ("two_variables", run_two_variables),
            ("simple_boolean", run_simple_boolean),
            ("boolean_real_simple", run_boolean_real_simple),
            ("complex_boolean", run_complex_boolean),
            ("boolean_example", run_boolean_example),
        ]

        for name, run_func in examples:
            with self.subTest(example=name):
                start_time = time.time()
                result = run_func(
                    eps=0.5, delta=0.2, verbose=False
                )  # Looser tolerances for speed
                end_time = time.time()

                execution_time = end_time - start_time

                # Each example should complete within 30 seconds
                self.assertLess(
                    execution_time,
                    30.0,
                    f"{name} took too long: {execution_time:.2f} seconds",
                )

                # Check that the result is valid
                self.assertIsNotNone(result)
                self.assertIn("result", result)
                self.assertIn("execution_time", result)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)

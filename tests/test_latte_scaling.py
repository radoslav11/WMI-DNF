import unittest
import numpy as np
from utils.run_latte import _write_latte_input_file
from utils.reals_universe import RealsUniverse
import os
import sys
from io import StringIO


class TestLatteScaling(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files if it doesn't exist
        os.makedirs("temp", exist_ok=True)

    def tearDown(self):
        # Clean up temporary files after each test
        for f in os.listdir("temp"):
            if f.startswith("test_") and f.endswith(".latte"):
                os.remove(os.path.join("temp", f))

    def test_latte_scaling_integer_coefficients(self):
        # Test with integer coefficients, no scaling needed
        lraAtoms = [[(1, 1), (2, 2), ("<=", 10)], [(1, -1), (2, 1), (">=", 0)]]
        nbBools = 0
        nbReals = 2
        universeReals = RealsUniverse(
            nbReals=nbReals, lowerBound=-10, upperBound=10
        )
        latte_file_path = "temp/test_integer_coeffs.hrep.latte"

        _, _, scaling_factor, _ = _write_latte_input_file(
            latte_file_path, lraAtoms, nbBools, nbReals, universeReals
        )

        self.assertEqual(scaling_factor, 1)
        with open(latte_file_path, "r") as f:
            content = f.read().replace("\\n", "\n")
            lines = [
                line.strip() for line in content.split("\n") if line.strip()
            ]

            # Expected 2 original constraints + 2*nbReals bounds = 2 + 2*2 = 6 constraints
            self.assertIn(
                "6 3", lines[0]
            )  # Number of constraints and variables
            self.assertIn(
                "10 -1 -2", lines
            )  # <= 10 becomes -x1 -2x2 + 10 >= 0
            self.assertIn("0 -1 1", lines)  # >= 0 becomes -x1 + x2 + 0 >= 0
            self.assertIn(
                "10 -1 0", lines
            )  # Upper bound for x1: x1 <= 10 becomes -x1 + 10 >= 0
            self.assertIn(
                "10 1 0", lines
            )  # Lower bound for x1: x1 >= -10 becomes x1 + 10 >= 0
            self.assertIn(
                "10 0 -1", lines
            )  # Upper bound for x2: x2 <= 10 becomes -x2 + 10 >= 0
            self.assertIn(
                "10 0 1", lines
            )  # Lower bound for x2: x2 >= -10 becomes x2 + 10 >= 0

    def test_latte_scaling_float_coefficients(self):
        # Test with float coefficients, scaling needed
        lraAtoms = [
            [(1, 0.5), (2, 1.5), ("<=", 10.0)],
            [(1, -0.25), (2, 0.75), (">=", 0.0)],
        ]
        nbBools = 0
        nbReals = 2
        universeReals = RealsUniverse(
            nbReals=nbReals, lowerBound=-10, upperBound=10
        )
        latte_file_path = "temp/test_float_coeffs.hrep.latte"

        _, _, scaling_factor, _ = _write_latte_input_file(
            latte_file_path, lraAtoms, nbBools, nbReals, universeReals
        )

        self.assertEqual(
            scaling_factor, 100
        )  # Max precision is 2 decimal places (0.25, 0.75)
        with open(latte_file_path, "r") as f:
            content = f.read().replace("\\n", "\n")
            lines = [
                line.strip() for line in content.split("\n") if line.strip()
            ]

            self.assertIn("6 3", lines[0])
            self.assertIn(
                "1000 -50 -150", lines
            )  # <= 10.0 becomes -50x1 -150x2 + 1000 >= 0
            self.assertIn("0 -25 75", lines)
            self.assertIn("1000 -100 0", lines)  # Upper bound for x1
            self.assertIn("1000 100 0", lines)  # Lower bound for x1
            self.assertIn("1000 0 -100", lines)  # Upper bound for x2
            self.assertIn("1000 0 100", lines)  # Lower bound for x2

    def test_latte_scaling_high_precision_warning(self):
        # Test with high precision coefficients, should trigger warning
        lraAtoms = [
            [(1, 0.1234567891), (2, 0.0000000001), ("<=", 1.0)],
        ]
        nbBools = 0
        nbReals = 2
        universeReals = RealsUniverse(
            nbReals=nbReals, lowerBound=-1, upperBound=1
        )
        latte_file_path = "temp/test_high_precision.hrep.latte"

        # Redirect stdout to capture the warning message
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        _write_latte_input_file(
            latte_file_path, lraAtoms, nbBools, nbReals, universeReals
        )

        captured_output = sys.stdout.getvalue()
        sys.stdout = old_stdout  # Restore stdout

        self.assertIn(
            (
                "Warning: Required precision for LattE coefficients (10) is"
                " greater than 9. This may lead to loss of precision."
            ),
            captured_output,
        )

    def test_latte_scaling_bounds(self):
        # Test with float bounds, scaling should apply
        lraAtoms = []
        nbBools = 0
        nbReals = 1
        universeReals = RealsUniverse(
            nbReals=nbReals, lowerBound=-0.5, upperBound=0.5
        )
        latte_file_path = "temp/test_float_bounds.hrep.latte"

        _, _, scaling_factor, _ = _write_latte_input_file(
            latte_file_path, lraAtoms, nbBools, nbReals, universeReals
        )

        self.assertEqual(
            scaling_factor, 10
        )  # Max precision is 1 decimal place (0.5)
        with open(latte_file_path, "r") as f:
            content = f.read().replace("\\n", "\n")
            lines = [
                line.strip() for line in content.split("\n") if line.strip()
            ]

            # Expected 0 original constraints + 2*nbReals bounds = 0 + 2*1 = 2 constraints
            self.assertIn(
                "2 2", lines[0]
            )  # 2 constraints (bounds), 2 variables (constant + x1)
            self.assertIn(
                "5 -10", lines
            )  # 0.5 * 10 = 5, -1 * 10 = -10 (Upper bound: x1 <= 0.5 becomes -x1 + 0.5 >= 0)
            self.assertIn(
                "5 10", lines
            )  # -(-0.5) * 10 = 5, 1 * 10 = 10 (Lower bound: x1 >= -0.5 becomes x1 + 0.5 >= 0)

    def test_latte_scaling_custom_float_coefficients(self):
        # Test with custom float coefficients and bounds, scaling needed
        lraAtoms = [
            [(1, 0.7), (2, 0.6), ("<=", 98)],
            [(1, 0.7), (2, 0.6), (">=", 6)],
            [(1, 0.6), (2, -0.7), (">=", 42)],
            [(1, 0.6), (2, -0.7), ("<=", 52)],
        ]
        nbBools = 0
        nbReals = 2
        universeReals = RealsUniverse(
            nbReals=nbReals, lowerBound=-1000, upperBound=1000
        )
        latte_file_path = "temp/test_custom_float_coeffs.hrep.latte"

        _, _, scaling_factor, _ = _write_latte_input_file(
            latte_file_path, lraAtoms, nbBools, nbReals, universeReals
        )

        self.assertEqual(scaling_factor, 10)

        with open(latte_file_path, "r") as f:
            content = f.read().replace("\\n", "\n")
            lines = [
                line.strip() for line in content.split("\n") if line.strip()
            ]

            expected_lines = [
                "8 3",
                "980 -7 -6",
                "-60 7 6",
                "-420 6 -7",
                "520 -6 7",
                "10000 -10 0",
                "10000 10 0",
                "10000 0 -10",
                "10000 0 10",
            ]

            self.assertEqual(len(lines), len(expected_lines))
            for i in range(len(expected_lines)):
                self.assertEqual(lines[i], expected_lines[i])


if __name__ == "__main__":
    unittest.main()
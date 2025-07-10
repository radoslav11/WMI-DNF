# WMI-DNF Tests

This directory contains comprehensive unit tests for the WMI-DNF examples.

## Running Tests

### Run all tests:
```bash
python -m unittest tests.examples -v
```

### Run specific test class:
```bash
python -m unittest tests.examples.TestExamples -v
```

### Run specific test method:
```bash
python -m unittest tests.examples.TestExamples.test_simple_constant -v
```

### Run performance tests:
```bash
python -m unittest tests.examples.TestExamplePerformance -v
```

## Test Structure

- **TestExamples**: Unit tests for individual examples
- **TestExamplePerformance**: Performance tests ensuring examples complete in reasonable time

## Test Coverage

The test suite covers all examples in the `examples/` directory:

1. `simple_constant.py` - Basic constant weight integration
2. `linear_weight.py` - Linear weight function
3. `two_variables.py` - Two-variable polynomial integration
4. `simple_boolean.py` - Boolean variable integration
5. `boolean_real_simple.py` - Mixed boolean and real variables
6. `complex_boolean.py` - Complex boolean logic
7. `boolean_example.py` - Alternative boolean example
8. `advanced_latte.py` - Advanced LattE integration (requires LattE)
9. `two_variables_simple_square.py` - Simple square integration (requires LattE)
10. `four_variables_mixed.py` - Mixed active/free variables (requires LattE)

## Notes

- Tests that require LattE integration will gracefully handle the case where LattE is not installed
- All tests use relaxed tolerances to account for the probabilistic nature of the WMI algorithm
- Performance tests ensure examples complete within 30 seconds using looser parameters
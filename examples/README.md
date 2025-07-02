# WMI Examples

This directory contains self-contained examples demonstrating how to use the Weighted Model Integration (WMI) solver without requiring external input files.

## Running Examples

Each example can be run directly:

```bash
cd examples/
python simple_constant.py
python linear_weight.py
python two_variables.py
python boolean_real_simple.py
python complex_boolean.py
python advanced_latte.py        # Requires LattE installation
```

Or from the repository root:

```bash
python examples/simple_constant.py
```

## Example Descriptions

### `simple_constant.py`
- **Variables**: Single real variable x ∈ [0, 2]
- **DNF Formula**: Always true (empty clause)
- **Weight Function**: Constant 1
- **Expected Result**: ≈ 2.0 (volume of interval [0,2])

This is the basic example requested, demonstrating the fundamental setup.

### `linear_weight.py`
- **Variables**: Single real variable x ∈ [0, 2]
- **DNF Formula**: Always true
- **Weight Function**: x (linear)
- **Expected Result**: ≈ 2.0 (∫[0,2] x dx = 2)

Shows how to define polynomial weight functions.

### `two_variables.py`
- **Variables**: Two real variables x, y ∈ [0, 1]
- **DNF Formula**: x + y ≤ 1.5
- **Weight Function**: 1 + x + y
- **Expected Result**: Integration over constrained region

Demonstrates multi-variable real constraints and polynomial weights.

### `boolean_real_simple.py`
- **Variables**: One boolean b, one real x ∈ [0, 1]
- **DNF Formula**: Always true
- **Weight Function**: Constant 1
- **Boolean Weights**: P(b=true) = 0.7
- **Expected Result**: ≈ 1.0

Shows how to combine boolean and real variables with probabilistic boolean weights.

### `complex_boolean.py`
- **Variables**: Three booleans a, b, c + one dummy real x ∈ [0, 1]
- **DNF Formula**: (a ∧ b) ∨ (a ∧ c) ∨ (b ∧ c)
- **Weight Function**: Constant 1
- **Boolean Weights**: P(a=true) = P(b=true) = P(c=true) = 0.5
- **Expected Result**: 4/8 = 0.5

Demonstrates complex boolean logic. By enumeration: 4 out of 8 boolean combinations satisfy the formula, each with probability 1/8.

### `advanced_latte.py`
- **Variables**: Two reals x, y ∈ [0, 3]
- **DNF Formula**: (x + 2y ≤ 4 ∧ x - y ≥ 0) ∨ (2x + y ≤ 5 ∧ x + y ≥ 2)
- **Weight Function**: x² + y² + xy + 1
- **Requirements**: LattE integration tool must be installed

Shows the full power of WMI with complex polytopic constraints and polynomial weights. **Requires LattE installation**.


## Key Concepts

### Variable Indexing
- Boolean variables: indices 0 to `cntBools-1`
- Negated booleans: indices `cntBools` to `2*cntBools-1`
- Real variables: indices `cntBools` onwards

### DNF Formula Format
- List of clauses (disjunction)
- Each clause is a list of literals (conjunction)
- Boolean literals: integers (variable indices)
- Real constraints: `[(var_idx, coeff), ..., (op, bound)]`

### Weight Functions
- Monomials format: `[coefficient, [powers_for_each_real_var]]`
- Boolean weights: array of probabilities for each boolean variable

### Expected Results
All examples include expected theoretical results for validation. The approximation algorithm should produce results close to these values, within the specified epsilon and delta parameters.
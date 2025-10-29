# Sinkhorn JKO Module - Refactoring Documentation

## Overview

This module refactors the Sinkhorn algorithm code from the original files into a clean, JAX-based implementation that follows the same stylistic patterns as `pdhg_jko.py`.

## Key Changes

### 1. **Style Alignment**
The new code follows the same patterns as `pdhg_jko.py`:
- Uses `@flstr.dataclass` decorators for main classes
- Uses `@jax.jit` for compiled methods
- Pure functional style with no side effects
- Type hints throughout
- Clear docstrings in the same format
- Helper functions defined outside classes

### 2. **Pure JAX Implementation**
- Removed dependencies on: `pytorch`, `tqdm`, `matplotlib`, `scipy`, `loguru`, `multiprocessing`
- Uses only: `jax`, `jax.numpy`, `flax.struct`
- All iterations use `jax.lax.scan`, `jax.lax.fori_loop`, or `jax.lax.while_loop` for efficiency

### 3. **Functional Organization**

#### **Sinkhorn Class**
Standard two-marginal entropy-regularized optimal transport:
```python
solver = Sinkhorn(
    C=cost_matrix,      # (n, m) cost matrix
    a=source_marginal,   # (n,) distribution
    b=target_marginal,   # (m,) distribution
    epsilon=0.1,         # entropy regularization
    max_iters=1000,
    tol=1e-9
)
pi, u, v, error, iters = solver.solve()
```

#### **MultiMarginalSinkhorn Class**
N-marginal entropy-regularized optimal transport:
```python
solver = MultiMarginalSinkhorn(
    C=cost_tensor,        # (n, n, ..., n) N-dimensional
    marginals=[mu1, mu2, mu3],  # List of N marginals
    epsilon=0.1,
    max_iters=1000,
    tol=1e-9
)
P, scalings, error, iters = solver.solve()
```

#### **SinkhornJKO Class**
JKO gradient flow using entropic OT (similar structure to PrimalDualJKO):
```python
solver = SinkhornJKO(
    C=cost_matrix,       # (n, n) squared distance matrix
    rho0=initial_dist,   # (n,) initial distribution
    eta=0.1,             # JKO time step
    epsilon=0.1,         # entropy regularization
    inner_steps=1000,
    tol=1e-9
)
rhos, errors = solver.compute_flow(num_steps=100)
```

## Mapping from Original Code

### Original `classical_eot.py` Functions → New Classes

| Original Function | New Implementation | Notes |
|-------------------|-------------------|-------|
| `shannon_sinkhorn` | `Sinkhorn.solve()` | Cleaner interface, JIT-compiled |
| `quadratic_*` methods | Not included | These used quadratic regularization, not entropy; already handled by `PrimalDualJKO` |
| `sinkhorn_compute_error` | `_marginal_error()` | Helper function |
| `compute_P` | `_compute_coupling()` | Helper function |

### Original Utility Functions → Removed/Integrated

- `utils.py` plotting functions → Removed (use external visualization)
- `make_cost_tensor.py` → User responsibility (provide cost matrix)
- `make_marginal_tensor.py` → User responsibility (provide marginals)
- Error logging → Return values for user to handle

## Usage Example

Here's how to replace the original `run_classical_eot.py` workflow:

### Original Workflow:
```python
# Load data
cost = jnp.load('cost.npy')
marg = [jnp.load('mu1.npy'), jnp.load('mu2.npy')]

# Run Sinkhorn
P, error, iterations, time = shannon_sinkhorn(
    marg, cost, epsilon=0.1, convergence_error=1e-8, max_iters=50000
)
```

### New Workflow:
```python
from sinkhorn_jko import Sinkhorn

# Load data
cost = jnp.load('cost.npy')
a = jnp.load('mu1.npy')
b = jnp.load('mu2.npy')

# Create solver
solver = Sinkhorn(C=cost, a=a, b=b, epsilon=0.1, tol=1e-8, max_iters=50000)

# Solve
pi, u, v, error, iterations = solver.solve()
```

## Advantages of the New Implementation

1. **Performance**: Fully JIT-compiled, no Python loops
2. **Composability**: Easy to integrate with other JAX code
3. **Consistency**: Same style as `pdhg_jko.py`
4. **Simplicity**: No external dependencies for core algorithms
5. **Warm-starting**: Built-in support for warm-starting dual variables
6. **Fixed iterations**: `solve_fixed_iters()` for better JIT compilation

## Key Design Decisions

### Why these classes?

1. **Sinkhorn**: The core algorithm for entropy-regularized OT
2. **MultiMarginalSinkhorn**: Extension to N marginals (generalizes `shannon_sinkhorn` from original)
3. **SinkhornJKO**: Gradient flow using entropic OT (parallels `PrimalDualJKO`)

### Why remove quadratic methods?

The quadratic regularization methods (`quadratic_cyclic_projection`, etc.) from the original code are already handled by the existing `PrimalDualJKO` class with `proxF_quadratic`. Including them in `sinkhorn_jko.py` would be redundant and violate the naming convention (Sinkhorn specifically refers to entropy regularization).

### Convergence Checking

The new implementation uses `jax.lax.while_loop` with convergence checking, which is more efficient than Python loops but requires careful handling of termination conditions. For fixed-iteration needs, use `solve_fixed_iters()`.

## Testing the New Module

```python
import jax.numpy as jnp
from sinkhorn_jko import Sinkhorn

# Simple test
n = 100
C = jnp.zeros((n, n))
for i in range(n):
    for j in range(n):
        C = C.at[i, j].set((i - j) ** 2)

a = jnp.ones(n) / n
b = jnp.ones(n) / n

solver = Sinkhorn(C=C, a=a, b=b, epsilon=1.0, max_iters=1000, tol=1e-9)
pi, u, v, error, iters = solver.solve()

print(f"Converged in {iters} iterations with error {error}")
```

## Future Extensions

Possible additions while maintaining the current style:
- Unbalanced OT with KL divergence penalties
- Log-domain stabilized Sinkhorn
- Acceleration techniques (momentum, extrapolation)
- Adaptive epsilon scheduling

## Compatibility Note

This module is designed to be used alongside `pdhg_jko.py`, not replace it. Both solve different types of regularized optimal transport problems:
- **Sinkhorn**: Entropy regularization (this module)
- **PDHG**: General functionals with quadratic/entropy terms (existing module)
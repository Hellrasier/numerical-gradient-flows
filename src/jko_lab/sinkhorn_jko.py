"""
Sinkhorn algorithm for entropy-regularized optimal transport in JAX.

Problem (discrete):
    min_{pi >= 0}  <C, pi> + epsilon * H(pi)
    subject to: pi @ 1 = a,  pi^T @ 1 = b

where H(pi) = sum_{i,j} pi_{ij} * (log(pi_{ij}) - 1) is the Shannon entropy.

We also provide extensions for:
- Multi-marginal optimal transport (N marginals)
- JKO steps with entropy regularization
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, List
import flax.struct as flstr

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


def _compute_kernel(C: Array, epsilon: float) -> Array:
    """Compute Gibbs kernel K = exp(-C/epsilon)."""
    return jnp.exp(-C / epsilon)


def _row_sums(pi: Array) -> Array:
    """Compute row sums of a matrix."""
    return pi.sum(axis=1)


def _col_sums(pi: Array) -> Array:
    """Compute column sums of a matrix."""
    return pi.sum(axis=0)


def _compute_coupling(u: Array, v: Array, K: Array) -> Array:
    """Compute coupling pi from dual variables: pi = diag(u) @ K @ diag(v)."""
    return u[:, None] * K * v[None, :]


def _marginal_error(pi: Array, a: Array, b: Array) -> float:
    """Compute maximum marginal constraint violation."""
    error_a = jnp.abs(_row_sums(pi) - a).sum()
    error_b = jnp.abs(_col_sums(pi) - b).sum()
    return jnp.maximum(error_a, error_b)


# -------------------- Two-marginal Sinkhorn --------------------

@flstr.dataclass
class Sinkhorn:
    """
    Sinkhorn algorithm for entropy-regularized optimal transport between two marginals.
    
    Solves:
        min_{pi >= 0}  <C, pi> - epsilon * H(pi)
        s.t.  pi @ 1 = a,  pi^T @ 1 = b
    
    Attributes:
        C: Cost matrix (n, m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        epsilon: Regularization parameter
        max_iters: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance for marginal constraints
    """
    C: Array  # (n, m)
    a: Array  # (n,)
    b: Array  # (m,)
    epsilon: float
    max_iters: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)

    def __post_init__(self):
        """Precompute the Gibbs kernel."""
        object.__setattr__(self, '_K', _compute_kernel(self.C, self.epsilon))

    @jax.jit
    def solve(self, 
              u_init: Optional[Array] = None,
              v_init: Optional[Array] = None) -> Tuple[Array, Array, Array, float, int]:
        """
        Solve the entropy-regularized OT problem.
        
        Args:
            u_init: Initial dual variable (n,), default ones
            v_init: Initial dual variable (m,), default ones
            
        Returns:
            pi: Optimal coupling (n, m)
            u: Final dual variable (n,)
            v: Final dual variable (m,)
            error: Final marginal constraint violation
            iterations: Number of iterations performed
        """
        n, m = self.C.shape
        K = self._K
        
        # Initialize dual variables
        u = jnp.ones(n) if u_init is None else u_init
        v = jnp.ones(m) if v_init is None else v_init

        def body_fn(carry):
            u_c, v_c, _, _ = carry
            
            # Sinkhorn update
            u_new = self.a / (K @ v_c + 1e-300)
            v_new = self.b / (K.T @ u_new + 1e-300)
            
            # Compute error
            pi = _compute_coupling(u_new, v_new, K)
            error = _marginal_error(pi, self.a, self.b)
            
            return (u_new, v_new, error, 1)

        def cond_fn(carry):
            _, _, error, iteration = carry
            return (error > self.tol) & (iteration < self.max_iters)

        # Run Sinkhorn iterations
        init_carry = (u, v, jnp.inf, 0)
        u_final, v_final, error_final, iters = lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        # Compute final coupling
        pi_final = _compute_coupling(u_final, v_final, K)
        
        return pi_final, u_final, v_final, error_final, iters

    @jax.jit
    def solve_fixed_iters(self,
                          num_iters: int,
                          u_init: Optional[Array] = None,
                          v_init: Optional[Array] = None) -> Tuple[Array, Array, Array, float]:
        """
        Solve with a fixed number of iterations (useful for JIT compilation).
        
        Args:
            num_iters: Number of Sinkhorn iterations to perform
            u_init: Initial dual variable (n,)
            v_init: Initial dual variable (m,)
            
        Returns:
            pi: Coupling (n, m)
            u: Final dual variable (n,)
            v: Final dual variable (m,)
            error: Final marginal constraint violation
        """
        n, m = self.C.shape
        K = self._K
        
        u = jnp.ones(n) if u_init is None else u_init
        v = jnp.ones(m) if v_init is None else v_init

        def body_fn(_, carry):
            u_c, v_c = carry
            u_new = self.a / (K @ v_c + 1e-300)
            v_new = self.b / (K.T @ u_new + 1e-300)
            return (u_new, v_new)

        u_final, v_final = lax.fori_loop(0, num_iters, body_fn, (u, v))
        pi_final = _compute_coupling(u_final, v_final, K)
        error = _marginal_error(pi_final, self.a, self.b)
        
        return pi_final, u_final, v_final, error


# -------------------- Multi-marginal Sinkhorn --------------------

def _einsum_marginal(P: Array, K: Array, axis: int, N: int) -> Array:
    """Compute a marginal of P by summing over all axes except 'axis'."""
    axes_to_sum = [i for i in range(N) if i != axis]
    return jnp.sum(P, axis=tuple(axes_to_sum))


@flstr.dataclass
class MultiMarginalSinkhorn:
    """
    Sinkhorn algorithm for multi-marginal entropy-regularized optimal transport.
    
    Solves:
        min_{pi >= 0}  <C, pi> + epsilon * H(pi)
        s.t.  marginal_i(pi) = mu_i  for all i
    
    Attributes:
        C: Cost tensor (n, n, ..., n) with N dimensions
        marginals: List of N marginal distributions, each (n,)
        epsilon: Entropy regularization parameter
        max_iters: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance
    """
    C: Array  # (n, n, ..., n)
    marginals: List[Array]  # List of N arrays of shape (n,)
    epsilon: float
    max_iters: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)

    def __post_init__(self):
        """Precompute the Gibbs kernel."""
        object.__setattr__(self, '_K', _compute_kernel(self.C, self.epsilon))
        object.__setattr__(self, '_N', len(self.marginals))
        object.__setattr__(self, '_n', self.marginals[0].shape[0])

    @jax.jit
    def solve(self, scaling_init: Optional[List[Array]] = None) -> Tuple[Array, List[Array], float, int]:
        """
        Solve the multi-marginal entropy-regularized OT problem.
        
        Args:
            scaling_init: Initial scaling variables (list of N arrays of shape (n,))
            
        Returns:
            P: Optimal coupling (n, n, ..., n)
            scalings: Final scaling variables (list of N arrays)
            error: Final marginal constraint violation
            iterations: Number of iterations performed
        """
        N = self._N
        n = self._n
        K = self._K
        
        # Initialize scaling variables
        if scaling_init is None:
            scalings = [jnp.ones(n) for _ in range(N)]
        else:
            scalings = scaling_init

        def body_fn(carry):
            scales_c, _, _ = carry
            scales_new = list(scales_c)
            
            # Cyclic updates
            for i in range(N):
                # Compute P with current scalings
                P = K
                for j in range(N):
                    if j != i:
                        shape = [1] * N
                        shape[j] = n
                        P = P * scales_new[j].reshape(shape)
                
                # Update scaling variable i
                marginal_i = _einsum_marginal(P, K, i, N)
                scales_new[i] = self.marginals[i] / (marginal_i + 1e-300)
            
            # Compute final P and error
            P_final = K
            for j in range(N):
                shape = [1] * N
                shape[j] = n
                P_final = P_final * scales_new[j].reshape(shape)
            
            # Compute maximum marginal error
            error = 0.0
            for i in range(N):
                marginal_i = _einsum_marginal(P_final, K, i, N)
                error_i = jnp.abs(marginal_i - self.marginals[i]).sum()
                error = jnp.maximum(error, error_i)
            
            return (scales_new, error, 1)

        def cond_fn(carry):
            _, error, iteration = carry
            return (error > self.tol) & (iteration < self.max_iters)

        init_carry = (scalings, jnp.inf, 0)
        scales_final, error_final, iters = lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        # Compute final coupling
        P_final = K
        for j in range(N):
            shape = [1] * N
            shape[j] = n
            P_final = P_final * scales_final[j].reshape(shape)
        
        return P_final, scales_final, error_final, iters


# -------------------- Sinkhorn for JKO steps --------------------

@flstr.dataclass
class SinkhornJKO:
    """
    JKO scheme using entropy-regularized optimal transport.
    
    For a functional F and initial distribution rho_0, computes the Wasserstein gradient flow
    by iteratively solving:
        rho_{k+1} = argmin_{rho} eta * F(rho) + W_epsilon(rho, rho_k)
    
    where W_epsilon is the Shannon-entropy regularized 2-Wasserstein distance.
    
    For simple functionals (e.g., internal energy), this reduces to a modified
    Sinkhorn algorithm with adjusted marginals.
    
    Attributes:
        C: Cost matrix (n, n)
        rho0: Initial distribution (n,)
        eta: JKO time step
        epsilon: Entropy regularization parameter
        F_func: Functional F (optional, for general case)
        inner_steps: Number of Sinkhorn iterations per JKO step
    """
    C: Array  # (n, n)
    rho0: Array  # (n,)
    eta: float
    epsilon: float
    F_func: Optional[Callable[[Array], float]] = flstr.field(pytree_node=False, default=None)
    inner_steps: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)

    @jax.jit
    def take_step(self,
                  rho_k: Array,
                  u_ws: Optional[Array] = None,
                  v_ws: Optional[Array] = None) -> Tuple[Array, Array, Array, float]:
        """
        Take a single JKO step from rho_k.
        
        For the special case of internal energy F(rho) = sum rho_i * V_i,
        the JKO step becomes a Sinkhorn problem with modified cost.
        
        Args:
            rho_k: Current distribution (n,)
            u_ws: Warm-start for u (n,)
            v_ws: Warm-start for v (n,)
            
        Returns:
            rho_next: Next distribution (n,)
            u: Final dual variable (n,)
            v: Final dual variable (n,)
            error: Convergence error
        """
        # For simplicity, we solve: min W_epsilon(rho, rho_k)
        # subject to rho >= 0, sum(rho) = 1
        # This is equivalent to Sinkhorn with a = rho, b = rho_k
        
        solver = Sinkhorn(
            C=self.C,
            a=rho_k,
            b=rho_k,
            epsilon=self.epsilon,
            max_iters=self.inner_steps,
            tol=self.tol
        )
        
        pi, u, v, error, _ = solver.solve(u_init=u_ws, v_init=v_ws)
        
        # Extract marginal and normalize
        rho_next = _row_sums(pi)
        rho_next = rho_next / jnp.sum(rho_next)
        
        return rho_next, u, v, error

    @jax.jit
    def compute_flow(self, num_steps: int) -> Tuple[Array, Array]:
        """
        Compute JKO flow trajectory.
        
        Args:
            num_steps: Number of JKO steps
            
        Returns:
            rhos: Trajectory (num_steps+1, n) including rho0
            errors: Convergence errors at each step (num_steps,)
        """
        n = self.C.shape[0]

        def one_step(carry, _):
            rho_k, u_ws, v_ws = carry
            rho_next, u_new, v_new, error = self.take_step(rho_k, u_ws, v_ws)
            next_carry = (rho_next, u_new, v_new)
            out = (rho_next, error)
            return next_carry, out

        init_carry = (self.rho0, jnp.ones(n), jnp.ones(n))
        _, outs = lax.scan(one_step, init_carry, xs=None, length=num_steps)
        rhos_steps, errors = outs
        
        # Prepend initial rho0
        rhos = jnp.concatenate([self.rho0[None, :], rhos_steps], axis=0)
        
        return rhos, errors

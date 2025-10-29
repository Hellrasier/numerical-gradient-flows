"""
Sinkhorn algorithm for entropy-regularized optimal transport

Upon discretization, entropy regularized optimal transport solves the following problem:
    min_{pi >= 0} <C, pi> + epsilon * H(pi)
    such that pi @ 1 = a, and pi.T @ 1 = b

where H(pi) = sum_{i,j} pi_{i,j} (log(pi_{i,j}) - 1) is the Shannon entropy of pi.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import flax.struct as flstr
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
Array = jnp.ndarray

def _sinkhorn_log_step(f: Array, g: Array, C: Array, a: Array, b: Array, 
                       epsilon: float) -> Tuple[Array, Array]:
    """
    Take a single step of log-sum-exp version of Sinkhorn iterations. The updates are given by:

        f = ε·log(a) - ε·logsumexp((g_j - C_ij)/ε, over j)
        g = ε·log(b) - ε·logsumexp((f_i - C_ij)/ε, over i)
    
    Args:
        f: Log-potential for rows (n,)
        g: Log-potential for columns (m,)
        C: Cost matrix (n, m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        epsilon: Regularization parameter
        
    Returns:
        f_new: Updated log-potential for rows (n,)
        g_new: Updated log-potential for columns (m,)
    """
    f_new = epsilon * jnp.log(a) - epsilon * logsumexp(
        (g[None, :] - C) / epsilon, axis=1
    )
    g_new = epsilon * jnp.log(b) - epsilon * logsumexp(
        (f_new[:, None] - C) / epsilon, axis=0
    )
    
    return f_new, g_new


def _compute_coupling_from_log(f: Array, g: Array, C: Array, epsilon: float) -> Array:
    """
    Recover the optimal transport coupling π from log-potentials f and g using:

        π_ij = u_i * K_ij * v_j
             = exp(log(u_i) + log(K_ij) + log(v_j))
             = exp((f_i + g_j - C_ij) / ε)
    
    where f = ε·log(u), g = ε·log(v), and K = exp(-C/ε).
    
    Args:
        f: Log-potential for rows (n,)
        g: Log-potential for columns (m,)
        C: Cost matrix (n, m)
        epsilon: Regularization parameter
        
    Returns:
        pi: Transport coupling (n, m) - satisfies π≥0, row sums=a, col sums=b
    """
    return jnp.exp((f[:, None] + g[None, :] - C) / epsilon)


def _row_sums(pi: Array) -> Array:
    """Compute row sums of coupling matrix: π @ 1."""
    return pi.sum(axis=1)


def _col_sums(pi: Array) -> Array:
    """Compute column sums of coupling matrix: π.T @ 1."""
    return pi.sum(axis=0)


def _marginal_error(pi: Array, a: Array, b: Array) -> float:
    """
    Compute maximum of errors on marginals.
    
    Args:
        pi: Transport coupling (n, m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        
    Returns:
        error: max(||π@1 - a||_∞, ||π.T@1 - b||_∞)
    """
    error_a = jnp.abs(_row_sums(pi) - a).max()
    error_b = jnp.abs(_col_sums(pi) - b).max()
    return jnp.maximum(error_a, error_b)


@flstr.dataclass
class Sinkhorn:
    """
    Log-sum-exp formulation of Sinkhorn solver for entropy-regularized optimal transport.
    
    Solves the discrete OT problem:
        min_{π≥0}  ⟨C, π⟩ + ε·H(π)
        subject to: π @ 1 = a,  π.T @ 1 = b
    
    where H(π) = Σᵢⱼ πᵢⱼ(log(πᵢⱼ) - 1) is Shannon entropy.

    Attributes:
        C: Cost matrix (n, m) - typically squared Euclidean distances
        a: Source marginal (n,) - probability vector (positive, sums to 1)
        b: Target marginal (m,) - probability vector (positive, sums to 1)
        epsilon: Regularization parameter - smaller values approximate true W₂
        max_iters: Maximum number of iterations
        tol: Convergence tolerance for marginal constraints
    """
    C: Array  # (n, m) cost matrix
    a: Array  # (n,) source marginal
    b: Array  # (m,) target marginal
    epsilon: float  # entropic regularization
    max_iters: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)

    @jax.jit
    def solve(self, 
              f_init: Optional[Array] = None,
              g_init: Optional[Array] = None) -> Tuple[Array, Array, Array, float, int]:
        """
        Solve entropy-regularized OT using log-domain Sinkhorn iterations.
        
        Args:
            f_init: Initial log-potential for rows (n,). If None, starts with zeros.
            g_init: Initial log-potential for cols (m,). If None, starts with zeros.
            
        Returns:
            pi: Optimal transport coupling (n, m)
            f: Final log-potential for rows (n,)
            g: Final log-potential for columns (m,)
            error: Final marginal constraint violation
            iters: Number of iterations performed
        """
        n, m = self.C.shape
        
        # Initialize log-potentials
        f = jnp.zeros(n) if f_init is None else f_init
        g = jnp.zeros(m) if g_init is None else g_init

        def body_fn(carry):
            """Single iteration: update f, g and check error."""
            f_c, g_c, _, iteration = carry
            
            # Perform one log-domain Sinkhorn step
            f_new, g_new = _sinkhorn_log_step(
                f_c, g_c, self.C, self.a, self.b, self.epsilon
            )
            
            # Compute marginal error
            pi = _compute_coupling_from_log(f_new, g_new, self.C, self.epsilon)
            error = _marginal_error(pi, self.a, self.b)
            
            return (f_new, g_new, error, iteration + 1)

        def cond_fn(carry):
            """Continue while error > tol and not exceeded max_iters."""
            _, _, error, iteration = carry
            return (error > self.tol) & (iteration < self.max_iters)

        # Run Sinkhorn iterations with early stopping
        init_carry = (f, g, jnp.inf, 0)
        f_final, g_final, error_final, iters = lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        # Compute final coupling from converged log-potentials
        pi_final = _compute_coupling_from_log(f_final, g_final, self.C, self.epsilon)
        
        return pi_final, f_final, g_final, error_final, iters

    @jax.jit
    def solve_fixed_iters(self,
                          num_iters: int,
                          f_init: Optional[Array] = None,
                          g_init: Optional[Array] = None) -> Tuple[Array, Array, Array, float]:
        """
        Solve with exactly num_iters iterations
        
        Args:
            num_iters: Exact number of iterations to perform
            f_init: Initial log-potential for rows (n,), default zeros
            g_init: Initial log-potential for columns (m,), default zeros
            
        Returns:
            pi: Transport coupling (n, m)
            f: Final log-potential for rows (n,)
            g: Final log-potential for columns (m,)
            error: Final marginal constraint violation
            
        Note:
            Unlike solve(), this method does NOT return iteration count
            since it always performs exactly num_iters iterations.
        """
        n, m = self.C.shape
        
        # Initialize log-potentials
        f = jnp.zeros(n) if f_init is None else f_init
        g = jnp.zeros(m) if g_init is None else g_init

        def body_fn(_, carry):
            """Single iteration: just update f and g."""
            f_c, g_c = carry
            f_new, g_new = _sinkhorn_log_step(
                f_c, g_c, self.C, self.a, self.b, self.epsilon
            )
            return (f_new, g_new)

        # Run fixed number of iterations
        f_final, g_final = lax.fori_loop(0, num_iters, body_fn, (f, g))
        
        # Compute final coupling and error
        pi_final = _compute_coupling_from_log(f_final, g_final, self.C, self.epsilon)
        error = _marginal_error(pi_final, self.a, self.b)
        
        return pi_final, f_final, g_final, error


@flstr.dataclass
class SinkhornJKO:
    """
    JKO (Jordan-Kinderlehrer-Otto) scheme using entropy-regularized OT.
    
    Computes discrete-time Wasserstein gradient flow for a functional F:
        rho_{k+1} = argmin_rho { F(rho) + (1/2*tau)W₂²(rho, rho_k) }
    
    where W₂ is approximated by Sinkhorn with regularization ε.
    
    Attributes:
        C: Cost matrix (n, n) - squared distances on domain
        rho0: Initial distribution (n,) - starting condition
        eta: JKO timestep - controls temporal discretization
        epsilon: Sinkhorn regularization - controls W₂ approximation quality
        F_func: Optional functional F to minimize (not yet implemented)
        inner_steps: Max Sinkhorn iterations per JKO step
        tol: Convergence tolerance for inner Sinkhorn solves
    """
    C: Array  # (n, n) cost matrix
    rho0: Array  # (n,) initial distribution
    eta: float  # JKO timestep
    epsilon: float  # Sinkhorn regularization
    F_func: Optional[Callable[[Array], float]] = flstr.field(
        pytree_node=False, default=None
    )
    inner_steps: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)

    @jax.jit
    def take_step(self,
                  rho_k: Array,
                  f_ws: Optional[Array] = None,
                  g_ws: Optional[Array] = None) -> Tuple[Array, Array, Array, float]:
        """
        Take a single JKO step: ρ_{k+1} from ρ_k.
        
        For the basic case (F=0), this solves:
            min_ρ W²_ε(ρ, ρ_k)  s.t. Σρ = 1, ρ ≥ 0
        
        which is equivalent to Sinkhorn OT from ρ_k to ρ_k (identity mapping).
        
        Args:
            rho_k: Current distribution (n,)
            f_ws: Warm-start log-potential for rows (n,), optional
            g_ws: Warm-start log-potential for columns (n,), optional
            
        Returns:
            rho_next: Next distribution (n,) - normalized to sum to 1
            f: Final log-potential from Sinkhorn solve
            g: Final log-potential from Sinkhorn solve
            error: Sinkhorn convergence error
            
        Note:
            The warm-start potentials f_ws, g_ws from previous JKO step
            significantly reduce the number of Sinkhorn iterations needed.
        """
        # Create Sinkhorn solver for this JKO step
        # Note: Using rho_k for both source and target in simple case
        solver = Sinkhorn(
            C=self.C,
            a=rho_k,
            b=rho_k,
            epsilon=self.epsilon,
            max_iters=self.inner_steps,
            tol=self.tol
        )
        
        # Solve inner OT problem
        pi, f, g, error, _ = solver.solve(f_init=f_ws, g_init=g_ws)
        
        # Extract next distribution as row marginal
        rho_next = _row_sums(pi)
        
        # Normalize to ensure exact mass conservation
        rho_next = rho_next / jnp.sum(rho_next)
        
        return rho_next, f, g, error

    def compute_flow(self, num_steps: int) -> Tuple[Array, Array]:
        """
        Compute full JKO flow trajectory over num_steps.
        
        Uses lax.scan for efficient computation with warm-starting between steps.
        
        Args:
            num_steps: Number of JKO timesteps to compute
            
        Returns:
            rhos: Full trajectory (num_steps+1, n)
                - rhos[0] is initial condition rho0
                - rhos[k] is distribution at time t = k*eta
            errors: Sinkhorn convergence errors (num_steps,)
                - errors[k] is error for JKO step k→k+1
                - Should all be < tol if Sinkhorn converged
                
        Example:
            >>> rhos, errors = jko.compute_flow(100)
            >>> converged = jnp.all(errors < 1e-9)
            >>> if converged:
            ...     print("All Sinkhorn solves converged!")
            >>> # Analyze flow: entropy, cost, etc.
        """
        n = self.C.shape[0]

        def one_step(carry, _):
            """Execute one JKO step with warm-starting."""
            rho_k, f_ws, g_ws = carry
            
            # Take JKO step with warm start
            rho_next, f_new, g_new, error = self.take_step(rho_k, f_ws, g_ws)
            
            # Prepare next carry
            next_carry = (rho_next, f_new, g_new)
            
            # Output this step's results
            out = (rho_next, error)
            
            return next_carry, out

        # Initialize with zeros for log-potentials
        init_carry = (self.rho0, jnp.zeros(n), jnp.zeros(n))
        
        # Scan over all JKO steps
        _, outs = lax.scan(one_step, init_carry, xs=None, length=num_steps)
        rhos_steps, errors = outs
        
        # Prepend initial condition
        rhos = jnp.concatenate([self.rho0[None, :], rhos_steps], axis=0)
        
        return rhos, errors
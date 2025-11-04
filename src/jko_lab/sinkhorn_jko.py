"""
Sinkhorn algorithm for entropy-regularized optimal transport

Upon discretization, entropy regularized optimal transport solves the following problem:
    min_{pi >= 0} <C, pi> + epsilon * H(pi)
    such that pi @ 1 = a, and pi.T @ 1 = b

where H(pi) = sum_{i,j} pi_{i,j} (log(pi_{i,j}) - 1) is the Shannon entropy of pi.
"""

from __future__ import annotations
from functools import partial
from typing import Callable, Optional, Tuple
import flax.struct as flstr
import jax
import jax.numpy as jnp
from jax import lax
import optax
from jax.scipy.special import logsumexp

Array = jnp.ndarray


"""
Sinkhorn-based JKO scheme with proper warm-starting across JKO steps

Key improvements:
1. Proper warm-starting of dual potentials across JKO steps
2. Diagnostic mode to track Sinkhorn convergence
3. Adaptive Sinkhorn iterations based on warm-start quality
4. Memory-efficient implementation
"""

# ==================== Sinkhorn Utilities ====================

def _sinkhorn_log_step(f: Array, g: Array, C: Array, a: Array, b: Array, 
                       epsilon: float) -> Tuple[Array, Array]:
    """Single Sinkhorn iteration in log-stabilized form."""
    f_new = epsilon * jnp.log(a + 1e-300) - epsilon * logsumexp(
        (g[None, :] - C) / epsilon, axis=1
    )
    g_new = epsilon * jnp.log(b + 1e-300) - epsilon * logsumexp(
        (f_new[:, None] - C) / epsilon, axis=0
    )
    return f_new, g_new


def _compute_coupling_from_log(f: Array, g: Array, C: Array, epsilon: float) -> Array:
    """Recover coupling from log-potentials."""
    return jnp.exp((f[:, None] + g[None, :] - C) / epsilon)


def row_sums(pi: Array) -> Array:
    return pi.sum(axis=1)


def col_sums(pi: Array) -> Array:
    return pi.sum(axis=0)


def marginal_error(pi: Array, a: Array, b: Array) -> float:
    """Compute marginal constraint violation (infinity norm)."""
    error_a = jnp.abs(row_sums(pi) - a).max()
    error_b = jnp.abs(col_sums(pi) - b).max()
    return jnp.maximum(error_a, error_b)


@flstr.dataclass
class Sinkhorn:
    """
    Sinkhorn solver with adaptive iterations based on warm-start quality.
    """
    C: Array
    a: Array
    b: Array
    epsilon: float
    max_iters: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)
    
    def solve(self, 
              f_init: Optional[Array] = None,
              g_init: Optional[Array] = None) -> Tuple[Array, Array, Array, float, int]:
        """
        Solve with adaptive early stopping.
        Returns: (pi, f, g, error, iterations)
        """
        n, m = self.C.shape
        
        f = jnp.zeros(n) if f_init is None else f_init
        g = jnp.zeros(m) if g_init is None else g_init
        
        # Check initial error for diagnostics
        pi_init = _compute_coupling_from_log(f, g, self.C, self.epsilon)
        error_init = marginal_error(pi_init, self.a, self.b)
        
        def body_fn(carry):
            f_c, g_c, _, iteration = carry
            f_new, g_new = _sinkhorn_log_step(
                f_c, g_c, self.C, self.a, self.b, self.epsilon
            )
            pi = _compute_coupling_from_log(f_new, g_new, self.C, self.epsilon)
            error = marginal_error(pi, self.a, self.b)
            return (f_new, g_new, error, iteration + 1)
        
        def cond_fn(carry):
            _, _, error, iteration = carry
            return (error > self.tol) & (iteration < self.max_iters)
        
        init_carry = (f, g, error_init, 0)
        f_final, g_final, error_final, iters = lax.while_loop(
            cond_fn, body_fn, init_carry
        )
        
        pi_final = _compute_coupling_from_log(f_final, g_final, self.C, self.epsilon)
        return pi_final, f_final, g_final, error_final, iters


# ==================== JKO Solver ====================

@flstr.dataclass
class SinkhornJKO:
    """
    JKO scheme with gradient descent and proper warm-starting.
    
    Minimizes: Ï^{k+1} = argmin { F[Ï] + (1/2Ï„) WÂ²(Ï, Ï^k) }
    
    Key feature: Carries dual potentials (f, g) across JKO steps for fast Sinkhorn convergence.
    """
    C: Array
    rho0: Array
    eta: float  # JKO time step (tau in formulation)
    epsilon: float
    F_func: Optional[Callable[[Array], float]] = flstr.field(
        pytree_node=False, default=None
    )
    inner_steps: int = flstr.field(pytree_node=False, default=20)
    sinkhorn_iters: int = flstr.field(pytree_node=False, default=500)
    tol: float = flstr.field(pytree_node=False, default=1e-9)
    learning_rate: float = flstr.field(pytree_node=False, default=0.01)
    optimizer_name: str = flstr.field(pytree_node=False, default='sgd')
    verbose: bool = flstr.field(pytree_node=False, default=False)
    
    def compute_W2_gradient(
        self,
        rho: Array,
        rho_k: Array,
        f_ws: Optional[Array] = None,
        g_ws: Optional[Array] = None
    ) -> Tuple[Array, Array, Array, int]:
        """
        Compute âˆ‡WÂ² using Sinkhorn dual potentials.
        
        Returns: (gradient, f_new, g_new, sinkhorn_iters)
        """
        solver = Sinkhorn(
            C=self.C,
            a=rho,
            b=rho_k,
            epsilon=self.epsilon,
            max_iters=self.sinkhorn_iters,
            tol=self.tol
        )
        
        _, f_new, g_new, error, iters = solver.solve(f_init=f_ws, g_init=g_ws)
        grad_W2 = 2.0 * f_new
        
        return grad_W2, f_new, g_new, iters
    
    def take_step(
        self,
        rho_k: Array,
        f_ws: Optional[Array] = None,
        g_ws: Optional[Array] = None
    ) -> Tuple[Array, Array, Array, dict]:
        """
        Take one JKO step with diagnostic info.
        
        Returns: (rho_next, f_final, g_final, diagnostics)
        """
        n = len(rho_k)
        
        # Initialize from previous distribution
        sigma_init = jnp.log(rho_k + 1e-10)
        
        # Setup optimizer
        if self.optimizer_name == 'adam':
            optimizer = optax.adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            optimizer = optax.sgd(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        opt_state = optimizer.init(sigma_init)
        
        # Initialize potentials (warm-start)
        f_current = jnp.zeros(n) if f_ws is None else f_ws
        g_current = jnp.zeros(n) if g_ws is None else g_ws
        
        # Gradient function for F
        if self.F_func is not None:
            grad_F_fn = jax.grad(self.F_func)
        else:
            grad_F_fn = lambda rho: jnp.zeros_like(rho)
        
        # Track Sinkhorn iterations for single JKO step
        total_sinkhorn_iters = jnp.array(0)
        
        def sgd_step(state, _):
            sigma, opt_s, f_pot, g_pot, sink_count = state
            rho = jax.nn.softmax(sigma)
            
            # Gradient computation
            grad_F_rho = grad_F_fn(rho)
            grad_W2_rho, f_new, g_new, sink_iters = self.compute_W2_gradient(
                rho, rho_k, f_pot, g_pot
            )
            
            # Combined gradient in rho-space
            grad_rho = grad_F_rho + grad_W2_rho / (2.0 * self.eta)
            
            # Convert to sigma-space via Jacobian of softmax
            grad_dot_rho = jnp.sum(grad_rho * rho)
            grad_sigma = rho * (grad_rho - grad_dot_rho)
            
            # Update
            updates, opt_s_new = optimizer.update(grad_sigma, opt_s)
            sigma_new = optax.apply_updates(sigma, updates)
            
            new_state = (sigma_new, opt_s_new, f_new, g_new, sink_count + sink_iters)
            return new_state, sink_iters  # ← Output per-step Sinkhorn iters
        
        # Run gradient descent
        init_state = (sigma_init, opt_state, f_current, g_current, total_sinkhorn_iters)
        final_state, total_sink_iters = lax.scan(
            sgd_step, init_state, xs=None, length=self.inner_steps
        )
        
        sigma_final, _, f_final, g_final, total_sink_iters = final_state
        rho_next = jax.nn.softmax(sigma_final)
        
        return rho_next, f_final, g_final, total_sink_iters
    
    @partial(jax.jit, static_argnames=['num_steps'])
    def compute_flow(self,
                     num_steps: int,
                     f_init: Optional[Array] = None,
                     g_init: Optional[Array] = None,
        ) -> Tuple[Array, dict]:
        """
        Compute JKO flow trajectory
        
        Args:
            num_steps: Number of JKO steps
            f_init: Initial f potential for warm-starting (optional)
            g_init: Initial g potential for warm-starting (optional)
            
        Returns:
            rhos: Trajectory (num_steps+1, n) including rho0
            diagnostics:
                - sinkhorn_iters_per_jko_step: number of sinkhorn iterations used in a single JKO step
                - f_final: Final f potential
                - g_final: Final g potential
        """
        n = self.C.shape[0]

        def one_step(carry, _):
            rho_k, u_ws, v_ws = carry
            rho_next, u_new, v_new, sinkhorn_iters_per_JKO = self.take_step(rho_k, u_ws, v_ws)
            next_carry = (rho_next, u_new, v_new)
            # Return diagnostics as-is for collection
            out = (rho_next, sinkhorn_iters_per_JKO)
            return next_carry, out

        # Initialize with provided potentials or default to zeros
        u = jnp.zeros(n) if f_init is None else f_init
        v = jnp.zeros(n) if g_init is None else g_init
        init_carry = (self.rho0, u, v)
        
        # Scan over JKO steps - CAPTURE final_carry!
        final_carry, outs = lax.scan(one_step, init_carry, xs=None, length=num_steps)
        rhos_steps, sinkhorn_iters_per_JKO = outs
        
        # Extract final potentials from carry
        _, u_final, v_final = final_carry
        
        # Prepend initial rho0
        rhos = jnp.concatenate([self.rho0[None, :], rhos_steps], axis=0)
        
        
        # Return diagnostics including final potentials
        diagnostics = {
            # 'sinkhorn_iters_per_jko_step': step_diagnostics,
            'sinkhorn_iters_per_jko_step': sinkhorn_iters_per_JKO,
            'f_final': u_final,  # Return final f potential
            'g_final': v_final   # Return final g potential
        }
        
        return rhos, diagnostics
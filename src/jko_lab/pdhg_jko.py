"""
PDHG for a single JKO step (discrete case) in JAX, with an auxiliary
variable rho = row_sums(pi) to simplify the updates.

Problem (discrete):
    min_{pi >= 0,  pi^T 1 = rho_k}  eta * F( rho ) + 1/2 <C, pi> ,  where rho = pi 1.

We implement PDHG on the equivalent saddle form with two linear maps:
  K1(pi) = row_sums(pi) = rho,   K2(pi) = col_sums(pi) = nu,
  g1(rho) = eta * F(rho),        g2(nu) = iota_{nu = rho_k}.

Dual variables: v for g1^* (size n), u for g2^* (size n).
Primal variable: pi (size n x n). We also carry rho explicitly.

Updates (theta=1):
  # Duals (using extrapolated bar variables)
  u^{k+1} = u^k - sigma * (rho_k - col_sums( bar_pi ))
  v^{k+1} = prox_{sigma * eta * F^*}( v^k + sigma * bar_rho )
          = s - sigma * prox_{(eta/sigma) * F}( s / sigma ),  where s = v^k + sigma * bar_rho

  # Primal (nonnegativity via hinge)
  pi^{k+1} = max(0, pi^k - tau * ( 0.5*C - (v^{k+1}[:,None] + u^{k+1}[None,:]) ))

  # Auxiliary
  rho^{k+1} = row_sums( pi^{k+1} )
  bar_pi = pi^{k+1} + (pi^{k+1} - pi^k)    (theta=1)
  bar_rho = rho^{k+1} + (rho^{k+1} - rho^k)

Stepsizes must satisfy: tau * sigma * ||K||^2 < 1. We estimate ||K|| via
power iteration on K^*K (see _estimate_operator_norm).

You can plug different F via a prox_F function that computes
  prox_{alpha * F}(z).
We also provide two examples: quadratic and (negative) entropy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
from functools import partial
import flax.struct as flstr

import jax
import jax.numpy as jnp
from jax import lax
from .lambert import lambertw

Array = jnp.ndarray

@dataclass
class PDHGState:
    pi: Array          # (n, n)
    rho: Array         # (n,)
    u: Array           # (n,)
    v: Array           # (n,)
    pi_prev: Array     # (n, n)
    rho_prev: Array    # (n,)

    primal_obj: Array = 0.0          # (T,)
    feas_row: Array = jnp.inf            # (T,)  = ||rho - row_sums(pi)|| (should be 0 numerically)
    feas_col: Array = jnp.inf           # (T,)  = ||col_sums(pi) - rho_k||

# -------------------- Prox helpers for common F -----------------------------

def proxF_quadratic(z: Array, alpha: float, b: Array, lam: float) -> Array:
    """prox_{alpha * (lam/2)||x - b||^2}(z) = (z + alpha*lam*b) / (1 + alpha*lam).
    Args:
      z: (n,)
      alpha: scalar >= 0
      b: (n,)
      lam: scalar >= 0
    """
    denom = 1.0 + alpha * lam
    return (z + (alpha * lam) * b) / denom


def proxF_entropy(z: Array, alpha: float, eps: float = 0.0) -> Array:
    """Prox for F(x) = sum_i ( x_i log x_i - x_i ) with domain x_i >= 0.
    Closed-form per-coo rdinate using the Lambert W:
        prox_{alpha*F}(z)_i solves: x - z + alpha * log x = 0  => x = alpha * W( exp(z/alpha) / alpha )
    We return the real branch and clamp to nonnegative.
    eps: small nonnegativity clamp if desired.
    """
    alpha = jnp.asarray(alpha)
    # Handle alpha=0: prox reduces to identity
    def _prox_one(z_i):
        return jnp.where(
            alpha > 0,
            (alpha * lambertw(jnp.exp(z_i / alpha) / alpha).real).astype(z.dtype),
            z_i,
        )
    x = jax.vmap(_prox_one)(z)
    if eps > 0:
        x = jnp.maximum(x, eps)
    return x

# --------------------- Operator norm estimation -----------------------------

def _K_apply(pi: Array) -> Tuple[Array, Array]:
    """K(pi) = (rho, nu) = (row_sums(pi), col_sums(pi))."""
    return _row_sums(pi), _col_sums(pi)


def _KT_apply(rho: Array, nu: Array) -> Array:
    """K^*(rho, nu) = broadcast_add(rho[:,None], nu[None,:])."""
    return rho[:, None] + nu[None, :]


def _estimate_operator_norm(n: int, iters: int = 20) -> Array:
    """Power iteration to estimate ||K||_2 for K: R^{n*n} -> R^{2n}.
    Works in the Frobenius/Euclidean norms used by PDHG.
    Returns an overestimate with a small safety factor.
    """
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, n))

    def body(_, x_curr):
        rho, nu = _K_apply(x_curr)
        x_next = _KT_apply(rho, nu)
        norm = jnp.linalg.norm(x_next)
        # Avoid division by zero
        x_next = jnp.where(norm > 0, x_next / norm, x_next)
        return x_next

    x = lax.fori_loop(0, iters, body, x)
    # One more application to get Rayleigh quotient
    rho, nu = _K_apply(x)
    Kx_norm = jnp.sqrt(jnp.linalg.norm(rho) ** 2 + jnp.linalg.norm(nu) ** 2)
    x_norm = jnp.linalg.norm(x)
    est = jnp.where(x_norm > 0, Kx_norm / x_norm, 0.0)
    # small safety margin
    return est * 1.05


def _row_sums(pi: Array) -> Array:
    return pi.sum(axis=1)

def _col_sums(pi: Array) -> Array:
    return pi.sum(axis=0)

def _choose_stepsizes(n: int) -> Tuple[float, float]:
    K = _estimate_operator_norm(n, iters=25)
    t = 0.9 / (K + 1e-12)
    return t, t


@flstr.dataclass
class PrimalDualJKO:
    C: Array # (n, n) squared-distance matrix
    rho0: Array # (n,) given probability vector (nonnegative, sums to 1)
    eta: float # positive JKO step coefficient
    reg: float

    # Static/non-traced fields (config, Python callables, etc.)
    proxF: Callable[[Array, float], Array] | Callable[..., Array] = flstr.field(pytree_node=False) # Proximal of the functional
    tau: Optional[float] = flstr.field(pytree_node=False, default=None)
    sigma: Optional[float] = flstr.field(pytree_node=False, default=None)
    inner_steps: int = flstr.field(pytree_node=False, default=1000) # inner PDHG iterations


    # ----------------- one JKO step via PDHG -----------------

    @jax.jit
    def take_step(self, rho_k: Array,
                  tau: float,
                  sigma: float,
                  reg: float,
                  pi_ws: Optional[Array] = None,
                  u_ws: Optional[Array] = None,
                  v_ws: Optional[Array] = None) -> Tuple[Array, Array, Array, Array]:
        """
        Run a single PDHG JKO step from rho_k with warm-starts (pi_ws, u_ws, v_ws).
        Returns (rho_next, pi, u, v).
        """

        C = self.C
        n = C.shape[0]

        # Initial state
        pi0 = jnp.zeros((n, n)) if pi_ws is None else pi_ws
        u0 = jnp.zeros((n,)) if u_ws is None else u_ws
        v0 = jnp.zeros((n,)) if v_ws is None else v_ws
        rho0 = _row_sums(pi0)

        def body_fun(_, carry):
            pi_c, rho_c, u_c, v_c = carry

            # Primal (pi)
            term = 0.5 * C + u_c[:, None] + v_c[None, :]
            pi_new = jnp.maximum((pi_c - tau * term)/(1 + tau*reg), 0.0)

            # Primal (rho) via prox of eta * F
            rho_new = self.proxF(rho_c + tau * u_c, self.eta * tau)

            # Over-relax
            pi_bar = 2.0 * pi_new - pi_c
            rho_bar = 2.0 * rho_new - rho_c

            # Dual updates
            u_new = u_c + sigma * (_row_sums(pi_bar) - rho_bar)
            v_new = v_c + sigma * (_col_sums(pi_bar) - rho_k)
            return (pi_new, rho_new, u_new, v_new)

        # run PDHG inner loop
        pi_f, rho_f, u_f, v_f = lax.fori_loop(0, self.inner_steps, body_fun, (pi0, rho0, u0, v0))

        # JKO normalization safeguard
        rho_next = rho_f / (jnp.sum(rho_f))

        return rho_next, pi_f, u_f, v_f

    # ----------------- full JKO flow (fully JIT) -----------------

    def compute_flow(self, num_steps: int) -> Tuple[Array, Array]:
        """
        Run num_iters JKO steps starting from self.rho0 with warm-starts.
        Returns:
            rhos: (num_iters+1, n) trajectory including rho0
            cols: (num_iters,) column-marginal residuals ||col_sums(pi_t) - rho_{t-1}||_2
        """

        n = self.C.shape[0]

        tau = self.tau
        sigma = self.sigma
        if tau is None or sigma is None:
            tau, sigma = _choose_stepsizes(num_steps)

        def one_step(carry, _):
            rho_k, pi_ws, u_ws, v_ws = carry
            rho_next, pi_f, u_f, v_f = self.take_step(rho_k, tau, sigma, self.reg, pi_ws, u_ws, v_ws)
            col_resid = jnp.linalg.norm(_col_sums(pi_f) - rho_k, ord=2)
            next_carry = (rho_next, pi_f, u_f, v_f)
            out = (rho_next, col_resid)
            return next_carry, out

        # scan over outer JKO steps
        init_carry = (self.rho0, jnp.zeros((n, n)), jnp.zeros((n,)), jnp.zeros((n,)))
        (final_carry, outs) = lax.scan(one_step, init_carry, xs=None, length=num_steps)
        rhos_steps, cols_steps = outs  # shapes: (num_iters, n), (num_iters,)

        # prepend initial rho0
        rhos = jnp.concatenate([self.rho0[None, :], rhos_steps], axis=0)
        return rhos, cols_steps
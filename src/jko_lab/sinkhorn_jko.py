"""
Sinkhorn algorithm for entropy-regularized optimal transport

Upon discretization, entropy regularized optimal transport solves the following problem:
    min_{pi >= 0} <C, pi> + epsilon * H(pi)
    such that pi @ 1 = a, and pi.T @ 1 = b

where H(pi) = sum_{i,j} pi_{i,j} (log(pi_{i,j}) - 1) is the Shannon entropy of pi.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import flax.struct as flstr
import jax
import jax.numpy as jnp
from jax import lax
import optax
import time
from jax.scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

Array = jnp.ndarray
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
HEATMAP_CMAP = 'YlOrRd'
TRANSPORT_CMAP = 'Blues'


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
    
    Minimizes: ρ^{k+1} = argmin { F[ρ] + (1/2τ) W²(ρ, ρ^k) }
    
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
        Compute ∇W² using Sinkhorn dual potentials.
        
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
        elif self.optimizer_name == 'rmsprop':
            optimizer = optax.rmsprop(learning_rate=self.learning_rate)
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
        
        # Track Sinkhorn iterations for diagnostics
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
        final_state, sinkhorn_iters_per_step = lax.scan(
            sgd_step, init_state, xs=None, length=self.inner_steps
        )
        
        sigma_final, _, f_final, g_final, total_sink_iters = final_state
        rho_next = jax.nn.softmax(sigma_final)
        
        # Diagnostics (keep as JAX arrays for JIT compatibility)
        diagnostics = {
            'total_sinkhorn_iters': total_sink_iters,
            'avg_sinkhorn_iters': jnp.mean(sinkhorn_iters_per_step),
            'first_sinkhorn_iters': sinkhorn_iters_per_step[0],
            'last_sinkhorn_iters': sinkhorn_iters_per_step[-1]
        }
        
        return rho_next, f_final, g_final, diagnostics
    
    def compute_flow(self, num_steps: int) -> Tuple[Array, dict]:
        """
        Compute JKO flow with warm-starting across steps.
        
        Returns: (rhos, diagnostics)
        """
        n = self.C.shape[0]
        
        def one_step(carry, step_idx):
            rho_k, f_ws, g_ws = carry
            
            # Take JKO step with warm-start
            rho_next, f_new, g_new, diag = self.take_step(rho_k, f_ws, g_ws)
            
            next_carry = (rho_next, f_new, g_new)
            out = (rho_next, diag['total_sinkhorn_iters'])
            
            return next_carry, out
        
        # Initialize with zero potentials for first step
        init_carry = (self.rho0, jnp.zeros(n), jnp.zeros(n))
        
        # Scan over JKO steps
        _, outs = lax.scan(one_step, init_carry, xs=jnp.arange(num_steps))
        rhos_steps, sinkhorn_iters_per_jko = outs
        
        # Prepend initial condition
        rhos = jnp.concatenate([self.rho0[None, :], rhos_steps], axis=0)
        
        diagnostics = {
            'sinkhorn_iters_per_jko_step': sinkhorn_iters_per_jko,
            'total_sinkhorn_iters': jnp.sum(sinkhorn_iters_per_jko),
            'avg_sinkhorn_per_jko': jnp.mean(sinkhorn_iters_per_jko)
        }
        
        return rhos, diagnostics

def plot_marginal(
    marginal: Array,
    x: Optional[Array] = None,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a single probability vector (marginal distribution).
    
    Args:
        marginal: Probability vector (n,)
        x: Domain points (n,). If None, uses indices 0, 1, ..., n-1
        label: Label for the plot legend
        ax: Matplotlib axes. If None, creates new figure
        color: Line color. If None, uses default color cycle
        **kwargs: Additional arguments passed to plt.plot
        
    Returns:
        Matplotlib axes object
    """
    marginal = np.asarray(marginal)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    if x is None:
        x = np.arange(len(marginal))
    else:
        x = np.asarray(x)
    
    # Plot
    plot_kwargs = {'linewidth': 2, 'alpha': 0.8}
    plot_kwargs.update(kwargs)
    
    if color is not None:
        plot_kwargs['color'] = color
    
    ax.plot(x, marginal, label=label, **plot_kwargs)
    ax.fill_between(x, marginal, alpha=0.3, color=plot_kwargs.get('color', None))
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if label is not None:
        ax.legend(fontsize=10)
    
    ax.set_ylim(bottom=0)
    
    return ax


def plot_marginals(
    marginals: List[Array],
    x: Optional[Array] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot multiple marginal distributions on the same axes.
    
    Args:
        marginals: List of probability vectors
        x: Domain points. If None, uses indices
        labels: List of labels for each marginal
        title: Plot title
        figsize: Figure size (width, height)
        colors: List of colors for each marginal
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f'Marginal {i+1}' for i in range(len(marginals))]
    
    if colors is None:
        colors = COLORS[:len(marginals)]
    
    for i, (marginal, label, color) in enumerate(zip(marginals, labels, colors)):
        plot_marginal(marginal, x=x, label=label, ax=ax, color=color)
    
    if title is not None:
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    return fig


def plot_cost_matrix(
    C: Array,
    title: str = 'Cost Matrix',
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = HEATMAP_CMAP,
    log_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True
) -> plt.Figure:
    """
    Visualize a cost matrix as a heatmap.
    
    Args:
        C: Cost matrix (n, m)
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap name
        log_scale: If True, use logarithmic color scale
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        show_colorbar: Whether to show colorbar
        
    Returns:
        Matplotlib figure object
    """
    C = np.asarray(C)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color scale
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    
    im = ax.imshow(
        C,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        origin='lower',
        norm=norm,
        vmin=vmin,
        vmax=vmax
    )
    
    ax.set_xlabel('Target index', fontsize=11)
    ax.set_ylabel('Source index', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cost', fontsize=11)
    
    plt.tight_layout()
    
    return fig


def plot_transport_plan(
    pi: Array,
    a: Optional[Array] = None,
    b: Optional[Array] = None,
    x_source: Optional[Array] = None,
    x_target: Optional[Array] = None,
    title: str = 'Optimal Transport Plan',
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = TRANSPORT_CMAP,
    show_marginals: bool = True,
    log_scale: bool = False
) -> plt.Figure:
    """
    Plot optimal transport plan with marginals on the sides.
    
    Creates a figure with the transport plan in the center and
    marginal distributions on the top and right sides.
    
    Args:
        pi: Transport plan matrix (n, m)
        a: Source marginal (n,). If None, computed from pi
        b: Target marginal (m,). If None, computed from pi
        x_source: Source domain points (n,)
        x_target: Target domain points (m,)
        title: Main plot title
        figsize: Figure size (width, height)
        cmap: Colormap for transport plan
        show_marginals: Whether to show marginal plots
        log_scale: If True, use log scale for transport plan colors
        
    Returns:
        Matplotlib figure object
    """
    pi = np.asarray(pi)
    
    if a is None:
        a = pi.sum(axis=1)
    else:
        a = np.asarray(a)
    
    if b is None:
        b = pi.sum(axis=0)
    else:
        b = np.asarray(b)
    
    n, m = pi.shape
    
    if x_source is None:
        x_source = np.arange(n)
    else:
        x_source = np.asarray(x_source)
    
    if x_target is None:
        x_target = np.arange(m)
    else:
        x_target = np.asarray(x_target)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    
    if show_marginals:
        gs = GridSpec(3, 3, figure=fig, 
                     width_ratios=[1, 4, 0.2],
                     height_ratios=[1, 4, 0.2],
                     hspace=0.05, wspace=0.05)
        
        ax_main = fig.add_subplot(gs[1, 1])
        ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 0], sharey=ax_main)
        ax_cbar = fig.add_subplot(gs[1, 2])
    else:
        gs = GridSpec(1, 2, figure=fig,
                     width_ratios=[10, 0.5],
                     wspace=0.05)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
    
    # Plot main transport plan
    norm = LogNorm(vmin=pi[pi > 0].min(), vmax=pi.max()) if log_scale and pi.max() > 0 else None
    
    extent = [x_target.min(), x_target.max(), x_source.min(), x_source.max()]
    im = ax_main.imshow(
        pi,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        origin='lower',
        extent=extent,
        norm=norm
    )
    
    ax_main.set_xlabel('Target (x)', fontsize=11)
    ax_main.set_ylabel('Source (y)', fontsize=11)
    ax_main.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Transport mass', fontsize=11)
    
    if show_marginals:
        # Plot marginals
        # Top: target marginal b
        ax_top.plot(x_target, b, color=COLORS[1], linewidth=2, label='Target')
        ax_top.fill_between(x_target, b, alpha=0.3, color=COLORS[1])
        ax_top.set_ylabel('Target\nmarginal', fontsize=10)
        ax_top.set_ylim(bottom=0)
        ax_top.grid(True, alpha=0.3, linestyle='--')
        ax_top.tick_params(labelbottom=False)
        
        # Right: source marginal a (rotated)
        ax_right.plot(a, x_source, color=COLORS[0], linewidth=2, label='Source')
        ax_right.fill_betweenx(x_source, a, alpha=0.3, color=COLORS[0])
        ax_right.set_xlabel('Source\nmarginal', fontsize=10)
        ax_right.set_xlim(left=0)
        ax_right.grid(True, alpha=0.3, linestyle='--')
        ax_right.tick_params(labelleft=False)
        ax_right.invert_xaxis()
    
    return fig

def plot_convergence(
    errors: Array,
    title: str = 'Convergence History',
    figsize: Tuple[int, int] = (10, 5),
    log_scale: bool = True,
    threshold: Optional[float] = None
) -> plt.Figure:
    """
    Plot convergence of error metrics over iterations.
    
    Args:
        errors: Array of error values (num_iters,)
        title: Plot title
        figsize: Figure size
        log_scale: Whether to use log scale for y-axis
        threshold: Optional horizontal line showing convergence threshold
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> _, errors = solver.compute_flow(num_steps=100)
        >>> plot_convergence(errors, threshold=1e-8)
    """
    errors = np.asarray(errors)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(len(errors))
    ax.plot(iterations, errors, linewidth=2, color=COLORS[0], marker='o', 
            markersize=4, markevery=max(1, len(errors) // 20))
    
    if log_scale:
        ax.set_yscale('log')
    
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                  label=f'Threshold = {threshold:.2e}')
        ax.legend(fontsize=10)
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig


def plot_sinkhorn_summary(
    C: Array,
    a: Array,
    b: Array,
    pi: Array,
    x_source: Optional[Array] = None,
    x_target: Optional[Array] = None,
    figsize: Tuple[int, int] = (16, 5)
) -> plt.Figure:
    """
    Create a comprehensive summary plot for Sinkhorn algorithm.
    
    Shows: marginals, cost matrix, and the corresponding optimal transport plan.
    
    Args:
        C: Cost matrix (n, m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        pi: Optimal transport plan (n, m)
        x_source: Source domain points
        x_target: Target domain points
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    if x_source is None:
        x_source = np.arange(n)
    if x_target is None:
        x_target = np.arange(m)
    axes[0].plot(x_source, a, color=COLORS[0], linewidth=2, label='Source (a)')
    axes[0].fill_between(x_source, a, alpha=0.3, color=COLORS[0])
    axes[0].plot(x_target, b, color=COLORS[1], linewidth=2, label='Target (b)')
    axes[0].fill_between(x_target, b, alpha=0.3, color=COLORS[1])
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Marginals', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim(bottom=0)
    
    C = np.asarray(C)
    im1 = axes[1].imshow(C, cmap=HEATMAP_CMAP, aspect='auto', 
                         interpolation='nearest', origin='lower')
    axes[1].set_xlabel('Y', fontsize=11)
    axes[1].set_ylabel('X', fontsize=11)
    axes[1].set_title('Cost Matrix C', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    pi = np.asarray(pi)
    im2 = axes[2].imshow(pi, cmap=TRANSPORT_CMAP, aspect='auto',
                         interpolation='nearest', origin='lower')
    axes[2].set_xlabel('Y', fontsize=11)
    axes[2].set_ylabel('X', fontsize=11)
    axes[2].set_title('Optimal Transport Plan', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig
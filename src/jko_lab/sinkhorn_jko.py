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


def _compute_jko_functional_value(
    sigma: Array,
    rho_k: Array,
    C: Array,
    epsilon: float,
    tau: float,
    V: Optional[Array] = None,
    sinkhorn_iters: int = 500,
    sinkhorn_tol: float = 1e-9
) -> float:
    """
    Compute JKO functional G(σ) = F(softmax(σ)) + (1/2τ) * W²(softmax(σ), ρ_k).
    
    This is the objective function we minimize using gradient descent.
    The softmax parametrization ensures ρ = softmax(σ) stays on the probability simplex.
    
    Args:
        sigma: Unconstrained parameters (n,)
        rho_k: Fixed previous distribution (n,)
        C: Cost matrix (n, n)
        epsilon: Sinkhorn regularization
        tau: JKO time step
        V: Potential function for F(ρ) = <V, ρ> (optional)
        sinkhorn_iters: Maximum iterations for Sinkhorn solver
        sinkhorn_tol: Convergence tolerance for Sinkhorn
        
    Returns:
        G(σ): Scalar functional value to be minimized
    """
    # Map unconstrained σ to probability simplex via softmax
    rho = jax.nn.softmax(sigma)
    
    # Functional term: F(ρ) = <V, ρ>
    F_value = 0.0 if V is None else jnp.sum(V * rho)
    
    # Wasserstein distance term using Sinkhorn class as black box
    W_squared = _compute_wasserstein_squared_from_sinkhorn(
        rho, rho_k, C, epsilon, sinkhorn_iters, sinkhorn_tol
    )
    
    # JKO functional: F(ρ) + (1/2τ) * W²(ρ, ρ_k)
    return F_value + W_squared / (2.0 * tau)

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


def row_sums(pi: Array) -> Array:
    """Compute row sums of coupling matrix: π @ 1."""
    return pi.sum(axis=1)


def col_sums(pi: Array) -> Array:
    """Compute column sums of coupling matrix: π.T @ 1."""
    return pi.sum(axis=0)


def marginal_error(pi: Array, a: Array, b: Array) -> float:
    """
    Compute maximum of errors on marginals.
    
    Args:
        pi: Transport coupling (n, m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        
    Returns:
        error: max(||π@1 - a||_∞, ||π.T@1 - b||_∞)
    """
    error_a = jnp.abs(row_sums(pi) - a).max()
    error_b = jnp.abs(col_sums(pi) - b).max()
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
        C: Cost matrix (n, m)
        a: Source marginal (n,)
        b: Target marginal (m,)
        f: Log-potential for rows (n,)
        g: Log-potential for columns (m,)
        epsilon: Regularization parameter
        max_iters: Maximum number of iterations
        tol: Convergence tolerance for pushforward errors
    """
    C: Array  # (n, m) cost matrix
    a: Array  # (n,) source marginal
    b: Array  # (m,) target marginal
    epsilon: float  # entropic regularization parameter
    max_iters: int = flstr.field(pytree_node=False, default=1000)
    tol: float = flstr.field(pytree_node=False, default=1e-9)
    f: Optional[Array] = flstr.field(pytree_node=False, default=None)  # (n,) log-potential for rows
    g: Optional[Array] = flstr.field(pytree_node=False, default=None)  # (m,) log-potential for cols

    @jax.jit
    def solve(self, 
              f_init: Optional[Array] = None,
              g_init: Optional[Array] = None) -> Tuple[Array, Array, Array, float, int]:
        """
        Solve entropy-regularized OT using log-sum-exp formulation of Sinkhorn iterations.
        
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
            
            # Perform single Sinkhorn step on log potentials f and g
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
            pi = _compute_coupling_from_log(f_new, g_new, self.C, self.epsilon)
            error = _marginal_error(pi, self.a, self.b)
            return (f_new, g_new)

        # Run fixed number of iterations
        f_final, g_final = lax.fori_loop(0, num_iters, body_fn, (f, g))
        
        # Compute final coupling and error
        pi_final = _compute_coupling_from_log(f_final, g_final, self.C, self.epsilon)
        error = _marginal_error(pi_final, self.a, self.b)
        
        return pi_final, f_final, g_final, error


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


@flstr.dataclass
class SinkhornJKO:
    """
    JKO scheme using entropy-regularized OT with gradient descent.
    
    Key insight: Gradient of W² is just the dual potential f from Sinkhorn!
    No need to autodiff through Sinkhorn - only autodiff the functional F.
    """
    C: Array  # (n, n) cost matrix
    rho0: Array  # (n,) initial distribution
    eta: float  # JKO timestep (this is tau in the formulation)
    epsilon: float  # Sinkhorn regularization
    F_func: Optional[Callable[[Array], float]] = flstr.field(
        pytree_node=False, default=None
    )  # Functional F: rho -> scalar (JAX will autodiff ONLY this)
    inner_steps: int = flstr.field(pytree_node=False, default=50)
    sinkhorn_iters: int = flstr.field(pytree_node=False, default=500)
    tol: float = flstr.field(pytree_node=False, default=1e-9)
    learning_rate: float = flstr.field(pytree_node=False, default=0.01)
    optimizer_name: str = flstr.field(pytree_node=False, default='adam')
    
    def compute_W2_gradient(
        self, 
        rho: Array, 
        rho_k: Array,
        f_ws: Optional[Array] = None,
        g_ws: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        """
        Compute gradient of W²_{2,ε}(ρ, ρ_k) w.r.t. ρ using dual potentials.
        
        Key insight: ∇_ρ W²(ρ, ρ_k) = 2f (the first dual potential!)
        
        Args:
            rho: Source distribution (n,)
            rho_k: Target distribution (n,)
            f_ws: Warm-start for first potential (n,)
            g_ws: Warm-start for second potential (n,)
            
        Returns:
            grad_W2: Gradient 2f (n,)
            f_new: Updated first potential (for next warm-start)
            g_new: Updated second potential (for next warm-start)
        """
        # Create Sinkhorn solver
        solver = Sinkhorn(
            C=self.C,
            a=rho,
            b=rho_k,
            epsilon=self.epsilon,
            max_iters=self.sinkhorn_iters,
            tol=self.tol
        )
        
        # Solve Sinkhorn with warm-start (can use regular solve, no autodiff needed!)
        _, f_new, g_new, error, iters = solver.solve(f_init=f_ws, g_init=g_ws)
        
        # The gradient of W²(ρ, ρ_k) w.r.t. ρ is simply 2f!
        grad_W2 = 2.0 * f_new
        
        return grad_W2, f_new, g_new
    
    def take_step(
        self, 
        rho_k: Array,
        f_ws: Optional[Array] = None,
        g_ws: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        """
        Take one JKO step from rho_k using gradient descent.
        
        Gradient computation:
        - F part: Use JAX autodiff
        - W² part: Use dual potential f from Sinkhorn (no autodiff needed!)
        
        Args:
            rho_k: Current distribution (n,)
            f_ws: Warm-start for first potential
            g_ws: Warm-start for second potential
            
        Returns:
            rho_next: Next distribution (n,)
            f_final: Final first potential
            g_final: Final second potential
        """
        n = len(rho_k)
        
        # Initialize sigma in log-space
        sigma_init = jnp.log(rho_k + 1e-10)
        
        # Create optimizer
        if self.optimizer_name == 'adam':
            optimizer = optax.adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            optimizer = optax.sgd(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            optimizer = optax.rmsprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        opt_state = optimizer.init(sigma_init)
        
        # Initialize warm-start potentials
        f_current = jnp.zeros(n) if f_ws is None else f_ws
        g_current = jnp.zeros(n) if g_ws is None else g_ws
        
        # Gradient function for F part only (autodiff only this!)
        if self.F_func is not None:
            grad_F_fn = jax.grad(self.F_func)
        else:
            grad_F_fn = lambda rho: jnp.zeros_like(rho)
        
        # Gradient descent loop
        def sgd_step(state, _):
            """Single gradient descent step."""
            sigma, opt_s, f_pot, g_pot = state
            
            # Convert to probability distribution
            rho = jax.nn.softmax(sigma)
            
            # --- Gradient computation ---
            
            # 1. Gradient of F(ρ) w.r.t. ρ (via autodiff)
            grad_F_rho = grad_F_fn(rho)
            
            # 2. Gradient of W²(ρ, ρ_k) w.r.t. ρ (via dual potential - NO autodiff!)
            grad_W2_rho, f_new, g_new = self.compute_W2_gradient(rho, rho_k, f_pot, g_pot)
            
            # 3. Combine gradients in ρ-space
            grad_rho = grad_F_rho + grad_W2_rho / (2.0 * self.eta)
            
            # 4. Convert to σ-space via Jacobian of softmax
            # ∇_σ g(softmax(σ)) = softmax(σ) ⊙ (∇_ρ g - <∇_ρ g, softmax(σ)>)
            grad_dot_rho = jnp.sum(grad_rho * rho)
            grad_sigma = rho * (grad_rho - grad_dot_rho)
            
            # --- Update parameters ---
            updates, opt_s_new = optimizer.update(grad_sigma, opt_s)
            sigma_new = optax.apply_updates(sigma, updates)
            
            new_state = (sigma_new, opt_s_new, f_new, g_new)
            return new_state, None
        
        # Run gradient descent using scan
        init_state = (sigma_init, opt_state, f_current, g_current)
        final_state, _ = lax.scan(sgd_step, init_state, xs=None, length=self.inner_steps)
        sigma_final, _, f_final, g_final = final_state
        
        # Extract final distribution via softmax
        rho_next = jax.nn.softmax(sigma_final)
        
        return rho_next, f_final, g_final
    
    def compute_flow(self, num_steps: int) -> Array:
        """
        Compute full JKO flow trajectory with warm-starting.
        
        Args:
            num_steps: Number of JKO timesteps to compute
            
        Returns:
            rhos: Full trajectory (num_steps+1, n) including rho0
        """
        n = self.C.shape[0]
        
        def one_step(carry, _):
            """Execute one JKO step with warm-starting."""
            rho_k, f_ws, g_ws = carry
            
            # Take JKO step with warm-start
            rho_next, f_new, g_new = self.take_step(rho_k, f_ws, g_ws)
            
            # Prepare next carry and output
            next_carry = (rho_next, f_new, g_new)
            out = rho_next
            
            return next_carry, out
        
        # Initialize with rho0 and zero potentials
        init_carry = (self.rho0, jnp.zeros(n), jnp.zeros(n))
        
        # Scan over all JKO steps with warm-starting
        _, rhos_steps = lax.scan(one_step, init_carry, xs=None, length=num_steps)
        
        # Prepend initial condition
        rhos = jnp.concatenate([self.rho0[None, :], rhos_steps], axis=0)
        
        return rhos
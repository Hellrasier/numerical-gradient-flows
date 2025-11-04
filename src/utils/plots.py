from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


Array = jnp.ndarray
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
HEATMAP_CMAP = 'YlOrRd'
TRANSPORT_CMAP = 'Blues'

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


def animate_hist_flow(
    mu_list,
    x,
    target=None,                 # optional target mass/density: same length as x
    interval=200,                # ms between frames
    title="Gradient Flow",
    xlabel="x",
    ylabel="Probability Density",
    bar_alpha=0.5,
    bar_edge="black",
    bar_color=None,              # None -> Matplotlib default
    line_color="red",
    ylim_pad=1.1,                # y-axis headroom factor
    return_html=True,            # return HTML string for display in notebooks
):
    """
    Animate a sequence of 1D measures as a bar 'histogram' over a fixed grid.

    Args
    ----
    mu_list : array-like, shape (T, n) or list of (n,)
        Sequence of measures (bin masses). Each frame will be normalized to sum=1.
    x : array-like, shape (n,)
        Bin centers (uniformly spaced).
    target : array-like, shape (n,), optional
        Target distribution. If supplied, it's plotted as a line (interpreted as mass; converted to density).
    interval : int
        Milliseconds between frames.
    title, xlabel, ylabel : str
        Labels.
    bar_alpha, bar_edge, bar_color : visual args for bars.
    line_color : color for the target curve.
    ylim_pad : float
        Multiplier for max y to add headroom.
    return_html : bool
        If True, return HTML string from ani.to_jshtml(); else return the Matplotlib Animation object.

    Returns
    -------
    HTML string (if return_html=True) or the Animation object.
    """
    # Convert to numpy (handles JAX arrays)
    mu_arr = np.asarray(mu_list)
    x = np.asarray(x)
    T, n = mu_arr.shape

    # Uniform bin width from centers
    if n < 2:
        raise ValueError("x must have at least 2 points to infer bin width.")
    bin_width = float(x[1] - x[0])

    # Densities for initial frame
    y0 = mu_arr[0] / np.sum(mu_arr[0])
    y0_density = y0 / bin_width

    # Optional target (mass or density -> convert to density)
    target_density = None
    if target is not None:
        tgt = np.asarray(target)
        if tgt.shape != (n,):
            raise ValueError("target must have shape (n,), same length as x.")
        # If target sums to ~1, treat as mass; otherwise assume it's already density
        if 0.9 <= float(np.sum(tgt)) <= 1.1:
            target_density = tgt / (np.sum(tgt) * bin_width)
        else:
            target_density = tgt

    # Figure/axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars at bin centers with width=bin_width
    bars = ax.bar(
        x, y0_density, width=bin_width, alpha=bar_alpha,
        align="center", edgecolor=bar_edge, color=bar_color, label="Distribution"
    )

    # Overlay target curve if provided
    if target_density is not None:
        (target_line,) = ax.plot(x, target_density, color=line_color, lw=2, label="Target Distribution")
    else:
        target_line = None

    # Axis limits
    y_max = y0_density.max()
    if target_density is not None:
        y_max = max(y_max, float(np.max(target_density)))
    ax.set_xlim(x.min() - 0.5 * bin_width, x.max() + 0.5 * bin_width)
    ax.set_ylim(0, y_max * ylim_pad)

    # Labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — step 0 / {T-1}")
    ax.legend()

    # Init function: zero bars (optional eye-candy)
    def init():
        for b in bars:
            b.set_height(0.0)
        return bars

    # Update per frame
    def animate(i):
        yi = mu_arr[i]
        yi = yi / np.sum(yi)            # normalize to probability mass 1
        yi_density = yi / bin_width      # convert to density
        for b, h in zip(bars, yi_density):
            b.set_height(float(h))
        ax.set_title(f"{title} — step {i} / {T-1}")
        return bars

    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(T), init_func=init,
        interval=interval, blit=True
    )
    plt.close(fig)  # avoid static figure in notebooks

    if return_html:
        return HTML(ani.to_jshtml())
    return ani

COLORS = ["red", "green", "blue", "orange", "purple"]

def plot_vectors(n: int, data: dict):
    """
    Plots vectors supported on [-5,5] with respect to labels.
    """
    x = jnp.linspace(-1, 1, n)
    fig = plt.figure(figsize=(8, 6))
    i = 0
    for label, vec in data.items():
        plt.plot(x, vec, label=rf'\{label}', color=COLORS[i])
        i += 1
    plt.legend()
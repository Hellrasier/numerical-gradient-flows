import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# def animate_hist_flow(
#     mu_list,
#     x,
#     target=None,                 # optional target mass/density: same length as x
#     interval=200,                # ms between frames
#     title="Gradient Flow",
#     xlabel="x",
#     ylabel="Probability Density",
#     bar_alpha=0.5,
#     bar_edge="black",
#     bar_color=None,              # None -> Matplotlib default
#     line_color="red",
#     ylim_pad=1.1,                # y-axis headroom factor
#     return_html=True,            # return HTML string for display in notebooks
# ):
#     """
#     Animate a sequence of 1D measures as a bar 'histogram' over a fixed grid.

#     Args
#     ----
#     mu_list : array-like, shape (T, n) or list of (n,)
#         Sequence of measures (bin masses). Each frame will be normalized to sum=1.
#     x : array-like, shape (n,)
#         Bin centers (uniformly spaced).
#     target : array-like, shape (n,), optional
#         Target distribution. If supplied, it's plotted as a line (interpreted as mass; converted to density).
#     interval : int
#         Milliseconds between frames.
#     title, xlabel, ylabel : str
#         Labels.
#     bar_alpha, bar_edge, bar_color : visual args for bars.
#     line_color : color for the target curve.
#     ylim_pad : float
#         Multiplier for max y to add headroom.
#     return_html : bool
#         If True, return HTML string from ani.to_jshtml(); else return the Matplotlib Animation object.

#     Returns
#     -------
#     HTML string (if return_html=True) or the Animation object.
#     """
#     # Convert to numpy (handles JAX arrays)
#     mu_arr = np.asarray(mu_list)
#     x = np.asarray(x)
#     T, n = mu_arr.shape

#     # Uniform bin width from centers
#     if n < 2:
#         raise ValueError("x must have at least 2 points to infer bin width.")
#     bin_width = float(x[1] - x[0])

#     # Densities for initial frame
#     y0 = mu_arr[0] / np.sum(mu_arr[0])
#     y0_density = y0 / bin_width

#     # Optional target (mass or density -> convert to density)
#     target_density = None
#     if target is not None:
#         tgt = np.asarray(target)
#         if tgt.shape != (n,):
#             raise ValueError("target must have shape (n,), same length as x.")
#         # If target sums to ~1, treat as mass; otherwise assume it's already density
#         if 0.9 <= float(np.sum(tgt)) <= 1.1:
#             target_density = tgt / (np.sum(tgt) * bin_width)
#         else:
#             target_density = tgt

#     # Figure/axes
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Bars at bin centers with width=bin_width
#     bars = ax.bar(
#         x, y0_density, width=bin_width, alpha=bar_alpha,
#         align="center", edgecolor=bar_edge, color=bar_color, label="Distribution"
#     )

#     # Overlay target curve if provided
#     if target_density is not None:
#         (target_line,) = ax.plot(x, target_density, color=line_color, lw=2, label="Target Distribution")
#     else:
#         target_line = None

#     # Axis limits
#     y_max = y0_density.max()
#     if target_density is not None:
#         y_max = max(y_max, float(np.max(target_density)))
#     ax.set_xlim(x.min() - 0.5 * bin_width, x.max() + 0.5 * bin_width)
#     ax.set_ylim(0, y_max * ylim_pad)

#     # Labels and legend
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(f"{title} — step 0 / {T-1}")
#     ax.legend()

#     # Init function: zero bars (optional eye-candy)
#     def init():
#         for b in bars:
#             b.set_height(0.0)
#         return bars

#     # Update per frame
#     def animate(i):
#         yi = mu_arr[i]
#         yi = yi / np.sum(yi)            # normalize to probability mass 1
#         yi_density = yi / bin_width      # convert to density
#         for b, h in zip(bars, yi_density):
#             b.set_height(float(h))
#         ax.set_title(f"{title} — step {i} / {T-1}")
#         return bars

#     ani = animation.FuncAnimation(
#         fig, animate, frames=np.arange(T), init_func=init,
#         interval=interval, blit=True
#     )
#     plt.close(fig)  # avoid static figure in notebooks

#     if return_html:
#         return HTML(ani.to_jshtml())
#     return ani

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

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
    yscale="linear",             # "linear" or "log"
    ymin_floor=1e-12,            # lower bound for log scale
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
    yscale : {"linear","log"}
        Y-axis scale mode. If "log", values are clipped to be >= ymin_floor.
    ymin_floor : float
        Strictly positive floor for log scale.
    return_html : bool
        If True, return HTML string from ani.to_jshtml(); else return the Matplotlib Animation object.

    Returns
    -------
    HTML string (if return_html=True) or the Animation object.
    """
    # Convert to numpy (handles JAX arrays)
    mu_arr = np.asarray(mu_list)
    x = np.asarray(x)

    if mu_arr.ndim == 1:
        mu_arr = mu_arr[None, :]
    if mu_arr.ndim != 2:
        raise ValueError("mu_list must be (T, n) or (n,)")

    T, n = mu_arr.shape

    # Uniform bin width from centers
    if n < 2:
        raise ValueError("x must have at least 2 points to infer bin width.")
    bin_width = float(x[1] - x[0])

    # Safety for log scale
    use_log = (yscale.lower() == "log")
    eps = float(ymin_floor) if use_log else 0.0
    if use_log and eps <= 0:
        raise ValueError("ymin_floor must be > 0 for log scale.")

    # Densities for initial frame
    y0 = mu_arr[0] / np.sum(mu_arr[0])
    y0_density = y0 / bin_width
    if use_log:
        y0_density = np.clip(y0_density, eps, None)

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
        if use_log:
            target_density = np.clip(target_density, eps, None)

    # Figure/axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set y scale early
    if use_log:
        ax.set_yscale("log")

    # Bars at bin centers with width=bin_width
    bars = ax.bar(
        x, y0_density, width=bin_width, alpha=bar_alpha,
        align="center", edgecolor=bar_edge, color=bar_color, label="Distribution"
    )

    # Overlay target curve if provided
    if target_density is not None:
        (target_line,) = ax.plot(
            x, target_density, color=line_color, lw=2, label="Target Distribution"
        )
    else:
        target_line = None

    # Axis limits
    y_max = float(np.max(y0_density)) if y0_density.size else (eps if use_log else 1.0)
    if target_density is not None:
        y_max = max(y_max, float(np.max(target_density)))

    ax.set_xlim(x.min() - 0.5 * bin_width, x.max() + 0.5 * bin_width)
    if use_log:
        # In log scale, lower bound must be > 0
        ax.set_ylim(max(eps, 1e-300), max(y_max * ylim_pad, eps * 10.0))
    else:
        ax.set_ylim(0.0, y_max * ylim_pad)

    # Labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel + (" (log scale)" if use_log else ""))
    ax.set_title(f"{title} — step 0 / {T-1}")
    ax.legend()

    # Init function: zero bars (optional eye-candy)
    def init():
        for b in bars:
            b.set_height(eps if use_log else 0.0)
        return bars

    # Update per frame
    def animate(i):
        yi = mu_arr[i]
        yi = yi / np.sum(yi)           # normalize to probability mass 1
        yi_density = yi / bin_width    # convert to density
        if use_log:
            yi_density = np.clip(yi_density, eps, None)
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

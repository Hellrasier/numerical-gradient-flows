from .pdhg_jko import PrimalDualJKO, pdhg_jko, jko_flow, proxF_entropy, proxF_quadratic
from .sinkhorn_jko import Sinkhorn, SinkhornJKO, col_sums, row_sums, plot_marginal, plot_marginals, plot_cost_matrix, plot_convergence, plot_transport_plan, plot_sinkhorn_summary

__all__ = [
    "PrimalDualJKO",
    "proxF_entropy",
    "proxF_quadratic",
    "pdhg_jko",
    "jko_flow",
    "Sinkhorn",
    "SinkhornJKO",
    "col_sums",
    "row_sums",
    "plot_marginal",
    "plot_marginals",
    "plot_cost_matrix",
    "plot_convergence",
    "plot_transport_plan",
    "plot_sinkhorn_summary",
]

__version__ = "0.1.0"
from __future__ import annotations
import jax
import jax.numpy as jnp
import optuna
import pandas as pd
from datetime import datetime
from src.jko_lab import * # solvers
from src.utils import * # graphing utilities

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray

n = 100             # spatial discretization                
T_TARGET = 50       # total time of the flow

x = jnp.linspace(0.0, 1.0, n)
X, Y = jnp.meshgrid(x, x, indexing="ij")
C = (X - Y) ** 2

# initial measure
rho0 = 0.5 * jax.scipy.stats.norm.pdf(x, 0.25, 0.03) + 0.5 * jax.scipy.stats.norm.pdf(x,0.75,0.04)
rho0 = jnp.clip(rho0, 1e-12, None); rho0 = rho0 / rho0.sum()

# target measure is uniform
b = jnp.ones_like(x)/n
b = b / b.sum()

# fixed parameters
sinkhorn_max_iters = 2000               # maximum number of inner Sinkhorn iterations
jko_tol = 1e-8                          # tolerance for inner Sinkhorn approximation
jko_lr = 0.01                           # learning rate of outer SGD iterations
jko_inner_steps = 10                    # number of outer SGD iterations

@jax.jit
def F_shannon_entropy(rho: jax.Array) -> jax.Array:
    """
    Shannon entropy of a probability measure rho
    """
    rho_safe = jnp.clip(rho, 1e-12, None) # prevent log(0)
    return jnp.sum(rho_safe * jnp.log(rho_safe) - rho_safe)
                   

def objective(trial: optuna.Trial) -> float:
    eta = trial.suggest_float('eta', 1e-4, 1e-1, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-5, 1, log=True)
    
    # calculate dependent parameters
    num_jko_steps = int(T_TARGET / eta)

    jko_flow = SinkhornJKO(
        C=C,
        rho0=rho0,
        eta=eta,
        epsilon=epsilon,
        F_func=F_shannon_entropy,
        sinkhorn_iters=sinkhorn_max_iters, 
        inner_steps=jko_inner_steps,
        tol=jko_tol,
        learning_rate=0.01,
        optimizer_name='sgd'
    )

    try:
        rhos, _ = jko_flow.compute_flow(num_steps=num_jko_steps)
        rho_final = rhos[-1]
        final_error = jnp.linalg.norm(rho_final - b)        
        return float(final_error)
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.TrialPruned()
    

# --- Run Optimization ---
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=800)

# --- Save Results ---
print("\n--- HPO Finished ---")
print(f"Best L2 error: {study.best_value}")
print("Best parameters found:")
print(study.best_params)

# Get dataframe of all trials
df = study.trials_dataframe()

# Create results directory if it doesn't exist
import os
os.makedirs('results', exist_ok=True)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'results/jko_hpo_results_{timestamp}.csv'
df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Also save best parameters separately
best_params_file = f'results/jko_hpo_best_params_{timestamp}.txt'
with open(best_params_file, 'w') as f:
    f.write(f"Best L2 error: {study.best_value}\n")
    f.write(f"Best parameters:\n")
    for key, value in study.best_params.items():
        f.write(f"  {key}: {value}\n")
print(f"Best parameters saved to: {best_params_file}")
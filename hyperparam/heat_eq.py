from __future__ import annotations
import jax
import jax.numpy as jnp
import optuna
from src.jko_lab import * # solvers
from src.utils import * # graphing utilities
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray

# problem setup
T_TARGET = 215
n = 100
x = jnp.linspace(0.0, 1.0, n)
X, Y = jnp.meshgrid(x, x, indexing="ij")
C = (X - Y) ** 2
b = jnp.ones_like(x)/n
rho0 = 0.5 * jax.scipy.stats.norm.pdf(x, 0.25, 0.03) + 0.5 * jax.scipy.stats.norm.pdf(x,0.75,0.04)
rho0 = jnp.clip(rho0, 1e-12, None); rho0 = rho0 / rho0.sum()

# fixed parameters
sinkhorn_max_iters = 2000
jko_inner_steps = 15
jko_tol=1e-9
optimizer_name = 'sgd'
learning_rate = 0.01

@jax.jit
def entropy(r):
    r = jnp.clip(r, 1e-12, None)
    return -jnp.sum(r * jnp.log(r))

# Choose F = entropy (negative Boltzmann-Shannon)
@jax.jit
def F_value_entropy(rho: jax.Array) -> jax.Array:
    rho_safe = jnp.clip(rho, 1e-12, None) # prevent log(0)
    return jnp.sum(rho_safe * (jnp.log(rho_safe)) - 1.0)

def objective(trial: optuna.Trial) -> float:
    eta = trial.suggest_float('eta', 1e-2, 1e-1)
    epsilon = trial.suggest_float('epsilon', 1e-5, 15, log=True)
    
    # calculate dependent parameters
    num_jko_steps = int(T_TARGET / eta)

    jko_flow = SinkhornJKO(
        C=C,
        rho0=rho0,
        eta=eta,
        epsilon=epsilon,
        F_func=F_value_entropy,
        sinkhorn_iters=sinkhorn_max_iters, 
        inner_steps=jko_inner_steps,
        tol=jko_tol,
        learning_rate=0.01,
        optimizer_name=optimizer_name
    )

    try:
        rhos, _ = jko_flow.compute_flow(num_steps=num_jko_steps)
        rho_final = rhos[-1]
        # Calculate the L2 norm (scalar)
        final_error = jnp.linalg.norm(rho_final - b)        
        return float(final_error)
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return optuna.TrialPruned()
    

optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# --- 3. Print the Results ---
print("\n--- HPO Finished ---")
print(f"Best L2 error: {study.best_value}")
print("Best parameters found:")
print(study.best_params)

# You can also get a dataframe of all trials
df = study.trials_dataframe()
print(df)
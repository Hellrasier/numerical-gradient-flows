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
T_TARGET = 6
n = 600
x = jnp.linspace(0.0, 1.0, n)
X, Y = jnp.meshgrid(x, x, indexing="ij")
C = (X - Y) ** 2
b = jax.nn.softmax(-((x-0.6)**2)/0.02)
b = b / jnp.sum(b)
rho0 = jax.scipy.stats.norm.pdf(x, 0.2, 0.05)
rho0 = jnp.clip(rho0, 1e-12, None); rho0 = rho0 / rho0.sum()

# fixed parameters
sinkhorn_max_iters = 2000
jko_inner_steps = 10
jko_tol=1e-9
optimizer_name = 'sgd'
learning_rate = 0.01

@jax.jit
def F_value_entropy(rho):
    b = jax.nn.softmax(-((x-0.6)**2)/0.02)
    b = jnp.clip(b, 1e-300, None) # avoid log(0)
    b = b / jnp.sum(b)
    return jnp.dot(rho, jnp.log(jnp.divide(rho,b)) - jnp.ones_like(b))

# set up new hyperparameters
eta = 1e-2                           

def objective(trial: optuna.Trial) -> float:
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
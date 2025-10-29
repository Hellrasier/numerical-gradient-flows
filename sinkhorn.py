import jax
import jax.numpy as jnp
from jax import lax

def prox_kl_g(K_Ta, v_vals, kappa, l):
    # Shannon entropy
    w = K_Ta * jnp.exp(-kappa * v_vals)
    prox = (w * l**kappa) ** (1.0 / (1.0 + kappa))
    return prox


def sinkhorn(r0, C, b0, v_vals, l, tau, reg=1e-10, iters=100, stop=0.0):
    K = jnp.exp(-C/reg)
    K_T = K.T

    kappa = 2*tau/reg

    r = r0
    b = b0

    for i in range(iters):
        Kb = K @ b
        a = r / Kb
        K_Ta = K_T @ a
        bp =  prox_kl_g(K_Ta, v_vals, kappa, l) / K_Ta

        if stop > 0 and jnp.linalg.norm(bp - b) < stop:
            b = bp
            break
        else:
            b = bp

    r = K_T @ a * b

    return r, b

def sinkhorn_flow(r0, C, b0, v_vals, l, tau, reg=1e-10, steps=10, iters=100, stop=0.0):
    flow = [r0]
    b = b0
    r = r0
    for i in range(steps):
        r, b = sinkhorn(r, C, b, v_vals, l, tau, reg, iters=iters, stop=stop)
        flow.append(r)
    return jnp.stack(flow, axis=0)





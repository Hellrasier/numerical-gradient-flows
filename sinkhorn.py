import jax
import jax.numpy as jnp
from jax import lax


def prox_kl_g(K_Tu, kappa, p):
    # heat equation placeholder
    return K_Tu ** (1.0 / (1.0 + kappa)) * (1/p)**(kappa/(1+kappa))


def sinkhorn(r0, C, v0, tau, reg=1e-10, iters=100, stop=0.0):
    K = jnp.exp(-C/reg)
    K_T = K.T

    kappa = 2*tau/reg
    p = len(r0)

    r = r0
    v = v0

    for i in range(iters):
        Kv = K @ v
        u = r / Kv
        K_Tu = K_T @ u
        vp =  prox_kl_g(K_Tu, kappa, p) / K_Tu

        if stop > 0 and jnp.linalg.norm(vp - v) < stop:
            v = vp
            break
        else:
            v = vp

    r = K_T @ u * v 

    return r, v

def sinkhorn_flow(r0, C, v0, tau, reg=1e-10, steps=10, iters=100, stop=0.0):
    flow = [r0]
    v = v0
    r = r0
    for i in range(steps):
        r, v = sinkhorn(r, C, v, tau, reg, iters=iters, stop=stop)
        flow.append(r)
    return jnp.stack(flow, axis=0)





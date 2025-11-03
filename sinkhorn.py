from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import lax
# from jax.scipy.special import logsumexp
# from jko_lab.lambert import lambertw

# def prox_kl_g(K_Ta, v_vals, kappa, l):
#     # Shannon entropy
#     w = K_Ta * jnp.exp(-kappa * v_vals)
#     prox = (w * l**kappa) ** (1.0 / (1.0 + kappa))
#     return prox

    
# def sinkhorn(r0, C, b0, v_vals, l, tau, reg=1e-10, iters=100, stop=0.0):
#     K = jnp.exp(-C/reg)
    
#     K_T = K.T

#     kappa = 2*tau/reg
#     # print(f"kappa: {kappa}")

#     r = r0
#     # print(f"r range: {jnp.min(r)}, {jnp.max(r)}")
#     b = b0
    

#     for i in range(iters):
#         # print(f"b range: {jnp.min(b)}, {jnp.max(b)}")
#         Kb = K @ b
#         a = r / Kb
#         # print(f"a range: {jnp.min(a)}, {jnp.max(a)}")
#         K_Ta = K_T @ a
#         bp =  prox_kl_g(K_Ta, v_vals, kappa, l) / K_Ta

#         if stop > 0 and jnp.linalg.norm(bp - b) < stop:
#             b = bp
#             break
#         else:
#             b = bp
        

#     r = K_T @ a * b

#     return r, b

# def sinkhorn_flow(r0, C, b0, v_vals, l, tau, reg=1e-10, steps=10, iters=100, stop=0.0):
#     flow = [r0]
#     b = b0
#     r = r0
#     for i in range(steps):
#         r, b = sinkhorn(r, C, b, v_vals, l, tau, reg, iters=iters, stop=stop)
#         flow.append(r)
#     return jnp.stack(flow, axis=0)



def prox_kl_g_log(K_Ta, v_vals, kappa, l):
    finfo = jnp.finfo(jnp.result_type(K_Ta))
    tiny = finfo.tiny
    # shannon entropy
    w = K_Ta * jnp.exp(-kappa * v_vals)
    w = jnp.clip(w, tiny, jnp.inf)
    z = w * l**kappa
    z = jnp.clip(z, tiny, jnp.inf)
    prox = z ** (1.0 / (1.0 + kappa))
    return prox

def sinkhorn_log(r0, C, X, v_vals, l, tau, b0, u, v, reg=1e-10, iters=100, alpha=0.5, stop=0.0):

    finfo = jnp.finfo(jnp.result_type(C))
    log_max = (jnp.log(finfo.max) - 10.0) * 0.9
    tiny = finfo.tiny / 0.9

    kappa = 2*tau/reg

    r = r0

    p = X.shape[0]
    q = X.shape[1]


    b = b0


    K = jnp.exp((u[:, None] + v[None, :] - C) / reg)
    K_T = K.T


    def body_fun(carry):
        i, r, b, u, v, K, K_T, done = carry

        Kb = jnp.clip(K @ b, tiny, jnp.inf)
        a = r / Kb
        K_Ta = jnp.clip(K_T @ a, tiny, jnp.inf)
        bp = prox_kl_g_log(K_Ta, v_vals, kappa, l) / K_Ta

        # stop criterion
        rel = jnp.linalg.norm(bp - b) / (jnp.linalg.norm(b) + tiny)
        hit = (stop > 0) & (rel < stop)

        new_b = lax.cond(done | hit,
                         lambda _: bp,
                         lambda _: (1.0 - alpha) * b + alpha * bp,
                         operand=None)
        new_done = done | hit

        log_a = jnp.log(jnp.clip(a, tiny, jnp.inf))
        log_b = jnp.log(jnp.clip(new_b, tiny, jnp.inf))

        need_rescale = jnp.any(jnp.abs(log_a) > log_max) | jnp.any(jnp.abs(log_b) > log_max)

        def rescale(_):
            u2 = u + reg * log_a
            v2 = v + reg * log_b
            K2 = jnp.exp((v2[None, :] + u2[:, None] - C) / reg)
            K_T2 = K2.T
            b2 = jnp.ones_like(new_b)
            return u2, v2, K2, K_T2, b2

        def no_rescale(_):
            return u, v, K, K_T, new_b

        u, v, K, K_T, new_b = lax.cond(need_rescale, rescale, no_rescale, operand=None)

        Kb2 = jnp.clip(K @ new_b, tiny, jnp.inf)
        a2  = r / Kb2
        r2  = (K_T @ a2) * new_b

        return (i + 1, r2, new_b, u, v, K, K_T, new_done)

    def cond_fun(carry):
        i, _, _, _, _, _, _, done = carry
        return (i < iters) & (~done)

    init_carry = (0, r, b, u, v, K, K_T, False)
    _, r, b, u, v, K, K_T, _ = lax.while_loop(cond_fun, body_fun, init_carry)

    return r, b, u, v

def sinkhorn_flow_log(r0, C, X, v_vals, l, tau_max, tau_min, 
                      reg_max=1e-2, reg_min=1e-10, steps=10, min_by=5, 
                      iters=100, alpha=0.5, stop=0.0):


    div = max(1, min(min_by, steps - 1))
    gamma = jnp.log(reg_max / reg_min) / div
    beta  = jnp.log(tau_max / tau_min) / div
    idx = jnp.arange(steps)
    regs = reg_max * jnp.exp(-gamma * idx)
    taus = tau_max * jnp.exp(-beta * idx)

    p = X.shape[0]; q = X.shape[1]
    init_u = jnp.zeros(p); init_v = jnp.zeros(q); init_b = jnp.ones(q)

    def scan_body(carry, inp):
        r, u, v, b = carry
        reg, tau = inp
        r, b, u, v = sinkhorn_log(
            r, C, X, v_vals, l, tau, b, u, v,
            reg, iters=iters, alpha=alpha, stop=stop
        )
        return (r, u, v, b), r

    init = (r0, init_u, init_v, init_b)
    (r_last, u, v, b), r_hist = lax.scan(scan_body, init, (regs, taus))
    flow = jnp.concatenate([r0[None, ...], r_hist], axis=0)
    return flow
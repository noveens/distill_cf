import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax

def make_loss_fn(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params['depth'],
        num_classes=hyper_params['num_items']
    )
    # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
    # kernel_fn = nt.batch(kernel_fn, batch_size=128)
    kernel_fn = functools.partial(kernel_fn, get='ntk')

    # This function does gumbel sampling *once* per row in the learnable user-item matrix
    gumbel_sample_once = jax.jit(lambda logits, key: jax.nn.softmax(
        (logits + jax.lax.stop_gradient(jax.random.gumbel(key, logits.shape))) / hyper_params['gumbel_tau']
    ))

    # This complicicated looking function does multi-gumbel sampling
    # i.e. does gumbel sampling multiple times per row (user) in a very efficient manner
    @jax.custom_vjp
    def final_function(x_raw, keys):
        ret = jax.lax.scan(
            lambda x, step: (x + gumbel_sample_once(x_raw, keys[step + 1]), None),
            gumbel_sample_once(x_raw, keys[1]),
            jnp.arange(hyper_params['num_per_user'] - 1), 
            length = hyper_params['num_per_user'] - 1,
        )[0]
        return ret
        
    def f_fwd(x_raw, keys):
        return final_function(x_raw, keys), (x_raw, keys)
    
    def f_bwd(res, g):
        x_raw, keys = res
        def gumbel_sample_once_vjp(x, key):
            _, vjp_fun = jax.vjp(gumbel_sample_once, x, key)
            return vjp_fun(g)[0] # don't need grad w.r.t. key
        
        return jax.lax.scan(
            lambda acc, step: (acc + gumbel_sample_once_vjp(x_raw, keys[step + 1]), None),
            gumbel_sample_once_vjp(x_raw, keys[1]),
            jnp.arange(hyper_params['num_per_user'] - 1), 
            length = hyper_params['num_per_user'] - 1,
        )[0], None # don't need grad w.r.t. keys
    
    final_function.defvjp(f_fwd, f_bwd)

    @jax.jit
    def multi_gumbel_sampling(key, x_support_raw):
        keys_final = jax.random.split(key, num = hyper_params['num_per_user'] + 2)
        x_support = jax.nn.hard_tanh(final_function(x_support_raw, keys_final))
        return x_support, keys_final[-1]

    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        K_train = kernel_fn(X_train, X_train)
        K_predict = kernel_fn(X_predict, X_train)
        K_reg = (K_train + jnp.abs(reg) * jnp.trace(K_train) * jnp.eye(K_train.shape[0]) / K_train.shape[0])     
        return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))

    @jax.jit
    def loss_fn(x_support_raw, x_target, key, reg=0.1):      
        x_support, key = multi_gumbel_sampling(key, x_support_raw)
        pred = kernelized_rr_forward(x_support, x_target, reg=reg)
        # BCE Loss
        bce_loss = jnp.mean(jnp.sum(jax.nn.log_softmax(pred, axis = -1) * x_target * -1.0, axis = -1))
        # Regularization & Cardinality Loss
        cardinality_loss = jnp.mean(jnp.sum(x_support, axis = -1)) / float(hyper_params['num_interactions'] / hyper_params['num_users'])
        return bce_loss + (hyper_params['cardinality_reg'] * cardinality_loss), (pred, key)

    return loss_fn, kernelized_rr_forward, multi_gumbel_sampling, kernel_fn

def FullyConnectedNetwork( 
    depth,
    W_std = 2 ** 0.5, 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk'
):
    activation_fn = stax.Relu()
    dense = functools.partial(stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth): layers += [dense(1024), activation_fn] 
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization)]

    return stax.serial(*layers)

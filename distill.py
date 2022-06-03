import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import copy
import random
import numpy as np

from utils import get_common_path, log_end_epoch, get_item_propensity

def get_update_functions(hyper_params, init_params):
    import jax
    import jax.numpy as jnp
    from jax.example_libraries import optimizers

    from model import make_loss_fn

    opt_init, opt_update, get_params = optimizers.adam(hyper_params['learning_rate']) 
    opt_state = opt_init(init_params)

    loss_fn, kernelized_rr_forward, multi_gumbel_sampling, kernel_fn = make_loss_fn(hyper_params)

    grad_loss = jax.grad(
        lambda params, x_target, key, reg: loss_fn(
            params['x'],
            x_target,
            key, reg = reg
        ), has_aux=True
    )

    @jax.jit
    def grad_zeros(accumulated_grad):
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), accumulated_grad)

    @jax.jit
    def grad_add(accumulated_grad, current_grad):
        return jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grad, current_grad)

    @jax.jit
    def grad_div(accumulated_grad, div_by):
        return jax.tree_util.tree_map(lambda x: x / div_by, accumulated_grad)

    @jax.jit
    def update_accumulated_grad(args):
        step, accumulated_grad, opt_state = args

        # Normalize
        accumulated_grad = grad_div(accumulated_grad, hyper_params['accumulate_steps'])

        opt_state = opt_update(step, accumulated_grad, opt_state)
        accumulated_grad = grad_zeros(accumulated_grad)
        return opt_state, accumulated_grad

    @jax.jit
    def update_fn(step, opt_state, params, accumulated_grad, target_data, key, reg = 1.0):
        # Full matrix gumbel-sampling
        current_grad, (_, key) = grad_loss(params, *target_data, key, reg)

        accumulated_grad = grad_add(accumulated_grad, current_grad)

        opt_state, accumulated_grad = jax.lax.cond(
            step % hyper_params['accumulate_steps'] == 0, 
            update_accumulated_grad, 
            lambda args: (args[2], args[1]), # Don't do anything
            (step // hyper_params['accumulate_steps'], accumulated_grad, opt_state)
        )

        return opt_state, accumulated_grad, key

    return opt_state, get_params, update_fn, kernelized_rr_forward, multi_gumbel_sampling, grad_zeros

def initialize(hyper_params, data):
    return { 'x': data.sample_users(hyper_params['user_support']) }

def train_complete(hyper_params, data):
    import jax

    from eval import evaluate_with_grid_search

    key = jax.random.PRNGKey(hyper_params['seed'])
    params_init = initialize(hyper_params, data) # Initialize data with random users
    opt_state, get_params, update_fn, kernelized_rr_forward, multi_gumbel_sampling, grad_zeros = get_update_functions(hyper_params, params_init)
    params = get_params(opt_state)

    orig_start_time = time.time()
    item_propensity = get_item_propensity(hyper_params, data)
    metrics, orig_lamda = evaluate_with_grid_search(hyper_params, kernelized_rr_forward, data, item_propensity, params['x'])
    log_end_epoch(hyper_params, metrics, 0, time.time() - orig_start_time, metrics_on = '(VAL)')
    start_time = time.time()

    VAL_METRIC = "HR@100"
    best_metric = None
    accumulated_grad, best_step, best_lamda = None, 1, orig_lamda

    # Train -- full batch gradient descent
    for i in range(1, hyper_params['train_steps'] + 1):
        target_data = data.sample_training_batch()

        if accumulated_grad is None: accumulated_grad = grad_zeros(params)
        opt_state, accumulated_grad, key_new = update_fn(i, opt_state, params, accumulated_grad, target_data, key, reg = best_lamda)

        # NOTE: This is important! Only update the key after accumulate_steps otherwise results will be super-bad
        # This is because of the randomness in Gumbel-sampling
        if i % hyper_params['accumulate_steps'] == 0: key = key_new

        params = get_params(opt_state)

        # Validate
        if (i in [ 25, 50, 75, 100, 125, 150, 175, 200 ]) or (i % hyper_params['log_freq'] == 0):
            metrics, best_lamda = evaluate_with_grid_search(hyper_params, kernelized_rr_forward, data, item_propensity, multi_gumbel_sampling(key, params['x'])[0])
            log_end_epoch(hyper_params, metrics, i, time.time() - start_time, metrics_on = '(VAL)')
            start_time = time.time()

            if (best_metric is None) or (metrics[VAL_METRIC] > best_metric): 
                final_data = multi_gumbel_sampling(key, params['x'])[0].copy()
                best_metric, best_step = metrics[VAL_METRIC], i

            elif 'patience' in hyper_params and (i - best_step) >= hyper_params['patience']:
                print("Exiting early...")
                break

    # Test on the test-set
    params['x'] = final_data
    test_metrics, _ = evaluate_with_grid_search(hyper_params, kernelized_rr_forward, data, item_propensity, params['x'], test_set_eval = True)
    log_end_epoch(hyper_params, test_metrics, best_step, time.time() - orig_start_time)
    return params, test_metrics

def main(hyper_params, gpu_id = None):
    if gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax.config import config
    if 'float64' in hyper_params and hyper_params['float64'] == True: config.update('jax_enable_x64', True)

    from data import Dataset

    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])
    
    hyper_params['log_file'] = "./results/logs/" + get_common_path(hyper_params) + ".txt"
    hyper_params['distill_data_path'] = "./results/distilled_data/" + get_common_path(hyper_params) + ".npz"

    os.makedirs("./results/logs/", exist_ok=True)
    os.makedirs("./results/distilled_data/", exist_ok=True)
    
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params) # Updated w/ data stats

    # Data distillation
    params_final, test_metrics = train_complete(hyper_params, data)

    # Save data
    with open(hyper_params['distill_data_path'], 'wb') as f: np.savez_compressed(f, data = np.array(params_final['x']))

    return np.array(params_final['x'])

if __name__ == "__main__":
    from hyper_params import hyper_params
    main(hyper_params)

hyper_params = {
	'dataset': 'ml-1m', 
	'float64': False,

	# Distill-CF params
	'user_support': 500, # -1 implies use all users
	'num_per_user': 200,
	'gumbel_tau': 10.0,
	'cardinality_reg': 0.001,

	# Infinite-AE params
	'depth': 1,

	# Generic SGD params
	'train_steps': 500,
	'batch_size': -1, # Can't be greater than # users
	'accumulate_steps': 1, # Gradient accumulation
	'learning_rate': 0.04,
	'patience': 150, # Stop training after these steps if no improvement
	'log_freq': 40,
	'seed': 42,
}

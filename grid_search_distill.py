import os
import gc
import copy
import time
import datetime
import traceback

from distill import main
from utils import get_common_path
from data import Dataset

# NOTE: Specify all possible combinations of hyper-parameters you want to search on.
# NOTE: A list of values for a hyper-parameter means that you want to train all possible combinations of them
common_hyper_params = {
    'float64': False,
    
    'depth': [ 1 ],

    'gumbel_tau': [ 0.3, 0.5, 0.7, 5.0 ], 
    'cardinality_reg': [ 1e-3, 10.0 ],
    'learning_rate': 0.04,

    'patience': 150, # Stop training after these steps if no improvement
    'seed': 42,
}

ml_1m_hyper_params = {
    'dataset': 'ml-1m', 
    'num_per_user': [ 200, 500, 700 ],
    'user_support': [ 10, 20, 40, 80, 100, 200, 500, 800, 1600, 3200, 6000 ],

    'train_steps': 1000,
    'batch_size': -1, # Can't be greater than # users
    'accumulate_steps': 1,
    'log_freq': 40,
}

final_search = [
    [ common_hyper_params ],
    [ 
        ml_1m_hyper_params,
    ]
]

gpu_ids = [ 0 ]

################## CONFIGURATION INPUT ENDS ###################

# STEP-1: Count processes 
def get_all_jobs_recursive(task):
    ret, single_proc = [], True

    for key in task:
        if type(task[key]) != list: continue

        single_proc = False
        for val in task[key]:
            send = copy.deepcopy(task) ; send[key] = val
            ret += get_all_jobs_recursive(send)

        break # All sub-jobs are already counted

    return ret if not single_proc else [ task ]

def get_all_jobs(already, final_search):
    if len(final_search) == 0: return get_all_jobs_recursive(already)

    ret = []
    for at, i in enumerate(final_search):

        for j in i:
            send = copy.deepcopy(already) ; send.update(j)
            ret += get_all_jobs(send, final_search[at + 1:])

        break # All sub-jobs are already counted
    return ret

duplicate_tasks = get_all_jobs({}, final_search)
print("Total processes before unique:", len(duplicate_tasks))

temp = set()
covered_tasks, all_tasks = set(), []
for task in duplicate_tasks:
    log_file = get_common_path(task)

    if log_file is None: continue
    if log_file in covered_tasks: continue

    temp.add(log_file)

    ##### TEMP: Checking if job has already been done
    log_file_path = "./results/logs/" + log_file + ".txt"
    if os.path.exists(log_file_path):
        f = open(log_file_path, 'r')
        lines = f.readlines() ; f.close()
        # Trained for at least 200 DD steps
        exists = sum(map(lambda x: int('end of step  200' in x.strip()), lines))
        if exists != 0: continue

    all_tasks.append(task)
    covered_tasks.add(log_file)
print("Total processes after unique:", len(temp))
print("Total processes after removing already finished jobs:", len(all_tasks))
print(set(list(map(lambda x: x['dataset'], all_tasks))))
# exit()

# STEP-2: Assign individual GPU processes
gpu_jobs = [ [] for _ in range(len(gpu_ids)) ]
for i, task in enumerate(all_tasks): gpu_jobs[i % len(gpu_ids)].append(task)

# Step-3: Spawn jobs
def file_write(log_file, s):
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def run_tasks(tasks, gpu_id):
    start_time = time.time()

    unique_datasets = sorted(list(set(list(map(lambda x: x['dataset'], tasks)))))

    for d in unique_datasets:
        this_dataset_tasks = list(filter(lambda task: task['dataset'] == d, tasks))

        unique_bsz = list(set(list(map(lambda x: x['batch_size'], this_dataset_tasks))))
        data = None
        if len(unique_bsz) == 1: 
            print("Loading", d)
            data = Dataset({ 'dataset': d, 'batch_size': unique_bsz[0], 'seed': 42 })

        for num, task in enumerate(this_dataset_tasks):
            percent_done = max(0.00001, float(num) / float(len(this_dataset_tasks)))
            time_elapsed = time.time() - start_time
            file_write(
                "results/logs/grid_search_log.txt", 
                str(task) + "\nGPU_ID = " + str(gpu_id) + "; dataset = " + task['dataset'] + "; [{} / {}] ".format(num, len(this_dataset_tasks)) +
                str(round(100.0 * percent_done, 2)) + "% done; " +
                "ETA = " + str(datetime.timedelta(seconds=int((time_elapsed / percent_done) - time_elapsed)))
            )
            try: main(task, data = data, gpu_id = gpu_id)
            except Exception as e:
                file_write(
                    "results/logs/grid_search_log.txt", "GPU_ID = " + str(gpu_id) + \
                    "; ERROR [" + str(num) + "/" + str(len(this_dataset_tasks)) + "]\nJOB: " + str(task) + "\n" + str(traceback.format_exc())
                )
            gc.collect()

for gpu in range(len(gpu_ids)):
    run_tasks(gpu_jobs[gpu], gpu_ids[gpu])
    
	# NOTE: We should ideally do this parallely, but python's multi-processing doesn't work well with JAX
    # p = multiprocessing.Process(target=run_tasks, args=(gpu_jobs[gpu], gpu_ids[gpu], ))
    # p.start()
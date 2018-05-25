import tensorflow as tf
import subprocess
import time
import os
import json


def launch_GPU_exp(script, run_kwargs, id_gpu, init_cpu, end_cpu):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    kwargs_path = '/tmp/exp_variant.json'
    json.dump(run_kwargs, open(kwargs_path, 'w'))
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(id_gpu)
    call_str = "taskset -c {}-{} python {} {}".format(init_cpu, end_cpu, script, kwargs_path)
    call_list = call_str.split(" ")
    print("\ncall string:\n", call_str)
    p = subprocess.Popen(call_list, env=run_env)
    return p


def run_multi_gpu(script, exp_kwargs, n_gpu, ctx_per_gpu):
    n_run_slots = n_gpu * ctx_per_gpu
    frac_gpu = 0.85/ctx_per_gpu
    n_cpus = int(os.popen('grep -c cores /proc/cpuinfo').read())
    n_parallel = n_cpus//n_gpu
    print("n_parallel: ", n_parallel, "   n_cpus: ", n_cpus, "   n_run_slots: ", n_run_slots)
    procs = [None] * n_run_slots
    run_kwargs = exp_kwargs.copy()
    del run_kwargs['variants']
    run_kwargs['n_parallel'] = n_parallel
    for v in exp_kwargs['variants']:
        v['frac_gpu'] = frac_gpu
        run_kwargs['variant'] = v
        run_kwargs['seed'] = v.get('seed', None)
        launched = False
        while not launched:
            for run_slot, p in enumerate(procs):
                if p is None or p.poll() is not None:
                    init_cpu = n_parallel * (run_slot // ctx_per_gpu)
                    end_cpu = init_cpu + n_parallel
                    id_gpu = run_slot//ctx_per_gpu
                    procs[run_slot] = launch_GPU_exp(script, run_kwargs, id_gpu, init_cpu, end_cpu)
                    launched = True
                    break
            if not launched:
                time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # (don't return until they are all done)
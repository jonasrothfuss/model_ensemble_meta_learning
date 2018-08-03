import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
from sandbox_maml.rocky.tf.misc import tensor_utils

def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    parent_remote.close()
    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    ts = np.zeros(n_envs, dtype='int')
    np.random.seed(seed)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            actions, reset_args = data
            all_results = [env.step(a) for (a, env) in zip(actions, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    obs[i] = envs[i].reset(reset_args[i])
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))
        elif cmd == 'reset':
            obs = [env.reset(data_i) for env, data_i in zip(envs, data)]
            ts[:] = 0
            remote.send(obs)
        elif cmd == 'set_params':
            for env in envs:
                env.set_param_values(data)
            remote.send(None)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((envs[0].observation_space, envs[0].action_space))
        else:
            raise NotImplementedError

class MAMLParallelVecEnvExecutor(object):
    def __init__(self, env, n_tasks, n_envs, max_path_length):
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_tasks)])
        self.n_envs = n_envs
        assert n_envs % n_tasks == 0
        seeds = np.random.choice(int(1e5), n_tasks, replace=False)
        self.ps = [Process(target=worker, args=(work_remote, remote, pickle.dumps(env), n_envs // n_tasks, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)] # Why pass work remotes?
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, reset_args=None):
        if reset_args is None:
            reset_args = [None] * self.n_envs
        reset_args_split = np.split(np.asarray(reset_args), len(self.remotes))
        actions = np.split(np.asarray(actions), len(self.remotes))
        for remote, action_list, reset_args_list in zip(self.remotes, actions, reset_args_split):
            remote.send(('step', (action_list, reset_args_list)))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)

        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self, reset_args=None):
        if reset_args is None:
            reset_args = [None] * self.n_envs
        reset_args_split = np.split(np.asarray(reset_args), len(self.remotes))
        for remote, args in zip(self.remotes, reset_args_split):
            remote.send(('reset', args))
        return sum([remote.recv() for remote in self.remotes], [])

    def set_params(self, params=None):
        for remote in self.remotes:
            remote.send(('set_params', params))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        return self.n_envs

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass

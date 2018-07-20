import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
from sandbox_maml.rocky.tf.misc import tensor_utils

def worker(remote, parent_remote, env_pickle):
    parent_remote.close()
    env = pickle.loads(env_pickle)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'set_params':
            env.env.set_param_values(data)
            remote.send(None)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class MAMLParallelVecEnvExecutor(object):
    def __init__(self, env, n_envs, max_path_length):
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.ts = np.zeros(n_envs, dtype='int')
        self.max_path_length = max_path_length
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, pickle.dumps(env)))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)] # Why pass work remotes?
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, reset_args=None):
        if reset_args is None:
            reset_args = [None]*len(self.remotes)

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(list, zip(*results))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for i in np.where(dones)[0]:
            self.remotes[i].send(('reset', None))
            self.ts[i] = 0
            obs[i] = self.remotes[i].recv()
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self, reset_args=None):
        for remote in self.remotes:
            remote.send(('reset', None))
        self.ts[:] = 0
        return [remote.recv() for remote in self.remotes]

    def set_params(self, params=None):
        for remote in self.remotes:
            remote.send(('set_params', params))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        return len(self.remotes)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass

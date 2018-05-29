import pickle

from sandbox.jonas.sampler.base import ModelBaseSampler
from sandbox.ignasi.samplers.MAML_VINE_model_sampler.maml_vine_model_vec_env_executor import MAMLVINEModelVecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools
import time


class MAMLVINEModelVectorizedSampler(ModelBaseSampler):

    def __init__(self, algo):
        super(MAMLVINEModelVectorizedSampler, self).__init__(algo)
        self.n_models = self.algo.dynamics_model.num_models
        self.n_branch_per_model = self.algo.n_vine_branch
        self.init_obs = None
        self.n_parallel = self.algo.n_vine_init_obs * self.n_models * self.n_branch_per_model

    def start_worker(self):
        env = pickle.loads(pickle.dumps(self.algo.env))

        self.vec_env = MAMLVINEModelVecEnvExecutor(
            env=env,
            model=self.algo.dynamics_model,
            max_path_length=self.algo.vine_max_path_length,
            n_parallel=self.n_parallel
        )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, init_obs, return_dict=False, log=True, log_prefix=''):
        # return_dict: whether or not to return a dictionary or list form of paths

        self.vec_env.current_obs = self.vec_env.init_obs =\
            self.init_obs = np.repeat(init_obs, self.n_branch_per_model * self.n_models, axis=0)
        self.n_parallel = len(self.init_obs)
        self.vec_env.n_parallel = self.n_parallel

        # todo: put the reset function the observations they need to reset
        obses = self.init_obs
        self.vec_env.ts = np.zeros(self.n_parallel, dtype='int')
        dones = np.asarray([True] * self.n_parallel)

        pbar = ProgBarCounter(self.algo.vine_max_path_length)
        policy_time = 0
        env_time = 0
        process_time = 0
        _time_step = 0

        policy = self.algo.policy
        returns = np.zeros((self.n_parallel,))

        while _time_step < self.algo.vine_max_path_length:
            t = time.time()
            policy.reset(dones)

            # get actions from MAML policy
            obs_per_task = np.split(np.asarray(obses), self.n_models)
            actions, agent_infos = policy.get_actions_batch(obs_per_task)
            if _time_step == 0:
                init_actions = actions
                init_agent_infos = agent_infos

            assert actions.shape[0] == self.n_parallel

            policy_time += time.time() - t
            t = time.time()

            next_obses, rewards, ones, env_infos = self.vec_env.step(actions)
            returns += self.algo.discount * rewards
            env_time += time.time() - t

            pbar.inc(1)
            obses = next_obses
            _time_step += 1

        pbar.stop()

        if log:
            logger.record_tabular(log_prefix + "PolicyExecTime", policy_time)
            logger.record_tabular(log_prefix + "EnvExecTime", env_time)
            logger.record_tabular(log_prefix + "ProcessExecTime", process_time)

        return dict(observations=self.init_obs, actions=init_actions, agent_infos=init_agent_infos, returns=returns)

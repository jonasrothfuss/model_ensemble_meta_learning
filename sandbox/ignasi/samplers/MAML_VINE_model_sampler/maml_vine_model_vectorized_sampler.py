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

        paths = dict([(i, []) for i in range(self.n_models)])

        self.vec_env.current_obs = self.vec_env.init_obs =\
            self.init_obs = np.repeat(init_obs, self.n_branch_per_model * self.n_models, axis=0)
        self.n_parallel = len(self.init_obs)
        self.vec_env.n_parallel = self.n_parallel
        n_samples = 0
        n_parallel_per_task = self.vec_env.num_envs // self.n_models

        # todo: put the reset function the observations they need to reset
        obses = self.init_obs
        dones = np.asarray([True] * self.n_parallel)
        running_paths = [None] * self.n_parallel

        pbar = ProgBarCounter(self.algo.vine_max_path_length)
        policy_time = 0
        env_time = 0
        process_time = 0
        _time_step = 0

        policy = self.algo.policy


        while _time_step < self.algo.vine_max_path_length:
            t = time.time()
            _time_step += 1
            policy.reset(dones)

            # get actions from MAML policy
            obs_per_task = np.split(np.asarray(obses), self.n_models)
            actions, agent_infos = policy.get_actions_batch(obs_per_task)

            assert actions.shape[0] == self.n_parallel

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[idx // n_parallel_per_task].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(1)
            obses = next_obses

        pbar.stop()

        if log:
            logger.record_tabular(log_prefix + "PolicyExecTime", policy_time)
            logger.record_tabular(log_prefix + "EnvExecTime", env_time)
            logger.record_tabular(log_prefix + "ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            # path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])
        else:
            assert len(paths) == self.n_models
        return paths

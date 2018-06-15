#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger


def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    from sandbox.ours.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 12
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                         intra_op_parallelism_threads=ncpu,
    #                         inter_op_parallelism_threads=ncpu)
    #tf.Session(config=config).__enter__()
    tf.Session().__enter__()

    def make_env():
        env = AntEnvRandParams(log_scale_limit=0.0, max_path_length=10)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env for i in range(200)])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=250, nminibatches=25,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps)

    return model, env


def main():
    args = mujoco_arg_parser().parse_args()

    logger.configure(dir='/home/jonasrothfuss/Dropbox/Eigene_Dateien/UC_Berkley/2_Code/model_ensemble_meta_learning/data/local/ppo-baselines')
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:]  = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()

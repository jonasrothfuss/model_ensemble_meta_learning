from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import VariantGenerator
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.gym_mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.gym_mujoco.hopper_env import HopperEnv
from sandbox.jonas.envs.mujoco.hopper_env_random_param import HopperEnvRandParams
from rllab.envs.gym_mujoco.walker2d_env import Walker2DEnv
from rllab.envs.gym_mujoco.ant_env import AntEnv
from rllab.envs.gym_mujoco.humanoid_env import HumanoidEnv
from rllab.envs.gym_mujoco.swimmer_env import SwimmerEnv
from rllab import config
import random

def main(variant):
    real_env = TfEnv(normalize(
        variant['env'](log_scale_limit=0)
    ))


    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=real_env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=real_env.spec)

    algo = TRPO(
        env=real_env,
        policy=policy,
        baseline=baseline,
        n_itr=2000,
        discount=variant['discount'],
        step_size=0.05,
        batch_size=50000,
        max_path_length=variant['max_path_length'],
    )

    algo.train()

if __name__ == '__main__':
    log_dir = 'trpo-baselines-paper-hopper' #osp.join('vine_trpo', datetime.datetime.today().)
    mode = 'ec2'

    vg = VariantGenerator()
    vg.add('env', ['HopperEnvRandParams'])
    # vg.add('env', ['HumanoidEnv'])
    vg.add('discount', [0.99])
    vg.add('max_path_length', [200])
    vg.add('seed', [0, 30, 60])


    subnets = [
        'us-west-2b', 'us-west-2c',
    ]
    ec2_instance = 'm4.2xlarge'
    # configure instan
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    if mode == 'ec2':
        n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
        use_gpu = False
    else:
        n_parallel = 12
        use_gpu = True
    #
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(vg.variants())))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    subnet = random.choice(subnets)
    config.AWS_REGION_NAME = subnet[:-1]
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
        config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
        config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = \
        config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
            config.AWS_REGION_NAME]
    config.AWS_NETWORK_INTERFACES = [
        dict(
            SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
            Groups=config.AWS_SECURITY_GROUP_IDS,
            DeviceIndex=0,
            AssociatePublicIpAddress=True,
        )
    ]

    for v in vg.variants(randomized=True):
        v['env'] = globals()[v['env']]
        run_experiment_lite(
            main,
            use_gpu=use_gpu,
            sync_s3_pkl=True,
            periodic_sync=True,
            variant=v,
            snapshot_mode="last",
            mode=mode,
            n_parallel=n_parallel,
            seed=v['seed'],
            use_cloudpickle=True,
            # exp_name='vine_trpo'
            pre_commands=["yes | pip install tensorflow=='1.6.0'",
                          "pip list",
                          "yes | pip install --upgrade cloudpickle"],
            exp_prefix=log_dir,
            # log_dir=log_dir
        )

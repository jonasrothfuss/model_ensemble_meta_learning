from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.jonas.envs.mujoco import AntEnvRandParams, HalfCheetahEnvRandParams, HopperEnvRandParams, WalkerEnvRandomParams
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import VariantGenerator
from rllab.misc.instrument import run_experiment_lite
from rllab import config
import random

def main(variant):
    real_env = TfEnv(normalize(
        variant['env']()
    ))


    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=real_env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=real_env.spec)

    algo = VPG(
        env=real_env,
        policy=policy,
        baseline=baseline,
        n_itr=10000,
        discount=0.99,
        optimizer_args=dict(learning_rate=5e-3),
        batch_size=50000,
        max_path_length=1000,
    )

    algo.train()

if __name__ == '__main__':
    log_dir = 'vpg-baselines' #osp.join('vine_trpo', datetime.datetime.today().)
    mode = 'ec2'

    vg = VariantGenerator()
    # vg.add('env', ['HalfCheetahEnv', 'HumanoidEnv', 'SnakeEnv', 'SwimmerEnv', 'HopperEnv', 'AntEnv', 'Walker2DEnv'])
    vg.add('env', ['HalfCheetahEnvRandParams', 'HopperEnvRandParams', 'WalkerEnvRandomParams'])
    vg.add('seed', [0, 30, 60])


    subnets = [
        'us-west-1b', 'us-west-1c'
    ]

    ec2_instance = 'm4.4xlarge'
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
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('VPG', len(vg.variants())))
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
            # exp_name='vine_trpo',
            pre_commands=["yes | pip install tensorflow=='1.6.0'",
                          "pip list",
                          "yes | pip install --upgrade cloudpickle"],
            exp_prefix=log_dir,
            # log_dir=log_dir
        )

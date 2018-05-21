from sandbox.ignasi.algos.trpo import VINETRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.ignasi.envs.com_half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.ignasi.envs.model_env import ModelEnv
from sandbox.ignasi.dynamics.dynamics_ensemble import MLPDynamicsEnsemble
from rllab.misc.instrument import VariantGenerator
from rllab.misc.instrument import run_experiment_lite


def main(*args, **kwargs):
    real_env = TfEnv(normalize(HalfCheetahEnv()))

    dynamics_model = MLPDynamicsEnsemble(
        'dynamics_model',
        real_env
    )

    model_env = TfEnv(normalize(ModelEnv(
        env=HalfCheetahEnv(),
        dynamics_model=dynamics_model,
    )))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=real_env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=real_env.spec)

    algo = VINETRPO(
        real_env=real_env,
        model_env=model_env,
        policy=policy,
        baseline=baseline,
        dynamics_model=dynamics_model,
        num_paths=5,
        n_itr=10001,
        discount=1,
        step_size=0.01,

    )

    mode = 'local'

    # subnets = [
    #     'ap-northeast-2a', 'ap-northeast-2c', 'ap-south-1a', 'us-west-1b', 'us-west-1c', 'ap-southeast-1a',
    #     'ap-southeast-1b', 'ap-southeast-1c'
    # ]
    # ec2_instance = 'm4.4xlarge'
    # configure instan
    # info = config.INSTANCE_TYPE_INFO[ec2_instance]
    # config.AWS_INSTANCE_TYPE = ec2_instance
    # config.AWS_SPOT_PRICE = str(info["price"])
    # if mode == 'ec2':
    #     n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    # else:
    n_parallel = 0

    # print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', vg.size))
    # print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
    #                                                                                config.AWS_SPOT_PRICE, n_parallel),
    #       *subnets)
    #
    # subnet = random.choice(subnets)
    # config.AWS_REGION_NAME = subnet[:-1]
    # config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
    #     config.AWS_REGION_NAME]
    # config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
    #     config.AWS_REGION_NAME]
    # config.AWS_SECURITY_GROUP_IDS = \
    #     config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
    #         config.AWS_REGION_NAME]
    # config.AWS_NETWORK_INTERFACES = [
    #     dict(
    #         SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
    #         Groups=config.AWS_SECURITY_GROUP_IDS,
    #         DeviceIndex=0,
    #         AssociatePublicIpAddress=True,
    #     )
    # ]
    # log_dir = osp.join('vine_trpo', datetime.datetime.today().strftime("%Y%m%d"))
    algo.train()

    # run_experiment_lite(
    #     algo.train,
    #     use_gpu=True,
    #     sync_s3_pkl=True,
    #     periodic_sync=True,
    #     variant=v,
        # snapshot_mode="last",
        # mode=mode,
        # n_parallel=n_parallel,
        # seed=v['seed'],
        # use_cloudpickle=True,
        # exp_name='vine_trpo'
        # log_dir=datetime.datetime.today().strftime("%Y%m%d")
    # )

if __name__ == '__main__':
    log_dir = 'vine_trpo' #osp.join('vine_trpo', datetime.datetime.today().)

    vg = VariantGenerator()
    # vg.add('env', ['HalfCheetahEnv', 'HumanoidEnv', 'SnakeEnv', 'SwimmerEnv', 'HopperEnv', 'AntEnv', 'Walker2DEnv'])
    vg.add('env', ['HalfCheetahEnv'])

    for v in vg.variants(randomized=True):
        run_experiment_lite(
            main,
            use_gpu=True,
            sync_s3_pkl=True,
            periodic_sync=True,
            variant=v,
            snapshot_mode="last",
            mode='local',
            n_parallel=0,
            # seed=v['seed'],
            use_cloudpickle=True,
            # exp_name='vine_trpo'
            exp_prefix=log_dir,
            # log_dir=log_dir
        )

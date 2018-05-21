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
from rllab import config
import random
import sys


def main(variant):
    default_dict = dict(num_paths=5,
                        step_size=0.01,
                        discount=1,
                        num_branches=10,
                        opt_model_itr=10,
                        model_max_path_length=50,
                        )
    default_dict.update(variant)
    real_env = TfEnv(normalize(HalfCheetahEnv()))

    dynamics_model = MLPDynamicsEnsemble(
        'dynamics_model',
        real_env,
        hidden_sizes=(512,)*3,
        num_models=5,
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
        num_paths=default_dict['num_paths'],
        n_itr=1001,
        discount=default_dict['discount'],
        step_size=default_dict['step_size'],
        num_branches=default_dict['num_branches'],
        model_max_path_length=default_dict['model_max_path_length'],
        max_path_length=1000,
        opt_model_itr=default_dict['opt_model_itr']
    )


    algo.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ec2', type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.ec2:
        mode = 'ec2'
    else:
        mode = 'local'
    log_dir = 'vine_trpo' #osp.join('vine_trpo', datetime.datetime.today().)
    subnets = [
        'ap-northeast-2a', 'ap-northeast-2c', 'ap-south-1a', 'us-west-1b', 'us-west-1c', 'ap-southeast-1a',
        'ap-southeast-1b',
    ]
    ec2_instance = 'm4.2xlarge'
    # configure instan
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = 0
    #
    # print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', vg.size))
    # print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
    #                                                                                config.AWS_SPOT_PRICE, n_parallel),
    #       *subnets)

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

    vg = VariantGenerator()
    vg.add('env', ['HalfCheetahEnv'])
    vg.add('num_paths', [5])
    vg.add('discount', [1.])
    vg.add('model_max_path_length', [30])
    vg.add('num_branches', [20])
    vg.add('seed', [0])

    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format('TRPO', len(vg.variants())))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    for v in vg.variants(randomized=True):
        run_experiment_lite(
            main,
            use_gpu=False,
            sync_s3_pkl=True,
            periodic_sync=True,
            variant=v,
            snapshot_mode="last",
            mode=mode,
            n_parallel=0,
            seed=v['seed'],
            use_cloudpickle=True,
            # exp_name='vine_trpo'
            exp_prefix=log_dir,
            pre_commands=[
                          "yes | pip install tensorflow=='1.4.1'",
                          "yes | pip install --upgrade cloudpickle"],
            # log_dir=log_dir
        )

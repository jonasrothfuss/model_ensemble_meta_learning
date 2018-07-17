from vendor.cassie_mujoco_sim.test.cassiemujoco import CassieSim, CassieVis, pd_in_t, state_out_t
import numpy as np
import time
from rllab.spaces import Box
from rllab.envs.base import Env
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger

CASSIE_TORQUE_LIMITS = np.array([4.5*25, 4.5*25, 12.2*16, 12.2*16, 0.9*50]) # ctrl_limit * gear_ratio
CASSIE_MOTOR_VEL_LIMIT = np.array([2900, 2900, 1300, 1300, 5500]) / 60 / (2*np.pi) # max_rpm / 60 / 2*pi
P_GAIN_RANGE = [10, 10000]
D_GAIN_RANGE = [1, 100]
MODEL_TIMESTEP = 0.001

DEFAULT_P_GAIN = 200
DEFAULT_D_GAIN = 20

NUM_QPOS = 34
NUM_QVEL = 32

CTRL_COST_COEF = 0.001
STABILISTY_COST_COEF = 0.01



class CassieEnv(Env, Serializable):

    # TODO: add randomization of initial state

    def __init__(self, render=False, fix_pelvis=False, frame_skip=20, fixed_gains=True,
                 stability_cost_coef=1e-2, ctrl_cost_coef=1e-3, alive_bonus=0.2, impact_cost_coef=1e-5):

        self.sim = CassieSim()
        if render:
            self.vis = CassieVis()
        else:
            self.vis = None

        self.fix_pelvis = fix_pelvis
        self.model_timestep = 0.001
        self.frame_skip = frame_skip
        self.fixed_gains = fixed_gains

        # action and observation space specs
        self.act_limits_array = self._build_act_limits_array()
        self.act_dim = self.act_limits_array.shape[0]

        self.num_qpos = NUM_QPOS
        self.num_qvel = NUM_QVEL
        self.obs_dim = self.num_qpos + self.num_qvel

        # reward function coeffs
        self.stability_cost_coef = stability_cost_coef
        self.ctrl_cost_coef = ctrl_cost_coef
        self.impact_cost_coef = impact_cost_coef
        self.alive_bonus = alive_bonus

        if fix_pelvis: self.sim.hold()

        Serializable.quick_init(self, locals())

    def reset(self):
        self.sim = CassieSim()
        if self.fix_pelvis: self.sim.hold()
        state = self.sim.get_state()
        return self._cassie_state_to_obs(state)

    def reward(self, obs, action, obs_next):
        raise NotImplementedError #TODO: ctrl_cost requires motor torques which are neither in the state space nor in the

    @property
    def torque_limits(self):
        return np.concatenate([CASSIE_TORQUE_LIMITS]*2)

    def get_foot_forces(self, internal_state):
        left_toe = _to_np(internal_state.leftFoot.toeForce)
        left_heel = _to_np(internal_state.leftFoot.heelForce)
        right_toe = _to_np(internal_state.rightFoot.toeForce)
        right_heel = _to_np(internal_state.rightFoot.heelForce)
        return np.concatenate([left_toe, left_heel, right_toe, right_heel])


    def done(self, obs):
        if obs.ndim == 1 or obs.ndim == 2:
            height = pelvis_hight_from_obs(obs)
            return height < 0.65
        else:
            raise AssertionError('obs must be 1d or 2d numpy array')

    def step(self, action):
        assert action.ndim == 1 and action.shape == (self.act_dim,)
        u = self._action_to_pd_u(action)
        state, internal_state = self.do_simulation(u, self.frame_skip)
        obs = self._cassie_state_to_obs(state)

        # reward fct
        pelvis_vel = obs[self.num_qpos:self.num_qpos+3]

        foot_forces = self.get_foot_forces(internal_state)
        motor_torques = _to_np(internal_state.motor.torque)
        forward_vel = pelvis_vel[0]
        ctrl_cost = self.ctrl_cost_coef * 0.5 * np.mean(np.square(motor_torques/self.torque_limits))
        stability_cost = self.stability_cost_coef * 0.5 * np.mean(np.square(pelvis_vel[1:])) # quadratic velocity of pelvis in y and z direction ->
                                                                                # enforces to hold the pelvis in same position while walking
        impact_cost = self.impact_cost_coef * 0.5 * np.sum(np.square(np.clip(foot_forces, -1, 1)))
        reward = forward_vel - ctrl_cost - stability_cost - impact_cost + self.alive_bonus

        done = self.done(obs)
        info = {'forward_vel': forward_vel, 'ctrl_cost': ctrl_cost, 'stability_cost': stability_cost}

        return obs, reward, done, info

    def do_simulation(self, u, n_frames):
        assert n_frames >= 1
        for _ in range(n_frames):
            internal_state_obj = self.sim.step_pd(u) # step_pd returns state_out_t structure -> however this structure is still not fully understood
        joint_state = self.sim.get_state() # get CassieState object
        return joint_state, internal_state_obj

    def render(self):
        if self.vis is None:
            print('Setting up cassie visualizer')
            self.setup_cassie_vis()
        self.vis.draw(self.sim)

    @property
    def dt(self):
        return self.model_timestep

    @property
    def action_space(self):
        return Box(low=self.act_limits_array[:,0], high=self.act_limits_array[:,1])

    @property
    def observation_space(self):
        obs_limit = np.inf * np.ones(self.obs_dim)
        return Box(-obs_limit, obs_limit)

    @overrides
    def log_diagnostics(self, paths):
        forward_vel = [np.mean(path['env_infos']['forward_vel']) for path in paths]
        ctrl_cost = [np.mean(path['env_infos']['ctrl_cost']) for path in paths]
        stability_cost = [np.mean(path['env_infos']['stability_cost']) for path in paths]
        path_length = [path["observations"].shape[0] for path in paths]

        logger.record_tabular('AvgForwardVel', np.mean(forward_vel))
        logger.record_tabular('StdForwardVel', np.std(forward_vel))
        logger.record_tabular('AvgCtrlCost', np.mean(ctrl_cost))
        logger.record_tabular('AvgStabilityCost', np.mean(stability_cost))
        logger.record_tabular('AvgPathLength', np.mean(path_length))

    def _cassie_state_to_obs(self, state):
        qpos = np.asarray(state.qpos()[1:])
        assert self.num_qpos == qpos.shape[0]
        qvel = np.asarray(state.qvel())
        assert self.num_qvel == qvel.shape[0]
        return np.concatenate([qpos, qvel], axis=0)

    def setup_cassie_vis(self):
        self.vis = CassieVis()

    def _action_to_pd_u(self, action):
        u = pd_in_t()

        # motors:
        # 0: hip abduction
        # 1: hip twist
        # 2: hip pitch -> lift leg up
        # 3: knee
        # 4: foot pitch

        # Typical pGain ~ 200 [100, 10000]
        # Typical dGain ~ 20
        # Typical feedforward torque > 0

        i = 0
        for leg_name in ['leftLeg', 'rightLeg']:
            leg = getattr(u, leg_name)
            for motor_id in range(5):
                for pd_param in ['torque', 'pTarget', 'dTarget', 'pGain', 'dGain']:
                    if self.fixed_gains and pd_param == 'pGain':
                        getattr(leg.motorPd, pd_param)[motor_id] = DEFAULT_P_GAIN
                    elif self.fixed_gains and pd_param == 'dGain':
                        getattr(leg.motorPd, pd_param)[motor_id] = DEFAULT_D_GAIN
                    else:
                        getattr(leg.motorPd, pd_param)[motor_id] = action[i]
                        i += 1
        return u

    def _build_act_limits_array(self):
        limits = []

        if self.fixed_gains:
            pd_params_to_set = ['torque', 'pTarget', 'dTarget']
        else:
            pd_params_to_set = ['torque', 'pTarget', 'dTarget', 'pGain', 'dGain']

        for leg_name in ['leftLeg', 'rightLeg']:
            for motor_id in range(5):
                for pd_param in pd_params_to_set:
                    if pd_param == 'torque':
                        low, high = (-CASSIE_TORQUE_LIMITS[motor_id], CASSIE_TORQUE_LIMITS[motor_id])
                    elif pd_param == 'pTarget':
                        low, high = (-2 * np.pi, 2 * np.pi)
                    elif pd_param == 'dTarget':
                        low, high = (-CASSIE_MOTOR_VEL_LIMIT[motor_id], CASSIE_MOTOR_VEL_LIMIT[motor_id])
                    elif pd_param == 'pGain':
                        low, high = P_GAIN_RANGE
                    elif pd_param == 'dGain':
                        low, high = D_GAIN_RANGE
                    else:
                        raise AssertionError('Unknown pd_param %s' % pd_param)
                    limits.append(np.array([low, high]))
        limits_array = np.stack(limits, axis=0)
        assert limits_array.ndim == 2 and limits_array.shape[1] == 2
        return limits_array

    # #TODO: sth. is wrong with the pelvis_pos -> should be absolute from world coord. e.g. (0,0,1) in the beginning but that is not the case
    # def _cassie_state_to_obs(self, state):
    #     # pelvis
    #     pelvis_ori = _to_np(state.pelvis.orientation)  # TODO: figure out if this is a quaternion or axis % angle
    #     pelvis_pos = _to_np(state.pelvis.position)
    #     pelvis_rot_vel = _to_np(state.pelvis.rotationalVelocity)
    #     pelvis_transl_vel = _to_np(state.pelvis.translationalVelocity)
    #     print(pelvis_pos)
    #
    #     # joints
    #     joints_pos = _to_np(state.joint.position)
    #     joints_vel = _to_np(state.joint.velocity)
    #
    #     # motors
    #     motor_pos = _to_np(state.motor.position)
    #     motor_vel = _to_np(state.motor.position)
    #
    #     obs = np.concatenate(
    #         [pelvis_pos[1:], pelvis_ori, pelvis_transl_vel, pelvis_rot_vel, joints_pos, joints_vel, motor_pos,
    #          motor_vel], axis=0)
    #     assert obs.shape == (self.obs_dim,)
    #     return obs



def pelvis_hight_from_obs(obs):
    if obs.ndim == 1:
        return obs[1]
    elif obs.ndim == 2:
        return obs[:, 1]
    else:
        raise NotImplementedError

def _to_np(o, dtype=np.float32):
    return np.array([o[i] for i in range(len(o))], dtype=dtype)


if __name__ == '__main__':
    render = True
    env = CassieEnv(render=render, fix_pelvis=False, fixed_gains=False)

    for j in range(5):
        obs = env.reset()
        for j in range(500):
            cum_forward_vel = 0
            act = env.action_space.sample()
            print(act.shape)
            obs, reward, done, info = env.step(act)
            if render: env.render()
            if done: break
            time.sleep(env.dt * env.frame_skip)
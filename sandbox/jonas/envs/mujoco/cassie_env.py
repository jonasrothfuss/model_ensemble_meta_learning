from vendor.cassie_mujoco_sim.test.cassiemujoco import CassieSim, CassieVis, pd_in_t
import numpy as np
import time

class CassieEnv():

    def __init__(self, render=False, fix_pelvis=False): #TODO
        self.sim = CassieSim()
        if render:
            self.vis = CassieVis()
        else:
            self.vis = None

        self.fix_pelvis = fix_pelvis
        if fix_pelvis: self.sim.hold()


    def reset(self): #TODO
        self.sim = CassieSim()
        if self.fix_pelvis: self.sim.hold()
        return self.sim.get_state()

    def reward(self, obs, action, obs_next): #TODO
        return 1.0

    def done(self, obs): #TODO
        return False

    def step(self, action): #TODO: for now, the action is just the z-axis target of the feet
        assert action.ndim == 1 and action.shape == (2,)
        u = pd_in_t()

        u.leftLeg.motorPd.torque[3] = 0  # Feedforward torque
        u.leftLeg.motorPd.pGain[3] = 100  # proportional gain
        u.leftLeg.motorPd.dTarget[3] = -3 #-2
        u.leftLeg.motorPd.dGain[3] = 10  # differential gain
        u.rightLeg.motorPd = u.leftLeg.motorPd

        u.leftLeg.motorPd.pTarget[3] = action[0]
        u.rightLeg.motorPd.pTarget[3] = action[1]

        self.sim.step_pd(u)

    def render(self):
        assert self.vis is not None, 'render attribute must be set true in __init__'
        self.vis.draw(self.sim)

if __name__ == '__main__':
    env = CassieEnv(render=True, fix_pelvis=True)
    s = env.reset()

    for j in range(4000):
        freq = 0.001
        z_left = -1 + np.sin(2*np.pi * freq * j)
        z_right = -1 + np.cos(2*np.pi * freq * j)

        a = np.asarray([z_left, z_right])
        print(a)
        o = env.step(a)
        env.render()
        time.sleep(0.001)
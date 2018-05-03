from sandbox.jonas.controllers.base import Controller

class RandomController(Controller):
    def __init__(self, env):
        self.env = env
        super().__init__()

    def get_action(self, state):
        """ randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()
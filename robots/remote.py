import zerorpc
import gym

from .robot import Controller


class ZeroRPCController(Controller):
    """
    A simple ZeroRPC Controller that communicates with a controller that runs natively on the NUC.
    This perhaps isn't best practice, but many controllers require it:
    1. PolyMetis (since its python verisons are deprecated)
    2. R2D2 (it natively does something like this to get over the woes of polymetis.)
    """

    def __init__(self, ip_address: str = "127.16.0.1", port: int = 4242):
        self.ip_address = ip_address
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect("tcp://" + self.ip_address + ":" + str(port))
        self._observation_space = gym.spaces.Dict(self.server.observation_space())
        self._action_space = gym.spaces.Box(**self.server.action_space())

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def update(self, action):
        """
        Updates the robot controller with the action
        """
        self.server.update(action)

    def get_state(self):
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        Also updates the controllers internal state for delta actions.
        """
        return self.server.get_state()

    def reset(self, randomize=False):
        """
        Reset the robot to HOME, randomize if asked for.
        """
        self.server.reset(randomize=randomize)

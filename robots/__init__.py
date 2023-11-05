from robot import RobotEnv

try:
    from remote import ZeroRPCController
except ImportError:
    print("[robots] ZeroRPC controllers not loaded.")

try:
    from vr import VRController
except ImportError:
    print("[robots] VRController not loaded. oculus_reader likely missing.")

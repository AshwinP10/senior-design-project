from .airhockey_box2d import AirHockeyBox2D
try:
    from .air_hockey_real import AirHockeyReal
except (ImportError, ModuleNotFoundError):
    AirHockeyReal = None  # e.g. Windows (termios) or real deps not installed
try:
    from .airhockey_robosuite import AirHockeyRobosuite
    from robosuite.environments.base import register_env
    register_env(AirHockeyRobosuite)
except:
    print('Robosuite not loaded. Cannot use Robosuite environment on Aple Silicon')
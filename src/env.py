from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import random

SEED = random.uniform(1, 100)
DISCOUNT = 0.99  # how much past rewards are "discounted"
EPS_MIN = 0.1
EPS_MAX = 1.0
BATCH_SIZE = 32  # replay batch size
MAX_EPISODE_LENGTH = 10**4
LEARNING_RATE = 2.5e-4
CLIPNORM = 1.0
EPSILON_RANDOM_FRAMES = 50000
EPSILON_GREEDY_FRAMES = 1.0e6

eps_int = EPS_MAX - EPS_MIN  # rate at which epsilon decreases
eps = EPS_MAX


def make_atari_env():
    env = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(SEED)
    return env


env = make_atari_env()

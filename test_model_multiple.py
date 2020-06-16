
import retro
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.bench import Monitor

from utils import SaveOnBestTrainingRewardCallback, TimeLimitWrapper
from mario_wrappers import Discretizer

from utils import SaveOnBestTrainingRewardCallback, TimeLimitWrapper
from mario_wrappers import *
from retro_wrappers import wrap_deepmind_retro

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--game', default='SonicTheHedgehog-Genesis')
# parser.add_argument('--state', default=retro.State.DEFAULT)
# parser.add_argument('--scenario', default='scenario')
parser.add_argument('--load')
args = parser.parse_args()

workers = 4
steps = 1000

RUNS = 10
#state="RainbowRoad_M"
state = retro.State.DEFAULT
# state= "MarioCircuit2_M"

def wrapper(env):
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    # env= ReduceBinaryActions(env,BinaryActions.BREAK)
    env=TimeLimitWrapperMarioKart(env, minutes=3,seconds=0)
    env = CutMarioMap(env,show_map=False)
    env = wrap_deepmind_retro(env)
    return env


env = retro_make_vec_env('SuperMarioKart-Snes', scenario='scenario', state=state, n_envs=1,
                         vec_env_cls=lambda x: x[0](), max_episode_steps=4000,
                         wrapper_class=wrapper, seed=0,record=True)


model = PPO2.load(args.load)

obs = env.reset()

time = (float("inf"), float("inf"), float("inf"))
for i in range(RUNS):
    sum_reward = 0

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        sum_reward += reward

        env.render()
        if done:

            laps = env.data.lookup_value("lap") - 128
            min = env.data.lookup_value("currMin")
            sec = env.data.lookup_value("currSec")
            ms = ((env.data.lookup_value("currMiliSec") - 300) % 10000) / 100
            currTime = (min, sec, ms)
            if (laps >= 5 and currTime < time):
                time = currTime
                bestRun = i
            else:
                print("FAIL")

            print("Run %d Final time:%2d:%2d:%2d Total Reward: %f" % (i, min, sec, ms, sum_reward))
            env.reset()
            break

print("Best time: %2d:%2d:%2d in run %d\n" % (time + (bestRun,)))

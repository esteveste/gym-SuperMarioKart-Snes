# import gym
# # from stable_baselines import A2C, SAC, PPO2, TD3

# import os

# # Create save dir
# save_dir = "/tmp/gym/"
# os.makedirs(save_dir, exist_ok=True)

import retro
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecFrameStack,DummyVecEnv,SubprocVecEnv,VecVideoRecorder
from stable_baselines.common.policies import MlpPolicy,CnnPolicy
# from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.bench import Monitor

from utils import SaveOnBestTrainingRewardCallback,TimeLimitWrapper
from mario_wrappers import Discretizer

from utils import SaveOnBestTrainingRewardCallback,TimeLimitWrapper
from mario_wrappers import Discretizer, retro_make_vec_env,make_mario_env,CutMarioMap,DiscretizerActions
from retro_wrappers import wrap_deepmind_retro

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--game', default='SonicTheHedgehog-Genesis')
# parser.add_argument('--state', default=retro.State.DEFAULT)
# parser.add_argument('--scenario', default='scenario')
parser.add_argument('--load')
args = parser.parse_args()

workers=4
steps=1000


# state="Rainbow_solo_DK"
state=retro.State.DEFAULT
# state = ["BowserCastle_M", "BowserCastle2_M", "BowserCastle3_M", "ChocoIsland_M", "ChocoIsland2_M", "DonutPlains_M",
#           "DonutPlains2_M", "DonutPlains3_M", "GhostValley_M", "GhostValley2_M", "GhostValley3_M", "KoopaBeach_M",
#           "KoopaBeach2_M", "MarioCircuit_M", "MarioCircuit2_M", "MarioCircuit3_M", "MarioCircuit4_M", "RainbowRoad_M",
#           "VanillaLake_M", "VanillaLake2_M"]

def wrapper(env):
    env=Discretizer(env,DiscretizerActions.SIMPLE)
    # env=TimeLimitWrapper(env, max_steps=9000)
    env=CutMarioMap(env,show_map=False)
    env=wrap_deepmind_retro(env)
    return env



env = retro_make_vec_env('SuperMarioKart-Snes',scenario='training_check',state=state, n_envs=1,
                            vec_env_cls=lambda x: x[0](),max_episode_steps=4000,
                            wrapper_class=wrapper, seed=0,record=True)

model=PPO2.load(args.load)

obs=env.reset()

sum_reward=0
while True:
    action, _states = model.predict(obs)
    obs,reward ,done, info = env.step(action)

    sum_reward+=reward

    print(action, reward)
    env.render()
    if done:
        print("total reward:",sum_reward)
        print("Final time: %2d:%2d:%2d\n" % (env.data.lookup_value("currMin"), env.data.lookup_value("currSec"),
                                 ((env.data.lookup_value("currMiliSec") - 300) % 10000) / 100))

        obs=env.reset()
        sum_reward=0
"""
Helpers for scripts like run_atari.py.
"""

import os
import warnings
from enum import Enum
import numpy as np

import gym
import retro

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, ClipRewardEnv
from stable_baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from stable_baselines.common.atari_wrappers import EpisodicLifeEnv, ScaledFloatFrame, FrameStack
from stable_baselines.common.misc_util import mpi_rank_or_zero
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from retro_wrappers import *


# FIXME DO STATE LIST FOR FUTURE MULTIPLE TRAININGS
def retro_make_vec_env(env_id, state=retro.State.DEFAULT, scenario=None, n_envs=1, seed=None, start_index=0,
                       monitor_dir=None, wrapper_class=None, max_episode_steps=9000,
                       env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None, record=False,record_path='./movies'):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    initial_state=state[0] if isinstance(state,list) else state

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                # env = retro.make(env_id, state, scenario=scenario)
                if record:
                    env = make_retro(game=env_id, state=initial_state, scenario=scenario, max_episode_steps=max_episode_steps, record=record_path)
                else:
                    env = make_retro(game=env_id, state=initial_state, scenario=scenario, max_episode_steps=max_episode_steps)

                if len(env_kwargs) > 0:
                    warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)

            #if multiple states provided, wrap
            if isinstance(state,list):
                env=RandomStateReset(env,state,seed=seed)

            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    # makes sure movie folder exists
    if record:
        os.makedirs(record_path, exist_ok=True)
    

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def make_mario_env(env_id,
                   num_env,
                   seed,
                   actions=None,
                   cut_map=False,
                   do_wrap_dm=True,
                   wrapper_kwargs=None,
                   start_index=0,
                   allow_early_resets=True,
                   start_method=None,
                   use_subprocess=False):
    # FIXME do actions set up on env
    """
    Create a wrapped, monitored VecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_mario(env_id)
            env.seed(seed + rank)

            if cut_map:
                env = CutMarioMap(env)

            env = Monitor(env,
                          logger.get_dir()
                          and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)

            # FIXME do if wrap deepmind, create other methods
            return wrap_deepmind_custom(
                env, **wrapper_kwargs)  # converts to 84*84 bw, keep for now

        return _thunk

    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)


def make_mario(env_id):
    """
    Create a wrapped atari Environment

    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = retro.make(env_id)
    # assert 'NoFrameskip' in env.spec.id
    # env = NoopResetEnv(env, noop_max=10) # this could introduce some randomness. FIXME READ MORE ABOUT

    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind_custom(env, scale=True, frame_stack=4):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if scale:
        env = ScaledFloatFrame(env)
    return env



class DiscretizerActions(Enum):
    SIMPLE = [["B"], ["B", "LEFT"], ["B", "RIGHT"], ["B", "L"]]
    BREAK = [["B"], ["B", "LEFT"], ["B", "RIGHT"], ["B", "L"], ['X']]


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env, actions=DiscretizerActions.SIMPLE):
        super(Discretizer, self).__init__(env)
        # wrong button names though
        # buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

        buttons = ['B',
                   'Y',
                   'SELECT',
                   'START',
                   'UP',
                   'DOWN',
                   'LEFT',
                   'RIGHT',
                   'A', 'X', 'L', 'R']

        actions = actions.value
        # actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
        #            ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class BinaryActions(Enum):
    SIMPLE = ["B", "LEFT","RIGHT", "L"] # also hop
    BREAK = ["B", "LEFT","RIGHT", "L","Y"]


class ReduceBinaryActions(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use multi discrete actions
    """

    def __init__(self, env, actions=BinaryActions.SIMPLE):
        super().__init__(env)

        buttons = ['B',
                   'Y',
                   'SELECT',
                   'START',
                   'UP',
                   'DOWN',
                   'LEFT',
                   'RIGHT',
                   'A', 'X', 'L', 'R']

        actions = actions.value

        self.key_mappings=np.array([buttons.index(action) for action in actions])


        self.action_space = gym.spaces.MultiBinary(len(actions))

    def action(self, a):  # pylint: disable=W0221

        pressed_buttons=np.where(a==1)

        key_map=self.key_mappings[pressed_buttons]

        action=np.zeros(12)

        # set pressed keys
        action[key_map]=1


        return action

class RandomStateReset(gym.Wrapper):
    '''
    FIXME random seed, we could set a seed
    '''

    def __init__(self, env,states,seed=None):
        super().__init__(env)

        self.states=states


    def reset(self):
        
        new_state=np.random.choice(self.states)
        
        self.env.load_state(new_state)

        return self.env.reset()


class EarlyNegRewardTermination(gym.Wrapper):

	def __init__(self, env,max_steps_no_reward=40):
		# Call the parent constructor, so we can access self.env later
		super(EarlyNegRewardTermination, self).__init__(env)

		self.neg_count=0
		self.max_count=max_steps_no_reward

	def reset(self):
		# Reset the counter
		self.neg_count=0
		return self.env.reset()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)

		# count timesteps with negative reward, considering 0 negative
		self.neg_count = self.neg_count + 1 if reward < 0 else 0

		# Overwrite the done signal when
		if self.neg_count>self.max_count:

			done = True
			reward=-1 # negative reward dying

		return obs, reward, done, info

class TimeLimitWrapperMarioKart(gym.Wrapper):
	def __init__(self, env, minutes=3,seconds=0):
		super().__init__(env)
		self.minutes=minutes
		self.seconds=seconds


	def step(self, action):
		obs, reward, done, info = self.env.step(action)

		if self.env.data.lookup_value("currMin")>= self.minutes and \
		   self.env.data.lookup_value("currSec")>=self.seconds:
			done = True
			# Update the info dict to signal that the limit was exceeded
			info['time_limit_reached'] = True
		return obs, reward, done, info

class CutMarioMap(gym.Wrapper):  # FIXME BAD CODE PATTERN METHODS,USE OBSWRAPPER
    def __init__(self, env, show_map=False):
        # Call the parent constructor, so we can access self.env later
        super(CutMarioMap, self).__init__(env)
        self.show_map=show_map

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        actions need to be a 1 or 0 vector
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        obs, reward, done, info = self.env.step(action)
        if self.show_map:
            return obs[110:], reward, done, info
        else:
            return obs[:110], reward, done, info

### Useful templates

# class ObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def observation(self, obs):
#         # modify obs
#         return obs

# class RewardWrapper(gym.RewardWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def reward(self, rew):
#         # modify rew
#         return rew

# class ActionWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def action(self, act):
#         # modify act
#         return act

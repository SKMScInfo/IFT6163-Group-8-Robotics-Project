import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from dmc_vision_benchmark.environments.antmaze_env import make_env
import shimmy
from dmc_vision_benchmark.data.dmc_vb_info import get_action_dim, get_state_dim

class CustomActionRepeatWrapper(gym.Env):
    def __init__(self, env, repeat=4, max_episode_steps=1000):
        super(CustomActionRepeatWrapper, self).__init__()
        self.env = env
        self.repeat = repeat
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        # Define observation_space and action_space from the original environment
        self.observation_space = gym.Space((get_state_dim("ant"), 1))
        self.action_space = gym.Space((get_action_dim("ant"), 1))

    def reset(self):
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.current_step += 1
            if done or self.current_step >= self.max_episode_steps:
                break
        return state, total_reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

def create_ant_maze_env():
    env = make_env(
        seed=42,
        maze_name="easy7x7a",
        train_visual_styles=True,
        random_start_end=True,
        propagate_seed_to_env=True
    )
    
    env = CustomActionRepeatWrapper(env, repeat=4, max_episode_steps=1000)
    env = DummyVecEnv([lambda: env]) #BUG: compatibility not working
    return env

env = create_ant_maze_env()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ant_maze_tensorboard/")
model.learn(total_timesteps=1_000_000)
model.save("ppo_ant_maze_expert")

# after fixing current issues, the model will be loaded into `dadagger.py` for expert queries:
# model = PPO.load("ppo_ant_maze_expert", env=env)

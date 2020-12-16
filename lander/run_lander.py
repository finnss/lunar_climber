import gym
from lunar_lander_env import LunarLander
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# env = gym.make('LunarLander-v2')
env = LunarLander()
# Load the trained agent
model = DQN.load("lunar_lander_dqn")
# model = PPO2.load("lunar_lander_ppo2")

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(
# model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # print('reward', rewards)
    # print('done', dones)
    env.render()

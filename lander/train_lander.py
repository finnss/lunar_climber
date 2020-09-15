import gym
from lunar_lander_env import LunarLander
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv


def train():

    # Create environment
    env = gym.make('LunarLander-v2')
    # n_envs = 4
    # env = SubprocVecEnv([lambda: LunarLander() for i in range(n_envs)])
    # env = LunarLander()

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3,
                prioritized_replay=True,
                verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(2e5))
    # Save the agent
    model.save("lunar_lander_dqn")
    # mean_reward, std_reward = evaluate_policy(
    #     model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    # obs = env.reset()
    print("Model trained!")


if __name__ == '__main__':
    train()

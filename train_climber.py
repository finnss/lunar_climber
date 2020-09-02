# from gym.gym.envs.box2d.lunar_lander import LunarLander
from custom_env.lunar_climber_env import LunarLander

from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt


def train():
    env = LunarLander()

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=0.01,
                # prioritized_replay=True,
                verbose=1)

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10)
    # Train the agent
    model.learn(total_timesteps=int(1e6), log_interval=10)
    # Save the agent
    model.save("lunar_climber")
    # print('evaluating...')
    # rewards = evaluate_policy(
    #     model, model.get_env(), n_eval_episodes=50, return_episode_rewards=True)
    plt.plot(env.all_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Timesteps')
    plt.savefig('stats.png')

    print("Model trained!")


if __name__ == '__main__':
    train()

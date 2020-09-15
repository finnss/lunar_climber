# from gym.gym.envs.box2d.lunar_lander import LunarLander
import argparse
from custom_env.lunar_climber_env import LunarLander
import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from datetime import datetime
import os


def train(algorithm='dqn', timesteps=2e5):
    # env = gym.make('LunarLander-v2')  # This uses the library version of the Lunar Lander env.
    print('algorithm: ', algorithm)
    print('timesteps: ', timesteps)

    learning_rate = 0.001

    if algorithm.lower() == 'dqn':
        env = LunarLander()
        model = DQN('MlpPolicy', env, learning_rate=learning_rate,
                    prioritized_replay=True,
                    verbose=1)
    elif algorithm.lower() == 'ppo2':
        n_envs = 4
        env = SubprocVecEnv([lambda: LunarLander() for i in range(n_envs)])
        model = PPO2('MlpPolicy', env, learning_rate=learning_rate,
                     verbose=1)
    else:
        raise RuntimeError("Unknown algorithm. %s" % algorithm)

    # mean_reward, std_reward = evaluate_policy(
    #     model, model.get_env(), n_eval_episodes=10)

    # Train the agent
    model.learn(total_timesteps=int(float(timesteps)), log_interval=10)
    # Save the agent
    model.save("trained_models/latest")

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    model.save("trained_models/lunar_climber_%s-%s" %
               (algorithm.lower(), dt_string))

    # #lot training progress
    # plt.plot(env.all_rewards)
    # plt.ylabel('Reward')
    # plt.xlabel('Timesteps')
    # plt.savefig('figures/stats-%s.png' % dt_string)

    print("Model trained!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--alg", type=str, default='dqn')
    parser.add_argument("--steps", type=str, default='2e5')
    args = parser.parse_args()
    train(args.alg, args.steps)

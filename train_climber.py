# from gym.gym.envs.box2d.lunar_lander import LunarLander
from custom_env.lunar_climber_env import LunarLander
import gym
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from datetime import datetime
import os
print(os.environ)

def train():
    # env = LunarLander()
    env = gym.make('LunarLander-v2')

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=0.001,
                prioritized_replay=True,
                verbose=1)

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10)
    # Train the agent
    timesteps = os.environ.get('TIMESTEPS')
    timesteps = int(float(timesteps)) if timesteps is not None else 1e6
    print('timesteps %s' % timesteps)
    model.learn(total_timesteps=int(timesteps), log_interval=10)
    # Save the agent
    model.save("trained_models/latest")

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    model.save("trained_models/lunar_climber-%s" % dt_string)
    
    # Plot training progress
    plt.plot(env.all_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Timesteps')
    plt.savefig('figures/stats-%s.png' % dt_string)

    print("Model trained!")


if __name__ == '__main__':
    train()

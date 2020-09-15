# from gym.gym.envs.box2d.lunar_lander import LunarLander
from custom_env.lunar_climber_env import LunarLander
import argparse
import numpy as np
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv


def run(alg='dqn', model_file='latest'):
    alg = alg.lower()
    env = None
    model = None
    state = None
    done = None
    model_path = 'trained_models/%s' % model_file

    if alg == 'dqn':
        env = LunarLander()
        model = DQN.load(model_path)
        obs = env.reset()
    elif alg == 'ppo2':
        env = LunarLander()
        model = PPO2.load(model_path)
        obs = env.reset()
    elif alg == 'ppo2multi':
        n_envs = 4
        env = DummyVecEnv([LunarLander])
        model = PPO2.load(model_path)

        t_obs = env.reset()
        obs = np.zeros((n_envs,) + env.observation_space.shape)
        obs[0, :] = t_obs
        done = [False for _ in range(n_envs)]
    else:
        raise RuntimeError("Undefined algorithm selected (%s)." % alg)
    # mean_reward, std_reward = evaluate_policy(
    #     model, env, n_eval_episodes=10)

    # Enjoy trained agent

    for i in range(1500):
        action, _states = model.predict(obs, state, mask=done)
        new_obs, rewards, dones, info = env.step(action)
        env.render()

        if alg == 'ppo2multi':
            obs[0, :] = new_obs
        else:
            obs = new_obs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--alg", type=str, default='dqn')
    parser.add_argument("--model", type=str, default='latest')
    args = parser.parse_args()
    run(args.alg, args.model)

# from gym.gym.envs.box2d.lunar_lander import LunarLander
from custom_env.lunar_climber_env import LunarLander

from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy


def run():
    env = LunarLander()
    # Load the trained agent
    model = PPO2.load("trained_models/latest")
    # model = DQN.load("trained_models/good_climber_equal_weights")

    # mean_reward, std_reward = evaluate_policy(
    #     model, env, n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print('reward', rewards)
        env.render()


if __name__ == '__main__':
    run()

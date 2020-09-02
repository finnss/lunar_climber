import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3,
            prioritized_replay=True, verbose=1)
# Train the agent
model.learn(total_timesteps=int(3e5))
# Save the agent
model.save("lunar_lander")
print("Model trained!")
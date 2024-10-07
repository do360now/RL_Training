import gymnasium as gym
from stable_baselines3 import PPO
import os

# Model file name
model_filename = "ppo_CartPole-v1.zip"

# Create the environment
env = gym.make('CartPole-v1')

# Load or create a model
if os.path.exists(model_filename):
    print(f"Loading existing model from {model_filename}")
    model = PPO.load(model_filename, env=env)
else:
    print("Creating a new model")
    model = PPO('MlpPolicy', env, verbose=1)

# Train the model (without rendering)
total_timesteps = 1024000
model.learn(total_timesteps=total_timesteps)

# Save the model after training
model.save(model_filename)
print(f"Model saved as {model_filename}")

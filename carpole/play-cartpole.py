import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Model file name
model_filename = "ppo_CartPole-v1.zip"

# Check if model exists
if not os.path.exists(model_filename):
    print(f"Model file {model_filename} not found. Please train the model first.")
    exit()

# Load the pre-trained model
model = PPO.load(model_filename)

# Create the environment with rendering enabled
env = gym.make('CartPole-v1', render_mode='human')

# Play the game for 5 episodes
for episode in range(5):
    obs, info = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)  # Slow down rendering to make it watchable

# Close the environment
env.close()

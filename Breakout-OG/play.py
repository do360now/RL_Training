import gymnasium as gym
from stable_baselines3 import PPO

# Load the pre-trained model
model = PPO.load("ppo_Breakout-v4")

# Load the environment (make sure render mode is 'human' to display the game)
env = gym.make("Breakout-v4", render_mode='human')

# Reset the environment and extract the observation
obs, info = env.reset()  # New Gymnasium reset returns (obs, info)

# Let the model play for some episodes
for episode in range(5):  # Run for 5 episodes
    done = False
    obs, info = env.reset()  # Reset the environment and get obs, info tuple
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Model decides action
        obs, reward, done, truncated, info = env.step(action)  # Perform action
        env.render()  # Display the game window

env.close()

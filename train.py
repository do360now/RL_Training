import gymnasium as gym
from stable_baselines3 import PPO
from asteroid_dodge_env import AsteroidDodgeEnv  # Import your custom environment if needed
import argparse
import os

def choose_environment(env_name, render_mode=None):
    if env_name == "asteroid":
        return AsteroidDodgeEnv()
    else:
        # Set the render mode if specified (e.g., 'human' to display the game)
        return gym.make(env_name, render_mode=render_mode)

def train_model(env, model_name="ppo_model", total_timesteps=100000, render_game=False):
    model = PPO("CnnPolicy", env, verbose=1)

    if os.path.exists(model_name + ".zip"):
        print(f"Loading existing model from {model_name}.zip")
        model = PPO.load(model_name, env=env)  # Load the existing model
    else:
        print(f"Creating new model {model_name}")
        model = PPO("CnnPolicy", env, verbose=1)  # Create a new model if none exists

    # Total steps loop, rendering the game if requested
    for i in range(0, total_timesteps, 2048):  # PPO default batch size is 2048 steps
        model.learn(total_timesteps=2048, reset_num_timesteps=False)
        
        # Render the game every 5000 timesteps to reduce overhead
        if render_game and i % 5000 == 0:
            env.render()  # Render with the correct mode

        # Save model checkpoint every 10,000 timesteps
        if i % 10000 == 0:
            model.save(f"{model_name}_checkpoint_{i}")
            print(f"Checkpoint saved at timestep {i}")

    model.save(model_name)
    print(f"Model saved as {model_name}.zip")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Train PPO model on a given environment")
    parser.add_argument('--env', type=str, default="Breakout-v4", help="Name of the environment (e.g., Breakout-v4)")
    parser.add_argument('--timesteps', type=int, default=100000, help="Total timesteps for training")
    parser.add_argument('--render', action='store_true', help="Render the game during training")
    args = parser.parse_args()

    # Set render mode to 'human' if rendering is enabled
    render_mode = 'human' if args.render else None

    # Choose the environment with the appropriate render mode
    env_name = args.env
    env = choose_environment(env_name, render_mode=render_mode)

    # Train the model
    train_model(env, model_name="ppo_" + env_name, total_timesteps=args.timesteps, render_game=args.render)

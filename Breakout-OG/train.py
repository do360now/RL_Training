import gymnasium as gym
from stable_baselines3 import PPO

import argparse
import os

def choose_environment(env_name, render_mode=None):
    # Set the render mode if specified (e.g., 'human' to display the game)
    return gym.make(env_name, render_mode=render_mode)

def train_model(env, model_name="ppo_model", total_timesteps=51200, render_game=False):
    # Load the existing model if it exists, otherwise create a new one
    if os.path.exists(model_name + ".zip"):
        print(f"Loading existing model from {model_name}.zip")
        model = PPO.load(model_name, env=env)  # Load the existing model
        model.ent_coef = 0.005  # Manually update entropy coefficient to encourage more exploration
        model.learning_rate=1e-4
        model.n_steps=2048
        model.batch_size=512
        model.clip_range = lambda _: 0.2  # Set clip range as a callable that returns a fixed value  # Manually update clip range to encourage more exploration
        model.gamma=0.99
    else:
        print(f"Creating new model {model_name}")
        model = PPO("CnnPolicy", env, verbose=1)  # Create a new model if none exists

    total_trained_timesteps = 0
    render_interval = 2048  # Render every 2048 steps, adjust as needed

    # Training parameters
    total_timesteps = 1024000  # Increase total timesteps for better convergence
    checkpoint_interval = 2048  # Match n_steps for saving checkpoints
    total_trained_timesteps = 0
    target_reward = 6  # Target reward for stopping training early

    # Train the model in chunks, saving every 8192 steps
    while total_trained_timesteps < total_timesteps:
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        total_trained_timesteps += checkpoint_interval

        # Save the checkpoint by overwriting the same model file
        model.save(model_name)
        
    # Print progress
    print(f"Training progress: {total_trained_timesteps}/{total_timesteps} timesteps completed.")

    # Render the game every 2048 timesteps if requested
    if render_game and total_trained_timesteps % render_interval == 0:
        env.render()  # Render with the correct mode


    # Save the final model after training
    model.save(model_name)
    print(f"Final model saved as {model_name}.zip")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Train PPO model on a given environment")
    parser.add_argument('--env', type=str, default="Breakout-v4", help="Name of the environment (e.g., Breakout-v4)")
    parser.add_argument('--timesteps', type=int, default=51200, help="Total timesteps for training")
    parser.add_argument('--render', action='store_true', help="Render the game during training")
    args = parser.parse_args()

    # Set render mode to 'human' if rendering is enabled
    render_mode = 'human' if args.render else None

    # Choose the environment with the appropriate render mode
    env = choose_environment(args.env, render_mode=render_mode)

    # Train the model
    train_model(env, model_name="ppo_" + args.env, total_timesteps=args.timesteps, render_game=args.render)

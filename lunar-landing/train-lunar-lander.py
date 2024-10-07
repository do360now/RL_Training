import gymnasium as gym
from stable_baselines3 import PPO
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Train PPO model on a given environment")
    parser.add_argument('--env', type=str, default="LunarLander-v2", help="Name of the environment (e.g., LunarLander-v2)")
    parser.add_argument('--timesteps', type=int, default=10240000, help="Total timesteps for training")
    parser.add_argument('--checkpoint_interval', type=int, default=8192, help="Checkpoint interval for saving model checkpoints")
    args = parser.parse_args()
    logger.info(f"Arguments received: env={args.env}, timesteps={args.timesteps}, checkpoint_interval={args.checkpoint_interval}")

    # Initialize environment
    env = gym.make(args.env)

    # Check for pre-trained model or create a new one
    model_name = f"ppo_{args.env}"
    model = check_model_file(model_name=model_name, env=env)

    # Train the model
    train_model(model, env, model_name=model_name, total_timesteps=args.timesteps, checkpoint_interval=args.checkpoint_interval)

    # Close the environment
    env.close()


def check_model_file(model_name, env):
    # Check if a pre-trained model exists; if not, create a new one
    model_filename = f"{model_name}.zip"
    if os.path.exists(model_filename):
        logger.info(f"Loading existing model from {model_filename}")
        model = PPO.load(model_filename, env=env)
    else:
        logger.info(f"Creating a new model {model_name}")
        # Instantiate the PPO model
        model = PPO('MlpPolicy', env, verbose=1,
                    learning_rate=1e-4,
                    n_steps=2048,
                    batch_size=128,
                    ent_coef=0.01,
                    gamma=0.99,
                    clip_range=0.2)
    return model


def train_model(model, env, model_name, total_timesteps, checkpoint_interval):
    total_trained_timesteps = 0
    model_filename = f"{model_name}.zip"

    # Train the model in chunks, saving every checkpoint_interval steps
    while total_trained_timesteps < total_timesteps:
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        total_trained_timesteps += checkpoint_interval

        # Save the checkpoint by overwriting the same model file
        model.save(model_filename)
        logger.info(f"Checkpoint saved at {total_trained_timesteps}/{total_timesteps} timesteps")

    # Save the final model
    model.save(model_name)
    logger.info(f"Final model saved as {model_name}.zip")


if __name__ == "__main__":
    main()

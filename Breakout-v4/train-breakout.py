import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env(env_name, render_mode=None):
    def _init():
        logger.info(f"Initializing environment: {env_name} with render_mode={render_mode}")
        return gym.make(env_name, render_mode=render_mode)
    return _init

def train_model(env, model_name="ppo_model", total_timesteps=10240000, checkpoint_interval=2048, render_game=False):
    # Load existing model if it exists
    if os.path.exists(model_name + ".zip"):
        logger.info(f"Loading existing model from {model_name}.zip")
        model = PPO.load(model_name, env=env)
    else:
        logger.info(f"Creating new model {model_name}")
        policy_kwargs = dict(
            net_arch=dict(pi=[1024, 512, 256], vf=[1024, 512, 256])  # Increased network complexity
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=512,
            gamma=0.99,
            ent_coef=0.005,
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device='cuda'
        )
        logger.info(f"New model created with the following parameters: {policy_kwargs}")

    # Train in chunks, save checkpoint after every `checkpoint_interval` timesteps
    total_trained_timesteps = 0
    while total_trained_timesteps < total_timesteps:
        logger.info(f"Starting training from timestep {total_trained_timesteps} to {total_trained_timesteps + checkpoint_interval}")
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        total_trained_timesteps += checkpoint_interval

        # Save the checkpoint by overwriting the same model file
        model.save(model_name)
        logger.info(f"Checkpoint saved at {total_trained_timesteps}/{total_timesteps} timesteps")

    # Save the final model
    model.save(model_name)
    logger.info(f"Final model saved as {model_name}.zip")

if __name__ == "__main__":
    logger.info("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Train PPO model on a given environment")
    parser.add_argument('--env', type=str, default="Breakout-v4", help="Name of the environment (e.g., Breakout-v4)")
    parser.add_argument('--timesteps', type=int, default=10240000, help="Total timesteps for training")
    parser.add_argument('--render', action='store_true', help="Render the game during training")
    args = parser.parse_args()
    logger.info(f"Arguments received: env={args.env}, timesteps={args.timesteps}, render={args.render}")

    render_mode = 'human' if args.render else None
    logger.info(f"Render mode set to: {render_mode}")

    # Create vectorized environment with reduced n_envs for memory control
    logger.info("Creating vectorized environment...")
    env = make_vec_env(make_env(args.env, render_mode=render_mode), n_envs=2)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    logger.info("Environment created and normalized.")

    # Train the model with checkpoint saving
    logger.info("Starting model training...")
    train_model(env, model_name="ppo_" + args.env, total_timesteps=args.timesteps, render_game=args.render)
    logger.info("Training completed.")

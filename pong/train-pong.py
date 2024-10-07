import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
import argparse
import os

# Custom callback to implement a learning rate schedule
class CustomLearningRateCallback(BaseCallback):
    def __init__(self, initial_lr: float, total_timesteps: int, verbose=0):
        super(CustomLearningRateCallback, self).__init__(verbose)
        self.initial_lr = initial_lr
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Update the learning rate based on training progress
        progress = self.num_timesteps / self.total_timesteps
        new_lr = self.initial_lr * (1 - progress)

        # Set new learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr

        if self.verbose > 0:
            print(f"Learning rate updated to {new_lr:.6f}")

        return True

def choose_environment(env_name, render_mode=None):
    if env_name == "asteroid":
        return AsteroidDodgeEnv()  # Assuming a custom environment
    else:
        return gym.make(env_name, render_mode=render_mode)

def make_env(env_name, render_mode=None):
    def _init():
        return choose_environment(env_name, render_mode=render_mode)
    return _init

def train_model(env, model_name="ppo_model", total_timesteps=10240, checkpoint_interval=2048, render_game=False):
    # Load existing model if it exists
    if os.path.exists(model_name + ".zip"):
        print(f"Loading existing model from {model_name}.zip")
        model = PPO.load(model_name, env=env)
        # Update parameters manually after loading the model
        model.ent_coef = 0.5  # Manually update entropy coefficient to encourage more exploration
        model.clip_range = lambda _: 0.2  # Set clip range as a callable that returns a fixed value  # Manually update clip range to encourage more exploration
    else:
        print(f"Creating new model {model_name}")
        policy_kwargs = dict(
            net_arch=dict(pi=[1024, 512, 256], vf=[1024, 512, 256])  # Increased network complexity
        )
        model = PPO(
            "MlpPolicy",  # Switch to "CnnPolicy" if dealing with image input MlpPolicy
            env,
            verbose=1,
            learning_rate=2e-5,
            n_steps=2048,  # Reduced n_steps to control memory usage
            batch_size=256,  # Reduced batch size to control memory usage
            gamma=0.99,
            ent_coef=0.5,  # Increased entropy coefficient to encourage more exploration
            clip_range=0.2,  # Increased clip range to encourage more exploration
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device='cuda'  # Explicitly set device to CUDA
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=4098,
        deterministic=True,
        render=render_game,
        callback_on_new_best=StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5)
    )

    # Custom learning rate callback
    lr_callback = CustomLearningRateCallback(initial_lr=1e-5, total_timesteps=total_timesteps)

    # Train in chunks, save checkpoint after every `checkpoint_interval` timesteps
    total_trained_timesteps = 0
    while total_trained_timesteps < total_timesteps:
        # Train the model for a specified number of timesteps
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False, callback=[eval_callback, lr_callback])
        total_trained_timesteps += checkpoint_interval

        # Save the checkpoint by overwriting the same model file
        model.save(model_name)
        print(f"Checkpoint saved at {total_trained_timesteps}/{total_timesteps} timesteps")

    # Save the final model
    model.save(model_name)
    print(f"Final model saved as {model_name}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO model on a given environment")
    parser.add_argument('--env', type=str, default="Pong-v4", help="Name of the environment (e.g., Pong-v4)")
    parser.add_argument('--timesteps', type=int, default=1024000, help="Total timesteps for training")
    parser.add_argument('--render', action='store_true', help="Render the game during training")
    args = parser.parse_args()

    render_mode = 'human' if args.render else None

    # Create vectorized environment with reduced n_envs for memory control
    env = make_vec_env(make_env(args.env, render_mode=render_mode), n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Train the model with checkpoint saving
    train_model(env, model_name="ppo_" + args.env, total_timesteps=args.timesteps, render_game=args.render)
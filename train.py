"""
Training script for Kuka grasping with PPO (Stable-Baselines3).

Uses Monitor wrapper for logging and EvalCallback for saving the best model.
Logs are written to ./logs/ for TensorBoard and Monitor CSV.
Best model is saved to ./best_model/.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import KukaGraspEnv


def make_env(render_mode="headless"):
    """Create a monitored environment."""
    env = KukaGraspEnv(render_mode=render_mode)
    log_dir = os.path.join("logs", "monitor")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    return env


def main():
    # Training environment (headless for speed)
    env = make_env("headless")

    # Separate eval environment
    eval_env = KukaGraspEnv(render_mode="headless")
    eval_log_dir = os.path.join("logs", "eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_env = Monitor(eval_env, eval_log_dir)

    # EvalCallback: evaluate every 10k steps, save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/eval_results/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # PPO with reasonable defaults for a continuous control task
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # slight entropy bonus to encourage exploration
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    total_timesteps = 300_000  # ~300k steps; increase if needed
    print(f"Starting training for {total_timesteps} timesteps...")
    print("Monitor logs: ./logs/monitor/")
    print("TensorBoard:  tensorboard --logdir ./logs/tensorboard/")
    print("Best model:   ./best_model/best_model.zip")
    print("-" * 50)

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model too
    model.save("final_model")
    print("Training complete. Final model saved to final_model.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

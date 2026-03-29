"""
Evaluate a trained grasping policy.

Loads the best model (or a specified model) and runs N episodes,
printing success rate and average reward. Optionally renders in GUI.
Results are saved to results.txt.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from env import KukaGraspEnv


def evaluate(model_path, n_episodes=20, render=False):
    render_mode = "human" if render else "headless"
    env = KukaGraspEnv(render_mode=render_mode)

    model = PPO.load(model_path)

    rewards = []
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        success = info.get("is_success", False)
        successes.append(success)
        rewards.append(total_reward)
        status = "GRASPED" if success else "failed"
        print(f"Episode {ep + 1:3d}: reward={total_reward:8.2f}  [{status}]")

    env.close()

    avg_reward = np.mean(rewards)
    success_rate = np.mean(successes) * 100

    print("\n" + "=" * 40)
    print(f"Episodes:     {n_episodes}")
    print(f"Avg reward:   {avg_reward:.2f}")
    print(f"Success rate: {success_rate:.1f}%")
    print("=" * 40)

    # Save results
    with open("results.txt", "w") as f:
        f.write("Kuka Grasping Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Average reward: {avg_reward:.2f}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write(f"Min reward: {np.min(rewards):.2f}\n")
        f.write(f"Max reward: {np.max(rewards):.2f}\n")
        f.write(f"Std reward: {np.std(rewards):.2f}\n")
    print("Results saved to results.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained grasping policy")
    parser.add_argument("--model", default="best_model/best_model.zip",
                        help="Path to model file (default: best_model/best_model.zip)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--render", action="store_true",
                        help="Render in GUI mode")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)
